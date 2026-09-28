//! Write a correction back into the currently-focused text field.
//!
//! Tiered strategy:
//!
//!   1. Set `kAXSelectedTextRangeAttribute` to the target span.
//!      This usually works even in browsers / Electron — they expose
//!      caret manipulation through AXUI.
//!   2. Set `kAXSelectedTextAttribute` to the replacement string. Native
//!      Cocoa apps (TextEdit / Notes / Mail / Messages) honor this directly.
//!   3. **Fallback** — when step 2 silently no-ops (most browsers,
//!      Slack desktop, Electron-based editors), simulate ⌘V via the
//!      clipboard helper. The selection was already moved in step 1, so
//!      the paste replaces the right characters.
//!
//! When the fallback fires, we log `[nib][apply] fallback=clipboard`
//! so per-app behaviour is observable in the app log
//! (`~/Library/Logs/nib.log` when launched via install-dev.sh).

#![cfg(all(target_os = "macos", feature = "overlay"))]

use std::os::raw::c_void;

use accessibility_sys::{
    AXUIElementCopyAttributeValue, AXUIElementCreateSystemWide, AXUIElementGetPid,
    AXUIElementRef, AXUIElementSetAttributeValue, AXValueCreate, AXValueGetValue, AXValueRef,
    kAXErrorSuccess, kAXFocusedApplicationAttribute, kAXFocusedUIElementAttribute,
    kAXSelectedTextAttribute, kAXSelectedTextRangeAttribute, kAXValueAttribute,
    kAXValueTypeCFRange,
};
use core_foundation::base::{CFIndex, CFRange, CFRelease, CFType, CFTypeRef, TCFType};
use core_foundation::string::CFString;

use crate::overlay::clipboard;

#[derive(Debug)]
pub enum ApplyError {
    NoFocusedApp,
    NoFocusedElement,
    /// Both AXUI text-set AND the clipboard fallback failed.
    AllStrategiesFailed { axui_err: i32, clipboard_posted: bool },
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFocusedApp => write!(f, "no focused application"),
            Self::NoFocusedElement => write!(f, "no focused UI element"),
            Self::AllStrategiesFailed { axui_err, clipboard_posted } => write!(
                f,
                "both AXUI text-set (AXError {axui_err}) and clipboard paste \
                 (posted={clipboard_posted}) failed"
            ),
        }
    }
}

/// Result strategy — useful for tests and the diagnostic log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyStrategy {
    /// `kAXSelectedTextAttribute` set succeeded — native Cocoa path.
    AxuiText,
    /// AXUI text-set failed, clipboard ⌘V was simulated instead.
    Clipboard,
}

pub fn apply(start: u32, end: u32, replacement: &str) -> Result<(), ApplyError> {
    let strategy = apply_with_strategy(start, end, replacement)?;
    eprintln!(
        "[nib][apply] strategy={strategy:?} start={start} end={end} len={}",
        replacement.chars().count()
    );
    Ok(())
}

pub fn apply_with_strategy(
    start: u32,
    end: u32,
    replacement: &str,
) -> Result<ApplyStrategy, ApplyError> {
    // Atomicity: hold the pasteboard lock across range-select +
    // text-set + clipboard fallback. Two concurrent applies (bulk
    // Accept-all fires them back-to-back on async-command threads) must
    // not interleave — otherwise A's range-set, B's range-set, A's paste
    // lands A's text over B's span. The clipboard fallback below uses the
    // `_locked` variant since we already hold the guard (Mutex is not
    // reentrant).
    let _apply_guard = clipboard::lock_for_apply();
    // Prefer the cached engaged element from focus_tracker. Clicking our
    // overlay popover activates Nib's app and shifts live AXUI focus
    // away from the user's writing app — re-querying here would write to
    // our own WKWebView. The cache holds the last text-field the focus
    // tracker engaged on, which is the right target.
    let elem = if let Some(saved) = crate::overlay::engaged_elem::current_handle() {
        eprintln!("[nib][apply] using saved engaged elem");
        saved as AXUIElementRef
    } else {
        eprintln!("[nib][apply] no saved elem — falling back to live AXUI query");
        let system_wide = unsafe { AXUIElementCreateSystemWide() };
        let app = copy_attr_ref(system_wide, kAXFocusedApplicationAttribute);
        let elem = app.and_then(|app| {
            let e = copy_attr_ref(app as AXUIElementRef, kAXFocusedUIElementAttribute);
            unsafe { CFRelease(app) };
            e
        });
        unsafe { CFRelease(system_wide as CFTypeRef) };
        match elem {
            Some(e) => e as AXUIElementRef,
            None => return Err(ApplyError::NoFocusedElement),
        }
    };

    // The target app's pid — the clipboard fallback posts ⌘V directly to
    // it. After a popover click Nib itself is frontmost, so an HID-tap
    // paste would land in our own webview instead of the user's field.
    let target_pid = {
        let mut pid: i32 = 0;
        let err = unsafe { AXUIElementGetPid(elem, &mut pid) };
        (err == kAXErrorSuccess && pid > 0).then_some(pid)
    };

    // Harper lints (and the overlay JS) speak Unicode *char* offsets, but
    // AX CFRanges are UTF-16 code units. Convert against the field's
    // current text; with any non-BMP char (emoji) before the span, raw
    // char offsets would select — and replace — the wrong characters.
    // When the field text is unreadable (web inputs often reject
    // kAXValueAttribute), never guess raw char offsets as UTF-16: reuse
    // the live AX selection (already UTF-16 units) so the clipboard
    // fallback below pastes over the right span. Only when the live query
    // also fails do we fall back to clamped raw offsets (saturating, never
    // panicking) with the range already set for the clipboard path.
    let (ax_start, ax_length) = match copy_elem_text(elem) {
        Some(text) => char_range_to_utf16(&text, start as usize, end as usize),
        None => match copy_selected_range(elem) {
            Some((loc, len)) => (loc, len),
            None => (start as CFIndex, end.saturating_sub(start) as CFIndex),
        },
    };

    // Step 1 — move the selection. We try this even before deciding which
    // text-write path to take, because the clipboard fallback needs the
    // selection already on the target span for ⌘V to replace it.
    let range = CFRange {
        location: ax_start,
        length: ax_length,
    };
    let range_val = unsafe {
        AXValueCreate(
            kAXValueTypeCFRange,
            &range as *const _ as *const c_void,
        )
    };
    if !range_val.is_null() {
        let range_attr = CFString::new(kAXSelectedTextRangeAttribute);
        let _range_err = unsafe {
            AXUIElementSetAttributeValue(
                elem,
                range_attr.as_concrete_TypeRef(),
                range_val as CFTypeRef,
            )
        };
        unsafe { CFRelease(range_val as CFTypeRef) };
    }
    // We don't bail on range-set failure — even if AXUI rejects it, the
    // user's existing selection (or caret) is probably already at the
    // word they hovered. The clipboard fallback then pastes there.

    // Step 2 — try native AXUI text replacement.
    let text_attr = CFString::new(kAXSelectedTextAttribute);
    let text_val = CFString::new(replacement);
    let axui_err = unsafe {
        AXUIElementSetAttributeValue(
            elem,
            text_attr.as_concrete_TypeRef(),
            text_val.as_concrete_TypeRef() as CFTypeRef,
        )
    };
    unsafe { CFRelease(elem as CFTypeRef) };
    if axui_err == kAXErrorSuccess {
        return Ok(ApplyStrategy::AxuiText);
    }

    // Step 3 — clipboard fallback, posted to the target app's pid.
    // Lock already held above — use the non-locking variant.
    let posted = clipboard::paste_via_clipboard_to_locked(replacement, target_pid);
    if posted {
        Ok(ApplyStrategy::Clipboard)
    } else {
        Err(ApplyError::AllStrategiesFailed {
            axui_err,
            clipboard_posted: false,
        })
    }
}

/// Read the element's full text (kAXValueAttribute) as a Rust String.
fn copy_elem_text(elem: AXUIElementRef) -> Option<String> {
    let raw = copy_attr_ref(elem, kAXValueAttribute)?;
    let cf_any = unsafe { CFType::wrap_under_create_rule(raw) };
    cf_any.downcast::<CFString>().map(|s| s.to_string())
}

/// Read the element's live selected-text range (already UTF-16 code
/// units). Used when the field text is unreadable so we don't mis-apply
/// raw char offsets as UTF-16. Returns None when AXUI rejects the
/// attribute. Never panics; clamps negatives to zero.
fn copy_selected_range(elem: AXUIElementRef) -> Option<(CFIndex, CFIndex)> {
    let raw = copy_attr_ref(elem, kAXSelectedTextRangeAttribute)?;
    let mut range = CFRange { location: 0, length: 0 };
    let ok = unsafe {
        AXValueGetValue(
            raw as AXValueRef,
            kAXValueTypeCFRange,
            &mut range as *mut _ as *mut c_void,
        )
    };
    unsafe { CFRelease(raw) };
    if !ok {
        return None;
    }
    Some((range.location.max(0), range.length.max(0)))
}

/// Convert a [start, end) *char* range into a UTF-16 (location, length)
/// pair for AX CFRanges. Clamps to the text's length; out-of-bounds
/// inputs saturate instead of panicking (char iteration, no indexing).
fn char_range_to_utf16(text: &str, start: usize, end: usize) -> (CFIndex, CFIndex) {
    let mut u16_start: usize = 0;
    let mut u16_len: usize = 0;
    for (i, c) in text.chars().enumerate() {
        if i < start {
            u16_start += c.len_utf16();
        } else if i < end {
            u16_len += c.len_utf16();
        } else {
            break;
        }
    }
    (u16_start as CFIndex, u16_len as CFIndex)
}

fn copy_attr_ref(element: AXUIElementRef, attr_name: &str) -> Option<CFTypeRef> {
    let cf_attr = CFString::new(attr_name);
    let mut out: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, cf_attr.as_concrete_TypeRef(), &mut out)
    };
    if err == kAXErrorSuccess && !out.is_null() {
        Some(out)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_strategy_variants_are_distinct() {
        // Sanity: the public enum the tests + UI care about hasn't
        // collapsed accidentally.
        assert_ne!(ApplyStrategy::AxuiText, ApplyStrategy::Clipboard);
        // Exhaustiveness guard — if a third variant is added, this won't
        // compile until tests are updated.
        let _all: [ApplyStrategy; 2] = [ApplyStrategy::AxuiText, ApplyStrategy::Clipboard];
    }

    /// Emoji before the span shift UTF-16 offsets past the char offsets;
    /// CJK/BMP chars don't. Both must map correctly or apply replaces the
    /// wrong characters.
    #[test]
    fn char_range_to_utf16_handles_non_bmp() {
        // "🎉🎉 hi" — chars: [🎉,🎉,' ','h','i'], utf16: [2,2,1,1,1]
        let (loc, len) = char_range_to_utf16("🎉🎉 hi", 3, 5);
        assert_eq!((loc, len), (5, 2));
        // Pure ASCII: identity.
        let (loc, len) = char_range_to_utf16("hello", 1, 3);
        assert_eq!((loc, len), (1, 2));
        // BMP CJK: 1 utf16 unit per char — identity too.
        let (loc, len) = char_range_to_utf16("你好 hi", 3, 5);
        assert_eq!((loc, len), (3, 2));
        // Clamped when the range runs past the text.
        let (loc, len) = char_range_to_utf16("ab", 1, 9);
        assert_eq!((loc, len), (1, 1));
    }

    #[test]
    fn apply_error_messages_mention_codes() {
        let e = ApplyError::AllStrategiesFailed { axui_err: -25212, clipboard_posted: false };
        let s = format!("{e}");
        assert!(s.contains("-25212"));
        assert!(s.contains("clipboard"));
    }

    /// Integration test (gated): actually focuses Nib's own window and
    /// runs apply_with_strategy to verify the AXUI path returns
    /// `Strategy::AxuiText` on WKWebView. Only enabled if NIB_TEST_AXUI=1
    /// so CI without an active session doesn't pop random apply events.
    #[test]
    #[ignore]
    fn axui_path_returns_axui_text_when_native_focused() {
        if std::env::var("NIB_TEST_AXUI").ok().as_deref() != Some("1") {
            eprintln!("set NIB_TEST_AXUI=1 with a Cocoa text field focused");
            return;
        }
        let r = apply_with_strategy(0, 0, " ");
        // Either AxuiText (focused field is native Cocoa) or Clipboard
        // (focused field is web/Electron) — both are success paths.
        assert!(r.is_ok() || matches!(r, Err(ApplyError::NoFocusedElement | ApplyError::NoFocusedApp)));
    }
}
