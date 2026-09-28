//! Polls the macOS Accessibility API every 150ms (~6.6 Hz) for the focused
//! UI element's screen bounds and emits Tauri `focus-update` events to the overlay window.
//!
//! Requires Accessibility permission. First launch will prompt; user must grant
//! in System Settings → Privacy & Security → Accessibility.

#![cfg(all(target_os = "macos", feature = "overlay"))]

use std::thread;
use std::time::Duration;

use accessibility_sys::{
    AXIsProcessTrustedWithOptions, AXUIElementCopyAttributeValue,
    AXUIElementCopyParameterizedAttributeValue, AXUIElementCreateSystemWide, AXUIElementGetPid,
    AXUIElementRef, AXValueCreate, AXValueGetValue, AXValueRef,
    kAXBoundsForRangeParameterizedAttribute, kAXErrorSuccess, kAXFocusedApplicationAttribute,
    kAXFocusedUIElementAttribute, kAXPositionAttribute, kAXRoleAttribute,
    kAXRoleDescriptionAttribute, kAXSelectedTextRangeAttribute, kAXSizeAttribute,
    kAXSubroleAttribute, kAXTrustedCheckOptionPrompt, kAXValueAttribute,
    kAXValueTypeCFRange, kAXValueTypeCGPoint, kAXValueTypeCGRect, kAXValueTypeCGSize,
};
use core_foundation::base::{CFIndex, CFRange, CFType};
use core_graphics::geometry::CGRect;
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::CFDictionary;
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::geometry::{CGPoint, CGSize};
use serde::Serialize;
use tauri::{AppHandle, Emitter};
/// Primary overlay label (kept for JS compat; secondaries are overlay-1...).
/// Emits use broadcast so every overlay window receives them.
#[allow(dead_code)]
const OVERLAY_LABEL: &str = "overlay";

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FocusBounds {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_text_field_bounds_pass() {
        assert!(is_plausible_text_field(&FocusBounds { x: 418.0, y: 239.0, w: 521.0, h: 497.0 }));
        assert!(is_plausible_text_field(&FocusBounds { x: 0.0, y: 0.0, w: 200.0, h: 24.0 }));
        // Secondary display: negative x (left-of-primary) is legitimate.
        assert!(is_plausible_text_field(&FocusBounds { x: -2500.0, y: 500.0, w: 800.0, h: 40.0 }));
        assert!(is_plausible_text_field(&FocusBounds { x: -7000.0, y: 300.0, w: 500.0, h: 30.0 }));
    }

    #[test]
    fn axui_garbage_rejected() {
        // Real example seen in the wild — outer scrollview reporting itself.
        assert!(!is_plausible_text_field(&FocusBounds { x: -1.0, y: -17899.0, w: 1711.0, h: 19017.0 }));
        // Tiny zero-sized element (e.g., empty label)
        assert!(!is_plausible_text_field(&FocusBounds { x: 0.0, y: 0.0, w: 4.0, h: 4.0 }));
        // Absurdly tall column (over the 8000px multi-display cap)
        assert!(!is_plausible_text_field(&FocusBounds { x: 0.0, y: 0.0, w: 200.0, h: 10_000.0 }));
        // Just over the widened caps still rejected.
        assert!(!is_plausible_text_field(&FocusBounds { x: 0.0, y: 0.0, w: 8100.0, h: 24.0 }));
        assert!(!is_plausible_text_field(&FocusBounds { x: -9000.0, y: 500.0, w: 800.0, h: 40.0 }));
    }

    #[test]
    fn char_range_to_utf16_after_emoji_prefix() {
        // "🎉 hello" — chars: [🎉,' ','h','e','l','l','o'], utf16: [2,1,1,1,1,1,1].
        // A lint over "hello" is chars [2..7) but UTF-16 [3..8).
        assert_eq!(char_range_to_utf16("🎉 hello", 2, 7), (3, 5));
        // Pure ASCII: identity.
        assert_eq!(char_range_to_utf16("hello", 1, 3), (1, 2));
        // BMP CJK: 1 utf16 unit per char — identity too.
        assert_eq!(char_range_to_utf16("你好 hi", 3, 5), (3, 2));
        // Clamped when the range runs past the text.
        assert_eq!(char_range_to_utf16("ab", 1, 9), (1, 1));
    }

    #[test]
    fn utf16_range_to_char_after_emoji_prefix() {
        // Inverse of the above: AXUI reports UTF-16 [3..8) for "hello".
        assert_eq!(utf16_range_to_char("🎉 hello", 3, 5), (2, 7));
        // Pure ASCII: identity.
        assert_eq!(utf16_range_to_char("hello", 1, 2), (1, 3));
        // BMP CJK: identity too.
        assert_eq!(utf16_range_to_char("你好 hi", 3, 2), (3, 5));
        // Mid-surrogate offset rounds to the containing char, never panics.
        assert_eq!(utf16_offset_to_char("🎉 hello", 1), 0);
        // Out-of-bounds clamps to the char length, never panics.
        assert_eq!(utf16_range_to_char("ab", 99, 5), (2, 2));
        assert_eq!(utf16_range_to_char("🎉 hello", 0, 99), (0, 7));
    }
}

/// A WireLint plus the precomputed screen rect for its character span.
/// `rect` is `None` if AXUI's `kAXBoundsForRangeParameterizedAttribute` failed
/// for this element (common in web text inputs).
#[derive(Serialize, Clone, Debug)]
pub struct PositionedLint {
    #[serde(flatten)]
    pub lint: crate::wire::WireLint,
    pub rect: Option<FocusBounds>,
}

#[derive(Serialize, Clone, Debug)]
pub struct FocusEvent {
    pub bounds: Option<FocusBounds>,
    pub text: Option<String>,
    pub lints: Vec<PositionedLint>,
}

/// Snapshot of the user's current text selection in the focused field.
/// Emitted as `selection-update` events to the overlay so JS can render
/// a Grammarly-style trigger button at the selection's edge.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SelectionEvent {
    /// On-screen bounds of the selected text span. None when nothing is
    /// selected — JS uses this to hide the trigger.
    pub rect: Option<FocusBounds>,
    /// The selected text (capped at 4000 chars to avoid IPC bloat).
    pub text: Option<String>,
    /// Character offsets within the focused field, for apply round-trip.
    pub start: Option<u32>,
    pub end: Option<u32>,
    /// True when the selection exceeded the 4000-char cap — `text` holds
    /// only a prefix, so applying a rewrite of it over the FULL start..end
    /// range would destroy the tail. JS must not offer rewrite for these.
    pub truncated: bool,
}

/// Spawn the polling thread. Returns immediately; logs Accessibility-permission
/// status to stderr.
pub fn spawn(app: AppHandle, config: std::sync::Arc<crate::config::ConfigStore>) {
    thread::Builder::new()
        .name("nib-focus-tracker".into())
        .spawn(move || run(app, config))
        .expect("spawn focus-tracker thread");
}

/// Build a private LintGroup for the tracker thread. Harper's `concurrent`
/// feature is enabled in Cargo.toml so the dictionary is `Send`. Mirrors
/// `state::build_linter` — both call sites must enable the same extra rules
/// so the main-window panel and the overlay surface identical lints.
fn fresh_linter() -> harper_core::linting::LintGroup {
    crate::state::build_linter()
}

fn run(app: AppHandle, config: std::sync::Arc<crate::config::ConfigStore>) {
    // First check WITH prompt: triggers the system "grant Accessibility?"
    // dialog the first time. AXIsProcessTrustedWithOptions does a fresh read
    // each call (unlike AXIsProcessTrusted, which caches per-process).
    if !is_trusted(true) {
        eprintln!(
            "[nib] Accessibility permission NOT granted. \
             System Settings → Privacy & Security → Accessibility — toggle Nib on."
        );
        // Poll every 2s using the fresh-read variant — picks up the grant
        // without requiring a relaunch.
        while !is_trusted(false) {
            thread::sleep(Duration::from_secs(2));
        }
        eprintln!("[nib] Accessibility permission granted; focus tracker starting");
    } else {
        eprintln!("[nib] AXUI trusted; focus tracker starting");
    }

    let system_wide = unsafe { AXUIElementCreateSystemWide() };
    let mut linter = fresh_linter();
    let mut last_bounds: Option<FocusBounds> = None;
    let mut last_text_hash: u64 = 0;
    let mut last_skip: Option<SkipContext> = None;
    let mut last_bundle_id: Option<String> = None;
    let mut tick = 0u32;

    let mut last_paused_state: Option<bool> = None;
    let mut last_selection: Option<SelectionEvent> = None;
    loop {
        thread::sleep(Duration::from_millis(150));
        tick = tick.wrapping_add(1);

        // Pause short-circuit — when the user toggled Nib paused (via tray
        // or settings), we skip every AXUI read.
        let cfg = config.snapshot();
        if cfg.is_paused_now() {
            if last_paused_state != Some(true) {
                eprintln!("[nib] paused — overlay silent until resumed");
                last_paused_state = Some(true);
                last_bounds = None;
                last_text_hash = 0;
                last_skip = None;
                last_bundle_id = None;
                // Clear saved element so a stale handle doesn't outlive pause.
                crate::overlay::engaged_elem::clear();
                // Tell the overlays to hide (broadcast reaches
                // overlay + overlay-1...).
                let _ = app.emit("focus-update", &FocusEvent {
                    bounds: None, text: None, lints: vec![],
                });
            }
            continue;
        } else if last_paused_state == Some(true) {
            eprintln!("[nib] resumed");
            last_paused_state = Some(false);
        }

        let snapshot = focused_snapshot(system_wide, &cfg);

        // Log on every NEW skip context — gives users visible evidence that
        // the engagement filter is working without spamming on every poll.
        //
        // SPECIAL CASE: when the focused app is Nib itself (the user clicked
        // our overlay popover, which activated us), suppress the focus
        // update entirely. Otherwise the empty event clears currentLints in
        // JS and leaves stale underlines on screen when the user later
        // switches to a different app. The cached engaged_elem keeps apply
        // pointing at the right text field.
        if let SnapshotResult::Skip(ctx) = &snapshot {
            if last_skip.as_ref() != Some(ctx) {
                eprintln!(
                    "[nib] focus skipped: bundle={:?} role={:?} subrole={:?} role_desc={:?}",
                    ctx.bundle_id, ctx.role, ctx.subrole, ctx.role_description
                );
                last_skip = Some(ctx.clone());
            }
            if ctx.bundle_id.as_deref() == Some("app.nib") {
                // Don't propagate — let JS keep its prior focused-field state.
                continue;
            }
        } else {
            last_skip = None;
        }

        // Stale-handle guard: the engaged-elem cache must not outlive the
        // focus that produced it. Clear on bundle change and on any
        // non-engageable (Skip) or lost-focus (Empty) snapshot — otherwise
        // apply keeps writing to the previous app's dead element. The
        // app.nib self-focus case above `continue`s early on purpose: the
        // cache must survive our own popover click.
        let cur_bundle: Option<String> = match &snapshot {
            SnapshotResult::Engage(s) => s.bundle_id.clone(),
            SnapshotResult::Skip(ctx) => ctx.bundle_id.clone(),
            SnapshotResult::Empty => None,
        };
        if cur_bundle != last_bundle_id {
            crate::overlay::engaged_elem::clear();
            last_bundle_id = cur_bundle.clone();
        }
        if matches!(&snapshot, SnapshotResult::Skip(_) | SnapshotResult::Empty) {
            crate::overlay::engaged_elem::clear();
        }

        let snap_opt = match snapshot {
            SnapshotResult::Engage(s) => Some(s),
            SnapshotResult::Skip(_) | SnapshotResult::Empty => None,
        };
        let bounds_changed = snap_opt.as_ref().map(|s| &s.bounds) != last_bounds.as_ref();
        let text_hash = snap_opt
            .as_ref()
            .and_then(|s| s.text.as_deref())
            .map(simple_hash)
            .unwrap_or(0);
        let text_changed = text_hash != last_text_hash;

        // Extract elem early so we can poll selection on EVERY tick — the
        // user can drag a selection without changing bounds or text, and
        // we still need to surface the selection-update event so the JS
        // trigger button appears.
        let (bounds, text, elem_ref) = match snap_opt {
            Some(s) => (Some(s.bounds), s.text, s.elem),
            None => (None, None, std::ptr::null_mut()),
        };

        // Selection — read on EVERY tick, emit selection-update event so JS
        // can show/hide the Grammarly-style trigger button. This runs even
        // when bounds/text are unchanged (the common "user drags selection"
        // path), which is why it must come BEFORE the dedupe `continue`.
        let selection = if !elem_ref.is_null() {
            read_selection(elem_ref, text.as_deref())
        } else {
            None
        };
        let sel_event = match &selection {
            Some(s) if s.length >= 3 => SelectionEvent {
                rect: s.rect.clone(),
                text: s.text.clone(),
                start: Some(s.start),
                end: Some(s.start + s.length),
                truncated: s.truncated,
            },
            _ => SelectionEvent {
                rect: None,
                text: None,
                start: None,
                end: None,
                truncated: false,
            },
        };
        if last_selection.as_ref() != Some(&sel_event) {
            // Broadcast so every overlay window (overlay, overlay-1...) stays in sync.
            let _ = app.emit("selection-update", &sel_event);
            last_selection = Some(sel_event.clone());
        }

        if !bounds_changed && !text_changed {
            if tick.is_multiple_of(60) {
                eprintln!("[nib] focus-tracker heartbeat (no change in 9s)");
            }
            // Still need to release / store the elem_ref. If we just
            // CFRelease via the cache, the focus-update path that owns
            // it stays consistent.
            if !elem_ref.is_null() {
                crate::overlay::engaged_elem::store(elem_ref as *mut std::ffi::c_void);
            }
            continue;
        }

        // Lint the text if we got any. Harper takes ~5-30ms per check on
        // typical sentence-length input. Honor the user's personal
        // dictionary so e.g. "BitNet" or "abgunaydin" doesn't get flagged.
        let raw_lints: Vec<crate::wire::WireLint> = match &text {
            Some(t) if !t.is_empty() => {
                crate::wire::check_text_filtered(&mut linter, t, &cfg.ignored_words)
            }
            _ => Vec::new(),
        };

        // For each lint, ask AXUI where on the screen those characters sit.
        // Many text fields (Cocoa NSTextView, AppKit) implement
        // kAXBoundsForRangeParameterizedAttribute; web text inputs often don't.
        let lints: Vec<PositionedLint> = raw_lints
            .into_iter()
            .map(|lint| {
                let rect = if !elem_ref.is_null() {
                    match &text {
                        Some(t) => {
                            let (u16_start, u16_len) =
                                char_range_to_utf16(t, lint.start, lint.end);
                            bounds_for_range(elem_ref, u16_start, u16_len)
                        }
                        None => None,
                    }
                } else {
                    None
                };
                PositionedLint { lint, rect }
            })
            .collect();

        // Transfer the AXUIElement handle into the shared engaged-elem cache
        // so apply.rs can write to it even after the user clicks our overlay
        // (which would shift live AXUI focus away from their text field).
        // store() takes ownership of the retain — no CFRelease here.
        if !elem_ref.is_null() {
            crate::overlay::engaged_elem::store(elem_ref as *mut std::ffi::c_void);
        }

        let rects_resolved = lints.iter().filter(|l| l.rect.is_some()).count();
        eprintln!(
            "[nib] focus-update: bounds={} text_len={} lints={} rects={}/{}",
            bounds
                .as_ref()
                .map(|b| format!("x={:.0} y={:.0} w={:.0} h={:.0}", b.x, b.y, b.w, b.h))
                .unwrap_or_else(|| "<none>".into()),
            text.as_deref().map(|t| t.chars().count()).unwrap_or(0),
            lints.len(),
            rects_resolved,
            lints.len()
        );

        let payload = FocusEvent {
            bounds: bounds.clone(),
            text: text.clone(),
            lints,
        };
        // Broadcast to all overlay windows (overlay, overlay-1...).
        // Each window's JS filters by its own viewport.
        if let Err(e) = app.emit("focus-update", &payload) {
            eprintln!("[nib] emit focus-update failed: {e}");
        }

        last_bounds = bounds;
        last_text_hash = text_hash;
    }
}

/// Internal snapshot of the user's selection in the focused field.
struct SelectionSnapshot {
    start: u32,
    length: u32,
    rect: Option<FocusBounds>,
    text: Option<String>,
    /// Selection exceeded the text-snapshot cap — `text` is a prefix only.
    truncated: bool,
}

/// Read `kAXSelectedTextRangeAttribute` from the focused element and
/// resolve it to screen bounds + selected text. Returns None when
/// nothing is selected (or AXUI rejects the attribute).
fn read_selection(elem: AXUIElementRef, full_text: Option<&str>) -> Option<SelectionSnapshot> {
    let val = copy_attr(elem, kAXSelectedTextRangeAttribute)?;
    let mut range = core_foundation::base::CFRange { location: 0, length: 0 };
    let ok = unsafe {
        AXValueGetValue(
            val as AXValueRef,
            kAXValueTypeCFRange,
            &mut range as *mut _ as *mut std::ffi::c_void,
        )
    };
    unsafe { CFRelease(val as core_foundation::base::CFTypeRef) };
    if !ok || range.length < 1 {
        return None;
    }
    // AXUI ranges are UTF-16 code units — keep them as-is for the
    // bounds_for_range call below, convert to char offsets for slicing
    // and the wire format.
    let u16_start = range.location.max(0) as usize;
    let u16_len = range.length.max(0) as usize;

    // Bounds: same parameterized AXUI call we use for lint rendering.
    // Takes UTF-16 offsets, so pass the raw AXUI range through.
    let rect = bounds_for_range(elem, u16_start, u16_len);

    // Convert UTF-16 -> char offsets before slicing `chars()` and before
    // building the wire start/end (apply round-trips in char offsets).
    // Without the field text we can't convert — fall back to the raw
    // offsets (best effort) so the trigger still appears.
    let (char_start, char_end) = match full_text {
        Some(t) => utf16_range_to_char(t, u16_start, u16_len),
        None => (u16_start, u16_start.saturating_add(u16_len)),
    };
    let char_len = char_end.saturating_sub(char_start);

    // Selected text: slice the full field text by character (Unicode-safe).
    // Cap at 4000 chars to avoid IPC bloat for accidental "Cmd+A" cases —
    // and FLAG the cap, because rewriting a prefix while replacing the
    // full range would silently delete the selection's tail.
    let truncated = char_len > 4000;
    let selected_text = full_text.and_then(|t| {
        let chars: Vec<char> = t.chars().collect();
        let s = char_start.min(chars.len());
        let e = char_end.min(chars.len()).max(s);
        if s >= chars.len() && char_len > 0 {
            return None;
        }
        let len = (e - s).min(4000);
        Some(chars[s..s + len].iter().collect::<String>())
    });

    Some(SelectionSnapshot {
        start: char_start as u32,
        length: char_len as u32,
        rect,
        text: selected_text,
        truncated,
    })
}

/// Fresh-read trust check. `prompt=true` shows the system dialog if not
/// already granted (use sparingly — only on startup).
fn is_trusted(prompt: bool) -> bool {
    let key = unsafe { CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt as CFStringRef) };
    let val = CFBoolean::from(prompt);
    let opts = CFDictionary::from_CFType_pairs(&[(key, val)]);
    unsafe { AXIsProcessTrustedWithOptions(opts.as_concrete_TypeRef() as _) }
}

struct FocusSnapshot {
    bounds: FocusBounds,
    text: Option<String>,
    /// Borrowed AXUIElementRef — caller must CFRelease when done with it.
    elem: AXUIElementRef,
    /// Owning app's bundle id — used to drop the cached engaged-elem on
    /// app switch so apply can't write to a stale handle.
    bundle_id: Option<String>,
}

/// Identifies a focus context the engagement policy rejected. The run loop
/// uses this for log-on-change diagnostics so users can see *why* an app
/// was skipped without spamming the log on every poll.
#[derive(Clone, PartialEq, Debug)]
pub struct SkipContext {
    pub bundle_id: Option<String>,
    pub role: Option<String>,
    pub subrole: Option<String>,
    pub role_description: Option<String>,
}

enum SnapshotResult {
    Engage(FocusSnapshot),
    Skip(SkipContext),
    Empty,
}

fn focused_snapshot(
    system_wide: AXUIElementRef,
    config_snap: &crate::config::Config,
) -> SnapshotResult {
    let Some(focused_app) = copy_attr(system_wide, kAXFocusedApplicationAttribute) else {
        return SnapshotResult::Empty;
    };
    let bundle_id = bundle_id_for_app(focused_app as AXUIElementRef);
    let focused_elem = copy_attr(focused_app as AXUIElementRef, kAXFocusedUIElementAttribute);
    unsafe { CFRelease(focused_app) };
    let Some(focused_elem) = focused_elem else {
        return SnapshotResult::Empty;
    };
    let elem_ref = focused_elem as AXUIElementRef;

    let role = copy_string_attr(elem_ref, kAXRoleAttribute);
    let subrole = copy_string_attr(elem_ref, kAXSubroleAttribute);
    let role_description = copy_string_attr(elem_ref, kAXRoleDescriptionAttribute);

    // Password / secure fields are non-negotiable — checked BEFORE the
    // per-app override so a ForceAllow can never make Nib read, lint,
    // broadcast, or journal secure text entry.
    let is_secure = crate::overlay::engagement_policy::is_secure_field(
        role.as_deref(),
        subrole.as_deref(),
    );
    // Per-app override (set via Settings UI). ForceDeny always skips;
    // ForceAllow bypasses the rest of the engagement policy.
    let user_override = bundle_id
        .as_deref()
        .and_then(|bid| config_snap.app_override(bid));
    let engage = !is_secure
        && match user_override {
            Some(crate::config::AppOverride::ForceDeny) => false,
            Some(crate::config::AppOverride::ForceAllow) => true,
            None => crate::overlay::engagement_policy::is_engageable(
                role.as_deref(),
                subrole.as_deref(),
                role_description.as_deref(),
                bundle_id.as_deref(),
            ),
        };
    if !engage {
        unsafe { CFRelease(focused_elem) };
        return SnapshotResult::Skip(SkipContext { bundle_id, role, subrole, role_description });
    }

    let pos = copy_axvalue_cgpoint(elem_ref, kAXPositionAttribute);
    let size = copy_axvalue_cgsize(elem_ref, kAXSizeAttribute);
    let text = copy_string_attr(elem_ref, kAXValueAttribute);
    // NOTE: we don't release focused_elem here — caller takes ownership and
    // releases after using it for per-lint bounds lookups.

    let (p, s) = match (pos, size) {
        (Some(p), Some(s)) => (p, s),
        _ => {
            unsafe { CFRelease(focused_elem) };
            return SnapshotResult::Empty;
        }
    };
    let b = FocusBounds {
        x: p.x,
        y: p.y,
        w: s.width,
        h: s.height,
    };
    if !is_plausible_text_field(&b) {
        unsafe { CFRelease(focused_elem) };
        return SnapshotResult::Empty;
    }
    SnapshotResult::Engage(FocusSnapshot {
        bounds: b,
        text,
        elem: elem_ref,
        bundle_id,
    })
}

/// Convert a [start, end) *char* range into UTF-16 (location, length) for
/// AX CFRanges. Harper lints speak char offsets but AXUI speaks UTF-16 code
/// units — with any non-BMP char (emoji) before the span, raw char offsets
/// point at the wrong characters. Clamps to the text length. Mirrors
/// `overlay::apply::char_range_to_utf16`.
fn char_range_to_utf16(text: &str, start: usize, end: usize) -> (usize, usize) {
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
    (u16_start, u16_len)
}

/// Map a single UTF-16 code-unit offset to a char offset, clamping to
/// `chars().count()`. A mid-surrogate offset (invalid from AXUI, but cheap
/// to guard) rounds down to the containing char. Never panics.
fn utf16_offset_to_char(text: &str, u16_off: usize) -> usize {
    let mut cum: usize = 0;
    for (i, c) in text.chars().enumerate() {
        if u16_off == cum {
            return i;
        }
        let w = c.len_utf16();
        if u16_off < cum + w {
            return i;
        }
        cum += w;
    }
    text.chars().count()
}

/// Convert an AXUI UTF-16 (location, length) range into a char
/// [start, end) range for slicing `chars()` and for the SelectionEvent wire
/// format. Clamps safely, never panics.
fn utf16_range_to_char(text: &str, u16_start: usize, u16_len: usize) -> (usize, usize) {
    let u16_end = u16_start.saturating_add(u16_len);
    let start = utf16_offset_to_char(text, u16_start);
    let end = utf16_offset_to_char(text, u16_end);
    (start, end.max(start))
}

/// Ask AXUI: where on the screen does character range [start..start+length)
/// of this element sit? Used for inline underline rendering. Private —
/// takes a raw AXUIElementRef whose validity only this module's poll
/// loop can guarantee.
fn bounds_for_range(
    element: AXUIElementRef,
    start: usize,
    length: usize,
) -> Option<FocusBounds> {
    if length == 0 {
        return None;
    }
    let range = CFRange {
        location: start as CFIndex,
        length: length as CFIndex,
    };
    let range_val: AXValueRef = unsafe {
        AXValueCreate(
            kAXValueTypeCFRange,
            &range as *const _ as *const std::ffi::c_void,
        )
    };
    if range_val.is_null() {
        return None;
    }

    let attr_cf = CFString::new(kAXBoundsForRangeParameterizedAttribute);
    let mut out: core_foundation::base::CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyParameterizedAttributeValue(
            element,
            attr_cf.as_concrete_TypeRef(),
            range_val as core_foundation::base::CFTypeRef,
            &mut out,
        )
    };
    unsafe { CFRelease(range_val as core_foundation::base::CFTypeRef) };
    if err != kAXErrorSuccess || out.is_null() {
        return None;
    }
    let mut rect = CGRect::new(
        &core_graphics::geometry::CGPoint::new(0.0, 0.0),
        &core_graphics::geometry::CGSize::new(0.0, 0.0),
    );
    let ok = unsafe {
        AXValueGetValue(
            out as AXValueRef,
            kAXValueTypeCGRect,
            &mut rect as *mut _ as *mut std::ffi::c_void,
        )
    };
    unsafe { CFRelease(out) };
    if !ok {
        return None;
    }
    Some(FocusBounds {
        x: rect.origin.x,
        y: rect.origin.y,
        w: rect.size.width,
        h: rect.size.height,
    })
}

/// Resolve the bundle identifier of the app behind an AX application
/// element via pid → NSRunningApplication. Returns None for processes
/// without a registered bundle (rare; daemons, helper procs).
fn bundle_id_for_app(app_elem: AXUIElementRef) -> Option<String> {
    use objc2_app_kit::NSRunningApplication;
    // pid_t is i32 on macOS; we're cfg-gated to macOS only.
    let mut pid: i32 = 0;
    let err = unsafe { AXUIElementGetPid(app_elem, &mut pid) };
    if err != kAXErrorSuccess || pid <= 0 {
        return None;
    }
    let app = NSRunningApplication::runningApplicationWithProcessIdentifier(pid)?;
    let s = app.bundleIdentifier()?;
    Some(s.to_string())
}

/// Read a CFString-valued attribute off the focused element. Returns None
/// for non-text elements (the attribute is wrong type or absent).
fn copy_string_attr(element: AXUIElementRef, attr_name: &str) -> Option<String> {
    let raw = copy_attr(element, attr_name)?;
    // Verify it's actually a CFString before unsafe-wrapping.
    let cf_any = unsafe { CFType::wrap_under_create_rule(raw) };
    cf_any
        .downcast::<CFString>()
        .map(|s| s.to_string())
}

/// Cheap hash for change detection on potentially-large text snapshots.
fn simple_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Reject AXUI bounds that obviously aren't a real text input — outer
/// scrollviews and window background elements like to report
/// `x=-1 y=-17899 w=1711 h=19017` (giant rectangle stretching off-screen).
/// Bounds are global desktop coords, so secondary displays legitimately sit
/// at negative x (left-of-primary) with widths/heights up to a full 8K
/// display. Still rejects absurd areas (e.g. 1711x19017).
fn is_plausible_text_field(b: &FocusBounds) -> bool {
    // Multi-display aware: x may sit a full display left of primary,
    // y spans stacked displays, and w/h cover up to 8K panels.
    let on_screen_y = b.y > -8000.0 && b.y < 8000.0;
    let on_screen_x = b.x > -8000.0 && b.x < 8000.0;
    let sane_w = b.w >= 16.0 && b.w <= 8000.0;
    let sane_h = b.h >= 8.0 && b.h <= 8000.0;
    on_screen_x && on_screen_y && sane_w && sane_h
}

/// Copy an attribute and return the raw CFTypeRef (caller MUST release).
/// Returns None on any AX error.
fn copy_attr(element: AXUIElementRef, attr_name: &str) -> Option<CFTypeRef> {
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

fn copy_axvalue_cgpoint(element: AXUIElementRef, attr: &str) -> Option<CGPoint> {
    let val = copy_attr(element, attr)? as AXValueRef;
    let mut p = CGPoint { x: 0.0, y: 0.0 };
    let ok = unsafe {
        AXValueGetValue(
            val,
            kAXValueTypeCGPoint,
            &mut p as *mut _ as *mut std::ffi::c_void,
        )
    };
    unsafe { CFRelease(val as CFTypeRef) };
    if ok { Some(p) } else { None }
}

fn copy_axvalue_cgsize(element: AXUIElementRef, attr: &str) -> Option<CGSize> {
    let val = copy_attr(element, attr)? as AXValueRef;
    let mut s = CGSize { width: 0.0, height: 0.0 };
    let ok = unsafe {
        AXValueGetValue(
            val,
            kAXValueTypeCGSize,
            &mut s as *mut _ as *mut std::ffi::c_void,
        )
    };
    unsafe { CFRelease(val as CFTypeRef) };
    if ok { Some(s) } else { None }
}
