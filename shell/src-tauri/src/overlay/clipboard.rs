//! NSPasteboard save/set/restore + simulated ⌘V for the apply fallback.
//!
//! Used when AXUI's `kAXSelectedTextAttribute` write silently no-ops
//! (Safari, Chrome, every Electron app). The trick: select the span via
//! `kAXSelectedTextRangeAttribute` (this DOES work in most browsers),
//! stash the user's pasteboard, push our replacement, simulate ⌘V, then
//! restore.
//!
//! Correctness rules learned the hard way:
//!   - The whole save→set→paste→restore sequence holds a process-wide
//!     lock and restores SYNCHRONOUSLY. The old fire-and-forget restore
//!     timer raced back-to-back applies ("Accept all") — apply N's
//!     restore fired between apply N+1's set and its paste, so the wrong
//!     text landed in the user's document.
//!   - The snapshot captures EVERY pasteboard type's data, not just the
//!     string — a copied image/file/rich-text no longer gets destroyed.
//!   - Everything we place on the pasteboard is marked with
//!     `org.nspasteboard.ConcealedType` so clipboard managers (Maccy,
//!     Paste, Alfred, Raycast) don't permanently record the user's
//!     selections and rewrites.
//!
//! All operations are best-effort — never panic, never break the flow.

#![cfg(all(target_os = "macos", feature = "overlay"))]

use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use objc2::rc::Retained;
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_foundation::{NSArray, NSData, NSString};

/// Virtual key code for the "V" key on a US ANSI keyboard. From
/// `<HIToolbox/Events.h>` (`kVK_ANSI_V`).
const KEY_V: CGKeyCode = 0x09;

/// Virtual key code for the "C" key on a US ANSI keyboard.
const KEY_C: CGKeyCode = 0x08;

/// Community-standard marker type: clipboard managers skip history
/// entries that carry it. See <http://nspasteboard.org>.
const CONCEALED_TYPE: &str = "org.nspasteboard.ConcealedType";

/// Serializes every save→mutate→paste→restore sequence. Two concurrent
/// sequences interleaving is exactly the "pasted the wrong content" bug.
static PASTEBOARD_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the pasteboard lock for a full apply sequence (range-select +
/// text-set + clipboard fallback). `apply::apply_with_strategy` holds this
/// guard across all three steps so two concurrent applies cannot interleave
/// (range-set of A, range-set of B, paste of A → wrong span). Callers that
/// go through [`paste_via_clipboard_to`] don't need this — it locks
/// internally. Callers that already hold the guard must use
/// [`paste_via_clipboard_to_locked`] to avoid deadlocking on the non-
/// reentrant mutex.
pub fn lock_for_apply() -> MutexGuard<'static, ()> {
    PASTEBOARD_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Full-fidelity snapshot of the general pasteboard: every declared type
/// and its data. (First pasteboard item only — multi-item boards are
/// vanishingly rare in practice.)
struct PasteboardSnapshot {
    entries: Vec<(Retained<NSString>, Retained<NSData>)>,
}

fn snapshot_all() -> PasteboardSnapshot {
    let pb = NSPasteboard::generalPasteboard();
    let mut entries = Vec::new();
    if let Some(types) = pb.types() {
        for i in 0..types.count() {
            let t = types.objectAtIndex(i);
            // Don't re-save (and thus re-conceal-cycle) our own marker.
            if t.to_string() == CONCEALED_TYPE {
                continue;
            }
            if let Some(data) = pb.dataForType(&t) {
                entries.push((t, data));
            }
        }
    }
    PasteboardSnapshot { entries }
}

fn restore_snapshot(snap: &PasteboardSnapshot) {
    let pb = NSPasteboard::generalPasteboard();
    pb.clearContents();
    if snap.entries.is_empty() {
        return;
    }
    let type_refs: Vec<&NSString> = snap.entries.iter().map(|(t, _)| &**t).collect();
    let types = NSArray::from_slice(&type_refs);
    unsafe {
        pb.declareTypes_owner(&types, None);
        for (t, d) in &snap.entries {
            let _ = pb.setData_forType(Some(d), t);
        }
    }
}

/// Read the general pasteboard's plain-string content, if any.
pub fn snapshot_string() -> Option<String> {
    let pb = NSPasteboard::generalPasteboard();
    let ns_str = unsafe { pb.stringForType(NSPasteboardTypeString) }?;
    Some(ns_str.to_string())
}

/// Replace the pasteboard with a transient string of ours, marked
/// concealed so clipboard-history managers don't record it.
pub fn set_string(s: &str) {
    let pb = NSPasteboard::generalPasteboard();
    pb.clearContents();
    let ns = NSString::from_str(s);
    let concealed = NSString::from_str(CONCEALED_TYPE);
    unsafe {
        let types = NSArray::from_slice(&[NSPasteboardTypeString, &*concealed]);
        pb.declareTypes_owner(&types, None);
        let _ = pb.setString_forType(&ns, NSPasteboardTypeString);
        // Presence of the type is the marker; the value is irrelevant.
        let _ = pb.setString_forType(&NSString::from_str(""), &concealed);
    }
}

fn simulate_chord(key: CGKeyCode, target_pid: Option<i32>) -> bool {
    let src = match CGEventSource::new(CGEventSourceStateID::CombinedSessionState) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let down = match CGEvent::new_keyboard_event(src.clone(), key, true) {
        Ok(e) => e,
        Err(_) => return false,
    };
    down.set_flags(CGEventFlags::CGEventFlagCommand);
    let up = match CGEvent::new_keyboard_event(src, key, false) {
        Ok(e) => e,
        Err(_) => return false,
    };
    up.set_flags(CGEventFlags::CGEventFlagCommand);
    match target_pid {
        // Post straight to the target app. Crucial for popover-click
        // applies: clicking our overlay made Nib frontmost, so an HID-tap
        // ⌘V would land in Nib's own webview instead of the user's app.
        Some(pid) => {
            down.post_to_pid(pid);
            up.post_to_pid(pid);
        }
        None => {
            down.post(CGEventTapLocation::HID);
            up.post(CGEventTapLocation::HID);
        }
    }
    true
}

/// Simulate ⌘C in the focused app.
pub fn simulate_copy() -> bool {
    simulate_chord(KEY_C, None)
}

/// Read the currently-selected text via simulated ⌘C, restoring the
/// original clipboard (all types) afterward. Returns None if no string
/// was selected or copy failed. Blocks ~160 ms.
pub fn read_selection_via_copy() -> Option<String> {
    let _guard = PASTEBOARD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let saved = snapshot_all();
    // Clear so we can tell if the copy produced new content.
    let pb = NSPasteboard::generalPasteboard();
    pb.clearContents();
    if !simulate_copy() {
        restore_snapshot(&saved);
        return None;
    }
    thread::sleep(Duration::from_millis(120));
    let got = snapshot_string();
    restore_snapshot(&saved);
    got.filter(|s| !s.is_empty())
}

/// Synthesize ⌘V — to a specific pid when given (the reliable path after
/// a popover click), else at the system HID tap.
pub fn simulate_paste_to(target_pid: Option<i32>) -> bool {
    simulate_chord(KEY_V, target_pid)
}

/// Backwards-compatible HID-tap variant.
pub fn simulate_paste() -> bool {
    simulate_paste_to(None)
}

/// Full fallback: save → push → paste → wait → restore, all inside the
/// pasteboard lock so concurrent applies serialize. Blocks ~150 ms —
/// callers run on background/async-command threads, never the UI loop.
pub fn paste_via_clipboard_to(replacement: &str, target_pid: Option<i32>) -> bool {
    let _guard = lock_for_apply();
    paste_via_clipboard_to_locked(replacement, target_pid)
}

/// Same as [`paste_via_clipboard_to`] but assumes the caller already holds
/// the [`lock_for_apply`] guard (i.e. `apply_with_strategy`, which keeps the
/// lock across range-select + text-set + paste so the whole apply is
/// atomic). Must not lock again — `Mutex` is non-reentrant.
pub fn paste_via_clipboard_to_locked(replacement: &str, target_pid: Option<i32>) -> bool {
    let saved = snapshot_all();
    set_string(replacement);
    let posted = simulate_paste_to(target_pid);
    // Give the target app time to consume the paste before restoring.
    // 120ms was the long-standing empirical window; 150 adds margin for
    // slow Electron apps without being perceptible.
    thread::sleep(Duration::from_millis(150));
    restore_snapshot(&saved);
    posted
}

/// HID-tap variant used by the ⌘⇧R hotkey (the user's app is frontmost).
pub fn paste_via_clipboard(replacement: &str) -> bool {
    paste_via_clipboard_to(replacement, None)
}
