//! Shared cache of the AXUI element currently engaged by the focus tracker.
//!
//! Problem: when the user clicks a suggestion in our overlay popover, the
//! click activates Nib's own app and shifts AXUI focus away from the
//! writing app. By the time `apply::apply()` re-queries
//! `kAXFocusedApplicationAttribute`, the answer is "app.nib" — so the
//! AXUI text-write hits Nib's WKWebView (no-op) instead of the user's
//! text field.
//!
//! Fix: `focus_tracker` writes the most recently engaged element handle
//! here on every Engage snapshot. `apply` reads from here first, falls
//! back to the live AXUI query only when the cache is empty.
//!
//! Lifetimes: the focus tracker hands us a CFRetained `AXUIElementRef`
//! (one strong ref). We hold it in [`RetainedElem`], which CFReleases on
//! Drop. When `current_handle()` is called we bump the retain count so
//! the caller gets a fresh ref to release independently — the cache
//! keeps its own.

#![cfg(all(target_os = "macos", feature = "overlay"))]

use std::ffi::c_void;
use std::sync::Mutex;

use core_foundation::base::{CFRetain, CFRelease, CFTypeRef};

use accessibility_sys::{AXUIElementGetPid, AXUIElementRef, kAXErrorSuccess};

/// Owned AXUIElementRef — releases its retain count on drop.
struct RetainedElem(*mut c_void);

unsafe impl Send for RetainedElem {}
unsafe impl Sync for RetainedElem {}

impl Drop for RetainedElem {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0 as CFTypeRef); }
        }
    }
}

static LAST_ENGAGED: Mutex<Option<RetainedElem>> = Mutex::new(None);

/// Store the currently-engaged AXUIElement. Takes ownership of one retain
/// count — the caller must NOT CFRelease the ref after calling this. Any
/// previously-stored handle is dropped (its retain released).
///
/// `null` clears the cache.
pub fn store(elem: *mut c_void) {
    let new = if elem.is_null() { None } else { Some(RetainedElem(elem)) };
    let mut g = match LAST_ENGAGED.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // poisoned — recover anyway
    };
    *g = new;
}

/// Validated store: only caches `elem` when its owning pid matches
/// `expected_pid` (when given). Takes ownership on success exactly like
/// [`store`]; on validation failure it releases the caller's ref, clears
/// any stale cached handle, and returns false so the caller never caches
/// an element from the wrong app (e.g. a focus change raced the snapshot).
///
/// `null` always clears and returns false.
pub fn store_validated(elem: *mut c_void, expected_pid: Option<i32>) -> bool {
    if elem.is_null() {
        clear();
        return false;
    }
    if let Some(expected) = expected_pid {
        let mut actual: i32 = 0;
        let err = unsafe { AXUIElementGetPid(elem as AXUIElementRef, &mut actual) };
        if err != kAXErrorSuccess || actual <= 0 || actual != expected {
            // Reject: release the transferred retain, drop the stale cache.
            unsafe { CFRelease(elem as CFTypeRef) };
            clear();
            return false;
        }
    }
    store(elem);
    true
}

/// Pid of the currently-cached element, if any. Lets callers detect a
/// stale handle (cached pid no longer matches the focused app) without
/// bumping the retain count.
pub fn current_pid() -> Option<i32> {
    let g = LAST_ENGAGED.lock().ok()?;
    let owned = g.as_ref()?;
    let mut pid: i32 = 0;
    let err = unsafe { AXUIElementGetPid(owned.0 as AXUIElementRef, &mut pid) };
    (err == kAXErrorSuccess && pid > 0).then_some(pid)
}

/// Borrow the cached handle, bumping its retain count so the caller owns
/// an independent reference. Caller MUST CFRelease when done.
///
/// Returns `None` when no element is cached (no recent engagement, or
/// it was explicitly cleared via `store(null)`).
pub fn current_handle() -> Option<*mut c_void> {
    let g = LAST_ENGAGED.lock().ok()?;
    let owned = g.as_ref()?;
    unsafe { CFRetain(owned.0 as CFTypeRef); }
    Some(owned.0)
}

/// Clear the cache. Equivalent to `store(null)`. Useful when the user
/// pauses Nib or the engagement filter rejects the focused app.
pub fn clear() {
    let mut g = match LAST_ENGAGED.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    *g = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_resets_to_none() {
        clear();
        assert!(current_handle().is_none());
    }

    #[test]
    fn store_then_current_then_clear() {
        // We can't safely fabricate an AXUIElement pointer for unit tests
        // (CFRelease on a non-CF pointer crashes the process). So we only
        // exercise the null path here — round-trip the empty state.
        store(std::ptr::null_mut());
        assert!(current_handle().is_none());
        clear();
        assert!(current_handle().is_none());
    }

    #[test]
    fn store_clear_current_is_none() {
        // Regression guard for stale engaged-elem: once cleared, no
        // handle (and no pid) may survive — apply must fall back to the
        // live AXUI query instead of writing to a dead element.
        store(std::ptr::null_mut());
        clear();
        assert!(current_handle().is_none());
        assert!(current_pid().is_none());
    }

    #[test]
    fn store_validated_null_clears() {
        store(std::ptr::null_mut());
        assert!(!store_validated(std::ptr::null_mut(), None));
        assert!(current_handle().is_none());
        clear();
        assert!(current_handle().is_none());
    }
}
