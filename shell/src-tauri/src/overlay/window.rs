//! Creates the always-on-top, click-through overlay window that renders a
//! border around the focused text element.

#![cfg(all(target_os = "macos", feature = "overlay"))]

use tauri::{AppHandle, WebviewUrl, WebviewWindowBuilder};

/// Spawn the overlay window. It covers the main display, is transparent,
/// click-through (`set_ignore_cursor_events`), and floats above other apps.
/// Frontend at `src/overlay.html` listens for `focus-update` events and
/// positions a `<div>` border at the reported bounds.
pub fn create(app: &AppHandle) -> tauri::Result<()> {
    // Per-display overlays: one click-through window per monitor so
    // secondary displays get underlines. The primary keeps the "overlay"
    // label for JS/IPC compat; secondaries are overlay-1, overlay-2, ...
    match app.available_monitors() {
        Ok(monitors) if !monitors.is_empty() => {
            // Ensure the primary monitor owns the "overlay" label.
            let mut ordered = monitors;
            if let Ok(Some(primary)) = app.primary_monitor() {
                ordered.sort_by_key(|m| {
                    if m.position() == primary.position() && m.size() == primary.size() {
                        0
                    } else {
                        1
                    }
                });
            }
            for (i, m) in ordered.iter().enumerate() {
                let label = if i == 0 {
                    "overlay".to_string()
                } else {
                    format!("overlay-{i}")
                };
                let w = m.size().width as f64;
                let h = m.size().height as f64;
                let x = m.position().x as f64;
                let y = m.position().y as f64;
                create_single(app, &label, w, h, x, y)?;
            }
            Ok(())
        }
        // Enumeration failed (or no monitors reported) — fall back to the
        // legacy single window covering the primary display area.
        _ => create_single(app, "overlay", 4096.0, 3072.0, 0.0, 0.0),
    }
}

fn create_single(
    app: &AppHandle,
    label: &str,
    width: f64,
    height: f64,
    x: f64,
    y: f64,
) -> tauri::Result<()> {
    let win = WebviewWindowBuilder::new(app, label, WebviewUrl::App("overlay.html".into()))
        .title("Nib Overlay")
        .inner_size(width, height)
        .position(x, y)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .resizable(false)
        .skip_taskbar(true)
        .shadow(false)
        .focused(false)
        .build()?;

    win.set_ignore_cursor_events(true)?;

    // Apply macOS-specific window behaviors so the overlay floats above
    // fullscreen apps and stays out of Cmd-Tab / Mission Control.
    apply_macos_window_styling(&win);

    Ok(())
}

fn apply_macos_window_styling(win: &tauri::WebviewWindow) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;

    let Ok(raw) = win.ns_window() else { return };
    let ns_window = raw as *mut AnyObject;
    if ns_window.is_null() {
        return;
    }

    // Levels and collection-behavior constants pulled from <AppKit/NSWindow.h>.
    // kCGMaximumWindowLevel ≈ NSScreenSaverWindowLevel keeps us above almost
    // everything except the menubar's own screen-shot HUD.
    const NS_SCREEN_SAVER_LEVEL: i64 = 1000;
    // canJoinAllSpaces | fullScreenAuxiliary | stationary | ignoresCycle
    const COLLECTION: u64 = (1 << 0) | (1 << 8) | (1 << 4) | (1 << 6);

    unsafe {
        let _: () = msg_send![ns_window, setLevel: NS_SCREEN_SAVER_LEVEL];
        let _: () = msg_send![ns_window, setCollectionBehavior: COLLECTION];
        let _: () = msg_send![ns_window, setHasShadow: false];
    }
}
