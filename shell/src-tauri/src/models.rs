//! Registry of LLM models Nib can run + download orchestration.
//!
//! The default `qwen3-0.6b` ships bundled inside the .app. Additional
//! models are downloaded on demand into
//!   `~/Library/Application Support/Nib/models/<id>.gguf`
//! and selected via `config.selected_model`. The focus tracker / inference
//! engine resolves the path through [`resolve_path`].

use std::path::PathBuf;
use std::sync::Mutex;

use serde::Serialize;

/// Static metadata for one supported model.
#[derive(Serialize, Clone, Debug)]
pub struct ModelInfo {
    /// Stable identifier (e.g. `"lfm2.5-350m"`). Used as the on-disk
    /// filename stem and the config value.
    pub id: &'static str,
    /// User-facing display name.
    pub display_name: &'static str,
    /// Params, e.g. `"350M"`, `"1.2B"`.
    pub params: &'static str,
    /// Approximate Q4_K_M file size in MB.
    pub size_mb: u64,
    /// One-line description for the settings UI.
    pub blurb: &'static str,
    /// True when the .gguf ships inside the .app bundle. False = needs
    /// download to `~/Library/Application Support/Nib/models/`.
    pub bundled: bool,
    /// Download URL (HuggingFace direct GGUF). None when no download
    /// fallback exists; bundled entries may keep a URL as fallback.
    pub url: Option<&'static str>,
    /// On-disk filename (also used as bundle resource name when bundled).
    pub filename: &'static str,
    /// Set on adapter-only entries: the registry id of the base model
    /// this adapter layers on top of. `None` for standalone models.
    /// When set, [`resolve_paths`] returns the base path + the adapter
    /// path separately, and `is_installed` requires *both* files present.
    pub requires_base: Option<&'static str>,
}

/// Bundle of paths needed to load a registry entry. For standalone
/// models, `adapter` is `None`; for adapter entries it carries the
/// LoRA `.gguf` on top of the base.
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub base: PathBuf,
    pub adapter: Option<PathBuf>,
}

/// All models Nib can run. Order = display order in the settings panel.
/// Base entries (no `requires_base`) come first so adapter entries can
/// reference them by id.
pub const REGISTRY: &[ModelInfo] = &[
    // Stock Qwen3 0.6B — thinking disabled in-engine (no_think).
    // Bundled default.
    ModelInfo {
        id: "qwen3-0.6b",
        display_name: "Qwen3 0.6B (no-think)",
        params: "0.6B",
        size_mb: 462,
        blurb: "Bundled default. 83.3% strict holdout90; thinking disabled in-engine",
        bundled: true,
        url: Some("https://huggingface.co/lm-kit/qwen-3-0.6b-instruct-gguf/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true"),
        filename: "Qwen3-0.6B-Q4_K_M.gguf",
        requires_base: None,
    },
    ModelInfo {
        id: "lfm2.5-350m",
        display_name: "LFM2.5 350M",
        params: "350M",
        size_mb: 219,
        blurb: "Legacy default, download removed – use Qwen3. Fast and light. \
                Best for grammar fixes; rewrites may pad or invent content.",
        bundled: false,
        url: None,
        filename: "lfm2.5-350m-q4_k_m.gguf",
        requires_base: None,
    },
    // Stock Qwen 2.5-1.5B base — the substrate every Nib adapter v2.x+
    // layers on top of. Standalone-usable but the registry hides it
    // unless an adapter that needs it is selected.
    ModelInfo {
        id: "qwen2.5-1.5b-instruct",
        display_name: "Qwen 2.5 1.5B Instruct (base)",
        params: "1.5B",
        size_mb: 940,
        blurb: "Stock Qwen base. Nib adapters (v2.x+) layer on top of \
                this — download once, reuse across every future Nib LoRA.",
        bundled: false,
        url: Some("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"),
        filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        requires_base: None,
    },
    // Nib's faithful-rewrite LoRA, applied at runtime on top of the Qwen
    // base. Ships inside Full builds (`tauri.conf.full.json`); regular
    // builds download it from the GitHub release. Each future iteration
    // (v2.2, v2.3, …) ships as a tiny adapter swap, no base re-download.
    //
    // NOTE: the v2.1.0 release with this asset is NOT published yet — the
    // only release is v2.0.0 (which carries the old merged model). Until
    // `gh release create v2.1.0 nib-faithful-f16.gguf` happens, this URL
    // 404s; the hardened downloader below surfaces that as a clear error
    // instead of installing an error page.
    ModelInfo {
        id: "nib-qwen-v2",
        display_name: "Nib-Faithful v2 (Qwen 1.5B + LoRA)",
        params: "1.5B + LoRA",
        size_mb: 36,
        blurb: "Premium. Nib's faithful-rewrite LoRA layered on Qwen 2.5-\
                1.5B — preserves facts, numbers, and technical tokens. \
                81.1% on the 90-case held-out benchmark (vs 64.4% for \
                stock Qwen). Adapter is ~36 MB; the 940 MB Qwen base \
                downloads once, reusable for any future Nib adapter.",
        bundled: true,
        url: Some("https://github.com/abgnydn/nib/releases/download/v2.1.0/nib-faithful-f16.gguf"),
        filename: "nib-faithful-f16.gguf",
        requires_base: Some("qwen2.5-1.5b-instruct"),
    },
];

/// Look up by ID. Falls back to the default (first registry entry) when
/// the id is unknown — keeps a stale config value from breaking startup.
pub fn lookup(id: &str) -> &'static ModelInfo {
    REGISTRY.iter().find(|m| m.id == id).unwrap_or(&REGISTRY[0])
}

/// Resolve the on-disk path for a given model. Checks BOTH the bundle
/// resources dir AND the downloaded-models dir — that way the Full
/// installer (which ships Nib-Qwen v2 inside the .app) and the regular
/// installer (which expects users to download it) both work via the
/// same code path. Bundle wins when present.
///
/// Generic over `Manager` so both `&tauri::App` (setup-time) and
/// `&tauri::AppHandle` (command-time) work without duplication.
pub fn resolve_path<R: tauri::Runtime, M: tauri::Manager<R>>(
    app: &M,
    id: &str,
) -> Option<PathBuf> {
    let info = lookup(id);
    if let Ok(p) = app.path().resolve(
        format!("resources/{}", info.filename),
        tauri::path::BaseDirectory::Resource,
    ) {
        if p.exists() {
            return Some(p);
        }
    }
    let p = downloaded_models_dir().ok()?.join(info.filename);
    if p.exists() { Some(p) } else { None }
}

/// Runtime check: is the model on disk anywhere we can load from?
/// For adapter entries this requires *both* the adapter and its base
/// to be present — anything else and the engine couldn't load it.
pub fn is_installed<R: tauri::Runtime, M: tauri::Manager<R>>(
    app: &M,
    id: &str,
) -> bool {
    resolve_paths(app, id).is_some()
}

/// Resolve every file needed to actually load `id`. For standalone
/// models this is just the one path; for adapter entries it's `(base,
/// Some(adapter))`. Returns `None` if any required file is missing.
pub fn resolve_paths<R: tauri::Runtime, M: tauri::Manager<R>>(
    app: &M,
    id: &str,
) -> Option<ModelPaths> {
    let info = lookup(id);
    match info.requires_base {
        Some(base_id) => {
            let base = resolve_path(app, base_id)?;
            let adapter = resolve_path(app, id)?;
            Some(ModelPaths { base, adapter: Some(adapter) })
        }
        None => {
            let base = resolve_path(app, id)?;
            Some(ModelPaths { base, adapter: None })
        }
    }
}

/// Everything that must be downloaded before `id` can load, in download
/// order (base first, then the adapter itself when it has a URL and is
/// missing — `bundled` only means it *may* ship inside Full builds).
/// Empty = fully installed. Entries without a URL that are missing can't
/// be fixed by downloading; [`missing_undownloadable`] reports those.
pub fn download_targets<R: tauri::Runtime, M: tauri::Manager<R>>(
    app: &M,
    id: &str,
) -> Vec<&'static str> {
    let info = lookup(id);
    let mut out = Vec::new();
    if let Some(base_id) = info.requires_base {
        let base = lookup(base_id);
        if resolve_path(app, base_id).is_none() && base.url.is_some() {
            out.push(base.id);
        }
    }
    if resolve_path(app, info.id).is_none() && info.url.is_some() {
        out.push(info.id);
    }
    out
}

/// Files `id` needs that are missing AND have no download URL — the user
/// can't fix these from the picker (e.g. a bundled-only file absent from
/// this build). Used to produce an honest error message.
pub fn missing_undownloadable<R: tauri::Runtime, M: tauri::Manager<R>>(
    app: &M,
    id: &str,
) -> Vec<&'static str> {
    let info = lookup(id);
    let mut out = Vec::new();
    if let Some(base_id) = info.requires_base {
        let base = lookup(base_id);
        if resolve_path(app, base_id).is_none() && base.url.is_none() {
            out.push(base.id);
        }
    }
    if resolve_path(app, info.id).is_none() && info.url.is_none() {
        out.push(info.id);
    }
    out
}

/// Extended ModelInfo with runtime "installed" + "loaded" flags. Used
/// by the model_list Tauri command so the UI can render "bundled",
/// "downloaded", "needs download" pills correctly per actual disk state.
#[derive(Serialize, Clone, Debug)]
pub struct ModelInfoExt {
    #[serde(flatten)]
    pub info: ModelInfo,
    /// File is present on disk (bundled in .app OR downloaded).
    pub installed: bool,
    /// True if this is the model currently selected in config.
    pub selected: bool,
}

/// `~/Library/Application Support/Nib/models/`, created if missing.
pub fn downloaded_models_dir() -> std::io::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME not set"))?;
    let mut p = PathBuf::from(home);
    // The data dir is "Nib"; existing "Quill" dirs are migrated on startup by
    // `config::migrate_legacy_data_dir`, so user data carries over.
    p.push("Library/Application Support/Nib/models");
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

/// Progress callback signature: (bytes_downloaded, total_bytes_or_zero).
pub type ProgressFn = dyn Fn(u64, u64) + Send + 'static;

/// Tracks the currently-running download so the UI can poll its state
/// and we don't start two downloads of the same model simultaneously.
#[derive(Serialize, Clone, Debug, Default)]
pub struct DownloadStatus {
    pub model_id: String,
    pub bytes_done: u64,
    pub total_bytes: u64,
    pub state: DownloadState,
    pub error: Option<String>,
}

#[derive(Serialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    #[default]
    Idle,
    Running,
    Done,
    Failed,
}

pub struct DownloadTracker {
    inner: Mutex<DownloadStatus>,
    /// PID of the currently-running curl child, so the quit path can kill
    /// it instead of leaving an orphan writing into the models dir.
    child_pid: Mutex<Option<u32>>,
}

impl DownloadTracker {
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(DownloadStatus {
                model_id: String::new(),
                bytes_done: 0,
                total_bytes: 0,
                state: DownloadState::Idle,
                error: None,
            }),
            child_pid: Mutex::new(None),
        }
    }

    pub fn snapshot(&self) -> DownloadStatus {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn set(&self, s: DownloadStatus) {
        if let Ok(mut g) = self.inner.lock() {
            *g = s;
        }
    }

    pub fn update<F: FnOnce(&mut DownloadStatus)>(&self, f: F) {
        if let Ok(mut g) = self.inner.lock() {
            f(&mut g);
        }
    }

    fn set_child_pid(&self, pid: Option<u32>) {
        if let Ok(mut g) = self.child_pid.lock() {
            *g = pid;
        }
    }

    /// Kill the in-flight curl child, if any. Called on app quit so a
    /// half-finished download doesn't keep writing after Nib exits.
    /// `/bin/kill` avoids pulling in libc just for one signal.
    pub fn kill_running(&self) {
        if let Ok(g) = self.child_pid.lock() {
            if let Some(pid) = *g {
                let _ = std::process::Command::new("/bin/kill")
                    .arg(pid.to_string())
                    .status();
            }
        }
    }
}

impl Default for DownloadTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Spawn a background thread that downloads each id in `ids` (in order)
/// via `curl`. Polls the destination file size for progress (curl streams
/// writes), updates the shared [`DownloadTracker`]. Stops the queue on
/// the first failure. Returns immediately.
///
/// curl is universally available on macOS 13+ — no extra deps. It also
/// handles HuggingFace's 302→CDN redirect chain cleanly via `-L`.
/// Hardening:
///   - absolute /usr/bin/curl (no PATH hijack)
///   - `--fail`: an HTTP 404/500 error page must never be renamed into a
///     "installed" .gguf (that would permanently brick the entry — the UI
///     has no re-download path once `is_installed` is true)
///   - `--proto =https --retry 3 -C -`: https-only, transient-error retry,
///     resume of a previous .part
///   - post-download size floor vs the registry's size_mb
pub fn spawn_download(
    ids: Vec<String>,
    tracker: std::sync::Arc<DownloadTracker>,
    on_complete: Option<Box<dyn Fn() + Send + 'static>>,
) {
    let first = match ids.first() {
        Some(id) => id.clone(),
        None => return,
    };
    tracker.set(DownloadStatus {
        model_id: first,
        bytes_done: 0,
        total_bytes: lookup(&ids[0]).size_mb * 1024 * 1024,
        state: DownloadState::Running,
        error: None,
    });

    std::thread::Builder::new()
        .name("nib-model-download".into())
        .spawn({
            let tracker = tracker.clone();
            move || {
                for (i, id) in ids.iter().enumerate() {
                    if !download_one(id, &tracker) {
                        return; // tracker already set to Failed
                    }
                    // Keep showing Running between queue items so the UI
                    // (and model_download's double-start guard) don't see
                    // a momentary Done mid-queue.
                    if i + 1 < ids.len() {
                        tracker.update(|s| s.state = DownloadState::Running);
                    }
                }
                if let Some(cb) = on_complete { cb(); }
            }
        })
        .expect("spawn download thread");
}

/// Download a single registry entry synchronously (called from the queue
/// thread). Returns true on success; on failure sets the tracker state.
fn download_one(id: &str, tracker: &DownloadTracker) -> bool {
    let fail = |msg: String| {
        tracker.update(|s| {
            s.state = DownloadState::Failed;
            s.error = Some(msg);
        });
        false
    };

    let info = lookup(id).clone();
    let url = match info.url {
        Some(u) => u.to_string(),
        None => return fail(format!("{id} has no download URL (bundled-only)")),
    };
    let dest_dir = match downloaded_models_dir() {
        Ok(d) => d,
        Err(e) => return fail(format!("dest dir: {e}")),
    };
    let dest_path = dest_dir.join(info.filename);
    let tmp_path = dest_dir.join(format!("{}.part", info.filename));

    // Guard against an orphaned curl (from a previous crashed/quit run)
    // still writing the same .part: if the file grows while we watch it,
    // someone else owns it.
    if let Ok(m0) = std::fs::metadata(&tmp_path) {
        std::thread::sleep(std::time::Duration::from_millis(750));
        if let Ok(m1) = std::fs::metadata(&tmp_path) {
            if m1.len() > m0.len() {
                return fail(format!(
                    "another process is already downloading {} — try again in a minute",
                    info.filename
                ));
            }
        }
    }

    tracker.set(DownloadStatus {
        model_id: id.to_string(),
        bytes_done: 0,
        total_bytes: info.size_mb * 1024 * 1024,
        state: DownloadState::Running,
        error: None,
    });

    let mut child = match std::process::Command::new("/usr/bin/curl")
        .arg("--fail")
        .arg("--proto").arg("=https")
        .arg("--retry").arg("3")
        .arg("-C").arg("-")
        .arg("-L")
        .arg("-o").arg(&tmp_path)
        .arg(&url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return fail(format!("curl spawn: {e}")),
    };
    tracker.set_child_pid(Some(child.id()));

    // Poll file size every 500ms while curl runs.
    let result = loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
        tracker.update(|s| s.bytes_done = size);
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break Ok(()),
            Ok(Some(status)) => {
                // 22 = curl's HTTP-error exit under --fail; keep the .part
                // for `-C -` resume on transient failures, but a 404'd
                // .part is useless — drop it so retries start clean.
                let _ = std::fs::remove_file(&tmp_path);
                break Err(format!(
                    "curl exited {status} downloading {url} (HTTP error or network failure)"
                ));
            }
            Ok(None) => continue,
            Err(e) => break Err(format!("wait: {e}")),
        }
    };
    tracker.set_child_pid(None);
    if let Err(e) = result {
        return fail(e);
    }

    // Sanity floor: a real GGUF is within ~2× of the registry estimate;
    // an HTML error page or truncated CDN response is nowhere close.
    let got = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
    let floor = (info.size_mb * 1024 * 1024) / 2;
    if got < floor {
        let _ = std::fs::remove_file(&tmp_path);
        return fail(format!(
            "{} downloaded {got} bytes but ~{} MB expected — server likely returned an error body",
            info.filename, info.size_mb
        ));
    }

    if let Err(e) = std::fs::rename(&tmp_path, &dest_path) {
        return fail(format!("rename: {e}"));
    }
    tracker.update(|s| {
        s.state = DownloadState::Done;
        s.bytes_done = got;
        s.total_bytes = got;
    });
    eprintln!("[nib][model] download complete: {id}");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_a_bundled_default() {
        let first = &REGISTRY[0];
        assert!(first.bundled, "first registry entry must be bundled");
        assert_eq!(first.id, "qwen3-0.6b");
    }

    #[test]
    fn lookup_falls_back_to_default_on_unknown_id() {
        let m = lookup("not-a-real-model");
        assert_eq!(m.id, REGISTRY[0].id);
    }

    #[test]
    fn lookup_finds_known_models() {
        assert_eq!(lookup("nib-qwen-v2").id, "nib-qwen-v2");
        assert_eq!(lookup("qwen2.5-1.5b-instruct").id, "qwen2.5-1.5b-instruct");
    }

    #[test]
    fn qwen3_0_6b_lookup_resolves() {
        let m = lookup("qwen3-0.6b");
        assert_eq!(m.id, "qwen3-0.6b");
        assert_eq!(m.filename, "Qwen3-0.6B-Q4_K_M.gguf");
        assert!(
            m.filename.ends_with(".gguf"),
            "filename must match resources gguf convention",
        );
        assert!(m.bundled, "qwen3 is the bundled default");
        assert!(m.url.is_some(), "bundled default keeps a download fallback url");
        assert!(m.requires_base.is_none());
    }

    #[test]
    fn adapter_entries_point_at_real_base() {
        for m in REGISTRY {
            if let Some(base_id) = m.requires_base {
                let base = lookup(base_id);
                assert_eq!(
                    base.id, base_id,
                    "adapter {} → base {} is not in registry",
                    m.id, base_id,
                );
                assert!(
                    base.requires_base.is_none(),
                    "adapter base must be standalone (no nested adapters)",
                );
            }
        }
    }
}
