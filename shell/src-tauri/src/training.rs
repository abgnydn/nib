//! In-app personal LoRA training trigger.
//!
//! Spawns `modal run modal_train_personal.py` as a subprocess from the Nib
//! main window, polls its progress, and on success copies the resulting
//! `personal-adapter.gguf` into the spot Nib auto-detects at startup.
//!
//! Why subprocess rather than calling Modal's Python API directly: Modal's
//! orchestration is its CLI tool; replicating that from Rust is far more
//! work than just exec'ing the binary the user already has.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Idle,
    Running,
    Succeeded,
    Failed,
}

#[derive(Serialize, Clone, Debug)]
pub struct TrainingStatus {
    pub state: JobState,
    pub elapsed_secs: f64,
    /// Last meaningful line from stdout we managed to scrape, useful for
    /// telling the user what stage they're in ("Loading model…",
    /// "trained in X min", etc.).
    pub stage: Option<String>,
    pub error: Option<String>,
    /// Set when state == Succeeded — the local path Nib will install from.
    pub output_adapter: Option<String>,
    /// Which backend ran this job — UI surfaces "local (free)" vs "Modal ($0.20)".
    pub backend: Backend,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            state: JobState::Idle,
            elapsed_secs: 0.0,
            stage: None,
            error: None,
            output_adapter: None,
            backend: Backend::None,
        }
    }
}

struct Job {
    child: Option<Child>,
    started_at: Option<Instant>,
    state: JobState,
    /// Last meaningful child-output line, updated by the drainer threads
    /// (which MUST run: an unread pipe fills at ~64KB and deadlocks the
    /// trainer). Shared so drainers don't contend on the Job mutex.
    stage: Arc<Mutex<Option<String>>>,
    error: Option<String>,
    output_adapter: Option<PathBuf>,
    /// Working dir of the spawned process — used by the Modal backend so
    /// install can find `checkpoints/personal-adapter.gguf` after the
    /// modal CLI downloads it. Unset for the local backend (we already
    /// know the exact output path).
    cwd: Option<PathBuf>,
    /// Pre-known output path for backends that produce a deterministic
    /// adapter file (i.e. the local llama-finetune-lora path). Set at
    /// spawn time so status() doesn't have to guess.
    expected_output: Option<PathBuf>,
    /// Where the trainer actually writes while running (local backend
    /// uses `<expected_output>.part` so a crashed run can't leave a
    /// truncated adapter at the live path). Renamed on success.
    tmp_output: Option<PathBuf>,
    /// Base model this job trains against — recorded into the adapter's
    /// sidecar meta at install time so startup can refuse a mismatched
    /// base. None for the legacy Modal backend (which trains Gemma).
    base_model: Option<PathBuf>,
    /// Which backend this job is running on — surfaced to the UI so the
    /// user can see "training locally" vs "training on Modal".
    backend: Backend,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Deprecated legacy Gemma Modal backend — trains an adapter that
    /// won't load on LFM2.5/Qwen bases. Retained only behind
    /// `NIB_ALLOW_LEGACY_MODAL=1` + `allow_cloud_training`; prefer Local (QVAC).
    #[deprecated(note = "Modal Gemma legacy won't load on LFM2.5/Qwen — use local QVAC path instead")]
    Modal,
    Local,
    /// No job has run yet — distinguish from a default that lies.
    None,
}

impl Default for Job {
    fn default() -> Self {
        Self {
            child: None,
            started_at: None,
            state: JobState::Idle,
            stage: Arc::new(Mutex::new(None)),
            error: None,
            output_adapter: None,
            cwd: None,
            expected_output: None,
            tmp_output: None,
            base_model: None,
            backend: Backend::None,
        }
    }
}

#[derive(Default)]
pub struct TrainingState {
    inner: Mutex<Job>,
}

pub type SharedTraining = Arc<TrainingState>;

#[derive(Debug)]
pub enum StartError {
    AlreadyRunning,
    NoHfToken,
    ModalNotFound(String),
    TrainDirMissing(PathBuf),
    Spawn(String),
    /// Retired Modal-Gemma legacy path — Gemma adapter won't load on
    /// LFM2.5/Qwen. Set NIB_ALLOW_LEGACY_MODAL=1 + allow_cloud_training
    /// to override; prefer the local QVAC path.
    LegacyModalDeprecated,
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRunning => write!(f, "a training job is already running"),
            Self::NoHfToken => write!(
                f,
                "HF_TOKEN env var not set — launch Nib from a terminal with `export HF_TOKEN=hf_...`"
            ),
            Self::ModalNotFound(p) => write!(
                f,
                "modal CLI not found at {p} (legacy Gemma path — Gemma adapter won't load on LFM2.5/Qwen; use local QVAC)"
            ),
            Self::TrainDirMissing(p) => write!(
                f,
                "train dir missing: {} (checked NIB_TRAIN_DIR, ~/dev/nib/train, ~/quill/train legacy) — set NIB_TRAIN_DIR to your train checkout",
                p.display()
            ),
            Self::Spawn(e) => write!(
                f,
                "failed to spawn modal: {e} (legacy Gemma path won't load on LFM2.5/Qwen)"
            ),
            Self::LegacyModalDeprecated => write!(
                f,
                "Modal Gemma legacy is deprecated and won't load on LFM2.5/Qwen bases — use the local QVAC path instead; to override set NIB_ALLOW_LEGACY_MODAL=1 and enable allow_cloud_training"
            ),
        }
    }
}

/// Resolve the train directory for the retired legacy Modal backend (Gemma
/// adapter won't load on LFM2.5/Qwen — requires NIB_ALLOW_LEGACY_MODAL=1).
/// Checks `NIB_TRAIN_DIR`, then the current `~/dev/nib/train` checkout layout,
/// then the author's historical `~/quill/train` clone as a legacy fallback
/// (only if it exists). Returns None when nothing exists — callers must
/// surface the checked paths. Single-user assumption —
/// this whole path is opt-in legacy (see `allow_cloud_training`).
pub fn default_train_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("NIB_TRAIN_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    let candidates = [home.join("dev/nib/train"), home.join("quill/train")];
    candidates.into_iter().find(|p| p.exists())
}

/// Try the venv binary first, fall back to PATH lookup.
fn resolve_modal_bin(train_dir: &Path) -> Option<PathBuf> {
    let venv_modal = train_dir.join(".venv/bin/modal");
    if venv_modal.exists() {
        return Some(venv_modal);
    }
    // Fallback — `which modal`. We do a small PATH walk rather than pulling
    // in the `which` crate just for this.
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("modal");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

impl TrainingState {
    /// Snapshot the current status — cheap, called from the JS poller every
    /// 2-3 seconds. Reaps the child if it has exited.
    pub fn status(&self) -> TrainingStatus {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => {
                return TrainingStatus {
                    state: JobState::Failed,
                    error: Some("training mutex poisoned".into()),
                    ..Default::default()
                };
            }
        };

        // If we were running, see if the child finished since last poll.
        if g.state == JobState::Running {
            let mut transition: Option<(JobState, Option<String>)> = None;
            if let Some(child) = g.child.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        if status.success() {
                            transition = Some((JobState::Succeeded, None));
                        } else {
                            let code = status.code().unwrap_or(-1);
                            transition = Some((
                                JobState::Failed,
                                Some(format!("modal exited with code {code}")),
                            ));
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        transition = Some((JobState::Failed, Some(format!("try_wait: {e}"))));
                    }
                }
            }
            if let Some((new_state, err)) = transition {
                g.state = new_state.clone();
                g.error = err;
                // On success, point the UI at the generated adapter.
                if new_state == JobState::Succeeded {
                    // Local backend trained into a .part file — promote it
                    // atomically now that the run definitely succeeded.
                    if let (Some(tmp), Some(fin)) = (&g.tmp_output, &g.expected_output) {
                        if tmp.exists() {
                            if let Err(e) = std::fs::rename(tmp, fin) {
                                eprintln!("[nib][train] adapter promote failed: {e}");
                                g.state = JobState::Failed;
                                g.error = Some(format!("adapter promote: {e}"));
                            }
                        }
                    }
                    if g.state == JobState::Succeeded {
                        // Prefer the pre-known output path (local backend);
                        // fall back to the Modal cwd-relative default.
                        let candidate = g.expected_output.clone().or_else(|| {
                            g.cwd.as_ref().map(|c| c.join("checkpoints/personal-adapter.gguf"))
                        });
                        if let Some(out) = candidate {
                            if out.exists() {
                                g.output_adapter = Some(out);
                            }
                        }
                    }
                } else if let Some(tmp) = &g.tmp_output {
                    // Failed run: drop the partial file so it can't be
                    // mistaken for a real adapter later.
                    let _ = std::fs::remove_file(tmp);
                }
                g.child = None;
            }
        }

        let elapsed = g
            .started_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        TrainingStatus {
            state: g.state.clone(),
            elapsed_secs: elapsed,
            stage: g.stage.lock().ok().and_then(|s| s.clone()),
            error: g.error.clone(),
            output_adapter: g.output_adapter.as_ref().map(|p| p.display().to_string()),
            backend: g.backend,
        }
    }

    /// Spawn the Modal training subprocess (retired legacy Gemma path).
    ///
    /// Requires `NIB_ALLOW_LEGACY_MODAL=1` in the environment — callers
    /// additionally gate on the `allow_cloud_training` config flag, so both
    /// are needed to even attempt a cloud spawn. Otherwise returns an
    /// actionable error pointing at the local QVAC path (Gemma legacy
    /// won't load on LFM2.5/Qwen).
    #[allow(deprecated)]
    pub fn start(&self, journal_path: PathBuf) -> Result<(), StartError> {
        let mut g = self.inner.lock().map_err(|_| StartError::Spawn("mutex".into()))?;
        if g.state == JobState::Running {
            return Err(StartError::AlreadyRunning);
        }
        let hf_token = std::env::var("HF_TOKEN").map_err(|_| StartError::NoHfToken)?;
        // Retire Modal-Gemma legacy: Gemma adapter won't load on LFM2.5/Qwen.
        // Require explicit opt-in to even attempt a cloud spawn; otherwise
        // point at the local QVAC path.
        if std::env::var("NIB_ALLOW_LEGACY_MODAL").as_deref() != Ok("1") {
            return Err(StartError::LegacyModalDeprecated);
        }
        let train_dir = default_train_dir().ok_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "<HOME unset>".into());
            StartError::TrainDirMissing(PathBuf::from(format!(
                "{home}/dev/nib/train or {home}/quill/train"
            )))
        })?;
        if !train_dir.exists() {
            return Err(StartError::TrainDirMissing(train_dir));
        }
        let modal_bin = resolve_modal_bin(&train_dir)
            .ok_or_else(|| StartError::ModalNotFound(format!("{}/.venv/bin/modal or in PATH", train_dir.display())))?;

        eprintln!(
            "[nib][train] spawning {} run modal_train_personal.py --journal {}",
            modal_bin.display(),
            journal_path.display()
        );

        let mut child = Command::new(&modal_bin)
            .current_dir(&train_dir)
            .env("HF_TOKEN", hf_token)
            .arg("run")
            .arg("modal_train_personal.py")
            .arg("--journal")
            .arg(&journal_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| StartError::Spawn(e.to_string()))?;

        let stage = Arc::new(Mutex::new(Some("starting modal job…".to_string())));
        spawn_drainers(&mut child, stage.clone());

        *g = Job {
            child: Some(child),
            started_at: Some(Instant::now()),
            state: JobState::Running,
            stage,
            error: None,
            output_adapter: None,
            cwd: Some(train_dir),
            expected_output: None,
            tmp_output: None,
            base_model: None,
            backend: Backend::Modal,
        };
        Ok(())
    }

    /// Spawn `llama-finetune-lora` from the bundled QVAC binaries — runs
    /// the whole training loop on the user's Mac, no Modal, no network.
    /// `output_adapter` is the destination GGUF path passed to QVAC via
    /// `--output-adapter`; we record it so `install()` can find it later.
    #[cfg(feature = "llm")]
    pub fn start_local(
        &self,
        qvac_bin: PathBuf,
        base_model: PathBuf,
        journal_export: PathBuf,
        output_adapter: PathBuf,
    ) -> Result<(), StartError> {
        let mut g = self.inner.lock().map_err(|_| StartError::Spawn("mutex".into()))?;
        if g.state == JobState::Running {
            return Err(StartError::AlreadyRunning);
        }
        // Train into a sibling .part file — writing straight to the live
        // personal-adapter.gguf would let an interrupted run leave a
        // truncated adapter that breaks engine load at next startup.
        let tmp_output = {
            let mut s = output_adapter.as_os_str().to_os_string();
            s.push(".part");
            PathBuf::from(s)
        };
        let mut child = crate::training_local::spawn(
            &qvac_bin,
            &base_model,
            &journal_export,
            &tmp_output,
        )
        .map_err(|e| StartError::Spawn(e.to_string()))?;

        let stage = Arc::new(Mutex::new(Some("starting local training on Metal…".to_string())));
        spawn_drainers(&mut child, stage.clone());

        *g = Job {
            child: Some(child),
            started_at: Some(Instant::now()),
            state: JobState::Running,
            stage,
            error: None,
            output_adapter: None,
            cwd: None,
            expected_output: Some(output_adapter),
            tmp_output: Some(tmp_output),
            base_model: Some(base_model),
            backend: Backend::Local,
        };
        Ok(())
    }

    /// Copy a previously-trained adapter into Nib's Application Support
    /// dir so it's auto-detected on the next launch. When the local
    /// backend wrote the adapter directly to `dest` (we pass it via
    /// `--output-adapter`), the copy is a no-op and we just return the
    /// existing size.
    pub fn install(&self, dest: &Path) -> Result<u64, String> {
        let (src, base_model) = {
            let g = self.inner.lock().map_err(|_| "mutex".to_string())?;
            let src = g.output_adapter
                .clone()
                .ok_or_else(|| "no adapter produced yet".to_string())?;
            (src, g.base_model.clone())
        };
        if !src.exists() {
            return Err(format!("source missing: {}", src.display()));
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        // Record which base this adapter was trained on so startup can
        // refuse to layer it on a different one. The legacy Modal backend
        // trains Gemma-3-270M — tag it as such so it never loads on the
        // shipped LFM2.5/Qwen bases.
        let trained_on = base_model
            .as_deref()
            .and_then(|p| p.file_name())
            .and_then(|f| f.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "gemma-3-270m-it (legacy modal)".to_string());
        let meta = serde_json::json!({
            "base_filename": trained_on,
            "installed_at": crate::journal::now_rfc3339(),
        });
        let meta_path = crate::state::adapter_meta_path(dest);
        if let Err(e) = std::fs::write(&meta_path, meta.to_string()) {
            eprintln!("[nib][train] could not write adapter meta {}: {e}", meta_path.display());
        }
        // Same file? Nothing to do — local backend already wrote here.
        if let (Ok(s), Ok(d)) = (src.canonicalize(), dest.canonicalize()) {
            if s == d {
                return std::fs::metadata(dest).map(|m| m.len()).map_err(|e| e.to_string());
            }
        }
        std::fs::copy(&src, dest).map_err(|e| e.to_string())
    }

    /// Reset state to Idle so the user can run another job.
    pub fn reset(&self) {
        if let Ok(mut g) = self.inner.lock() {
            // If a child is still alive (shouldn't be — only reset Idle/
            // Succeeded/Failed), kill AND reap it so we don't leave a
            // zombie process table entry until app exit.
            if let Some(child) = g.child.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            if let Some(tmp) = &g.tmp_output {
                let _ = std::fs::remove_file(tmp);
            }
            *g = Job::default();
        }
    }
}

/// Consume the child's stdout + stderr on dedicated threads. MANDATORY
/// for piped children: `modal run` and `llama-finetune-lora` both write
/// far more than the ~64KB pipe buffer, and an unread pipe blocks the
/// trainer forever (UI stuck on "training in progress…"). Meaningful
/// lines land in `stage` for the status poller; everything else is
/// forwarded to our stderr so it still shows up in the app log.
fn spawn_drainers(child: &mut Child, stage: Arc<Mutex<Option<String>>>) {
    let is_stage_line = |l: &str| {
        l.contains("[nib")
            || l.contains("trained in")
            || l.contains("MB")
            || l.contains("epoch")
            || l.contains("loss")
    };
    for (name, reader) in [
        ("stdout", child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)),
        ("stderr", child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)),
    ] {
        let Some(reader) = reader else { continue };
        let stage = stage.clone();
        let _ = std::thread::Builder::new()
            .name(format!("nib-train-drain-{name}"))
            .spawn(move || {
                for line in BufReader::new(reader).lines().map_while(Result::ok) {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    eprintln!("[nib][train:{name}] {trimmed}");
                    if is_stage_line(trimmed) {
                        if let Ok(mut s) = stage.lock() {
                            *s = Some(trimmed.to_string());
                        }
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_status_after_new() {
        let s = TrainingState::default();
        let st = s.status();
        assert_eq!(st.state, JobState::Idle);
        assert_eq!(st.elapsed_secs, 0.0);
        assert!(st.error.is_none());
    }

    #[test]
    fn start_errors_with_no_hf_token() {
        let _guard = crate::state::TEST_ENV_LOCK.lock().unwrap();
        // Save & restore HF_TOKEN — tests share env.
        let saved = std::env::var("HF_TOKEN").ok();
        unsafe { std::env::remove_var("HF_TOKEN"); }
        let s = TrainingState::default();
        let r = s.start(PathBuf::from("/tmp/nib-journal.jsonl"));
        match r {
            Err(StartError::NoHfToken) => {}
            other => panic!("expected NoHfToken, got {other:?}"),
        }
        unsafe {
            if let Some(v) = saved {
                std::env::set_var("HF_TOKEN", v);
            }
        }
    }

    #[test]
    fn start_without_legacy_env_returns_deprecated_not_traindir() {
        let _guard = crate::state::TEST_ENV_LOCK.lock().unwrap();
        let saved_legacy = std::env::var("NIB_ALLOW_LEGACY_MODAL").ok();
        let saved_hf = std::env::var("HF_TOKEN").ok();
        unsafe {
            std::env::remove_var("NIB_ALLOW_LEGACY_MODAL");
            std::env::set_var("HF_TOKEN", "hf_test_dummy");
        }
        let s = TrainingState::default();
        let r = s.start(PathBuf::from("/tmp/nib-journal.jsonl"));
        match r {
            Err(StartError::LegacyModalDeprecated) => {}
            other => panic!("expected LegacyModalDeprecated, got {other:?}"),
        }
        // Message must point at the local QVAC path + Gemma legacy note.
        let msg = format!("{}", StartError::LegacyModalDeprecated);
        assert!(msg.contains("LFM2.5") && msg.contains("Qwen"), "missing Gemma legacy note: {msg}");
        assert!(msg.contains("NIB_ALLOW_LEGACY_MODAL"), "missing env pointer: {msg}");
        assert!(
            msg.contains("QVAC") || msg.contains("local"),
            "missing local path pointer: {msg}"
        );
        unsafe {
            match saved_legacy {
                Some(v) => std::env::set_var("NIB_ALLOW_LEGACY_MODAL", v),
                None => std::env::remove_var("NIB_ALLOW_LEGACY_MODAL"),
            }
            match saved_hf {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
        }
    }

    #[test]
    fn default_train_dir_resolves_against_home() {
        let _guard = crate::state::TEST_ENV_LOCK.lock().unwrap();
        let saved_home = std::env::var("HOME").ok();
        let saved_train = std::env::var_os("NIB_TRAIN_DIR");
        unsafe { std::env::remove_var("NIB_TRAIN_DIR"); }
        // Isolated HOME with no train dirs → None (never a phantom
        // ~/quill/train).
        let base = std::env::temp_dir().join(format!("nib-train-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        unsafe { std::env::set_var("HOME", &base); }
        assert_eq!(default_train_dir(), None);
        // Current layout wins when present.
        let cur = base.join("dev/nib/train");
        std::fs::create_dir_all(&cur).unwrap();
        assert_eq!(default_train_dir(), Some(cur.clone()));
        // Legacy fallback only when current layout absent but legacy exists.
        std::fs::remove_dir_all(base.join("dev/nib")).unwrap();
        let legacy = base.join("quill/train");
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(default_train_dir(), Some(legacy.clone()));
        let _ = std::fs::remove_dir_all(&base);
        unsafe {
            match saved_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match saved_train {
                Some(v) => std::env::set_var("NIB_TRAIN_DIR", v),
                None => std::env::remove_var("NIB_TRAIN_DIR"),
            }
        }
    }

    #[test]
    fn install_errors_when_no_adapter() {
        let s = TrainingState::default();
        let r = s.install(&PathBuf::from("/tmp/nib-dst.gguf"));
        assert!(r.is_err(), "should error when no job has produced an adapter");
    }

    #[test]
    fn reset_returns_to_idle() {
        let s = TrainingState::default();
        // Force a non-Idle state via direct mutation (we can't easily start
        // a real subprocess in tests without modal).
        if let Ok(mut g) = s.inner.lock() {
            g.state = JobState::Failed;
            g.error = Some("simulated".into());
        }
        s.reset();
        assert_eq!(s.status().state, JobState::Idle);
    }

    #[test]
    fn start_error_messages_are_actionable() {
        let strs = [
            format!("{}", StartError::NoHfToken),
            format!("{}", StartError::AlreadyRunning),
            format!("{}", StartError::ModalNotFound("/x".into())),
            format!("{}", StartError::TrainDirMissing(PathBuf::from("/y"))),
            format!("{}", StartError::LegacyModalDeprecated),
        ];
        // Each error mentions what to do or where to look.
        assert!(strs[0].contains("HF_TOKEN"));
        assert!(strs[1].contains("already"));
        assert!(strs[2].contains("modal"));
        assert!(strs[3].contains("/y"));
        // Gemma legacy path must say it won't load on LFM2.5/Qwen.
        assert!(strs[2].contains("LFM2.5") && strs[2].contains("Qwen"));
        assert!(strs[4].contains("LFM2.5") && strs[4].contains("Qwen"));
        assert!(strs[4].contains("NIB_ALLOW_LEGACY_MODAL"));
    }
}
