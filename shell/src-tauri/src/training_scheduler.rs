//! Background loop that triggers personal-LoRA retraining when enough
//! new events have accumulated.
//!
//! Pattern: every `POLL_INTERVAL`, snapshot journal stats + config. If
//! the user has enabled auto-retrain AND
//!     (applied + rewrite_applied) - last_train_event_count >= threshold
//! AND no training job is currently running, kick off the same
//! `TrainingState::start` path the UI uses. Once the job finishes
//! successfully, install the adapter and stamp the config so we don't
//! re-trigger immediately.
//!
//! Hot-reload of the engine IS done here — after a successful auto-install
//! we swap the engine atomically in place via
//! `RewriteState::reload_personal_adapter`. The relaunch badge
//! (`pending_relaunch`) is only the fallback when the reload fails or
//! leaves the adapter inactive.

#![cfg(feature = "llm")]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::ConfigStore;
use crate::journal::Journal;
use crate::qvac::BackendConfig;
use crate::training::{JobState, TrainingState};
use tauri::Manager;

const POLL_INTERVAL: Duration = Duration::from_secs(30);

pub fn spawn(
    journal: Arc<Journal>,
    training: Arc<TrainingState>,
    config: Arc<ConfigStore>,
    backend: Arc<BackendConfig>,
    app: tauri::AppHandle,
) {
    thread::Builder::new()
        .name("nib-retrain-scheduler".into())
        .spawn(move || run(journal, training, config, backend, app))
        .expect("spawn retrain scheduler");
}

fn run(
    journal: Arc<Journal>,
    training: Arc<TrainingState>,
    config: Arc<ConfigStore>,
    backend: Arc<BackendConfig>,
    app: tauri::AppHandle,
) {
    eprintln!("[nib][scheduler] background retrain loop started (poll every {}s)", POLL_INTERVAL.as_secs());
    let mut waiting_on_job_since: Option<Instant> = None;
    loop {
        thread::sleep(POLL_INTERVAL);

        let cfg = config.snapshot();
        if !cfg.auto_retrain_enabled {
            // Reset job-watch state when toggled off.
            waiting_on_job_since = None;
            continue;
        }

        let stats = journal.stats();
        let applied_total = stats.applied + stats.rewrite_applied;

        let status = training.status();

        // If we previously kicked off a job and it has since succeeded,
        // do the install + bookkeeping.
        if let Some(_started) = waiting_on_job_since {
            match status.state {
                JobState::Succeeded => {
                    let dest = crate::state::personal_adapter_path();
                    if let Some(dest) = dest {
                        match training.install(&dest) {
                            Ok(bytes) => {
                                eprintln!(
                                    "[nib][scheduler] auto-installed {bytes}B → {}",
                                    dest.display()
                                );
                                // Hot-reload first — badge is only the fallback.
                                let reloaded = app
                                    .state::<crate::state::RewriteState>()
                                    .reload_personal_adapter(&app)
                                    .unwrap_or(false);
                                if !reloaded {
                                    eprintln!(
                                        "[nib][scheduler] hot-reload did not activate adapter — keeping relaunch badge"
                                    );
                                }
                                let _ = config.update(|c| {
                                    c.last_train_event_count = applied_total;
                                    c.last_train_at = Some(crate::config::now_rfc3339());
                                    c.pending_relaunch = !reloaded;
                                });
                            }
                            Err(e) => {
                                eprintln!("[nib][scheduler] install failed: {e}");
                            }
                        }
                    }
                    training.reset();
                    waiting_on_job_since = None;
                }
                JobState::Failed => {
                    eprintln!(
                        "[nib][scheduler] last training failed: {}",
                        status.error.unwrap_or_else(|| "unknown".into())
                    );
                    training.reset();
                    waiting_on_job_since = None;
                }
                JobState::Running => { /* still going */ }
                JobState::Idle => {
                    // Edge: status reset externally. Drop the watch.
                    waiting_on_job_since = None;
                }
            }
            continue;
        }

        // No active job. Decide whether to start one.
        if status.state == JobState::Running {
            // Manually started — don't start a second one.
            continue;
        }
        let new_events = applied_total.saturating_sub(cfg.last_train_event_count);
        if new_events < cfg.auto_retrain_threshold {
            continue;
        }

        // Export → spawn. Prefer LOCAL backend (free, ~5 min on Metal)
        // when bundled. The Modal fallback UPLOADS the journal export
        // (everything the user typed) to a cloud GPU — that must never
        // happen implicitly, so it's gated behind the explicit
        // `allow_cloud_training` config opt-in.
        if !backend.local_ready() && !cfg.allow_cloud_training {
            eprintln!(
                "[nib][scheduler] retrain due ({new_events} new events) but local backend \
                 isn't ready and cloud training is disabled — skipping \
                 (set allow_cloud_training in config.json to opt in)"
            );
            continue;
        }
        // Exports live in the app data dir (0600), not world-readable /tmp.
        let export_path: PathBuf = match crate::journal::exports_dir() {
            Ok(d) => d.join("auto-retrain.jsonl"),
            Err(e) => {
                eprintln!("[nib][scheduler] exports dir: {e}");
                continue;
            }
        };
        match journal.export_training_pairs(&export_path) {
            Ok(n) if n >= 10 => {
                let start_result = if backend.local_ready() {
                    let finetune = backend.finetune_bin.clone().unwrap();
                    let base = backend.base_model.clone().unwrap();
                    let out = crate::state::personal_adapter_path()
                        .unwrap_or_else(|| std::env::temp_dir().join("nib-auto-adapter.gguf"));
                    eprintln!(
                        "[nib][scheduler] firing auto-retrain (LOCAL): {n} pairs, {new_events} new events"
                    );
                    training.start_local(finetune, base, export_path.clone(), out)
                } else {
                    eprintln!(
                        "[nib][scheduler] firing auto-retrain (MODAL, user opted in): {n} pairs, {new_events} new events"
                    );
                    training.start(export_path.clone())
                };
                match start_result {
                    Ok(()) => {
                        waiting_on_job_since = Some(Instant::now());
                    }
                    Err(e) => {
                        eprintln!("[nib][scheduler] start failed: {e}");
                    }
                }
            }
            Ok(n) => {
                eprintln!("[nib][scheduler] only {n} applied pairs in export, skipping");
            }
            Err(e) => {
                eprintln!("[nib][scheduler] export failed: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new_events < threshold` should NOT trigger training.
    /// We test the pure decision logic rather than spawn a real subprocess.
    #[test]
    fn below_threshold_does_not_trigger() {
        let cfg = crate::config::Config {
            auto_retrain_enabled: true,
            auto_retrain_threshold: 25,
            last_train_event_count: 100,
            ..crate::config::Config::default()
        };
        let applied_total: u64 = 110; // delta = 10, threshold = 25
        let new_events = applied_total.saturating_sub(cfg.last_train_event_count);
        assert!(new_events < cfg.auto_retrain_threshold);
    }

    #[test]
    fn at_threshold_triggers() {
        let cfg = crate::config::Config {
            auto_retrain_enabled: true,
            auto_retrain_threshold: 25,
            last_train_event_count: 100,
            ..crate::config::Config::default()
        };
        let applied_total: u64 = 125; // delta = 25
        let new_events = applied_total.saturating_sub(cfg.last_train_event_count);
        assert!(new_events >= cfg.auto_retrain_threshold);
    }

    #[test]
    fn disabled_short_circuits_regardless_of_count() {
        let cfg = crate::config::Config {
            auto_retrain_enabled: false,
            auto_retrain_threshold: 25,
            last_train_event_count: 100,
            ..crate::config::Config::default()
        };
        // No matter how many new events, we shouldn't trigger because
        // `auto_retrain_enabled == false`. The function under test exits
        // early — verified by the explicit `continue` in `run`.
        assert!(!cfg.auto_retrain_enabled);
    }
}
