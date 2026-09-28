//! Tauri-managed application state.

use std::sync::Mutex;

use harper_core::Dialect;
use harper_core::linting::LintGroup;
use harper_core::spell::FstDictionary;

#[cfg(feature = "llm")]
use crate::inference;

/// Style / clarity rules that Harper 2.0 ships disabled by default but that we
/// want on in Nib. Single source of truth — both `CheckerState::new` and the
/// overlay's `fresh_linter()` route through `build_linter` so they stay in sync.
///
/// Selection rationale (from auditing `harper-core-2.0.0/src/linting/lint_group/mod.rs`,
/// the off-by-default set is small — only 7 rules):
/// - `BoringWords`     — flags "very / interesting / several / most / many",
///   low-noise nudge toward stronger word choice.
/// - `PossessiveNoun`  — catches missing-apostrophe possessives ("the dogs bone").
/// - `SpelledNumbers`  — suggests spelling out integers < 10
///   (`I have 3 cats.` -> `I have three cats.`).
///
/// Deliberately left OFF (would be noisy or wrong for casual writing):
/// - `NoOxfordComma` — conflicts with the curated `OxfordComma` rule.
/// - `AnotherThinkComing` / `ViciousCycle*` — pedantic / niche idiom corrections.
pub const EXTRA_RULES: &[&str] = &["BoringWords", "PossessiveNoun", "SpelledNumbers"];

/// Build a fresh `LintGroup` with Harper's curated defaults plus the small
/// set of style / clarity rules listed in [`EXTRA_RULES`].
pub fn build_linter() -> LintGroup {
    let mut lg = LintGroup::new_curated(FstDictionary::curated(), Dialect::American);
    for rule in EXTRA_RULES {
        lg.config.set_rule_enabled(*rule, true);
    }
    lg
}

pub struct CheckerState {
    pub linter: Mutex<LintGroup>,
}

impl CheckerState {
    pub fn new() -> Self {
        Self {
            linter: Mutex::new(build_linter()),
        }
    }
}

impl Default for CheckerState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "llm")]
pub struct RewriteState {
    pub engine: Mutex<Option<inference::RewriteEngine>>,
}

#[cfg(feature = "llm")]
impl RewriteState {
    pub fn from_path(path: Option<std::path::PathBuf>) -> Self {
        let adapter = personal_adapter_path().filter(|p| p.exists());
        // Skip a personal adapter trained against a different base model —
        // llama.cpp either refuses it or (worse) silently degrades output.
        let adapter = match (&path, adapter) {
            (Some(base), Some(ad)) if !personal_adapter_compatible(base, &ad) => {
                eprintln!(
                    "[nib] personal adapter at {} was trained on a different \
                     base model — skipping it (retrain to re-enable)",
                    ad.display()
                );
                None
            }
            (_, ad) => ad,
        };
        Self::from_paths(path, adapter)
    }

    /// Load the base model and optionally a personal LoRA adapter on top.
    /// `adapter_path` is only used when it exists on disk. If the adapter
    /// fails to load, we retry base-only — one bad personal-adapter.gguf
    /// must not disable all rewrite functionality.
    pub fn from_paths(
        path: Option<std::path::PathBuf>,
        adapter_path: Option<std::path::PathBuf>,
    ) -> Self {
        let adapter = adapter_path.filter(|p| p.exists());
        let engine = match path {
            Some(p) if p.exists() => match inference::RewriteEngine::load_with_adapter(&p, adapter.as_ref()) {
                Ok(e) => {
                    eprintln!("[nib] loaded model from {}", p.display());
                    Some(e)
                }
                Err(err) if adapter.is_some() => {
                    eprintln!(
                        "[nib] failed to load {} with adapter {:?}: {err:#} — retrying base-only",
                        p.display(),
                        adapter,
                    );
                    match inference::RewriteEngine::load(&p) {
                        Ok(e) => {
                            eprintln!("[nib] loaded model base-only from {}", p.display());
                            Some(e)
                        }
                        Err(err2) => {
                            eprintln!("[nib] base-only load also failed: {err2:#}");
                            None
                        }
                    }
                }
                Err(err) => {
                    eprintln!("[nib] failed to load {}: {err:#}", p.display());
                    None
                }
            },
            Some(p) => {
                eprintln!("[nib] model path does not exist: {}", p.display());
                None
            }
            None => {
                eprintln!(
                    "[nib] no model path resolved (NIB_MODEL unset, no bundled resource); rewrite disabled"
                );
                None
            }
        };
        Self {
            engine: Mutex::new(engine),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.engine.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    pub fn has_personal_adapter(&self) -> bool {
        self.engine
            .lock()
            .map(|g| g.as_ref().map(|e| e.has_adapter()).unwrap_or(false))
            .unwrap_or(false)
    }
}

#[cfg(not(feature = "llm"))]
pub struct RewriteState;

#[cfg(not(feature = "llm"))]
impl RewriteState {
    pub fn from_path(_: Option<std::path::PathBuf>) -> Self {
        Self
    }

    pub fn is_loaded(&self) -> bool {
        false
    }

    pub fn has_personal_adapter(&self) -> bool {
        false
    }
}

/// Resolve the model path. Priority:
///   1. `NIB_MODEL` env var (dev override — exact path, no adapter).
///   2. `~/Library/Application Support/Nib/config.json`'s
///      `selected_model` field, resolved via the `models::REGISTRY`.
///      Adapter entries return both base + adapter; standalone models
///      return just the base.
///   3. Fall back to the bundled default if the selected model isn't
///      fully installed on disk (e.g. user selected an adapter but the
///      base wasn't downloaded yet).
pub fn resolve_model_path(app: &tauri::App) -> Option<std::path::PathBuf> {
    resolve_model_paths(app).map(|p| p.base)
}

/// Like `resolve_model_path` but also returns the LoRA adapter path
/// when the selected model is an adapter entry. Returns `(base, None)`
/// for standalone models.
pub fn resolve_model_paths(app: &tauri::App) -> Option<crate::models::ModelPaths> {
    if let Ok(env) = std::env::var("NIB_MODEL") {
        let p = std::path::PathBuf::from(&env);
        if p.exists() {
            // Dev override: take env path as a raw base, no adapter wiring.
            return Some(crate::models::ModelPaths { base: p, adapter: None });
        }
        eprintln!(
            "[nib] NIB_MODEL={env} does not exist — falling through to config/bundled model"
        );
    }

    let selected_id = read_selected_model_id().unwrap_or_else(|| "lfm2.5-350m".to_string());
    if let Some(paths) = crate::models::resolve_paths(app, &selected_id) {
        eprintln!(
            "[nib] resolved selected model '{selected_id}' → {} (adapter={})",
            paths.base.display(),
            paths.adapter.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
        );
        return Some(paths);
    }
    eprintln!("[nib] selected model '{selected_id}' not fully installed — falling back");
    crate::models::resolve_paths(app, "lfm2.5-350m")
}

/// Peek at config.json's selected_model without spinning up the full
/// ConfigStore. Returns None if the file/key doesn't exist.
fn read_selected_model_id() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let path = std::path::PathBuf::from(home)
        .join("Library/Application Support/Nib/config.json");
    let raw = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("selected_model").and_then(|s| s.as_str()).map(|s| s.to_string())
}

/// Where Nib looks for an optional personal LoRA adapter on startup.
/// `~/Library/Application Support/Nib/personal-adapter.gguf` — same
/// directory as the journal so all per-user state is co-located.
pub fn personal_adapter_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push("Library/Application Support/Nib");
    p.push("personal-adapter.gguf");
    Some(p)
}

/// Sidecar next to the personal adapter recording which base model it was
/// trained on (written by `training::TrainingState::install`).
pub fn adapter_meta_path(adapter: &std::path::Path) -> std::path::PathBuf {
    let mut s = adapter.as_os_str().to_os_string();
    s.push(".meta.json");
    std::path::PathBuf::from(s)
}

/// True when the personal adapter can be layered on `base`. Adapters
/// trained before base-tagging existed have no sidecar — treated as
/// compatible (pre-tag behavior, at the user's own risk).
#[cfg(feature = "llm")]
fn personal_adapter_compatible(base: &std::path::Path, adapter: &std::path::Path) -> bool {
    let meta = adapter_meta_path(adapter);
    let Ok(raw) = std::fs::read_to_string(&meta) else { return true };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { return true };
    let Some(trained_on) = v.get("base_filename").and_then(|s| s.as_str()) else { return true };
    let current = base.file_name().and_then(|f| f.to_str()).unwrap_or("");
    trained_on == current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn personal_adapter_path_uses_home() {
        // Save & restore the real HOME so this test doesn't leak.
        let saved = std::env::var("HOME").ok();
        // SAFETY: tests are single-threaded by default for env mutation here.
        unsafe { std::env::set_var("HOME", "/tmp/nib-test-home"); }
        let got = personal_adapter_path();
        let expected = std::path::PathBuf::from(
            "/tmp/nib-test-home/Library/Application Support/Nib/personal-adapter.gguf",
        );
        assert_eq!(got, Some(expected));
        // Restore.
        unsafe {
            match saved {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[cfg(feature = "llm")]
    #[test]
    fn rewrite_state_with_no_model_path_is_not_loaded() {
        let s = RewriteState::from_paths(None, None);
        assert!(!s.is_loaded(), "no model path → should not be loaded");
        assert!(!s.has_personal_adapter());
    }

    #[cfg(feature = "llm")]
    #[test]
    fn rewrite_state_with_missing_model_path_is_not_loaded() {
        let s = RewriteState::from_paths(
            Some(std::path::PathBuf::from("/tmp/nib-no-such-model-xyz.gguf")),
            None,
        );
        assert!(!s.is_loaded(), "nonexistent model path → not loaded");
        assert!(!s.has_personal_adapter());
    }

    #[cfg(feature = "llm")]
    #[test]
    fn rewrite_state_with_missing_adapter_doesnt_break_init() {
        // No model AND no adapter — verifies the adapter-missing branch
        // doesn't panic / error out the state constructor.
        let s = RewriteState::from_paths(
            None,
            Some(std::path::PathBuf::from(
                "/tmp/nib-no-such-adapter-xyz.gguf",
            )),
        );
        assert!(!s.is_loaded());
        assert!(!s.has_personal_adapter(), "missing adapter file → no personal adapter");
    }

    /// End-to-end test gated behind NIB_TEST_MODEL env var. When the user
    /// has artifacts on disk, `cargo test -- --ignored` exercises the full
    /// model+adapter load + rewrite path. CI without artifacts skips this.
    #[cfg(feature = "llm")]
    #[test]
    #[ignore]
    fn full_model_load_and_rewrite_if_artifacts_present() {
        let Ok(model) = std::env::var("NIB_TEST_MODEL") else {
            eprintln!("NIB_TEST_MODEL not set; skipping");
            return;
        };
        let adapter = std::env::var("NIB_TEST_ADAPTER").ok().map(std::path::PathBuf::from);
        let s = RewriteState::from_paths(Some(std::path::PathBuf::from(&model)), adapter.clone());
        assert!(s.is_loaded(), "model at {model} should load");
        if adapter.is_some() {
            assert!(s.has_personal_adapter(), "adapter at {adapter:?} should load");
        }
    }
}
