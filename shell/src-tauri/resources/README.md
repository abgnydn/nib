# Bundled resources

Tauri's build script validates that every `bundle.resources` path in the
active config exists on disk. Model GGUFs are git-ignored (root `.gitignore`
has `*.gguf`), so the files below must be placed in this directory manually
before building each flavor:

| Build flavor | Config | Files required here |
|---|---|---|
| **Default** | `tauri.conf.json` | `Qwen3-0.6B-Q4_K_M.gguf` (~462 MB, bundled default model) |
| **Full** | `tauri.conf.full.json` (`cargo tauri build --config tauri.conf.full.json`) | `Qwen3-0.6B-Q4_K_M.gguf` + `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~940 MB premium base) + `nib-faithful-f16.gguf` (~36 MB premium LoRA adapter) |

Filenames must match the model registry in `src/models.rs` exactly —
`resolve_path` looks up `resources/{filename}` inside the built `.app`, so a
renamed file bundles fine but is invisible to the app at runtime. (The old
v2.0 `nib-qwen-v2-q4_k_m.gguf` merged model predates the base+adapter split
and no registry entry resolves it.)

`qvac/` holds the QVAC Fabric binaries (`llama-cli`, `llama-finetune-lora`,
dylibs, metallib) staged per-machine by `scripts/install-dev.sh`
(`prepare_qvac`); only its README placeholder is committed.

CI has no model files and stubs the default entry with
`touch resources/Qwen3-0.6B-Q4_K_M.gguf` (see `.github/workflows/ci.yml`).
