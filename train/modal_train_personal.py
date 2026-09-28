# ─── LEGACY + KNOWN MISMATCH — read before relying on this ───────────────────
# The Rust app's Modal personal-training FALLBACK (shell .../training.rs::start)
# still shells out to this script, but it trains a *Gemma 3 270M* adapter that
# will NOT load on the shipping LFM2.5-350M / Qwen 2.5-1.5B bases (different
# architectures + LoRA shapes). The supported personal-training path is now
# LOCAL QVAC (shell .../training_local.rs, base-model-agnostic, free, on-device);
# the scheduler prefers it whenever the QVAC binaries are bundled. Treat this
# Modal path as unmaintained until it's rebased onto the v2.x models.
# ─────────────────────────────────────────────────────────────────────────────
"""Personal LoRA fine-tune on top of Quill's CoEdIT-merged base model.

WARNING (legacy): this script trains a Gemma-3-270M adapter — NOT loadable
on the shipped LFM2.5-350M / Qwen 2.5-1.5B bases. Legacy path pending rebase.

Takes a user journal exported from Nib's "⤓ Export" button (CoEdIT-style
`{src, tgt}` JSONL), interleaves it with random CoEdIT pairs at 50/50 to
prevent catastrophic forgetting, runs a small LoRA on Modal L4, converts
to GGUF, and downloads the adapter to `./checkpoints/personal-adapter.gguf`.

Nib auto-detects an adapter at
`~/Library/Application Support/Nib/personal-adapter.gguf` on next launch.

USAGE (~15 min on Modal L4, ~$0.20):

    cd ~/quill/train
    HF_TOKEN=hf_xxx .venv/bin/modal run modal_train_personal.py \\
        --journal /tmp/nib-training-2026-05-21T15-30-45.jsonl

ARGUMENTS (tweakable):
    --epochs            default 2 — keep it small, this is a delta on top
    --lr                default 5e-5 — half the original CoEdIT run's LR
    --rank              default 16 — same as the original CoEdIT adapter
    --rehearsal-frac    default 0.5 — 50% CoEdIT rehearsal per batch
    --max-personal      default 2000 — cap to avoid overfit on tiny journals
"""

import os
import pathlib
import subprocess
import sys

# Retired Modal-Gemma legacy guard — prevent accidental Gemma training.
# This script trains a Gemma-3-270M adapter that won't load on LFM2.5/Qwen;
# the supported path is the local QVAC backend. Require explicit opt-in.
if os.environ.get("NIB_ALLOW_LEGACY_MODAL") != "1":
    print("LEGACY WARNING: modal_train_personal.py trains a Gemma-3-270M adapter")
    print("that won't load on the shipped LFM2.5-350M / Qwen 2.5-1.5B bases.")
    print("The supported personal-training path is the local QVAC backend.")
    print("To override, set NIB_ALLOW_LEGACY_MODAL=1 (and allow_cloud_training).")
    sys.exit(2)

import modal

APP_NAME = "nib-train-personal"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install("unsloth", "sentencepiece", "protobuf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("nib-artifacts", create_if_missing=True)

GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{src}<end_of_turn>\n"
    "<start_of_turn>model\n{tgt}<end_of_turn>"
)


@app.function(
    gpu="L4",
    timeout=60 * 60,
    volumes={"/artifacts": volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train_personal(
    journal_jsonl: bytes,
    epochs: int = 2,
    lr: float = 5e-5,
    rank: int = 16,
    rehearsal_frac: float = 0.5,
    max_personal: int = 2000,
    base_model: str = "unsloth/gemma-3-270m-it",
    base_adapter: str = "/artifacts/gemma3-270m-coedit-lora",
) -> dict:
    """Train a personal LoRA on top of the existing CoEdIT-trained model."""
    import json
    import random
    import time
    from pathlib import Path

    from datasets import Dataset, load_dataset
    from huggingface_hub import login
    from unsloth import FastLanguageModel, is_bfloat16_supported

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    login(token=token)

    # --- Parse personal journal ----------------------------------------
    personal: list[dict] = []
    for line in journal_jsonl.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("src") and row.get("tgt"):
            personal.append({"src": row["src"], "tgt": row["tgt"]})
    if len(personal) < 10:
        raise RuntimeError(f"need ≥10 personal pairs, got {len(personal)}")
    if len(personal) > max_personal:
        random.shuffle(personal)
        personal = personal[:max_personal]
    print(f"[quill.personal] {len(personal)} personal pairs")

    # --- Load + interleave with CoEdIT rehearsal -----------------------
    coedit = load_dataset("grammarly/coedit", split="train")
    n_rehearsal = int(len(personal) * rehearsal_frac / (1 - rehearsal_frac))
    rehearsal_idx = random.sample(range(len(coedit)), min(n_rehearsal, len(coedit)))
    rehearsal = [{"src": coedit[i]["src"], "tgt": coedit[i]["tgt"]} for i in rehearsal_idx]
    print(f"[quill.personal] {len(rehearsal)} rehearsal pairs ({rehearsal_frac*100:.0f}%)")

    combined = personal + rehearsal
    random.shuffle(combined)
    ds = Dataset.from_list(combined).map(
        lambda r: {"text": GEMMA_TEMPLATE.format(src=r["src"], tgt=r["tgt"])},
        remove_columns=["src", "tgt"],
    )

    # --- Load base + previous adapter ----------------------------------
    print(f"[quill.personal] loading {base_model} + existing adapter at {base_adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=256,
        dtype=None,
        load_in_4bit=True,
    )
    # If we have a previous adapter, merge it into the base before training
    # the new personal LoRA on top. (Otherwise we'd be stacking deltas
    # and overflowing the q4 base's expected weight magnitudes.)
    if Path(base_adapter).exists():
        from peft import PeftModel
        merged = PeftModel.from_pretrained(model, base_adapter)
        model = merged.merge_and_unload()
        print(f"[quill.personal] merged base adapter from {base_adapter}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=1337,
    )

    # --- Train ----------------------------------------------------------
    from trl import SFTConfig, SFTTrainer

    out_dir = Path("/artifacts/personal-lora")
    if out_dir.exists():
        subprocess.run(["rm", "-rf", str(out_dir)], check=True)
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=1337,
        max_length=256,
        packing=False,
        optim="adamw_8bit",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds, processing_class=tokenizer,
    )

    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    print(f"[quill.personal] trained in {train_secs/60:.1f} min  final_loss={result.training_loss:.3f}")
    trainer.save_model(str(out_dir))

    # --- Convert PEFT adapter → GGUF LoRA ------------------------------
    llama_dir = Path("/tmp/llama.cpp")
    if not llama_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(llama_dir)],
            check=True,
        )
    subprocess.run(["pip", "install", "-q", "-r", str(llama_dir / "requirements.txt")], check=True)

    gguf_out = Path("/artifacts/personal-adapter.gguf")
    subprocess.run(
        ["python", str(llama_dir / "convert_lora_to_gguf.py"),
         "--base", base_model, str(out_dir),
         "--outfile", str(gguf_out), "--outtype", "f16"],
        check=True,
    )
    size_mb = gguf_out.stat().st_size / (1024 * 1024)
    print(f"[quill.personal] GGUF LoRA written: {gguf_out} ({size_mb:.1f} MB)")
    volume.commit()
    return {
        "gguf_path": str(gguf_out),
        "gguf_size_mb": size_mb,
        "train_seconds": train_secs,
        "final_loss": result.training_loss,
        "personal_pairs": len(personal),
        "rehearsal_pairs": len(rehearsal),
    }


@app.local_entrypoint()
def main(journal: str, epochs: int = 2, lr: float = 5e-5):
    """Read the journal file locally, send the bytes to Modal, train,
    download the resulting GGUF LoRA into ./checkpoints/, and print the
    Nib install hint."""
    print("=" * 76)
    print("WARNING: this trains a Gemma-3-270M adapter — NOT loadable on the")
    print("shipped LFM2.5-350M / Qwen 2.5-1.5B bases; legacy path pending rebase.")
    print("The supported personal-training path is the local QVAC backend.")
    print("=" * 76)
    journal_path = pathlib.Path(journal).expanduser()
    if not journal_path.exists():
        raise SystemExit(f"journal not found: {journal_path}")
    journal_bytes = journal_path.read_bytes()
    print(f"[quill.personal] sending {journal_path} ({len(journal_bytes)} bytes) to Modal …")

    result = train_personal.remote(
        journal_jsonl=journal_bytes,
        epochs=epochs,
        lr=lr,
    )
    print(f"[quill.personal] result: {result}")

    local_dir = pathlib.Path("./checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "personal-adapter.gguf"
    print(f"[quill.personal] downloading GGUF → {local_path}")
    subprocess.run(
        ["modal", "volume", "get", "--force", "nib-artifacts",
         "personal-adapter.gguf", str(local_path)],
        check=True,
    )

    install_path = pathlib.Path.home() / "Library/Application Support/Nib/personal-adapter.gguf"
    install_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print(f"[quill.personal] DONE  {local_path}  ({result['gguf_size_mb']:.1f} MB)")
    print("[quill.personal] to use in Nib, drop the adapter into:")
    print(f"    {install_path}")
    print("[quill.personal] one-liner:")
    print(f"    cp {local_path} '{install_path}' && killall nib 2>/dev/null; open -a Nib")
    print("[quill.personal] NOTE: this adapter is Gemma-3-270M-based and will "
          "NOT load on the shipped LFM2.5/Qwen bases (legacy path pending rebase).")
