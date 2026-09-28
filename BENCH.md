# BENCH — Nib latency numbers

> Numbers below are **to-fill after the first run on your machine**.
> Do not copy them as claims until you have run the commands yourself.

## Harper (`wire::check_text_with` via `state::build_linter`)

Run:

```sh
cd shell/src-tauri
cargo bench --features llm
```

Compile-only check (what CI / `scripts/test.sh` runs — fast, no measurements):

```sh
cd shell/src-tauri
cargo bench --no-run --features llm
```

| input | chars | p50 | p95 | notes |
|-------|-------|-----|-----|-------|
| short | 50 | ~285 µs | ~306 µs | `harper_check/short`, 2026-09-28, Apple M2 Max (arm64) |
| medium | 500 | ~1.91 ms | ~2.08 ms | `harper_check/medium`, 2026-09-28, Apple M2 Max (arm64) |
| long | 2000 | ~6.86 ms | ~7.25 ms | `harper_check/long`, 2026-09-28, Apple M2 Max (arm64) |

Measured 2026-09-28 on MacBook Pro (Mac14,5, Apple M2 Max, arm64, 32 GB) via
`cargo bench --features llm --bench harper_bench -- --sample-size 20`
(median point-estimate = p50; p95 from per-iter `sample.json`).

Criterion reports live in `shell/src-tauri/target/criterion/harper_check/`
after a run (use the `report/index.html` per group for p50/p95).

## Rewrite tok/s (`nib-rewrite`, feature `llm`)

The rewrite binary prints wall-clock timing to stderr
(`loaded in …s`, `rewrote in …s (N chars in, M chars out)`).
Derive tok/s from a timed run:

```sh
cd shell/src-tauri
cargo build --features llm --bin nib-rewrite --release
time ./target/release/nib-rewrite \
  --model "$NIB_MODEL" \
  --text "This is an test of the Harper grammer checker."
```

`NIB_MODEL` example:

```sh
export NIB_MODEL=~/Library/Application\ Support/Nib/models/lfm2.5-350m-q4_k_m.gguf
# or any dev GGUF:
# export NIB_MODEL=~/quill/train/checkpoints/nib-q4_k_m.gguf
```

| model | prompt chars | rewrite s (wall) | tok/s | notes |
|-------|--------------|------------------|-------|-------|
| lfm2.5-350m-q4_k_m (cold, first run after build) | 46 | 0.36s | ~272 tok/s out (~32 tok/s in) | 2026-09-28, MacBook Pro Mac14,5 Apple M2 Max arm64 32GB, `rewrote in 0.36s (46 in, 392 out)`, `loaded in 9.11s`, `time` total 10.20s |
| lfm2.5-350m-q4_k_m (warm rerun) | 46 | 0.21s | ~467 tok/s out (~55 tok/s in) | 2026-09-28, same machine, `rewrote in 0.21s (46 in, 392 out)`, `loaded in 0.09s`, `/usr/bin/time -p real 0.54s` |

> tok/s here is approximate (chars → tokens varies); report the raw
> wall time alongside any derived rate so repeats stay comparable.
