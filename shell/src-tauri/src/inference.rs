//! GGUF-backed rewrite engine. Wraps llama-cpp-2.
//!
//! Compiled only when the `llm` feature is on. Loads the model once at
//! startup and reuses the backend across rewrite calls. Each rewrite
//! builds a fresh context (cheap for small models) so concurrent calls
//! don't share KV cache state.

use std::borrow::Cow;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// ChatML template — used by both LFM2.5-Instruct and Qwen 2.5-Instruct,
/// with proper system + user role separation.
/// BOS (`<|startoftext|>` for LFM2.5, none for Qwen) is added by
/// `AddBos::Always` at tokenize time per the model's GGUF metadata.
///
/// Why system vs user-only: when we crammed instruction + source into
/// ONE user message ("You are a copy editor. ... source text") the
/// 1.2B model started meta-narrating ("The user has requested a
/// rewrite..."). With proper roles, the model understands the system
/// message is its job and the user message is the data to operate on.
const PROMPT_TEMPLATE: &str =
    "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n";

/// Generation stop marker for LFM2.5 ChatML.
const STOP_MARKER: &str = "<|im_end|>";

/// Qwen3 no-think suffix. Appended to the user source (engine-level, NOT
/// in JS/PY `composeInstruction`) when the loaded model is a thinking
/// model (general.architecture == "qwen3"). Probe: 6/6 non-empty vs 2/6,
/// ~0.2s vs ~1.0s, never leaks. Non-thinking models (LFM2.5, Qwen2.5)
/// must NEVER see this — LFM echo risk untested.
const NO_THINK_SUFFIX: &str = " /no_think";

/// True only for thinking architectures. Exact match on purpose:
/// "qwen2", "qwen2.5", "lfm2", … must not match.
fn arch_disables_thinking(arch: &str) -> bool {
    arch == "qwen3"
}

/// Sniff the loaded GGUF for a thinking architecture. Missing/unreadable
/// metadata → false (never append the suffix blindly).
fn model_disables_thinking(model: &LlamaModel) -> bool {
    model
        .meta_val_str("general.architecture")
        .map(|a| arch_disables_thinking(&a))
        .unwrap_or(false)
}

/// Append [`NO_THINK_SUFFIX`] to the user source when thinking is
/// disabled for this engine. Returns borrowed when disabled=false so
/// non-thinking models see byte-identical prompts to before.
fn apply_no_think_suffix<'a>(source: &'a str, disable_thinking: bool) -> Cow<'a, str> {
    if disable_thinking {
        Cow::Owned(format!("{source}{NO_THINK_SUFFIX}"))
    } else {
        Cow::Borrowed(source)
    }
}

/// Strip Qwen3-style `<think>...</think>` reasoning blocks (model-agnostic
/// cleanup). Removes ALL blocks, case-sensitive exact tags, multiline-safe.
/// An unclosed `<think>` drops to end-of-text — a truncated thought must
/// never ship to the user.
fn strip_think_blocks(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find(OPEN) {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + OPEN.len()..];
                match after_open.find(CLOSE) {
                    None => break, // unclosed: drop to end-of-text
                    Some(end) => rest = &after_open[end + CLOSE.len()..],
                }
            }
        }
    }
    out
}

/// Apply post-generation cleanup: STOP_MARKER stripping, then
/// `<think>` stripping. If nothing remains (model ONLY thought), force
/// `truncated=true` — callers already refuse to paste truncated output.
fn finalize_generation(raw: &str, truncated: bool) -> Generation {
    let no_stop = raw.replace(STOP_MARKER, "");
    let text = strip_think_blocks(&no_stop).trim().to_string();
    if text.is_empty() {
        return Generation {
            text,
            truncated: true,
        };
    }
    Generation { text, truncated }
}

/// Default system message when no explicit instruction is supplied.
/// Kept short on purpose — small models degrade with long checklists
/// (max 3 sentences).
const DEFAULT_INSTRUCTION: &str = "You are a copy editor. Fix grammar and improve clarity while \
preserving facts, numbers and names verbatim and adding no new ideas, keeping word count within ±20%. \
Output only the corrected text, nothing else.";

/// Heuristic token estimate: ~4 chars per token (English average).
/// Used only for pre-splitting long inputs, never for the exact
/// context-size check (that uses real tokenization in `generate`).
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() + 3) / 4
}

/// Split a paragraph into sentences on `.`/`!`/`?` boundaries.
/// The terminator stays with the preceding sentence; splits only when
/// the terminator is followed by whitespace or end-of-input so
/// abbreviations don't shatter mid-paragraph.
fn split_sentences(paragraph: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = paragraph.char_indices().collect();
    for (idx, (byte_i, ch)) in chars.iter().enumerate() {
        if *ch == '.' || *ch == '!' || *ch == '?' {
            let next_is_boundary = match chars.get(idx + 1) {
                None => true,
                Some((_, nc)) => nc.is_whitespace(),
            };
            if next_is_boundary {
                let end = byte_i + ch.len_utf8();
                let s = paragraph[start..end].trim().to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                start = end;
            }
        }
    }
    let tail = paragraph[start..].trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    if out.is_empty() {
        let t = paragraph.trim().to_string();
        if !t.is_empty() {
            out.push(t);
        }
    }
    out
}

/// Pack whitespace-separated words into sub-chunks that each fit
/// `max_source_tokens`. Never splits inside a word; a single word
/// larger than the budget becomes its own (over-budget) chunk rather
/// than being cut.
fn split_long_sentence(sentence: &str, max_source_tokens: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in sentence.split_whitespace() {
        if cur.is_empty() {
            if estimate_tokens(w) > max_source_tokens {
                out.push(w.to_string());
                continue;
            }
            cur.push_str(w);
        } else {
            let joined = format!("{cur} {w}");
            if estimate_tokens(&joined) <= max_source_tokens {
                cur = joined;
            } else {
                out.push(std::mem::take(&mut cur));
                if estimate_tokens(w) > max_source_tokens {
                    out.push(w.to_string());
                } else {
                    cur.push_str(w);
                }
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Split `text` into chunks that each fit `max_source_tokens`
/// (by [`estimate_tokens`]). Splits on paragraph (`\n\n`) boundaries
/// first, then sentence (`.!?`) boundaries, then word boundaries —
/// never inside a word.
pub fn split_for_context(text: &str, max_source_tokens: usize) -> Vec<String> {
    if max_source_tokens == 0 {
        return vec![text.to_string()];
    }
    if estimate_tokens(text) <= max_source_tokens {
        return vec![text.to_string()];
    }
    let joiner = if text.contains("\n\n") {
        "\n\n"
    } else {
        " "
    };
    // 1. Break into fitting pieces (paragraph → sentence → words).
    let mut pieces: Vec<String> = Vec::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if estimate_tokens(para) <= max_source_tokens {
            pieces.push(para.to_string());
            continue;
        }
        for sent in split_sentences(para) {
            if estimate_tokens(&sent) <= max_source_tokens {
                pieces.push(sent);
            } else {
                pieces.extend(split_long_sentence(&sent, max_source_tokens));
            }
        }
    }
    if pieces.is_empty() {
        return vec![text.to_string()];
    }
    // 2. Greedily pack pieces into chunks.
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if cur.is_empty() {
            cur = p;
            continue;
        }
        let joined = format!("{cur}{joiner}{p}");
        if estimate_tokens(&joined) <= max_source_tokens {
            cur = joined;
        } else {
            chunks.push(std::mem::take(&mut cur));
            cur = p;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Send/Sync wrapper around the raw LoRA adapter handle.
///
/// `LlamaLoraAdapter` holds a `NonNull<llama_adapter_lora>` which the
/// upstream crate doesn't mark `Send` even though the underlying llama.cpp
/// adapter handle is process-local and thread-safe under external
/// synchronisation. We hold the inner adapter inside a `Mutex`, so all
/// mutating access is serialised — `unsafe impl Send + Sync` is sound for
/// our use.
struct AdapterCell(LlamaLoraAdapter);
unsafe impl Send for AdapterCell {}
unsafe impl Sync for AdapterCell {}

/// One finished generation. `truncated` = hit the max-new-tokens cap
/// without a natural stop (EOG token / stop marker) — the text is cut
/// off mid-thought and callers must not silently paste it over the
/// user's original.
#[derive(Clone, Debug)]
pub struct Generation {
    pub text: String,
    pub truncated: bool,
}

pub struct RewriteEngine {
    backend: LlamaBackend,
    model: LlamaModel,
    /// Optional personal LoRA adapter applied on top of the base model on
    /// every fresh context. `None` = base model only.
    adapter: Option<Mutex<AdapterCell>>,
    adapter_path: Option<PathBuf>,
    adapter_scale: f32,
    ctx_size: u32,
    max_new_tokens: i32,
    /// True when the loaded model is a thinking model (Qwen3). When set,
    /// [`NO_THINK_SUFFIX`] is appended to the user source in `generate()`
    /// before templating. Auto-detected from GGUF
    /// `general.architecture` at load; false for everything else.
    disable_thinking: bool,
}

impl RewriteEngine {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_adapter(model_path, None::<PathBuf>)
    }

    /// Load the base model and (optionally) attach a LoRA adapter. The
    /// adapter is loaded once at startup and re-applied to each new
    /// inference context.
    pub fn load_with_adapter<P: AsRef<Path>>(
        model_path: impl AsRef<Path>,
        adapter_path: Option<P>,
    ) -> Result<Self> {
        let backend = LlamaBackend::init().context("LlamaBackend::init")?;
        let model_params = pin!(LlamaModelParams::default());
        let model = LlamaModel::load_from_file(&backend, model_path.as_ref(), &model_params)
            .with_context(|| format!("loading GGUF at {}", model_path.as_ref().display()))?;

        let (adapter, adapter_path_owned) = match adapter_path {
            Some(p) => {
                let path: PathBuf = p.as_ref().to_path_buf();
                let ad = model
                    .lora_adapter_init(&path)
                    .with_context(|| format!("loading LoRA adapter at {}", path.display()))?;
                eprintln!("[nib] personal LoRA adapter loaded from {}", path.display());
                (Some(Mutex::new(AdapterCell(ad))), Some(path))
            }
            None => (None, None),
        };

        // Arch sniffing: Qwen3 thinks by default; disable it engine-level.
        // Anything else (LFM2.5, Qwen2.5, unknown) stays exactly as before.
        let disable_thinking = model_disables_thinking(&model);
        if disable_thinking {
            eprintln!("[nib] qwen3 thinking model detected — /no_think suffix enabled");
        }

        Ok(Self {
            backend,
            model,
            adapter,
            adapter_path: adapter_path_owned,
            adapter_scale: 1.0,
            ctx_size: 2048,
            max_new_tokens: 256,
            disable_thinking,
        })
    }

    /// True when this engine appends the Qwen3 `/no_think` suffix.
    pub fn disable_thinking(&self) -> bool {
        self.disable_thinking
    }

    /// Explicit override (tests / future registry wiring). Normal path is
    /// auto-detect at load; this only exists so callers can force the
    /// suffix on or off without reloading.
    pub fn set_disable_thinking(&mut self, v: bool) {
        self.disable_thinking = v;
    }

    pub fn has_adapter(&self) -> bool {
        self.adapter.is_some()
    }

    pub fn adapter_path(&self) -> Option<&PathBuf> {
        self.adapter_path.as_ref()
    }

    /// Run a single-shot rewrite. Convenience wrapper that buffers tokens
    /// from `rewrite_streaming` into a single String. Use the streaming
    /// variant when you want per-token UI updates.
    pub fn rewrite(&self, text: &str, instruction: Option<&str>) -> Result<Generation> {
        self.rewrite_streaming(text, instruction, |_| {})
    }

    /// Streaming rewrite. `on_token` is invoked with each piece of newly
    /// generated text as it's decoded; the same accumulated text is also
    /// returned at the end (with `<|im_end|>` stripped). Callbacks are
    /// invoked from the calling thread, in order, synchronously.
    /// Greedy decoding — deterministic; use [`rewrite_variants`] for
    /// sampled alternatives.
    pub fn rewrite_streaming<F>(
        &self,
        text: &str,
        instruction: Option<&str>,
        on_token: F,
    ) -> Result<Generation>
    where
        F: FnMut(&str),
    {
        self.generate(text, instruction, LlamaSampler::greedy(), on_token)
    }

    /// Generate up to `n` rewrite variants (capped at 3 to bound latency;
    /// cost is N × single-rewrite latency) using independent samplers and
    /// fresh contexts per variant. Variant 0 is always the deterministic
    /// greedy baseline (same as `rewrite`); subsequent variants use
    /// temp=0.7 / top_p=0.9 with distinct RNG seeds so the user gets real
    /// alternatives, not minor reshuffles. Identical outputs (after `trim()`)
    /// are deduped before returning — sometimes greedy + low-temp converge.
    ///
    /// Wall-clock is roughly N × single-rewrite latency since each variant
    /// builds its own context and runs its own decode loop. We deliberately
    /// don't reuse a single context across variants: KV-cache state would
    /// leak across samples and undermine the variance we're trying to
    /// produce.
    pub fn rewrite_variants(
        &self,
        text: &str,
        instruction: Option<&str>,
        n: usize,
    ) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let n = n.min(3);
        let mut outs: Vec<String> = Vec::with_capacity(n);
        for i in 0..n {
            let sampler = match i {
                0 => LlamaSampler::greedy(),
                1 => LlamaSampler::chain_simple([
                    LlamaSampler::temp(0.7),
                    LlamaSampler::top_p(0.9, 1),
                    LlamaSampler::dist(1337),
                ]),
                _ => LlamaSampler::chain_simple([
                    LlamaSampler::temp(0.7),
                    LlamaSampler::top_p(0.9, 1),
                    LlamaSampler::dist(2718),
                ]),
            };
            let out = self.rewrite_one(text, instruction, sampler)?;
            let trimmed = out.text.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            if !outs.iter().any(|prev| prev == &trimmed) {
                outs.push(trimmed);
            }
        }
        Ok(outs)
    }

    /// Single-shot rewrite with a caller-supplied sampler. Shared between
    /// `rewrite_streaming` (greedy), `rewrite_variants` (per-variant
    /// sampler), and the `nib-rewrite` CLI (when called with --temperature
    /// for RSFT data generation). Parameter-driven so the public APIs
    /// don't accidentally share sampler state.
    pub fn rewrite_one(
        &self,
        text: &str,
        instruction: Option<&str>,
        sampler: LlamaSampler,
    ) -> Result<Generation> {
        self.generate(text, instruction, sampler, |_| {})
    }

    /// Source-token budget for one prompt: what is left for `{source}`
    /// after the instruction + template overhead and the reserved
    /// `max_new_tokens` are subtracted from `ctx_size`.
    fn max_source_tokens(&self, instruction: Option<&str>) -> usize {
        let instr = instruction.unwrap_or(DEFAULT_INSTRUCTION);
        let overhead_prompt = PROMPT_TEMPLATE
            .replace("{instruction}", instr)
            .replace("{source}", "");
        let overhead = estimate_tokens(&overhead_prompt);
        (self.ctx_size as usize)
            .saturating_sub(self.max_new_tokens as usize)
            .saturating_sub(overhead)
    }

    /// Chunked rewrite for long documents. Splits `text` via
    /// [`split_for_context`] using this engine's context budget
    /// (`ctx_size - max_new_tokens - instruction overhead`), rewrites
    /// each chunk with a fresh greedy sampler (same as `rewrite`), and
    /// joins the results (`\n\n` when the source had paragraphs, else a
    /// space). `truncated` is false only if every chunk stopped
    /// naturally; if any chunk hit the token cap the whole result is
    /// marked truncated.
    pub fn rewrite_chunked(&self, text: &str, instruction: Option<&str>) -> Result<Generation> {
        let max_src = self.max_source_tokens(instruction);
        let chunks = split_for_context(text, max_src);
        if chunks.len() <= 1 {
            return self.rewrite(text, instruction);
        }
        let joiner = if text.contains("\n\n") {
            "\n\n"
        } else {
            " "
        };
        let mut outs = Vec::with_capacity(chunks.len());
        let mut any_truncated = false;
        for c in &chunks {
            let g = self.rewrite(c, instruction)?;
            any_truncated |= g.truncated;
            outs.push(g.text);
        }
        Ok(Generation {
            text: outs.join(joiner),
            truncated: any_truncated,
        })
    }

    /// The one decode loop everything routes through. Short inputs take
    /// the single-shot path unchanged; inputs whose prompt would exceed
    /// the context are split via [`split_for_context`] and each chunk is
    /// decoded with the same sampler (reused `&mut`, since
    /// `LlamaSampler` is not `Clone`) through the single-shot helper
    /// below that takes an already-built prompt.
    fn generate<F>(
        &self,
        text: &str,
        instruction: Option<&str>,
        mut sampler: LlamaSampler,
        mut on_token: F,
    ) -> Result<Generation>
    where
        F: FnMut(&str),
    {
        let max_src = self.max_source_tokens(instruction);
        if estimate_tokens(text) > max_src {
            let chunks = split_for_context(text, max_src);
            if chunks.len() > 1 {
                let joiner = if text.contains("\n\n") {
                    "\n\n"
                } else {
                    " "
                };
                let instr = instruction.unwrap_or(DEFAULT_INSTRUCTION);
                let mut outs = Vec::with_capacity(chunks.len());
                let mut any_truncated = false;
                for c in &chunks {
                    let src = apply_no_think_suffix(c, self.disable_thinking);
                    let prompt = PROMPT_TEMPLATE
                        .replace("{instruction}", instr)
                        .replace("{source}", &src);
                    let g = self.generate_prompt(&prompt, &mut sampler, &mut on_token)?;
                    any_truncated |= g.truncated;
                    outs.push(g.text);
                }
                return Ok(Generation {
                    text: outs.join(joiner),
                    truncated: any_truncated,
                });
            }
        }

        let src = apply_no_think_suffix(text, self.disable_thinking);
        let prompt = PROMPT_TEMPLATE
            .replace("{instruction}", instruction.unwrap_or(DEFAULT_INSTRUCTION))
            .replace("{source}", &src);
        self.generate_prompt(&prompt, &mut sampler, &mut on_token)
    }

    /// Single-shot decode for an already-built prompt. Each call builds
    /// a fresh context, so concurrent / sequential calls don't share KV
    /// cache state.
    fn generate_prompt<F>(
        &self,
        prompt: &str,
        sampler: &mut LlamaSampler,
        on_token: &mut F,
    ) -> Result<Generation>
    where
        F: FnMut(&str),
    {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.ctx_size));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("creating llama context")?;

        if let Some(adapter_mu) = &self.adapter {
            let mut cell = adapter_mu
                .lock()
                .map_err(|_| anyhow::anyhow!("adapter mutex poisoned"))?;
            ctx.lora_adapter_set(&mut cell.0, self.adapter_scale)
                .map_err(|e| anyhow::anyhow!("lora_adapter_set: {e}"))?;
        }

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("tokenizing prompt")?;
        let prompt_len = tokens.len() as i32;
        let n_len = prompt_len + self.max_new_tokens;
        if n_len > ctx.n_ctx() as i32 {
            bail!(
                "prompt + max_new_tokens ({n_len}) exceeds context size ({})",
                ctx.n_ctx()
            );
        }

        let mut batch = LlamaBatch::new(512.max(prompt_len as usize), 1);
        let last_idx = prompt_len - 1;
        for (i, tok) in tokens.into_iter().enumerate() {
            batch.add(tok, i as i32, &[0], i as i32 == last_idx)?;
        }
        ctx.decode(&mut batch).context("initial decode")?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut n_cur = batch.n_tokens();
        let mut truncated = true; // flipped off on a natural stop

        // Strictly `<`: with `<=` this generated max_new_tokens+1 tokens
        // and, when the prompt filled the context exactly, attempted a
        // decode at KV position n_ctx (out of range).
        while n_cur < n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                truncated = false;
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .context("token_to_piece")?;
            if piece.contains(STOP_MARKER) {
                truncated = false;
                break;
            }
            out.push_str(&piece);
            on_token(&piece);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch).context("decode step")?;
            n_cur += 1;
        }

        Ok(finalize_generation(&out, truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_INSTRUCTION, NO_THINK_SUFFIX, PROMPT_TEMPLATE, apply_no_think_suffix,
        arch_disables_thinking, estimate_tokens, finalize_generation, split_for_context,
    };

    #[test]
    fn estimate_tokens_is_chars_over_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a".repeat(400).as_str()), 100);
        assert_eq!(estimate_tokens("abcd"), 1);
    }

    #[test]
    fn long_doc_splits_into_bounded_chunks() {
        // ~5000+ char doc: repeated sentences with paragraph breaks.
        let para = "The quick brown fox jumps over the lazy dog. It barks! Does it run? Yes. ";
        let doc = format!("{}\n\n{}\n\n{}", para.repeat(25), para.repeat(25), para.repeat(25));
        assert!(
            doc.chars().count() >= 5000,
            "fixture should be a long doc, got {} chars",
            doc.chars().count()
        );
        let max = 300; // tokens ≈ 1200 chars
        let chunks = split_for_context(&doc, max);
        assert!(chunks.len() >= 2, "expected 2+ chunks, got {}", chunks.len());
        for c in &chunks {
            assert!(
                estimate_tokens(c) <= max,
                "chunk exceeds budget: {} tokens > {max}",
                estimate_tokens(c)
            );
        }
        // Join preserves words (whitespace-normalized comparison).
        let orig_words: Vec<&str> = doc.split_whitespace().collect();
        let joined = chunks.join("\n\n");
        let joined_words: Vec<&str> = joined.split_whitespace().collect();
        assert_eq!(orig_words, joined_words);
    }

    #[test]
    fn splitter_never_splits_inside_word() {
        // No sentence terminators: forces the word-boundary fallback path.
        let doc = "word ".repeat(1500); // 7500 chars
        let max = 300;
        let chunks = split_for_context(&doc, max);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(estimate_tokens(c) <= max);
        }
        // Compare as owned strings to avoid lifetime juggling.
        let orig: Vec<String> = doc.split_whitespace().map(|s| s.to_string()).collect();
        let joined: Vec<String> = chunks
            .join(" ")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(orig, joined);
    }

    #[test]
    fn short_input_stays_single_chunk() {
        let text = "Fix this short sentence, please.";
        let chunks = split_for_context(text, 1700);
        assert_eq!(chunks, vec![text.to_string()]);
    }

    #[test]
    fn default_instruction_guards_faithfulness() {
        let lower = DEFAULT_INSTRUCTION.to_lowercase();
        assert!(
            lower.contains("verbatim") || lower.contains("preserve"),
            "instruction should preserve facts verbatim: {DEFAULT_INSTRUCTION}"
        );
        assert!(
            lower.contains("word count") && lower.contains("20%"),
            "instruction should guard word count ±20%: {DEFAULT_INSTRUCTION}"
        );
    }

    #[test]
    fn think_block_removed() {
        let raw = "<think>reasoning\nacross\nlines</think>Fixed text.";
        let g = finalize_generation(raw, false);
        assert_eq!(g.text, "Fixed text.");
        assert!(!g.truncated);
    }

    #[test]
    fn unclosed_think_dropped() {
        let raw = "Fixed text.<think>truncated thought never ends";
        let g = finalize_generation(raw, false);
        assert_eq!(g.text, "Fixed text.");
        assert!(!g.truncated);
    }

    #[test]
    fn no_think_text_untouched() {
        let raw = "Just the rewrite, nothing else.";
        let g = finalize_generation(raw, false);
        assert_eq!(g.text, "Just the rewrite, nothing else.");
        assert!(!g.truncated);
    }

    #[test]
    fn think_only_yields_truncated() {
        let raw = "<think>only thinking, no answer</think>";
        let g = finalize_generation(raw, false);
        assert!(g.text.is_empty());
        assert!(g.truncated);
    }

    #[test]
    fn only_qwen3_arch_disables_thinking() {
        assert!(arch_disables_thinking("qwen3"));
        // Non-thinking models must never match (exact equality).
        for arch in ["qwen2", "qwen2.5", "lfm2", "llama", "", "Qwen3", "qwen3-foo"] {
            assert!(!arch_disables_thinking(arch), "arch {arch:?} must not disable thinking");
        }
    }

    #[test]
    fn no_think_suffix_present_only_when_disabled() {
        let src = "I has a apple.";
        let on = apply_no_think_suffix(src, true);
        assert_eq!(on.as_ref(), format!("{src}{NO_THINK_SUFFIX}"));
        assert!(on.ends_with(" /no_think"));

        let off = apply_no_think_suffix(src, false);
        assert_eq!(off.as_ref(), src, "non-thinking path must be byte-identical");
    }

    #[test]
    fn prompt_contains_suffix_only_for_qwen3_path() {
        let src = "I has a apple.";
        // Thinking path: suffix lands inside the user block before templating.
        let thinking_src = apply_no_think_suffix(src, true);
        let thinking_prompt = PROMPT_TEMPLATE
            .replace("{instruction}", DEFAULT_INSTRUCTION)
            .replace("{source}", &thinking_src);
        assert!(thinking_prompt.contains(" /no_think"));

        // Non-thinking path: template output contains no trace of the suffix.
        let plain_src = apply_no_think_suffix(src, false);
        let plain_prompt = PROMPT_TEMPLATE
            .replace("{instruction}", DEFAULT_INSTRUCTION)
            .replace("{source}", &plain_src);
        assert!(!plain_prompt.contains("no_think"));
        assert!(plain_prompt.contains(src));
    }
}
