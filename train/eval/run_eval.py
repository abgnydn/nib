"""
Faithful-rewrite eval harness for Nib.

Reads train/eval/cases.jsonl, runs each case through Nib's own
`nib-rewrite` binary against the chosen model + the production
instruction template, scores per-case:

  - WORDS:    word count within [min_words, max_words]
  - FORBID:   none of the forbidden substrings appear in output
  - KEEP:     every must_keep term appears in output (semantic variant
              expansion, word-boundary matched)

must_keep matching note: keep-term variants are anchored with word
boundaries by default, so numeric terms like "45" no longer match inside
unrelated numbers ("450", "145") and "nine" no longer matches inside
"ninety". All reports in train/reports/ were produced with the older
unanchored substring-containment matching — pass --legacy-keep-match to
reproduce those numbers.

Outputs a json report + a one-line summary.

Usage:
  python run_eval.py --model PATH_TO.gguf [--label baseline-1.2b]
  python run_eval.py --model PATH_TO.gguf --adapter nib-faithful.gguf
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# ───────────────────────── config ─────────────────────────

# Match the exact instruction template the rewrite panel uses
# in overlay.js → composeInstruction(). Diverging here invalidates the
# eval as a proxy for the user's experience.
def compose_instruction(tone: str | None, formality: str | None) -> str | None:
    if not tone and not formality:
        return None
    parts: list[str] = []
    if tone:
        parts.append(tone)
    if formality:
        parts.append(formality)
    style = ", ".join(parts)
    return (
        f"You are a copy editor. Restate the user's text in a {style} tone. "
        f"Keep the same number of words (±20%). "
        f"Do not introduce first or second person (I/you/we) unless the source uses them. "
        f"Do not add commitments, relationships, opinions, or context not in the source. "
        f"Keep all facts, numbers and names exactly as written; add no new ideas. "
        f"Do not pad with filler. Output only the rewritten text, nothing else."
    )

# ───────────────────────── runners ─────────────────────────

# Nib's own `nib-rewrite` Rust binary — uses the SAME llama-cpp-2
# engine the app uses at runtime, so eval results match user experience
# exactly. Avoids the dead-ends we hit otherwise:
#   - QVAC's llama-cli requires conversation mode (no -no-cnv)
#   - Vanilla llama.cpp from brew doesn't have LFM2.5 arch support
# Resolved relative to this repo checkout so a fresh `git clone` works;
# falls back to the author's historical ~/quill clone if that's where
# the binary actually is.
_REPO_BINARY = str(
    Path(__file__).resolve().parents[2]
    / "shell/src-tauri/target/release-dev/nib-rewrite"
)
_LEGACY_BINARY = os.path.expanduser(
    "~/quill/shell/src-tauri/target/release-dev/nib-rewrite"
)
NIB_REWRITE = (
    _REPO_BINARY
    if os.path.exists(_REPO_BINARY) or not os.path.exists(_LEGACY_BINARY)
    else _LEGACY_BINARY
)


def run_model(
    model_path: str,
    source: str,
    instruction: str | None,
    *,
    adapter_path: str | None = None,
    binary: str = NIB_REWRITE,
) -> str:
    """Single-shot generation via Nib's own nib-rewrite binary.
    Same engine as the running app — eval matches user experience.
    """
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"nib-rewrite not found: tried {_REPO_BINARY} and {_LEGACY_BINARY}"
            + (f" (requested: {binary})" if binary not in (_REPO_BINARY, _LEGACY_BINARY) else "")
            + ". Build it with: cd shell/src-tauri && "
            "cargo build --profile release-dev --features llm --bin nib-rewrite"
        )
    cmd = [binary, "-m", model_path, "-t", source]
    if instruction:
        cmd += ["-i", instruction]
    if adapter_path:
        cmd += ["--adapter", adapter_path]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            errors="replace",   # nib-rewrite stderr has Metal-init binary chars
        )
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    if proc.returncode != 0:
        # Infra failure (model/adapter failed to load, bad CLI arg, …) —
        # abort the run instead of scoring an empty output as a model
        # failure and publishing a bogus pass rate.
        stderr_tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        print(
            f"[eval] INFRA FAILURE: {binary} exited {proc.returncode} — "
            f"harness/model-loading problem, NOT a model quality failure.\n"
            f"[eval] stderr tail:\n{stderr_tail}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    raw = proc.stdout.strip()
    # nib-rewrite writes "[nib] rewrote in Xs ..." to stderr; the
    # rewritten text goes to stdout on the last line(s). Take the last
    # non-empty block.
    lines = [line for line in raw.splitlines() if line.strip()]
    return "\n".join(lines).strip()


# ───────────────────────── scoring ─────────────────────────

@dataclass
class Score:
    id: str
    ok: bool
    word_count: int
    word_count_ok: bool
    sentence_type_ok: bool = True
    forbidden_hits: list[str] = field(default_factory=list)
    missing_keeps: list[str] = field(default_factory=list)
    output: str = ""
    failure_reasons: list[str] = field(default_factory=list)
    # Optional LLM-judge result (see judge.py); None unless --judge is set.
    judge: dict[str, Any] | None = None


WORD_RE = re.compile(r"\w+", re.UNICODE)


# ───────────── semantic must_keep matching (harness v2) ─────────────
# A "must_keep" token has succeeded if any semantically-equivalent
# surface form appears in the output. Without this, a model that
# expands "Sept 9" → "September 9" or "$1.85M" → "$1.85 million" gets
# scored as a failure even though it preserved the fact perfectly.
#
# Forbidden lists stay strict — we don't want to accidentally allow
# banned filler under expansion.

_MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august",
    "sept": "september", "sep": "september",
    "oct": "october", "nov": "november", "dec": "december",
}

_ABBREVS = {
    "ppl": "people", "devs": "developers", "hrs": "hours", "hr": "hour",
    "mins": "minutes", "secs": "seconds", "yrs": "years",
    "mos": "months", "wks": "weeks", "pcs": "pieces",
}

_NUMBER_SCALE = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}
_NUMBER_SCALE_REV = {v: k for k, v in _NUMBER_SCALE.items()}

_DIGIT_WORD = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}
_WORD_DIGIT = {v: k for k, v in _DIGIT_WORD.items()}


def _normalize_for_match(s: str) -> str:
    """Lowercase + hyphens-to-spaces + collapse whitespace.
    Preserves digits, $, %, /, decimal points."""
    s = s.lower()
    s = re.sub(r"[-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keep_variants(term: str) -> set[str]:
    """Surface forms equivalent to `term` for must_keep matching."""
    base = _normalize_for_match(term)
    out = {base}

    # Bidirectional swap helper for word-token maps.
    def swap(mapping: dict[str, str]) -> None:
        for k, v in mapping.items():
            for s in list(out):
                if re.search(rf"\b{re.escape(k)}\b", s):
                    out.add(re.sub(rf"\b{re.escape(k)}\b", v, s))

    swap(_MONTHS)
    swap({v: k for k, v in _MONTHS.items() if v != "may"})  # avoid "may" verb collision
    swap(_ABBREVS)
    swap({v: k for k, v in _ABBREVS.items()})
    swap(_DIGIT_WORD)
    swap(_WORD_DIGIT)

    # Number scale: 1.85M ↔ 1.85 million, 8k ↔ 8 thousand.
    for s in list(out):
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([kmbt])\b", s):
            num, suf = m.group(1), m.group(2)
            out.add(s[:m.start()] + f"{num} {_NUMBER_SCALE[suf]}" + s[m.end():])
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(thousand|million|billion|trillion)\b", s):
            num, full = m.group(1), m.group(2)
            out.add(s[:m.start()] + f"{num}{_NUMBER_SCALE_REV[full]}" + s[m.end():])

    # Ordinal: "april 2" ↔ "april 2nd".
    for s in list(out):
        for m in re.finditer(r"\b(\d{1,3})\b", s):
            n = m.group(1)
            for suf in ("st", "nd", "rd", "th"):
                out.add(s[:m.start()] + f"{n}{suf}" + s[m.end():])

    return out


# CJK ranges treated as non-word for keep-match anchoring (CJK has no
# spaces, so terms glue to particles): Hiragana \u3040-\u309f, Katakana
# \u30a0-\u30ff, CJK Unified \u4e00-\u9fff + Extension A \u3400-\u4dbf,
# Hangul \uac00-\ud7af + Jamo \u1100-\u11ff, Halfwidth/Fullwidth \uff00-\uffef.
_CJK_RANGES = "\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\uff00-\uffef"


def _anchored(variant: str) -> re.Pattern[str]:
    """Word-boundary regex for a keep variant: anchor the ends that are
    word characters so "45" can't match inside "450"/"145" and "nine"
    can't match inside "ninety". Ends that are already non-word ("$",
    "%", "/") need no anchor — "$47.30", "12%", "/v2/search" still match
    as before.

    CJK note: Python \\w matches CJK chars, but CJK has no spaces so
    keep terms glue to particles (e.g. "HND" in "HND\\u304b\\u3089",
    "JAL 516" in "516\\u4fbf"). Anchors therefore check ASCII word chars
    only ([A-Za-z0-9_]); a CJK neighbor ([_CJK_RANGES] above) never blocks
    a match, while ASCII neighbors still do ("45" won't match "450",
    "nine" won't match "ninety").

    Self-test: "HND" matches "HND\\u304b\\u3089" ("HND" in "HND\\u304b\\u3089"),
    "JAL 516" matches "516\\u4fbf" context ("JAL 516" in "JAL 516\\u4fbf")."""
    prefix = r"(?<![A-Za-z0-9_])" if variant and re.match(r"\w", variant[0]) else ""
    suffix = r"(?![A-Za-z0-9_])" if variant and re.match(r"\w", variant[-1]) else ""
    return re.compile(prefix + re.escape(variant) + suffix)


def _keep_matches(term: str, output_normalized: str, *, legacy: bool = False) -> bool:
    """True if any semantic variant of `term` appears in the normalized
    output as a word-boundary match (default). `legacy=True` restores the
    old unanchored substring containment (what every report in
    train/reports/ was produced with). Used by score_output for the
    must_keep check."""
    variants = _keep_variants(term)
    if legacy:
        return any(v in output_normalized for v in variants)
    return any(_anchored(v).search(output_normalized) for v in variants)


def score_output(
    case: dict[str, Any], output: str, *, legacy_keep_match: bool = False
) -> Score:
    sc = Score(id=case["id"], ok=True, word_count=0, word_count_ok=True)
    sc.output = output

    words = WORD_RE.findall(output)
    sc.word_count = len(words)
    min_w = case.get("min_words", 1)
    max_w = case.get("max_words", 50)
    if not (min_w <= sc.word_count <= max_w):
        sc.word_count_ok = False
        sc.ok = False
        sc.failure_reasons.append(f"words: {sc.word_count} not in [{min_w},{max_w}]")

    # sentence-type gate (statement→question): a statement source must not
    # become a question in the output; a question source must stay a question.
    source_text = case.get("source", "")
    source_has_question = "?" in source_text
    source_ends_with_question = source_text.strip().endswith("?")
    output_has_question = "?" in output
    sentence_type_ok = True
    if not source_has_question and output_has_question:
        # statement-question mismatch: statement turned into a question
        sentence_type_ok = False
    elif source_ends_with_question and not output_has_question:
        # question-statement mismatch: question lost its question mark
        sentence_type_ok = False
    sc.sentence_type_ok = sentence_type_ok
    if not sentence_type_ok:
        sc.ok = False
        sc.failure_reasons.append(
            f"sentence-type: source_has_question={source_has_question} "
            f"output_has_question={output_has_question}"
        )

    out_lower = output.lower()

    # Forbidden: use word-boundary match for ALPHABETIC terms so a banned
    # "I" doesn't false-match "ass[i]gned" / "vers[i]on" / etc. Multi-word
    # phrases ("committed to ensuring") stay as case-insensitive substring.
    for term in case.get("forbidden", []):
        t_lower = term.lower()
        if " " in t_lower or not term.isalpha():
            hit = t_lower in out_lower
        else:
            # Word-boundary regex match for single alpha tokens.
            hit = bool(re.search(rf"\b{re.escape(t_lower)}\b", out_lower))
        if hit:
            sc.forbidden_hits.append(term)
    if sc.forbidden_hits:
        sc.ok = False
        sc.failure_reasons.append(f"forbidden: {sc.forbidden_hits}")

    # Must-keep: semantic match — accepts month abbrev ↔ full, k/M ↔
    # thousand/million, digit ↔ word for 1-12, hyphen tolerance, and
    # ordinal suffixes. Each variant is word-boundary matched against the
    # normalized output (unanchored containment with legacy_keep_match).
    # Preserves "$47.30", "12%", "/v2/search" since those don't trip any
    # of the variant rules.
    out_normalized = _normalize_for_match(output)
    for term in case.get("must_keep", []):
        if not _keep_matches(term, out_normalized, legacy=legacy_keep_match):
            sc.missing_keeps.append(term)
    if sc.missing_keeps:
        sc.ok = False
        sc.failure_reasons.append(f"missing_keep: {sc.missing_keeps}")

    return sc


# ───────────────────────── main ─────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .gguf base model")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter .gguf")
    ap.add_argument("--cases", default=str(Path(__file__).parent / "cases.jsonl"))
    ap.add_argument("--out", default=None, help="Write JSON report to PATH")
    ap.add_argument("--label", default="run", help="Human-readable label for report")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N cases (debug)")
    ap.add_argument("--verbose", action="store_true", help="Print each output")
    ap.add_argument("--binary", default=None,
                    help=f"Path to the nib-rewrite binary (default: {NIB_REWRITE})")
    ap.add_argument("--judge", action="store_true",
                    help="Also score each rewrite with the Claude LLM-judge "
                         "(needs ANTHROPIC_API_KEY + `pip install anthropic`)")
    ap.add_argument("--judge-model", default=None,
                    help="Judge model id (default: claude-opus-4-8)")
    ap.add_argument("--legacy-keep-match", action="store_true",
                    help="Score must_keep with the old unanchored substring "
                         "containment instead of word-boundary matching "
                         "(reproduces the reports in train/reports/)")
    args = ap.parse_args()

    binary = args.binary or NIB_REWRITE

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if args.adapter and not os.path.exists(args.adapter):
        print(f"adapter not found: {args.adapter}", file=sys.stderr)
        return 2
    if not os.path.exists(binary):
        print(f"nib-rewrite not found: {binary}\n"
              f"Build it with: cd shell/src-tauri && "
              f"cargo build --profile release-dev --features llm --bin nib-rewrite",
              file=sys.stderr)
        return 2

    cases = [json.loads(line) for line in open(args.cases) if line.strip()]
    if args.limit:
        cases = cases[: args.limit]

    # LLM-judge setup (lazy: only import/connect when --judge is on).
    judge_mod = None
    judge_client = None
    judge_model = None
    if args.judge:
        import judge as judge_mod  # same dir; on sys.path when run as a script
        judge_model = args.judge_model or judge_mod.DEFAULT_JUDGE_MODEL
        judge_client = judge_mod.make_client()
        print(f"[eval] judge={judge_model}", file=sys.stderr)

    print(f"[eval] model={args.model}", file=sys.stderr)
    if args.adapter:
        print(f"[eval] adapter={args.adapter}", file=sys.stderr)
    print(f"[eval] cases={len(cases)}", file=sys.stderr)

    scores: list[Score] = []
    t0 = time.time()
    for i, case in enumerate(cases):
        instr = compose_instruction(case.get("tone"), case.get("formality"))
        t_case = time.time()
        out = run_model(args.model, case["source"], instr, adapter_path=args.adapter, binary=binary)
        dt = time.time() - t_case
        sc = score_output(case, out, legacy_keep_match=args.legacy_keep_match)
        if args.judge:
            sc.judge = judge_mod.judge_rewrite(
                case["source"], out, instr, client=judge_client, model=judge_model,
            )
        scores.append(sc)
        mark = "✓" if sc.ok else "✗"
        judge_tag = ""
        if sc.judge:
            if "error" in sc.judge:
                judge_tag = f"  judge=ERR({sc.judge['error'][:24]})"
            else:
                axes = ("grammaticality", "faithfulness", "improvement", "fluency")
                avg = sum(sc.judge.get(a, 0) for a in axes) / len(axes)
                judge_tag = f"  judge={avg:.2f} {sc.judge.get('verdict', '?')}"
        print(
            f"[{i+1:2}/{len(cases)}] {mark} {sc.id:32} "
            f"words={sc.word_count:3} ({dt:.1f}s){judge_tag}",
            file=sys.stderr,
        )
        if not sc.ok:
            for r in sc.failure_reasons:
                print(f"        └─ {r}", file=sys.stderr)
        if args.verbose:
            print(f"        out: {out!r}", file=sys.stderr)

    dt_total = time.time() - t0
    pass_n = sum(1 for s in scores if s.ok)
    pass_rate = 100 * pass_n / max(1, len(scores))

    report = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "n_cases": len(scores),
        "n_pass": pass_n,
        "pass_rate": round(pass_rate, 1),
        "duration_s": round(dt_total, 1),
        "avg_words": round(statistics.mean(s.word_count for s in scores), 1),
        "scores": [asdict(s) for s in scores],
    }

    if args.judge:
        report["judge"] = judge_mod.summarize_judgments([s.judge for s in scores])

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[eval] wrote {args.out}", file=sys.stderr)

    print(
        f"\n{args.label}: {pass_n}/{len(scores)} pass ({pass_rate:.1f}%)  "
        f"avg_words={report['avg_words']}  total={dt_total:.1f}s"
    )
    if args.judge:
        j = report["judge"]
        if j.get("n_judged"):
            v = j.get("verdicts", {})
            print(
                f"{args.label}: judge overall={j.get('mean_overall')}/5  "
                f"(gram={j.get('mean_grammaticality')} faith={j.get('mean_faithfulness')} "
                f"impr={j.get('mean_improvement')} flu={j.get('mean_fluency')})  "
                f"verdict better/same/worse={v.get('better')}/{v.get('same')}/{v.get('worse')}  "
                f"win_rate={j.get('win_rate')}%  (judged {j['n_judged']}, errors {j.get('n_errors', 0)})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
