#!/usr/bin/env bash
# Nib — agentic loop oracle for Phase 1 + Phase 2 (Grammarly-parity plan).
#
# This script does NOT do work. It answers ONE question:
#   "is the loop done?"  →  exit 0 = ALL DONE (stop the loop)
#                          exit 1 = NOT DONE (keep working, see FAIL lines)
#
# Loop contract (for the agent running the loop):
#   1. Run `./scripts/agent-loop.sh`.
#   2. If exit 0 → STOP. Report DONE, commit nothing new.
#   3. If exit 1 → fix ONLY the first FAIL gate, re-run this script.
#   4. STOP ANYWAY after `--max-rounds` (default 10) or when two
#      consecutive rounds fix nothing (no-progress guard lives with
#      the loop runner — this script just reports per-gate status).
#   5. Never commit/push from inside the loop. Commit once, after DONE.
#
# Phase 1 gates (each independently verifiable, no model needed except G4):
#   G1  failure bank exists: train/eval/failure-bank.jsonl, >=10 rows,
#       every row has {id, source, bad_output, label, note}
#   G2  eval coverage: all train/eval/cases*.jsonl total >=500 valid rows
#       with a `source` field (holdout-90 + extended files count)
#   G3  automatic gates in harness: train/eval/run_eval.py enforces
#       length (±20%) AND sentence-type match (no statement→question)
#   G4  plain-first UI: overlay fallback/rewrite panel defaults to NO
#       tone + NO formality selected (tones are opt-in risk)
#   G5  whole suite green: ./scripts/test.sh exits 0
#
# Phase 2 gates (faithfulness push; P3 needs a local GGUF + ~1 min):
#   P1  tone-safe leash: the faithfulness sentence lives in BOTH the
#       panel template (overlay.js) and the eval template (run_eval.py)
#   P2  negative pairs: train/data/phase2-negatives.jsonl, 20 rows with
#       {id, source, chosen, rejected, label, note}, provenance marked
#   P3  measured baseline: train/reports/baseline-lfm250-round2-90.json
#       exists with 90 scored cases (the "before" number for Phase 2)
#
# Usage:
#     ./scripts/agent-loop.sh
#     ./scripts/agent-loop.sh --max-rounds 10   # documented for runners;
#                                               # this script checks gates once
#                                               # and always reports honestly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0

gate() { # $1 = id, $2 = description
  printf "GATE %s: %s ... " "$1" "$2";
}
pass() { printf "\033[32mPASS\033[0m %s\n" "$*"; }
fail() { printf "\033[31mFAIL\033[0m %s\n" "$*"; FAIL=1; }

for arg in "$@"; do
  case "$arg" in
    --max-rounds|--max-rounds=*) ;; # runner-level concern, accepted + ignored
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

# ── G1: failure bank ──────────────────────────────────────────────
gate "G1" "failure bank >=10 labeled rows"
FB="$REPO/train/eval/failure-bank.jsonl"
if [[ -f "$FB" ]]; then
  if python3 - "$FB" <<'EOF' 2>/dev/null; then
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(rows) >= 10, f"only {len(rows)} rows, need >=10"
need = {"id", "source", "bad_output", "label", "note"}
for r in rows:
    assert need <= set(r), f"row {r.get('id')} missing {need - set(r)}"
print(f"{len(rows)} rows ok")
EOF
    pass "$(python3 -c "import json; print(len([l for l in open('$FB') if l.strip()]))") rows"
  else
    fail "$FB schema/count check failed (need >=10 rows with id/source/bad_output/label/note)"
  fi
else
  fail "missing $FB"
fi

# ── G2: eval coverage >=500 ───────────────────────────────────────
gate "G2" "eval cases total >=500"
TOTAL=$(python3 - "$REPO/train/eval" <<'EOF' 2>/dev/null
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/cases*.jsonl"):
    for l in open(f):
        if l.strip():
            assert "source" in json.loads(l), f"row without source in {f}"
            n += 1
print(n)
EOF
) || TOTAL=0
if [[ "$TOTAL" -ge 500 ]]; then
  pass "$TOTAL rows"
else
  fail "only $TOTAL rows, need >=500 (add to train/eval/cases-*.jsonl)"
fi

# ── G3: automatic length + sentence-type gates ────────────────────
gate "G3" "harness enforces length + sentence-type"
if grep -q "word_count\|min_words.*max_words" "$REPO/train/eval/run_eval.py" 2>/dev/null \
  && grep -qiE "sentence.?type|statement.*question|ends_with_question|question.*statement" "$REPO/train/eval/run_eval.py" 2>/dev/null; then
  pass "length + sentence-type checks present in run_eval.py"
else
  fail "run_eval.py lacks length (±20%) and/or sentence-type (statement→question) gate"
fi

# ── G4: plain-first UI ────────────────────────────────────────────
gate "G4" "rewrite panel defaults to no tone/formality"
# Plain-first = no pill pre-selected on open: the JS must not mark any
# tone/formality active at init. We check the init path sets nothing
# active (search the panel init for a default-active assignment).
if grep -q "fbActiveIdx\s*=\s*-1\|selectedTone\s*=\s*null\|selectedTone\s*=\s*''" "$REPO/shell/src/overlay.js" 2>/dev/null \
  || grep -q "no pill pre-selected\|plain-first\|default.*no tone" "$REPO/shell/src/overlay.js" 2>/dev/null; then
  pass "no tone pre-selected at panel init"
else
  fail "overlay.js pre-selects a tone/formality (or no provable plain default)"
fi

# ── G5: full suite green ──────────────────────────────────────────
gate "G5" "./scripts/test.sh green"
if "$REPO/scripts/test.sh" >/tmp/nib-loop-test.log 2>&1; then
  RUST_LINE=$(grep -E "test result:" /tmp/nib-loop-test.log | head -1)
  pass "$RUST_LINE"
else
  fail "test.sh failed — see /tmp/nib-loop-test.log"
fi

# ── P1: tone-safe leash in both templates ─────────────────────────
gate "P1" "faithfulness leash in panel + eval templates"
LEASH="Keep all facts, numbers and names exactly as written"
if grep -q "$LEASH" "$REPO/shell/src/overlay.js" 2>/dev/null \
  && grep -q "$LEASH" "$REPO/train/eval/run_eval.py" 2>/dev/null; then
  pass "leash present in overlay.js + run_eval.py"
else
  fail "leash missing in overlay.js and/or run_eval.py (keep templates identical)"
fi

# ── P2: negative pairs dataset ────────────────────────────────────
gate "P2" "phase2-negatives.jsonl 20 labeled pairs"
NEG="$REPO/train/data/phase2-negatives.jsonl"
if [[ -f "$NEG" ]]; then
  if python3 - "$NEG" <<'EOF' 2>/dev/null; then
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(rows) >= 20, f"only {len(rows)} rows, need >=20"
need = {"id", "source", "chosen", "rejected", "label", "note"}
for r in rows:
    assert need <= set(r), f"row {r.get('id')} missing {need - set(r)}"
    assert r["chosen"] != r["rejected"], f"row {r.get('id')}: chosen==rejected"
    assert "OBSERVED-REAL" in r["note"] or "SYNTHETIC-REPRESENTATIVE" in r["note"], \
        f"row {r.get('id')}: provenance unmarked"
print(f"{len(rows)} pairs ok")
EOF
    pass "$(python3 -c "import json; print(len([l for l in open('$NEG') if l.strip()]))") pairs"
  else
    fail "$NEG schema/count check failed (need >=20 pairs with id/source/chosen/rejected/label/note)"
  fi
else
  fail "missing $NEG"
fi

# ── P3: measured baseline report ──────────────────────────────────
gate "P3" "baseline-lfm250-round2-90.json with 90 scored cases"
REP="$REPO/train/reports/baseline-lfm250-round2-90.json"
if [[ -f "$REP" ]]; then
  if python3 - "$REP" <<'EOF' 2>/dev/null; then
import json, sys
r = json.load(open(sys.argv[1]))
assert r.get("n_cases") == 90, f"n_cases={r.get('n_cases')}, need 90"
assert "n_pass" in r and "pass_rate" in r, "missing n_pass/pass_rate"
print(f"{r['n_pass']}/{r['n_cases']} = {r['pass_rate']}")
EOF
    pass "$(python3 -c "import json; r=json.load(open('$REP')); print(f\"{r['n_pass']}/{r['n_cases']}\")")"
  else
    fail "$REP invalid (need n_cases=90 with n_pass/pass_rate)"
  fi
else
  fail "missing $REP (run run_eval.py on cases-round2-90 with the bundled model)"
fi

# ── verdict ───────────────────────────────────────────────────────
if [[ "$FAIL" -eq 0 ]]; then
  printf "\n== LOOP DONE — all Phase-1 + Phase-2 gates green, stop working ==\n"
  exit 0
else
  printf "\n== NOT DONE — fix the first FAIL gate above, then re-run ./scripts/agent-loop.sh ==\n"
  exit 1
fi
