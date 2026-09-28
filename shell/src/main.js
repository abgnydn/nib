const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const editor = document.getElementById("editor");
const panel = document.getElementById("suggestions");
const status = document.getElementById("status");
const caps = document.getElementById("caps");
const rewriteBtn = document.getElementById("rewrite-btn");
const rewriteHint = document.getElementById("rewrite-hint");
const rewriteOutput = document.getElementById("rewrite-output");
const rewriteText = document.getElementById("rewrite-text");
const rewriteApply = document.getElementById("rewrite-apply");
const rewriteDismiss = document.getElementById("rewrite-dismiss");

// ---- Multi-variant rewrite UI (built dynamically — no HTML/CSS edits) ----
// We inject the variants button into the same `.actions` row as the single
// "Rewrite" button, plus a sibling panel after `#rewrite-output`. Styles are
// injected via a single `<style>` tag so we can reuse the project's CSS
// variables (--panel, --panel-2, --accent, --fg, --fg-dim, --radius).
(function injectVariantsStyles() {
  const style = document.createElement("style");
  style.textContent = `
    .rewrite-variants {
      background: var(--panel-2);
      border: 1px solid #2a3046;
      border-left: 3px solid var(--accent);
      border-radius: var(--radius);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .rewrite-variants.hidden { display: none; }
    .rewrite-variants-list { display: flex; flex-direction: column; gap: 6px; }
    .rewrite-variant-card {
      font: 13px/1.55 ui-monospace, "SF Mono", Menlo, monospace;
      white-space: pre-wrap;
      word-wrap: break-word;
      background: var(--panel);
      border: 1px solid #2a3046;
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      text-align: left;
      color: inherit;
      transition: border-color 0.12s, background 0.12s;
      width: 100%;
    }
    .rewrite-variant-card:hover,
    .rewrite-variant-card:focus {
      border-color: var(--accent);
      background: #161a25;
      outline: none;
    }
    .rewrite-variant-card .variant-tag {
      display: inline-block;
      font-family: ui-monospace, "SF Mono", monospace;
      font-size: 10px;
      color: var(--fg-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-right: 8px;
      padding: 1px 5px;
      background: #0c0e14;
      border-radius: 3px;
    }
    .rewrite-variants-loading {
      font-size: 12px;
      color: var(--fg-dim);
      padding: 8px 0;
    }
    .rewrite-variants-loading::after {
      content: "▋";
      animation: blink 0.7s steps(2) infinite;
      color: var(--accent);
      display: inline-block;
      margin-left: 4px;
    }
  `;
  document.head.appendChild(style);
})();

const rewriteVariantsBtn = document.createElement("button");
rewriteVariantsBtn.id = "rewrite-variants-btn";
rewriteVariantsBtn.textContent = "✦ Show 3 variants";
rewriteVariantsBtn.title =
  "Generate 3 alternative rewrites and pick the one you like (runs the model 3×)";
rewriteVariantsBtn.disabled = true;
rewriteBtn.insertAdjacentElement("afterend", rewriteVariantsBtn);

const rewriteVariants = document.createElement("div");
rewriteVariants.id = "rewrite-variants";
rewriteVariants.className = "rewrite-variants hidden";
const rewriteVariantsLabel = document.createElement("div");
rewriteVariantsLabel.className = "rewrite-label";
const rewriteVariantsTitle = document.createElement("span");
rewriteVariantsTitle.textContent = "variants";
const rewriteVariantsHint = document.createElement("span");
rewriteVariantsHint.className = "hint";
rewriteVariantsHint.style.marginLeft = "6px";
rewriteVariantsLabel.appendChild(rewriteVariantsTitle);
rewriteVariantsLabel.appendChild(rewriteVariantsHint);
const rewriteVariantsList = document.createElement("div");
rewriteVariantsList.id = "rewrite-variants-list";
rewriteVariantsList.className = "rewrite-variants-list";
const rewriteVariantsActions = document.createElement("div");
rewriteVariantsActions.className = "rewrite-actions";
const rewriteVariantsCancel = document.createElement("button");
rewriteVariantsCancel.textContent = "Cancel";
rewriteVariantsActions.appendChild(rewriteVariantsCancel);
rewriteVariants.appendChild(rewriteVariantsLabel);
rewriteVariants.appendChild(rewriteVariantsList);
rewriteVariants.appendChild(rewriteVariantsActions);
rewriteOutput.insertAdjacentElement("afterend", rewriteVariants);

let debounceTimer = null;
let inflight = false;
let pending = false;

function debounce(fn, ms) {
  return (...args) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => fn(...args), ms);
  };
}

async function probeCapabilities() {
  try {
    const c = await invoke("capabilities");
    let label = "harper-only";
    if (c.llm_built && c.model_loaded) {
      label = c.personal_adapter_loaded ? "harper + llm + personal" : "harper + llm";
      rewriteBtn.disabled = false;
      if (rewriteVariantsBtn) rewriteVariantsBtn.disabled = false;
      rewriteHint.textContent = "";
    } else if (c.llm_built && !c.model_loaded) {
      label = "harper + llm (no model)";
      rewriteHint.textContent = "set NIB_MODEL to your .gguf and relaunch";
    } else {
      rewriteHint.textContent = "rebuild with --features llm to enable";
    }
    if (c.qvac_available) {
      label += " · qvac";
    }
    caps.textContent = label;
    if (c.qvac_version) {
      caps.title = `QVAC Fabric: ${c.qvac_version}`;
    }
    // Pill in the personalization panel reflects the same state.
    if (personalPill) {
      if (c.personal_adapter_loaded) {
        personalPill.textContent = "personal";
        personalPill.className = "pill pill-personal";
      } else {
        personalPill.textContent = "base only";
        personalPill.className = "pill pill-base";
      }
    }
  } catch (e) {
    caps.textContent = `caps error: ${e}`;
  }
}

async function runCheck() {
  if (inflight) {
    pending = true;
    return;
  }
  inflight = true;
  status.textContent = "checking…";
  try {
    const text = editor.value;
    const t0 = performance.now();
    const lints = await invoke("check", { text });
    const dt = (performance.now() - t0).toFixed(1);
    render(lints, text);
    status.textContent = `${lints.length} lint${lints.length === 1 ? "" : "s"} · ${dt} ms`;
  } catch (err) {
    status.textContent = "error";
    panel.innerHTML = `<div class="empty">error: ${String(err)}</div>`;
  } finally {
    inflight = false;
    if (pending) {
      pending = false;
      runCheck();
    }
  }
}

function render(lints, source) {
  if (!lints.length) {
    panel.innerHTML = `<div class="empty">no suggestions</div>`;
    return;
  }
  const chars = [...source];
  panel.innerHTML = lints
    .map((l, i) => {
      const slice = chars.slice(l.start, l.end).join("");
      const suggs = l.suggestions
        .map(
          (s, j) =>
            `<button class="sugg" data-lint="${i}" data-sugg="${j}">${
              s.kind === "remove" ? "⌫ remove" : escapeHtml(s.text)
            }</button>`
        )
        .join("");
      return `<div class="lint">
        <div class="kind">${l.kind}<span class="span">${escapeHtml(slice)}</span></div>
        <div class="msg">${escapeHtml(l.message)}</div>
        ${suggs ? `<div class="suggs">${suggs}</div>` : ""}
      </div>`;
    })
    .join("");

  panel.querySelectorAll(".sugg").forEach((btn) => {
    btn.addEventListener("click", () => {
      const lintIdx = parseInt(btn.dataset.lint, 10);
      const suggIdx = parseInt(btn.dataset.sugg, 10);
      applySuggestion(lints[lintIdx], lints[lintIdx].suggestions[suggIdx]);
    });
  });
}

function applySuggestion(lint, sugg) {
  const before = editor.value;
  const chars = [...before];
  let start = lint.start, end = lint.end;
  let replacement = sugg.text || "";
  if (sugg.kind === "replace") {
    chars.splice(lint.start, lint.end - lint.start, ...sugg.text);
  } else if (sugg.kind === "insert_after") {
    chars.splice(lint.end, 0, ...sugg.text);
    start = lint.end; end = lint.end;
  } else if (sugg.kind === "remove") {
    chars.splice(lint.start, lint.end - lint.start);
    replacement = "";
  }
  const after = chars.join("");
  editor.value = after;
  // Record into the personalization journal. Main-window applies mutate
  // editor.value directly (no AXUI round-trip) so we journal explicitly.
  invoke("journal_log", {
    kind: "apply",
    sourceText: before,
    appliedText: after,
    suggested: replacement,
    lintKind: lint.kind,
    lintMessage: lint.message,
    lintStart: start,
    lintEnd: end,
  })
    .then(() => refreshPersonal())
    .catch(() => {});
  runCheck();
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function selectedOrAll() {
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  if (end > start) {
    return { text: editor.value.slice(start, end), start, end };
  }
  return { text: editor.value, start: 0, end: editor.value.length };
}

async function runRewrite() {
  const { text } = selectedOrAll();
  if (!text.trim()) return;
  rewriteBtn.disabled = true;
  rewriteOutput.classList.remove("hidden");
  rewriteText.textContent = "";
  rewriteText.classList.add("streaming");
  rewriteHint.textContent = "streaming…";
  const t0 = performance.now();

  const session = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
  const unlisten = await listen("rewrite-token", (evt) => {
    const p = evt.payload || {};
    if (p.session !== session) return;
    if (p.done) {
      rewriteText.classList.remove("streaming");
      return;
    }
    if (p.delta) rewriteText.textContent += p.delta;
  });

  try {
    // rewrite returns { text, truncated } — truncated means generation hit
    // the token cap and the tail is missing.
    const out = await invoke("rewrite", { text, instruction: null, session });
    const dt = (performance.now() - t0).toFixed(0);
    if (!rewriteText.textContent) rewriteText.textContent = out.text;
    rewriteHint.textContent = out.truncated
      ? `${dt} ms · ⚠ cut off at the length limit — try a shorter selection`
      : `${dt} ms`;
  } catch (err) {
    rewriteText.textContent = `error: ${err}`;
    rewriteHint.textContent = "";
  } finally {
    unlisten();
    rewriteText.classList.remove("streaming");
    rewriteBtn.disabled = false;
  }
}

rewriteBtn.addEventListener("click", runRewrite);
rewriteApply.addEventListener("click", () => {
  const out = rewriteText.textContent;
  if (!out) return;
  const sel = selectedOrAll();
  const before = editor.value;
  const after =
    editor.value.slice(0, sel.start) + out + editor.value.slice(sel.end);
  editor.value = after;
  rewriteOutput.classList.add("hidden");
  invoke("journal_log", {
    kind: "rewrite_apply",
    sourceText: before,
    appliedText: after,
    suggested: out,
  })
    .then(() => refreshPersonal())
    .catch(() => {});
  runCheck();
});
rewriteDismiss.addEventListener("click", () => {
  rewriteOutput.classList.add("hidden");
});

// ---- Multi-variant rewrite (3 alternatives, click to apply) -------------
function hideVariants() {
  rewriteVariants.classList.add("hidden");
  rewriteVariantsList.innerHTML = "";
  rewriteVariantsHint.textContent = "";
}

function applyVariant(variantText) {
  // Mirror the single-rewrite "Apply" path: replace the selection (or the
  // whole textarea when nothing is selected) with the chosen variant, and
  // journal the (before, after) pair so the personal adapter learns from
  // the user's pick.
  const sel = selectedOrAll();
  const before = editor.value;
  const after =
    editor.value.slice(0, sel.start) + variantText + editor.value.slice(sel.end);
  editor.value = after;
  hideVariants();
  invoke("journal_log", {
    kind: "rewrite_apply",
    sourceText: before,
    appliedText: after,
    suggested: variantText,
  })
    .then(() => refreshPersonal())
    .catch(() => {});
  runCheck();
}

function renderVariants(variants) {
  rewriteVariantsList.innerHTML = "";
  variants.forEach((v, i) => {
    const card = document.createElement("button");
    card.className = "rewrite-variant-card";
    card.type = "button";
    const tag = document.createElement("span");
    tag.className = "variant-tag";
    // Variant 0 is greedy (deterministic), 1+ are sampled.
    tag.textContent = i === 0 ? "greedy" : `alt ${i}`;
    card.appendChild(tag);
    card.appendChild(document.createTextNode(v));
    card.addEventListener("click", () => applyVariant(v));
    rewriteVariantsList.appendChild(card);
  });
}

async function runRewriteVariants() {
  const { text } = selectedOrAll();
  if (!text.trim()) return;
  // Hide the single-rewrite panel so the two UIs don't fight for the eye.
  rewriteOutput.classList.add("hidden");
  rewriteVariants.classList.remove("hidden");
  rewriteVariantsBtn.disabled = true;
  rewriteBtn.disabled = true;
  rewriteVariantsList.innerHTML =
    '<div class="rewrite-variants-loading">generating 3 variants — runs the model 3× so this is slower than a single rewrite</div>';
  rewriteVariantsHint.textContent = "";
  const t0 = performance.now();
  try {
    const variants = await invoke("rewrite_variants", {
      text,
      instruction: null,
      n: 3,
    });
    const dt = (performance.now() - t0).toFixed(0);
    if (!variants || variants.length === 0) {
      rewriteVariantsList.innerHTML =
        '<div class="rewrite-variants-loading">no variants returned</div>';
      rewriteVariantsHint.textContent = `${dt} ms`;
      return;
    }
    renderVariants(variants);
    const per = (dt / variants.length).toFixed(0);
    rewriteVariantsHint.textContent =
      variants.length === 1
        ? `${dt} ms · 1 variant`
        : `${dt} ms · ${variants.length} variants (~${per} ms each)`;
  } catch (err) {
    rewriteVariantsList.innerHTML = `<div class="rewrite-variants-loading">error: ${escapeHtml(
      String(err)
    )}</div>`;
  } finally {
    rewriteVariantsBtn.disabled = false;
    rewriteBtn.disabled = false;
  }
}

if (rewriteVariantsBtn) {
  rewriteVariantsBtn.addEventListener("click", runRewriteVariants);
}
if (rewriteVariantsCancel) {
  rewriteVariantsCancel.addEventListener("click", hideVariants);
}

editor.addEventListener("input", debounce(runCheck, 250));

// ---- Personalization panel (v0.5 phase 1) --------------------------------
const personalCount = document.getElementById("personal-count");
const personalApplied = document.getElementById("personal-applied");
const personalRewrite = document.getElementById("personal-rewrite");
const personalRange = document.getElementById("personal-range");
const personalExport = document.getElementById("personal-export");
const personalClear = document.getElementById("personal-clear");
const personalTrain = document.getElementById("personal-train");
const personalPill = document.getElementById("personal-pill");
const autoRetrain = document.getElementById("auto-retrain");
const autoThreshold = document.getElementById("auto-threshold");
const autoLast = document.getElementById("auto-last");
const relaunchBadge = document.getElementById("relaunch-badge");

const trainModal = document.getElementById("train-modal");
const trainState = document.getElementById("train-state");
const trainElapsed = document.getElementById("train-elapsed");
const trainStage = document.getElementById("train-stage");
const trainError = document.getElementById("train-error");
const trainInstall = document.getElementById("train-install");
const trainDismiss = document.getElementById("train-dismiss");

function fmtTs(t) {
  if (!t) return "—";
  return t.slice(0, 10); // YYYY-MM-DD
}

async function refreshPersonal() {
  try {
    const s = await invoke("journal_stats");
    personalCount.textContent = String(s.count || 0);
    personalApplied.textContent = String(s.applied || 0);
    personalRewrite.textContent = String(s.rewrite_applied || 0);
    if (s.oldest_ts && s.newest_ts) {
      personalRange.textContent =
        s.oldest_ts === s.newest_ts
          ? `since ${fmtTs(s.oldest_ts)}`
          : `${fmtTs(s.oldest_ts)} → ${fmtTs(s.newest_ts)}`;
    } else {
      personalRange.textContent = "no events yet";
    }
    // Enable Train button only when there's enough data.
    if (personalTrain) {
      const enough = (s.applied || 0) + (s.rewrite_applied || 0) >= 10;
      personalTrain.disabled = !enough;
      personalTrain.title = enough
        ? "Train a personal LoRA adapter (local, ~5 min on Metal; cloud fallback only if you opt in)"
        : `Need ≥10 applied edits to train (have ${(s.applied||0)+(s.rewrite_applied||0)})`;
    }
  } catch (e) {
    // journal not available — fail quiet
  }
}

personalExport.addEventListener("click", async () => {
  // Rust picks the destination (0600 inside the app data dir) — the
  // journal is the user's typed text and must not land in shared /tmp.
  try {
    const res = await invoke("journal_export");
    personalExport.textContent = `✓ ${res.pairs} pairs → ${res.path}`;
    setTimeout(() => (personalExport.textContent = "⤓ Export"), 6000);
  } catch (e) {
    personalExport.textContent = `error: ${e}`;
    setTimeout(() => (personalExport.textContent = "⤓ Export"), 4000);
  }
});

personalClear.addEventListener("click", async () => {
  if (!confirm("Clear all learned edits? This cannot be undone.")) return;
  try {
    const bytes = await invoke("journal_clear");
    personalClear.textContent = `✓ cleared ${bytes}B`;
    setTimeout(() => (personalClear.textContent = "⌫ Reset"), 2500);
    refreshPersonal();
  } catch (e) {
    personalClear.textContent = `error: ${e}`;
  }
});

// ---- Personal training (v0.5 phase 3) ----------------------------------
let trainPollTimer = null;
const STATE_LABEL = {
  idle: "idle",
  running: "training in progress…",
  succeeded: "✓ training complete",
  failed: "training failed",
};

function fmtElapsed(s) {
  if (!s) return "";
  const sec = Math.floor(s);
  const m = Math.floor(sec / 60);
  return m > 0 ? `${m}m ${sec % 60}s elapsed` : `${sec}s elapsed`;
}

const BACKEND_LABEL = {
  local: "on your Mac (Metal · free)",
  modal: "on Modal (cloud · ~$0.20)",
  none: "",
};

function renderTrainStatus(st) {
  let label = STATE_LABEL[st.state] || st.state;
  if (st.backend && st.backend !== "none" && (st.state === "running" || st.state === "succeeded")) {
    label += ` · ${BACKEND_LABEL[st.backend] || st.backend}`;
  }
  trainState.textContent = label;
  trainElapsed.textContent = fmtElapsed(st.elapsed_secs);
  trainStage.textContent = st.stage || "";
  trainError.textContent = st.error || "";
  if (st.state === "succeeded") {
    trainInstall.classList.remove("hidden");
  } else {
    trainInstall.classList.add("hidden");
  }
}

async function pollTrainOnce() {
  try {
    const st = await invoke("train_personal_status");
    renderTrainStatus(st);
    if (st.state === "succeeded" || st.state === "failed") {
      clearInterval(trainPollTimer);
      trainPollTimer = null;
    }
  } catch (e) {
    trainError.textContent = String(e);
  }
}

function startTrainPolling() {
  if (trainPollTimer) clearInterval(trainPollTimer);
  trainPollTimer = setInterval(pollTrainOnce, 3000);
}

// "Start new run" appears only when a finished job is still parked in the
// modal — so reopening never silently discards a result or double-starts.
const trainRestart = document.createElement("button");
trainRestart.textContent = "Start new run";
trainRestart.classList.add("hidden");
trainDismiss.insertAdjacentElement("beforebegin", trainRestart);

async function startTraining() {
  trainRestart.classList.add("hidden");
  trainError.textContent = "";
  trainStage.textContent = "";
  trainState.textContent = "starting…";
  try {
    const st = await invoke("train_personal_start");
    renderTrainStatus(st);
    startTrainPolling();
  } catch (e) {
    trainState.textContent = "failed to start";
    trainError.textContent = String(e);
  }
}

personalTrain.addEventListener("click", async () => {
  trainModal.classList.remove("hidden");
  trainError.textContent = "";
  // Re-attach to whatever is already happening instead of blindly calling
  // start (which showed a bogus "failed to start" for a running job, and
  // silently kicked off a fresh run over a finished one).
  let cur = null;
  try {
    cur = await invoke("train_personal_status");
  } catch (e) {
    /* fall through to a fresh start */
  }
  if (cur && cur.state === "running") {
    renderTrainStatus(cur);
    startTrainPolling();
    return;
  }
  if (cur && cur.state === "succeeded") {
    renderTrainStatus(cur);
    trainRestart.classList.remove("hidden");
    return;
  }
  await startTraining();
});

trainRestart.addEventListener("click", async () => {
  try {
    await invoke("train_personal_reset");
  } catch (e) {
    /* reset of a finished job can't meaningfully fail — proceed */
  }
  await startTraining();
});

trainInstall.addEventListener("click", async () => {
  try {
    const dest = await invoke("train_personal_install");
    // The install already attempted a hot-reload on the Rust side; this
    // second call is idempotent and picks up the live state for the UI.
    try {
      const live = await invoke("reload_personal_adapter");
      if (live === true) {
        trainStage.textContent = `installed → ${dest}\nPersonal adapter live – no relaunch needed`;
        trainInstall.disabled = true;
        trainInstall.textContent = "✓ Installed — live";
        if (relaunchBadge) relaunchBadge.classList.add("hidden");
        flashToast("Personal adapter live – no relaunch needed");
        await probeCapabilities();
        refreshConfig();
        return;
      }
    } catch (e) {
      /* reload failed — fall through to the relaunch badge */
    }
    trainStage.textContent = `installed → ${dest}\nQuit and relaunch Nib to load the adapter.`;
    trainInstall.disabled = true;
    trainInstall.textContent = "✓ Installed — relaunch Nib";
    refreshConfig();
  } catch (e) {
    trainError.textContent = String(e);
  }
});

trainDismiss.addEventListener("click", () => {
  trainModal.classList.add("hidden");
  if (trainPollTimer) {
    clearInterval(trainPollTimer);
    trainPollTimer = null;
  }
  // Don't reset the actual job — user can re-open via "Train" while it's
  // still running. They can also explicitly reset via the menu (future).
});

// ---- Config (auto-retrain + relaunch badge) ----------------------------
async function refreshConfig() {
  try {
    const c = await invoke("config_get");
    // Don't clobber an in-progress edit — this poller runs every 7s and
    // used to overwrite the input before the change event committed it.
    if (document.activeElement !== autoRetrain) {
      autoRetrain.checked = !!c.auto_retrain_enabled;
    }
    if (document.activeElement !== autoThreshold) {
      autoThreshold.value = c.auto_retrain_threshold || 25;
    }
    autoLast.textContent = c.last_train_at
      ? `last trained ${c.last_train_at.slice(0, 10)}`
      : "never trained";
    if (c.pending_relaunch) {
      relaunchBadge.classList.remove("hidden");
    } else {
      relaunchBadge.classList.add("hidden");
    }
  } catch (e) {
    /* config not available — fail quiet */
  }
}

async function pushAutoRetrain() {
  try {
    const enabled = !!autoRetrain.checked;
    const threshold = parseInt(autoThreshold.value || "25", 10);
    await invoke("config_set_auto_retrain", { enabled, threshold });
    refreshConfig();
  } catch (e) {
    /* shrug; UI will reflect server state on next refresh */
  }
}

autoRetrain.addEventListener("change", pushAutoRetrain);
autoThreshold.addEventListener("change", pushAutoRetrain);

// ---- Settings panel (dictionary + pause + per-app overrides) ----------
const settingsToast = document.getElementById("settings-toast");
const pauseToggle = document.getElementById("pause-toggle");
const pauseSubtitle = document.getElementById("pause-subtitle");
const dictInput = document.getElementById("dict-input");
const dictAddBtn = document.getElementById("dict-add");
const dictList = document.getElementById("dict-list");
const appBundleInput = document.getElementById("app-bundle-input");
const appKindSelect = document.getElementById("app-kind-select");
const appAddBtn = document.getElementById("app-add");
const appsTable = document.getElementById("apps-table");

let settingsToastTimer = null;
function flashToast(msg, isError = false) {
  if (!settingsToast) return;
  settingsToast.textContent = msg;
  settingsToast.classList.remove("hidden");
  settingsToast.classList.toggle("error", !!isError);
  if (settingsToastTimer) clearTimeout(settingsToastTimer);
  settingsToastTimer = setTimeout(() => {
    settingsToast.classList.add("hidden");
  }, 3500);
}

const PAUSED_SUBTITLE = "Overlay is silent. Tray icon shows the same state.";
const ACTIVE_SUBTITLE = "Overlay watches focused text fields normally.";

function renderPause(paused) {
  pauseToggle.checked = !!paused;
  pauseSubtitle.textContent = paused ? PAUSED_SUBTITLE : ACTIVE_SUBTITLE;
}

function renderDict(words) {
  if (!Array.isArray(words) || words.length === 0) {
    dictList.innerHTML =
      `<div class="settings-empty">No words yet. Add names/jargon Nib shouldn't flag.</div>`;
    return;
  }
  dictList.innerHTML = words
    .map(
      (w) => `<div class="dict-item">
        <span class="dict-word">${escapeHtml(w)}</span>
        <button class="dict-remove" data-word="${escapeHtml(w)}" title="Remove">Remove</button>
      </div>`
    )
    .join("");
  dictList.querySelectorAll(".dict-remove").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const word = btn.dataset.word;
      try {
        const updated = await invoke("dictionary_remove", { word });
        renderDict(updated);
      } catch (e) {
        flashToast(`remove failed: ${e}`, true);
      }
    });
  });
}

const OVERRIDE_LABEL = {
  force_allow: "Force Allow",
  force_deny: "Force Deny",
};

function renderAppOverrides(overrides) {
  const entries = Object.entries(overrides || {}).sort(([a], [b]) =>
    a.localeCompare(b)
  );
  if (entries.length === 0) {
    appsTable.innerHTML =
      `<div class="settings-empty">No per-app overrides yet.</div>`;
    return;
  }
  appsTable.innerHTML = entries
    .map(
      ([bundleId, kind]) => `<div class="apps-row" data-kind="${escapeHtml(kind)}">
        <span class="apps-bundle">${escapeHtml(bundleId)}</span>
        <span class="apps-kind apps-kind-${escapeHtml(kind)}">${
          OVERRIDE_LABEL[kind] || kind
        }</span>
        <button class="apps-remove" data-bundle="${escapeHtml(bundleId)}" title="Remove">Remove</button>
      </div>`
    )
    .join("");
  appsTable.querySelectorAll(".apps-remove").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const bundleId = btn.dataset.bundle;
      try {
        const cfg = await invoke("app_override_remove", { bundleId });
        renderAppOverrides(cfg.app_overrides || {});
      } catch (e) {
        flashToast(`remove failed: ${e}`, true);
      }
    });
  });
}

async function refreshDictionary() {
  try {
    const words = await invoke("dictionary_list");
    renderDict(words);
  } catch (e) {
    flashToast(`dictionary load failed: ${e}`, true);
  }
}

async function dictAdd() {
  const word = (dictInput.value || "").trim();
  if (!word) return;
  try {
    const updated = await invoke("dictionary_add", { word });
    renderDict(updated);
    dictInput.value = "";
    dictInput.focus();
  } catch (e) {
    flashToast(`add failed: ${e}`, true);
  }
}

async function appAdd() {
  const bundleId = (appBundleInput.value || "").trim();
  if (!bundleId) return;
  const kind = appKindSelect.value || "force_allow";
  try {
    const cfg = await invoke("app_override_set", { bundleId, kind });
    renderAppOverrides(cfg.app_overrides || {});
    appBundleInput.value = "";
    appBundleInput.focus();
  } catch (e) {
    flashToast(`add failed: ${e}`, true);
  }
}

pauseToggle.addEventListener("change", async () => {
  const paused = !!pauseToggle.checked;
  try {
    const newPaused = await invoke("pause_set", { paused });
    renderPause(newPaused);
  } catch (e) {
    // Revert and warn.
    pauseToggle.checked = !paused;
    renderPause(!paused);
    flashToast(`pause failed: ${e}`, true);
  }
});

dictAddBtn.addEventListener("click", dictAdd);
dictInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    dictAdd();
  }
});

appAddBtn.addEventListener("click", appAdd);
appBundleInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    appAdd();
  }
});

// Reflect pause + per-app overrides from the full config snapshot. Sits
// alongside refreshConfig() (which already calls config_get for the
// auto-retrain block) instead of monkey-patching it.
async function refreshSettings() {
  try {
    const c = await invoke("config_get");
    renderPause(!!c.paused);
    renderAppOverrides(c.app_overrides || {});
  } catch (e) {
    /* fail quiet on the poll loop — initial-load errors surface in console */
  }
}

// ───────── Model picker ─────────
let lastModelListSnapshot = "";
async function refreshModelList() {
  try {
    const models = await invoke("model_list");
    const dl = await invoke("model_download_status");
    // Rebuilding the DOM every poll churns radio buttons under the
    // user's cursor — skip when nothing actually changed.
    const snapshot = JSON.stringify([models, dl]);
    if (snapshot === lastModelListSnapshot) return;
    lastModelListSnapshot = snapshot;
    renderModelList(models, dl);
  } catch (e) {
    console.error("model list refresh failed:", e);
  }
}

function renderModelList(models, dlStatus) {
  const list = document.getElementById("model-list");
  if (!list) return;
  list.innerHTML = "";

  // Hide base models that exist only as substrates for adapter entries —
  // users pick the *capability* ("Nib-Faithful"), not the bytes underneath.
  const byId = new Map(models.map(m => [m.id, m]));
  const baseIds = new Set(
    models.filter(m => m.requires_base).map(m => m.requires_base)
  );
  const visible = models.filter(m => !baseIds.has(m.id));

  for (const m of visible) {
    // `m` is ModelInfoExt: info fields flattened in + installed/selected.
    const installed = m.installed;
    const baseInfo = m.requires_base ? byId.get(m.requires_base) : null;
    // Adapter entries download as a queue (base first, then the adapter
    // itself) — match the progress tracker on either id.
    const dlIds = baseInfo ? [baseInfo.id, m.id] : [m.id];
    const isDownloading =
      dlIds.includes(dlStatus.model_id) && dlStatus.state === "running";
    const dlFailed =
      dlIds.includes(dlStatus.model_id) && dlStatus.state === "failed";

    const row = document.createElement("label");
    row.className = "model-row" + (m.selected ? " selected" : "");

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model";
    radio.value = m.id;
    radio.checked = m.selected;
    radio.disabled = !installed;
    radio.addEventListener("change", async () => {
      if (!installed) return;
      try {
        await invoke("model_set_selected", { id: m.id });
        flashToast(`Selected ${m.display_name}. Quit and relaunch Nib to load it.`);
        refreshModelList();
      } catch (e) {
        flashToast(`Failed: ${e}`, true);
      }
    });
    row.appendChild(radio);

    const info = document.createElement("div");
    info.className = "model-info";

    const name = document.createElement("div");
    name.className = "model-name";
    let statusPill = '';
    if (m.bundled && installed) statusPill = '<span class="pill bundled">bundled</span>';
    else if (installed) statusPill = '<span class="pill downloaded">downloaded</span>';
    else if (baseInfo) statusPill = '<span class="pill">needs base</span>';
    name.innerHTML =
      `<span>${escHtml(m.display_name)}</span>` +
      `<span class="pill">${escHtml(m.params)}</span>` +
      `<span class="pill">${m.size_mb} MB</span>` +
      statusPill;
    info.appendChild(name);

    const blurb = document.createElement("div");
    blurb.className = "model-blurb";
    blurb.textContent = m.blurb;
    info.appendChild(blurb);

    // Download button: only when this entry needs network. The Rust side
    // resolves the full queue of missing files (base and/or adapter) via
    // `download_targets`.
    const downloadable = !installed && !isDownloading && (m.url || baseInfo?.url);
    if (downloadable) {
      const btn = document.createElement("button");
      btn.className = "model-action";
      let dlLabel;
      if (baseInfo && !baseInfo.installed) {
        dlLabel = `↓ Download (${baseInfo.size_mb} MB base, one-time + ${m.size_mb} MB adapter)`;
      } else if (baseInfo) {
        dlLabel = `↓ Download adapter (${m.size_mb} MB)`;
      } else {
        dlLabel = `↓ Download (${m.size_mb} MB)`;
      }
      btn.textContent = dlLabel;
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        try {
          await invoke("model_download", { id: m.id });
          refreshModelList();
        } catch (err) {
          flashToast(`Download failed: ${err}`, true);
        }
      });
      info.appendChild(btn);
    }
    if (isDownloading) {
      // total_bytes tracks the CURRENT queue item (base vs adapter), so
      // derive the label from the tracker, not this row's size.
      const totalMb = dlStatus.total_bytes > 0
        ? dlStatus.total_bytes / 1024 / 1024
        : (baseInfo ? baseInfo.size_mb : m.size_mb);
      const pct = dlStatus.total_bytes > 0
        ? Math.min(100, (dlStatus.bytes_done / dlStatus.total_bytes) * 100)
        : 0;
      const stageLabel = baseInfo && dlStatus.model_id === m.id ? " (adapter)" : "";
      const prog = document.createElement("div");
      prog.className = "model-progress";
      prog.innerHTML =
        `Downloading${stageLabel}… ${(dlStatus.bytes_done / 1024 / 1024).toFixed(1)} MB / ${totalMb.toFixed(0)} MB (${pct.toFixed(0)}%)` +
        `<div class="model-progress-bar"><div style="width:${pct}%"></div></div>`;
      info.appendChild(prog);
    }
    if (dlFailed) {
      const err = document.createElement("div");
      err.className = "model-progress";
      err.style.color = "#ff7878";
      err.textContent = `Download failed: ${dlStatus.error || "unknown error"}`;
      info.appendChild(err);
    }

    row.appendChild(info);
    list.appendChild(row);
  }
}

function escHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// "How training works" expandable explanation in the personalization panel.
(() => {
  const toggle = document.getElementById("train-explain-toggle");
  const body = document.getElementById("train-explain");
  if (!toggle || !body) return;
  toggle.addEventListener("click", (e) => {
    e.preventDefault();
    const open = !body.hidden;
    body.hidden = open;
    toggle.textContent = open ? "how training works ▾" : "how training works ▴";
  });
})();

probeCapabilities();
runCheck();
refreshPersonal();
refreshConfig();
refreshDictionary();
refreshSettings();
refreshModelList();
setInterval(refreshPersonal, 5000);
setInterval(refreshConfig, 7000);
setInterval(refreshSettings, 7000);
// 1.5s cadence keeps the download progress bar live; when nothing
// changed the refresh short-circuits before touching the DOM.
setInterval(refreshModelList, 1500);
