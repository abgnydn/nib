// Nib overlay — runtime.
//
// All state + DOM logic for the click-through overlay window. Receives
// `focus-update` events from the Rust focus_tracker, `cursor-{enter,move,
// leave}-hot` events from the mouse_arbiter, and round-trips diagnostic
// pings back via the `overlay_ping` Tauri command.

(() => {
  const $ = (id) => document.getElementById(id);
  const underlinesEl = $("underlines");
  const fallbackEl = $("fallback");
  const fbHeader = $("fb-header");
  const fbList = $("fb-list");
  const popover = $("popover");
  const popBulk = $("pop-bulk");
  const popBulkAccept = $("pop-bulk-accept");
  const popBulkDismiss = $("pop-bulk-dismiss");
  const popBulkCount = $("pop-bulk-count");
  const popKind = $("pop-kind");
  const popMsg = $("pop-msg");
  const popSuggs = $("pop-suggs");
  const popWhyToggle = $("pop-why-toggle");
  const popWhyBody = $("pop-why-body");
  // (AI rewrite UI moved out of the hover popover in v1.1.2 — handled
  // entirely by the selection trigger + dedicated rewrite panel below.)
  // Selection trigger + rewrite panel — Grammarly-style separate surface
  // for selection-scoped rewrites (independent of the hover popover).
  const selTrigger = $("sel-trigger");
  const rewritePanel = $("rewrite-panel");
  const rpClose = $("rp-close");
  const rpSource = $("rp-source");
  const rpStreaming = $("rp-streaming");
  const rpOut = $("rp-out");
  const rpGo = $("rp-go");
  const rpApply = $("rp-apply");
  const rpRegen = $("rp-regen");
  const rpToneRow = $("rp-tone-row");
  const rpFormalityRow = $("rp-formality-row");
  const rpInstr = $("rp-instr");

  if (!window.__TAURI__ || !window.__TAURI__.event || !window.__TAURI__.core) {
    throw new Error("no tauri api");
  }
  const { listen } = window.__TAURI__.event;
  const { invoke } = window.__TAURI__.core;

  // ---- diagnostic ping --------------------------------------------------
  const ping = (stage, count, detail) =>
    invoke("overlay_ping", { stage, count, detail: detail || null }).catch(() => {});
  ping("script-loaded", 0);

  // ---- state -----------------------------------------------------------
  let eventCount = 0;
  let currentText = "";
  let currentLints = [];
  let currentFieldBounds = null;
  let activeLintIdx = -1;
  // Timestamp of the last popover open. While the popover is "fresh"
  // (< 400ms old), ignore cursor-move-hot events that would swap to a
  // different lint — gives the user time to move toward THIS popover
  // without losing it to a nearby underline.
  let popoverShownAt = 0;
  const POPOVER_STICKY_MS = 400;
  let hoverHideTimer = null;
  let lastRewrite = "";

  // The floating "Nib · N" corner chip was removed in v1.0.5 — the
  // menubar tray icon is the only ambient surface. flashCorner /
  // updateCorner are stubs so existing call sites keep working.
  const flashCorner = () => {};
  const updateCorner = () => {};

  // ---- helpers --------------------------------------------------------
  const escapeHtml = (s) =>
    String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  // Word-level LCS diff for the AI rewrite output. Splits on whitespace
  // boundaries but preserves separators so reassembling is exact.
  function tokenizeWords(s) {
    return s.match(/\s+|\S+/g) || [];
  }
  function diffWords(oldStr, newStr) {
    const a = tokenizeWords(oldStr);
    const b = tokenizeWords(newStr);
    const m = a.length, n = b.length;
    const dp = Array(m + 1).fill(null).map(() => new Uint32Array(n + 1));
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        dp[i][j] = a[i] === b[j] ? dp[i+1][j+1] + 1 : Math.max(dp[i+1][j], dp[i][j+1]);
      }
    }
    const out = [];
    let i = 0, j = 0;
    while (i < m && j < n) {
      if (a[i] === b[j]) { out.push({ type: "same", text: a[i] }); i++; j++; }
      else if (dp[i+1][j] >= dp[i][j+1]) { out.push({ type: "del", text: a[i] }); i++; }
      else { out.push({ type: "ins", text: b[j] }); j++; }
    }
    while (i < m) { out.push({ type: "del", text: a[i++] }); }
    while (j < n) { out.push({ type: "ins", text: b[j++] }); }
    const merged = [];
    for (const seg of out) {
      const last = merged[merged.length - 1];
      if (last && last.type === seg.type) last.text += seg.text;
      else merged.push({ ...seg });
    }
    return merged;
  }
  function renderDiffHtml(oldStr, newStr) {
    return diffWords(oldStr, newStr).map((s) => {
      const t = escapeHtml(s.text);
      if (s.type === "del") return `<del>${t}</del>`;
      if (s.type === "ins") return `<ins>${t}</ins>`;
      return t;
    }).join("");
  }

  // Grammarly's 4-lane category model: each LintKind from Harper is
  // bucketed into one of four lanes with a distinct color. Matches what
  // the user gets in Grammarly's hover card — visually classifies issues
  // at a glance instead of presenting 35 same-color squigglies.
  //
  //   correctness  red     — must-fix: spelling, grammar, agreement
  //   clarity      blue    — style: passive voice, wordiness, readability
  //   engagement   green   — word choice, vocabulary boost
  //   delivery     purple  — tone, formality, inclusivity (future)
  const kindClass = (kind) => {
    const k = String(kind).toLowerCase();
    if (k.includes("spell") ||
        k.includes("typo") ||
        k.includes("agreement") ||
        k.includes("capital") ||
        k.includes("punct") ||
        k.includes("boundary") ||
        k.includes("malaprop") ||
        k.includes("nonstandard") ||
        k.includes("possessive") ||
        k.includes("eggcorn") ||
        k.includes("formatting") ||
        k.includes("quotes")) return "correctness";
    if (k.includes("style") ||
        k.includes("wordy") ||
        k.includes("passive") ||
        k.includes("readability") ||
        k.includes("repetit") ||
        k.includes("redundan") ||
        k.includes("longsent") ||
        k.includes("spellednum")) return "clarity";
    if (k.includes("enhanc") ||
        k.includes("wordchoice") ||
        k.includes("usage") ||
        k.includes("boring") ||
        k.includes("vocab")) return "engagement";
    if (k.includes("tone") ||
        k.includes("formality") ||
        k.includes("inclus") ||
        k.includes("regional")) return "delivery";
    return "correctness";  // safe default — flag rather than hide
  };

  // Static "Why?" lookup, keyed by Harper's `LintKind` Debug name (see
  // shell/src-tauri/src/wire.rs → `format!("{:?}", l.lint_kind)`). When a
  // kind isn't covered, the popover falls back to the lint's own `message`.
  const WHY_MAP = {
    Agreement: "Subjects and verbs must agree in number. 'I has' uses the wrong verb form for the singular 'I' — singular subjects take 'have'/'is', plural subjects take 'have'/'are'.",
    Spelling: "This word isn't in the dictionary or contains a typo. Dictionary words read as more credible and avoid distracting your reader.",
    Typo: "Your brain knew the right word but your fingers slipped — like 'can be seem' instead of 'can be seen'. The suggested form is the intended one.",
    Capitalization: "Proper nouns, sentence starts, and 'I' need capital letters. Inconsistent casing makes prose look careless.",
    Punctuation: "Punctuation guides the reader's pacing — missing or extra marks can change meaning or stall the eye. Hyphens also matter for compound adjectives (e.g. 'face-first' before a noun).",
    Grammar: "This construction breaks a syntactic rule (tense, case, agreement, word order, etc.). The suggested edit restores standard English.",
    Style: "Both forms are technically correct, but one is preferred for clarity or formal writing — e.g. expanding 'min' to 'minimum' in a formal context.",
    Redundancy: "Words that repeat meaning already expressed weaken your prose. 'Free gift' and 'basic fundamentals' say the same thing twice.",
    Repetition: "The same word or phrase appears twice in a row, almost always by accident. Removing the duplicate usually fixes it.",
    WordChoice: "A different word fits the context better — sharper meaning, fewer connotations, or more natural collocation.",
    Usage: "Standard English prefers a particular collocation here (e.g. 'by accident' vs. 'on accident'). The suggestion matches the conventional form.",
    Enhancement: "Not an error, but a tightening — this edit makes the sentence clearer or more impactful without changing its meaning.",
    Readability: "This sentence is harder to parse than it needs to be — shorter clauses, plainer words, or active voice usually help.",
    BoundaryError: "Words are joined or split at the wrong boundaries — like 'each and everyone' for 'each and every one'. The suggestion separates or joins them correctly.",
    Eggcorn: "A similar-sounding word or phrase has crept in that almost makes sense ('egg corn' for 'acorn'). The suggestion restores the original idiom.",
    Malapropism: "A similar-sounding word with a different meaning slipped in — like 'eluded to' instead of 'alluded to'. The suggested word is the intended one.",
    Nonstandard: "This form is recognized but falls outside standard written English. Use the suggestion when you want the conventional spelling or phrasing.",
    Regionalism: "This spelling or phrasing is standard in some regions but not others (e.g. 'colour' vs. 'color'). The suggestion matches the dialect Nib is set to.",
    Formatting: "Whitespace, quotes, dashes, or other formatting characters don't match prose conventions — like straight quotes where curly quotes are preferred.",
    WrongQuotes: "Smart quotes (curly) are preferred over straight quotes in prose. Most word processors auto-substitute them; the suggestion does the same.",
    Miscellaneous: "Harper flagged this against a rule that doesn't fit the other categories. The suggestion is the rule's recommended replacement.",
  };
  const whyFor = (lint) => {
    if (!lint) return "";
    return WHY_MAP[lint.kind] || lint.message || "";
  };

  const renderChip = (s, lintIdx, suggIdx) => {
    const label = s.kind === "remove" ? "⌫ remove" : escapeHtml(s.text || "");
    const removeClass = s.kind === "remove" ? " remove" : "";
    return `<button class="sugg${removeClass}" data-lint="${lintIdx}" data-sugg="${suggIdx}">${label}</button>`;
  };

  const lintAtPoint = (x, y) => {
    for (let i = 0; i < currentLints.length; i++) {
      const r = currentLints[i].rect;
      if (!r) continue;
      const PAD = 4, TOP = 16;
      if (x >= r.x - PAD && x <= r.x + r.w + PAD &&
          y >= r.y + r.h - TOP - PAD && y <= r.y + r.h + PAD) {
        return i;
      }
    }
    return -1;
  };

  // ---- hot regions (driver for the Rust mouse arbiter) ----------------
  // When the popover is visible, we add a *bridge* rect that fully encloses
  // the active underline + the popover + the gap between them. Without this,
  // the cursor briefly enters click-through territory while traveling
  // between the two, the mouse arbiter fires `cursor-leave-hot`, and the
  // popover dismisses before the user can click a suggestion.
  const pushHotRegions = () => {
    const rects = [];
    underlinesEl.querySelectorAll(".underline").forEach((u) => {
      const r = u.getBoundingClientRect();
      rects.push({ x: r.left - 2, y: r.top - 4, w: r.width + 4, h: r.height + 12 });
    });
    if (popover.classList.contains("visible")) {
      const p = popover.getBoundingClientRect();
      rects.push({ x: p.left - 4, y: p.top - 4, w: p.width + 8, h: p.height + 8 });
      // Bridge: span from the active underline to the popover so the cursor
      // can travel between them without leaving hot territory.
      if (activeLintIdx >= 0 && currentLints[activeLintIdx]) {
        const ar = currentLints[activeLintIdx].rect;
        if (ar && ar.w > 0 && ar.h > 0) {
          const x0 = Math.min(p.left, ar.x) - 8;
          const y0 = Math.min(p.top, ar.y) - 8;
          const x1 = Math.max(p.right, ar.x + ar.w) + 8;
          const y1 = Math.max(p.bottom, ar.y + ar.h) + 8;
          rects.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
        }
      }
    }
    if (fallbackEl.classList.contains("visible")) {
      const r = fallbackEl.getBoundingClientRect();
      rects.push({ x: r.left - 4, y: r.top - 4, w: r.width + 8, h: r.height + 8 });
    }
    // Selection trigger (small blue ↓ button) — needs to be clickable.
    if (!selTrigger.hidden) {
      const r = selTrigger.getBoundingClientRect();
      rects.push({ x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 });
    }
    // Rewrite panel (selection-scoped) — interactive surface.
    if (!rewritePanel.hidden) {
      const r = rewritePanel.getBoundingClientRect();
      rects.push({ x: r.left - 4, y: r.top - 4, w: r.width + 8, h: r.height + 8 });
    }
    invoke("overlay_set_hot_regions", { rects }).catch(() => {});
  };

  // ---- popover --------------------------------------------------------
  const hidePopover = () => {
    popover.classList.remove("visible");
    activeLintIdx = -1;
    requestAnimationFrame(pushHotRegions);
  };
  const showPopover = (lintIdx) => {
    const lint = currentLints[lintIdx];
    if (!lint || !lint.rect) return;
    activeLintIdx = lintIdx;
    popoverShownAt = performance.now();
    clearTimeout(hoverHideTimer);
    flashCorner();

    const slice = [...currentText].slice(lint.start, lint.end).join("");
    const cls = kindClass(lint.kind);
    popKind.className = "pop-kind " + cls;
    popKind.innerHTML =
      `<span>${escapeHtml(lint.kind)}</span>` +
      `<span class="pop-target">${escapeHtml(slice)}</span>`;
    popMsg.textContent = lint.message;
    popSuggs.innerHTML = (lint.suggestions || [])
      .map((s, j) => renderChip(s, lintIdx, j))
      .join("");

    // Why? — collapsed by default; click expands an explanation block.
    popWhyBody.textContent = whyFor(lint);
    popWhyBody.hidden = true;
    popWhyToggle.setAttribute("aria-expanded", "false");
    popWhyToggle.textContent = "Why?";

    // Bulk toolbar — only useful when there are 2+ lints in the field.
    const n = currentLints.length;
    if (n >= 2) {
      popBulk.hidden = false;
      popBulkCount.textContent = `${n} issues`;
      popBulkAccept.disabled = false;
      popBulkAccept.textContent = "Accept all";
    } else {
      popBulk.hidden = true;
    }

    // Position bottom-right of the word (Grammarly-style).
    //
    // Multi-display: one overlay window per monitor, so window.innerWidth/
    // Height equal this monitor's viewport — clamp to them.
    const r = lint.rect;
    const W = window.innerWidth;
    const H = window.innerHeight;
    const pw = 320 + 24;
    const gap = 6;

    // Provisional position, refined after we can measure actual height.
    let x = r.x + r.w + gap;
    let y = r.y + r.h + gap;
    if (x + pw > W - 8) {
      x = Math.max(8, Math.min(r.x + r.w - pw, W - pw - 8));
    }
    x = Math.max(8, Math.min(x, W - pw - 8));

    // Cap popover max-height to remaining viewport space (below word OR
    // above word, whichever flips end up using). max-height takes effect
    // BEFORE measurement so we don't render a 1000px popover then shrink.
    const spaceBelow = H - (r.y + r.h) - 16;
    const spaceAbove = r.y - 16;
    const maxH = Math.max(180, Math.max(spaceBelow, spaceAbove));
    popover.style.maxHeight = maxH + "px";

    popover.style.left = x + "px";
    popover.style.top = y + "px";
    popover.classList.add("visible");

    // After render, measure REAL height. If it overflows below the
    // visible screen, flip above the word. Last-resort clamp + scroll.
    requestAnimationFrame(() => {
      const pb = popover.getBoundingClientRect();
      const realH = pb.height || 200;
      if (y + realH > H - 8) {
        // Try flipping above the word.
        const aboveY = r.y - realH - gap;
        if (aboveY >= 8) {
          y = aboveY;
        } else {
          // Neither below nor above fits the full popover. Pick the side
          // with more room and let the CSS overflow-y scroll the rest.
          if (spaceBelow >= spaceAbove) {
            y = r.y + r.h + gap;
            popover.style.maxHeight = (spaceBelow - 8) + "px";
          } else {
            y = 8;
            popover.style.maxHeight = (spaceAbove - 8) + "px";
          }
        }
        popover.style.top = y + "px";
      }
      pushHotRegions();
    });
  };

  // Hide delay: 300ms. mouseenter on the popover cancels the timer
  // entirely so hovering it keeps it open until explicit dismissal.
  // (The AI rewrite has its own panel now — no more 3000ms "give them
  // time to read the streaming output" case here.)
  const scheduleHide = () => {
    clearTimeout(hoverHideTimer);
    hoverHideTimer = setTimeout(hidePopover, 300);
  };
  const cancelHide = () => clearTimeout(hoverHideTimer);

  popover.addEventListener("mouseenter", cancelHide);
  popover.addEventListener("mouseleave", scheduleHide);

  // ---- suggestion click → AXUI write-back -----------------------------
  // Builds the applied_text optimistically so the journal captures the
  // (pre, post) pair without needing a second AXUI read.
  const applySuggestion = async (lintIdx, suggIdx, btn) => {
    const lint = currentLints[lintIdx];
    if (!lint) return;
    const s = lint.suggestions[suggIdx];
    if (!s) return;
    let start = lint.start, end = lint.end, replacement = s.text || "";
    if (s.kind === "remove") replacement = "";
    else if (s.kind === "insert_after") { start = lint.end; end = lint.end; }

    const chars = [...currentText];
    const applied_text =
      chars.slice(0, start).join("") + replacement + chars.slice(end).join("");

    if (btn) btn.classList.add("applied");
    try {
      await invoke("apply_suggestion", {
        start, end, replacement,
        context: {
          kind: "apply",
          source_text: currentText,
          applied_text,
          lint_kind: lint.kind,
          lint_message: lint.message,
        },
      });
      ping("apply-ok", lintIdx, `${lint.kind} -> "${replacement.slice(0,40)}"`);
      setTimeout(hidePopover, 280);
    } catch (err) {
      if (btn) btn.classList.remove("applied");
      ping("apply-err", lintIdx, String(err));
    }
  };

  popSuggs.addEventListener("click", (e) => {
    const t = e.target;
    if (!t || !t.classList.contains("sugg")) return;
    applySuggestion(activeLintIdx, parseInt(t.dataset.sugg, 10), t);
  });
  fbList.addEventListener("click", (e) => {
    const t = e.target;
    if (!t || !t.classList.contains("sugg")) return;
    applySuggestion(parseInt(t.dataset.lint, 10), parseInt(t.dataset.sugg, 10), t);
  });

  // ---- "Why?" expansion -----------------------------------------------
  popWhyToggle.addEventListener("click", () => {
    const open = !popWhyBody.hidden;
    popWhyBody.hidden = open;
    popWhyToggle.setAttribute("aria-expanded", open ? "false" : "true");
    popWhyToggle.textContent = open ? "Why?" : "Why? ▾";
    // The popover changed height — refresh hot regions so the cursor
    // arbiter doesn't drop us before the user clicks Apply.
    requestAnimationFrame(pushHotRegions);
  });

  // ---- bulk Accept all / Dismiss all ----------------------------------
  // Apply the first suggestion of every lint that has one. Lints share a
  // single text buffer, so earlier edits shift later character offsets —
  // we apply in REVERSE start order to keep the remaining offsets valid.
  // Snapshot the (lint, sugg) pairs up-front: each AXUI write fires a
  // focus-update that rewrites `currentLints` mid-loop, so we can't trust
  // currentLints[idx] later — we need to capture the offsets eagerly.
  popBulkAccept.addEventListener("click", async () => {
    if (popBulkAccept.disabled) return;
    const sourceText = currentText;
    const snapshots = currentLints
      .filter((l) => l && l.suggestions && l.suggestions.length > 0)
      .map((l) => ({ lint: l, sugg: l.suggestions[0] }))
      .sort((a, b) => b.lint.start - a.lint.start);
    if (!snapshots.length) {
      hidePopover();
      return;
    }
    popBulkAccept.disabled = true;
    popBulkAccept.textContent = `Applying 0/${snapshots.length}…`;
    let ok = 0;
    let runningText = sourceText;
    for (let k = 0; k < snapshots.length; k++) {
      const { lint, sugg } = snapshots[k];
      let start = lint.start, end = lint.end, replacement = sugg.text || "";
      if (sugg.kind === "remove") replacement = "";
      else if (sugg.kind === "insert_after") { start = lint.end; end = lint.end; }
      const chars = [...runningText];
      const applied_text =
        chars.slice(0, start).join("") + replacement + chars.slice(end).join("");
      try {
        await invoke("apply_suggestion", {
          start, end, replacement,
          context: {
            kind: "apply",
            source_text: runningText,
            applied_text,
            lint_kind: lint.kind,
            lint_message: lint.message,
          },
        });
        runningText = applied_text;
        ok++;
      } catch (err) {
        ping("bulk-apply-err", k, String(err));
      }
      popBulkAccept.textContent = `Applying ${k + 1}/${snapshots.length}…`;
    }
    ping("bulk-apply-done", ok, `total=${snapshots.length}`);
    setTimeout(hidePopover, 200);
  });
  popBulkDismiss.addEventListener("click", () => {
    ping("bulk-dismiss", currentLints.length);
    hidePopover();
  });

  // ---- AI rewrite session id (used by selection rewrite panel) -------
  function makeSession() {
    return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));
  }

  // ---- inline underline rendering ------------------------------------
  const FAT = 14;  // hover-target height (visible wavy stroke is 4px)
  const renderUnderlines = () => {
    underlinesEl.innerHTML = "";
    let rendered = 0;
    for (let i = 0; i < currentLints.length; i++) {
      const l = currentLints[i];
      if (!l.rect || l.rect.w < 1 || l.rect.h < 1) continue;
      const u = document.createElement("div");
      u.className = "underline " + kindClass(l.kind);
      u.dataset.lintIdx = String(i);
      u.style.left = l.rect.x + "px";
      u.style.top = (l.rect.y + l.rect.h - FAT) + "px";
      u.style.width = l.rect.w + "px";
      u.style.height = FAT + "px";
      u.title = l.message;
      underlinesEl.appendChild(u);
      rendered++;
    }
    renderFallback(rendered);
    pushHotRegions();
  };

  // ---- browser fallback panel -----------------------------------------
  // When AXUI `bounds_for_range` returns None for every lint (browsers /
  // Electron web inputs), inline underlines can't render. The fallback
  // panel beside the field lists every suggestion instead.
  let fbActiveIdx = 0;
  const FB_CTX_MAX = 120;
  // Focusable + listbox semantics so ArrowDown/Up + Enter work when the
  // panel has focus; the document-level keydown below covers the rest.
  fallbackEl.tabIndex = 0;
  fallbackEl.setAttribute("role", "listbox");
  fallbackEl.setAttribute("aria-label", "Nib suggestions");
  fallbackEl.style.overflowY = "auto";

  const truncateStr = (s, max) => {
    const chars = [...String(s || "")];
    if (chars.length <= max) return String(s || "");
    return chars.slice(0, Math.max(0, max - 1)).join("") + "…";
  };

  // Source sentence around [start, end): scan out to sentence boundaries
  // (. ! ? newline), fall back to ±60 chars, then cap at FB_CTX_MAX.
  // `clipped` is the truncated guard — the row renders "…" at cut edges.
  const sentenceContext = (text, start, end) => {
    const chars = [...String(text || "")];
    const n = chars.length;
    const s = Math.max(0, Math.min(start | 0, n));
    const e = Math.max(s, Math.min(end | 0, n));
    const isBoundary = (c) => c === "." || c === "!" || c === "?" || c === "\n";
    let a = s;
    const backStop = Math.max(0, s - 80);
    while (a > backStop && !isBoundary(chars[a - 1])) a--;
    if (a === 0 || isBoundary(chars[a - 1])) {
      while (a < s && (chars[a] === " " || chars[a] === "\n")) a++;
    } else {
      a = Math.max(0, s - 60);
    }
    let b = e;
    const fwdStop = Math.min(n, e + 80);
    while (b < fwdStop && !isBoundary(chars[b])) b++;
    if (b < n && isBoundary(chars[b])) b += 1;
    else if (b === fwdStop) b = Math.min(n, e + 60);
    const clippedLeft = a > 0;
    const clippedRight = b < n;
    const hit = chars.slice(s, e).join("");
    let bArr = chars.slice(a, s);
    let aArr = chars.slice(e, b);
    const hLen = [...hit].length;
    let over = bArr.length + hLen + aArr.length - FB_CTX_MAX;
    let cutLeft = false, cutRight = false;
    if (over > 0) {
      const takeBefore = Math.min(bArr.length, Math.ceil(over / 2));
      if (takeBefore > 0) { bArr = bArr.slice(takeBefore); over -= takeBefore; cutLeft = true; }
      if (over > 0 && aArr.length > 0) { aArr = aArr.slice(0, Math.max(0, aArr.length - over)); cutRight = true; }
    }
    const before = ((clippedLeft || cutLeft) ? "…" : "") + bArr.join("");
    const after = aArr.join("") + ((clippedRight || cutRight) ? "…" : "");
    return { before, hit, after, clipped: clippedLeft || clippedRight || cutLeft || cutRight };
  };

  // Highlight the active row without a full re-render (no flicker).
  const setFbActive = (idx, scroll = true) => {
    if (!currentLints.length) return;
    const clamped = Math.max(0, Math.min(idx, currentLints.length - 1));
    fbActiveIdx = clamped;
    fbList.querySelectorAll(".fb-row").forEach((row) => {
      const active = parseInt(row.dataset.fbIdx, 10) === clamped;
      row.classList.toggle("active", active);
      row.setAttribute("aria-selected", active ? "true" : "false");
    });
    if (scroll) {
      const el = fbList.querySelector(`.fb-row[data-fb-idx="${clamped}"]`);
      if (el && el.scrollIntoView) {
        try { el.scrollIntoView({ block: "nearest" }); } catch (ignored) {}
      }
    }
  };

  // Lint overlapping [selStart, selEnd), else the nearest lint at/after
  // the caret. -1 when there is nothing to follow.
  const fbLintForSelection = (selStart, selEnd) => {
    for (let i = 0; i < currentLints.length; i++) {
      const l = currentLints[i];
      if (l.start < selEnd && l.end > selStart) return i;
    }
    let best = -1;
    for (let i = 0; i < currentLints.length; i++) {
      if (currentLints[i].start >= selEnd &&
          (best < 0 || currentLints[i].start < currentLints[best].start)) best = i;
    }
    return best;
  };

  const renderFallback = (renderedInlineCount) => {
    const shouldHide = !currentLints.length || renderedInlineCount > 0 || !currentFieldBounds;
    if (shouldHide) {
      // No flicker on empty: leave an already-hidden panel untouched
      // (don't clear innerHTML — avoids layout thrash on empty polls).
      if (!fallbackEl.classList.contains("visible")) return;
      fallbackEl.classList.remove("visible");
      fbActiveIdx = 0;
      return;
    }
    fbActiveIdx = Math.max(0, Math.min(fbActiveIdx, currentLints.length - 1));
    fbHeader.textContent =
      `Nib — ${currentLints.length} issue${currentLints.length === 1 ? "" : "s"} · ↑↓ + Enter`;
    fbList.innerHTML = currentLints.map((l, i) => {
      const slice = [...currentText].slice(l.start, l.end).join("");
      const suggs = (l.suggestions || []).map((s, j) => renderChip(s, i, j)).join("");
      const ctx = sentenceContext(currentText, l.start, l.end);
      const ctxHtml =
        `${escapeHtml(ctx.before)}<b>${escapeHtml(ctx.hit || slice)}</b>${escapeHtml(ctx.after)}`;
      const active = i === fbActiveIdx ? " active" : "";
      const sel = i === fbActiveIdx ? "true" : "false";
      return `<div class="fb-row ${kindClass(l.kind)}${active}" data-fb-idx="${i}" tabindex="0" role="option" aria-selected="${sel}">
        <div class="fb-msg"><b>${escapeHtml(truncateStr(slice, 60))}</b> — ${escapeHtml(l.message)}</div>
        <div class="fb-ctx">${ctxHtml}</div>
        ${suggs ? `<div class="fb-suggs">${suggs}</div>` : ""}
      </div>`;
    }).join("");
    const b = currentFieldBounds;
    // Per-monitor overlay window: clamp to this window's viewport.
    // The fallback panel stays for fields where bounds_for_range returns None.
    const W = window.innerWidth, H = window.innerHeight;
    const fw = 300 + 24;
    let x = b.x + b.w + 12;
    let y = b.y;
    if (x + fw > W) { x = b.x; y = b.y + b.h + 12; }
    if (y + 300 > H) y = Math.max(8, H - 320);
    fallbackEl.style.left = Math.max(8, x) + "px";
    fallbackEl.style.top  = Math.max(8, y) + "px";
    fallbackEl.style.maxHeight = Math.max(180, H - Math.max(8, y) - 8) + "px";
    fallbackEl.classList.add("visible");
    const activeEl = fbList.querySelector(`.fb-row[data-fb-idx="${fbActiveIdx}"]`);
    if (activeEl && activeEl.scrollIntoView) {
      try { activeEl.scrollIntoView({ block: "nearest" }); } catch (ignored) {}
    }
  };

  // Keyboard: ArrowDown/Up moves the active fallback row, Enter applies
  // its first suggestion. Skipped while the rewrite panel is open so the
  // two surfaces never fight over keys.
  document.addEventListener("keydown", (e) => {
    if (!fallbackEl.classList.contains("visible")) return;
    if (!rewritePanel.hidden) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Enter") return;
    if (!currentLints.length) return;
    e.preventDefault();
    if (e.key === "ArrowDown") {
      setFbActive(fbActiveIdx + 1 >= currentLints.length ? 0 : fbActiveIdx + 1);
    } else if (e.key === "ArrowUp") {
      setFbActive(fbActiveIdx - 1 < 0 ? currentLints.length - 1 : fbActiveIdx - 1);
    } else {
      const lint = currentLints[fbActiveIdx];
      if (!lint || !lint.suggestions || !lint.suggestions.length) return;
      const btn = fbList.querySelector(`.fb-row[data-fb-idx="${fbActiveIdx}"] .sugg`);
      applySuggestion(fbActiveIdx, 0, btn);
    }
  });

  // Clicking a row (outside its chips) moves keyboard focus there without
  // applying — the suggestion-chip path above still handles applies.
  fbList.addEventListener("click", (e) => {
    const t = e.target;
    if (!t || !t.closest) return;
    if (t.classList && t.classList.contains("sugg")) return;
    const row = t.closest(".fb-row");
    if (!row) return;
    setFbActive(parseInt(row.dataset.fbIdx, 10), false);
  });

  // ---- listeners ------------------------------------------------------
  listen("cursor-enter-hot", (evt) => {
    const { x, y } = evt.payload || {};
    const idx = lintAtPoint(x, y);
    ping("cursor-enter-hot", idx, `x=${x|0} y=${y|0}`);
    if (idx >= 0) showPopover(idx);
  });
  listen("cursor-move-hot", (evt) => {
    const { x, y } = evt.payload || {};
    const idx = lintAtPoint(x, y);
    if (idx < 0 || idx === activeLintIdx) return;
    // Stickiness window: don't swap popovers if the current one just
    // appeared. Lets the user move toward the popover past adjacent
    // underlines without losing it.
    if (popover.classList.contains("visible") &&
        performance.now() - popoverShownAt < POPOVER_STICKY_MS) {
      return;
    }
    showPopover(idx);
  });
  listen("cursor-leave-hot", () => {
    ping("cursor-leave-hot", activeLintIdx);
    scheduleHide();
  });

  listen("focus-update", (evt) => {
    eventCount++;
    const p = evt.payload || {};
    const newText = p.text || "";
    const newLints = p.lints || [];
    const newBounds = p.bounds || null;

    // (Rust focus_tracker now suppresses focus-updates when Nib itself is
    // the focused app — so an empty update here means the user actually
    // switched to a non-engaged app like Cursor/Terminal. Clear state +
    // hide the popover so stale underlines don't linger on the new window.)
    // The popover's lint offsets are stale the moment the text or field
    // changes — actually hide it (this comment used to lie).
    if (popover.classList.contains("visible") &&
        (!newBounds || newText !== currentText)) {
      hidePopover();
    }
    currentText = newText;
    currentLints = newLints;
    currentFieldBounds = newBounds;
    updateCorner();
    flashCorner();
    const inline = currentLints.filter((l) => l.rect).length;
    ping("event", eventCount,
      `lints=${currentLints.length} inline=${inline} text_len=${currentText.length}`);

    if (lastRewrite && currentText && lastRewrite !== currentText) {
      lastRewrite = "";
    }
    if (!currentFieldBounds) fallbackEl.classList.remove("visible");
    renderUnderlines();
  }).then(() => ping("listen-registered", 0))
    .catch((err) => ping("listen-error", 0, String(err)));

  // ---- Selection trigger + rewrite panel ------------------------------
  // Tracks the most recent selection-update so the rewrite panel knows
  // what text to send to the model and what range to apply over.
  let currentSelection = null;  // { rect, text, start, end } | null
  // Snapshot taken when the panel OPENS. Apply must write over the range
  // the user asked to rewrite — not whatever they happen to have selected
  // by the time they click Apply.
  let panelSelection = null;
  let rpSession = null;
  let rpRewrite = "";

  const hideTrigger = () => { selTrigger.hidden = true; };
  const hideRewritePanel = () => {
    rewritePanel.hidden = true;
    rpRewrite = "";
    panelSelection = null;
    rpApply.hidden = true;
    rpRegen.hidden = true;
    rpOut.textContent = "";
    rpStreaming.hidden = true;
    requestAnimationFrame(pushHotRegions);
  };

  const positionTrigger = (rect) => {
    // Anchor ~6px to the LEFT of the selection's left edge, vertically
    // centered. Falls back to clamping inside the visible viewport.
    const W = window.innerWidth, H = window.innerHeight;
    const size = 28;
    let x = rect.x - size - 6;
    let y = rect.y + (rect.h / 2) - (size / 2);
    if (x < 8) x = rect.x + rect.w + 6;       // flip to the right if no room
    x = Math.max(8, Math.min(x, W - size - 8));
    y = Math.max(8, Math.min(y, H - size - 8));
    selTrigger.style.left = x + "px";
    selTrigger.style.top  = y + "px";
  };

  const positionRewritePanel = (rect) => {
    // Position below-right of the selection's bottom-right corner, with
    // the same viewport clamping logic as the hover popover.
    const W = window.innerWidth, H = window.innerHeight;
    const pw = 380 + 24;
    let x = rect.x + rect.w + 12;
    let y = rect.y + rect.h + 8;
    if (x + pw > W - 8) x = Math.max(8, W - pw - 8);
    const spaceBelow = H - (rect.y + rect.h) - 16;
    rewritePanel.style.maxHeight = Math.max(220, spaceBelow) + "px";
    rewritePanel.style.left = x + "px";
    rewritePanel.style.top  = y + "px";
    requestAnimationFrame(() => {
      const pb = rewritePanel.getBoundingClientRect();
      if (y + pb.height > H - 8) {
        const aboveY = rect.y - pb.height - 8;
        rewritePanel.style.top = Math.max(8, aboveY) + "px";
      }
      pushHotRegions();
    });
  };

  listen("selection-update", (evt) => {
    const p = evt.payload || {};
    // `truncated` selections carry only a prefix of the real range —
    // rewriting them would replace the full range with a rewrite of the
    // prefix, deleting the tail. Never offer the trigger for those.
    currentSelection = (p.rect && p.text && !p.truncated)
      ? { rect: p.rect, text: p.text, start: p.start, end: p.end }
      : null;
    if (!currentSelection) {
      // No selection → hide trigger. Don't auto-hide the rewrite panel
      // mid-rewrite (user might still want to apply the result).
      hideTrigger();
      return;
    }
    positionTrigger(currentSelection.rect);
    selTrigger.hidden = false;
    requestAnimationFrame(pushHotRegions);
    // Fallback auto-follow: move the active row to the lint under the
    // caret/selection so keyboard Enter applies the relevant fix.
    // Truncated guard: a truncated selection carries only a prefix —
    // its offsets don't map to visible text, so never follow it.
    if (!p.truncated && typeof p.start === "number" && typeof p.end === "number" &&
        fallbackEl.classList.contains("visible") && currentLints.length) {
      const idx = fbLintForSelection(p.start, p.end);
      if (idx >= 0) setFbActive(idx);
    }
  }).catch((err) => ping("sel-listen-err", 0, String(err)));

  selTrigger.addEventListener("click", () => {
    if (!currentSelection) return;
    // Freeze the selection the panel operates on.
    panelSelection = { ...currentSelection };
    ping("sel-trigger-click", panelSelection.text.length, "");
    rpSource.textContent = panelSelection.text;
    rpOut.textContent = "";
    rpStreaming.hidden = true;
    rpApply.hidden = true;
    rpRegen.hidden = true;
    rpGo.hidden = false;
    rpGo.disabled = false;
    rpGo.textContent = "Rewrite";
    // Reset chip selection on each open — tone/formality are per-session.
    rpToneRow.querySelectorAll(".rp-chip").forEach(c => c.classList.remove("active"));
    rpFormalityRow.querySelectorAll(".rp-chip").forEach(c => c.classList.remove("active"));
    refreshInstrBadge();
    positionRewritePanel(panelSelection.rect);
    rewritePanel.hidden = false;
    hideTrigger();
    requestAnimationFrame(pushHotRegions);
  });

  // Single-select chip toggle: clicking sets active, clicking again
  // clears. Click another chip in the same row → swaps active.
  // Each click also refreshes the instruction badge so the user can see
  // what we'll send to the model.
  const wireChipRow = (row) => {
    row.addEventListener("click", (e) => {
      const t = e.target;
      if (!t.classList.contains("rp-chip")) return;
      const wasActive = t.classList.contains("active");
      row.querySelectorAll(".rp-chip").forEach(c => c.classList.remove("active"));
      if (!wasActive) t.classList.add("active");
      refreshInstrBadge();
    });
  };
  wireChipRow(rpToneRow);
  wireChipRow(rpFormalityRow);

  // Compose an LLM instruction string from the currently-active chips.
  // Returns null when no chip is active — the caller falls back to the
  // model's default editing instruction ("Fix the grammar and improve
  // clarity:").
  //
  // Instruction design (after observing both LFM2.5-350M and 1.2B
  // hallucinate "I/you" relationships + corporate filler):
  //   - Lead with role ("copy editor") — anchors the model away from
  //     "helpful assistant" persona that wants to elaborate.
  //   - Numeric constraint ("same number of words, ±20%") — small
  //     models follow short numeric rules better than long prose
  //     guards.
  //   - Explicit deny list — first/second person, commitments,
  //     relationships, padding. Each one is a real failure mode we
  //     saw in testing.
  //   - "Output only the rewritten text" — kills preamble like
  //     "Sure, here is your rewrite:".
  const composeInstruction = () => {
    const tone = rpToneRow.querySelector(".rp-chip.active")?.dataset.tone;
    const formality = rpFormalityRow.querySelector(".rp-chip.active")?.dataset.formality;
    if (!tone && !formality) return null;
    const parts = [];
    if (tone) parts.push(tone);
    if (formality) parts.push(formality);
    const style = parts.join(", ");
    return `You are a copy editor. Restate the user's text in a ${style} tone. ` +
           `Keep the same number of words (±20%). ` +
           `Do not introduce first or second person (I/you/we) unless the source uses them. ` +
           `Do not add commitments, relationships, opinions, or context not in the source. ` +
           `Do not pad with filler. Output only the rewritten text, nothing else.`;
  };

  // Update the instruction badge whenever chips change. Hidden when no
  // chips are active (we'll use the default model instruction silently).
  const refreshInstrBadge = () => {
    const tone = rpToneRow.querySelector(".rp-chip.active")?.dataset.tone;
    const formality = rpFormalityRow.querySelector(".rp-chip.active")?.dataset.formality;
    if (!tone && !formality) {
      rpInstr.hidden = true;
      return;
    }
    const labels = [];
    if (tone) labels.push(tone);
    if (formality) labels.push(formality);
    rpInstr.innerHTML =
      '<span class="rp-instr-arrow">→ asking for:</span>' +
      labels.join(" · ");
    rpInstr.hidden = false;
  };

  // Minimum selection length we'll send to the model. Below this, the
  // model has no context to ground the rewrite and will hallucinate
  // (turning "manage users: 1 Day" into "Confidently manage user
  // inquiries…"). 30 chars covers the smallest useful sentence.
  const MIN_REWRITE_CHARS = 30;

  rpClose.addEventListener("click", hideRewritePanel);

  const runRewritePanel = async () => {
    if (!panelSelection) return;
    // Min-length guard: under 30 chars the model has no grounding and
    // hallucinates. Show a friendly message instead of trying.
    if (panelSelection.text.trim().length < MIN_REWRITE_CHARS) {
      rpOut.innerHTML =
        '<div class="rp-too-short">Select a full sentence (at least ' +
        MIN_REWRITE_CHARS + ' characters) — the model needs context to ' +
        'rewrite faithfully. Fragments tend to get hallucinated into ' +
        'unrelated prose.</div>';
      rpApply.hidden = true;
      rpRegen.hidden = true;
      rpGo.disabled = false;
      ping("rp-rewrite-too-short", panelSelection.text.length, "");
      return;
    }
    rpGo.disabled = true;
    rpGo.textContent = "Streaming…";
    rpStreaming.hidden = false;
    rpStreaming.textContent = "model generating…";
    rpOut.textContent = "";
    rpApply.hidden = true;
    rpRegen.hidden = true;
    const session = makeSession();
    rpSession = session;
    const unlisten = await listen("rewrite-token", (evt) => {
      const p = evt.payload || {};
      if (p.session !== session) return;
      if (p.done) { rpStreaming.hidden = true; return; }
      if (p.delta) {
        rpOut.textContent += p.delta;
        requestAnimationFrame(pushHotRegions);
      }
    });
    try {
      // rewrite returns { text, truncated }.
      const out = await invoke("rewrite", {
        text: panelSelection.text,
        instruction: composeInstruction(),
        session,
      });
      rpRewrite = String(out.text || "");
      if (!rpOut.textContent) rpOut.textContent = rpRewrite;
      // Show inline diff against the original selection.
      if (panelSelection.text && rpRewrite && panelSelection.text !== rpRewrite) {
        rpOut.innerHTML = renderDiffHtml(panelSelection.text, rpRewrite);
      }
      if (out.truncated) {
        // A cut-off rewrite must never be applied over the selection —
        // it would replace the user's text with a fragment.
        rpRewrite = "";
        rpOut.innerHTML +=
          '<div class="rp-too-short">⚠ The rewrite was cut off at the ' +
          'length limit — try a shorter selection.</div>';
        rpApply.hidden = true;
        rpRegen.hidden = false;
        rpGo.hidden = true;
      } else {
        rpApply.hidden = false;
        rpRegen.hidden = false;
        rpGo.hidden = true;
      }
    } catch (err) {
      rpOut.textContent = "error: " + String(err);
      rpGo.disabled = false;
      rpGo.textContent = "Retry";
      ping("rp-rewrite-err", 0, String(err));
    } finally {
      unlisten();
      rpStreaming.hidden = true;
    }
  };

  rpGo.addEventListener("click", runRewritePanel);
  rpRegen.addEventListener("click", runRewritePanel);
  rpApply.addEventListener("click", async () => {
    // Apply over the SNAPSHOT taken when the panel opened — the live
    // currentSelection may have moved to a different span since.
    if (!panelSelection || !rpRewrite) return;
    const { start, end, text: orig } = panelSelection;
    const chars = [...currentText];
    const applied = chars.slice(0, start).join("") + rpRewrite + chars.slice(end).join("");
    try {
      await invoke("apply_suggestion", {
        start, end,
        replacement: rpRewrite,
        context: {
          kind: "rewrite_apply",
          source_text: currentText,
          applied_text: applied,
        },
      });
      ping("rp-apply-ok", rpRewrite.length, `${orig.length}→${rpRewrite.length}`);
      hideRewritePanel();
    } catch (err) {
      ping("rp-apply-err", 0, String(err));
    }
  });
})();
