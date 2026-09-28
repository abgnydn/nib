#!/usr/bin/env bash
# Install + relaunch dance for the Nib .app during local development.
#
#   Why this exists: each rebuild needs to (a) kill any running Nib, (b)
#   replace ~/Applications/Nib.app, (c) re-codesign with the stable
#   `app.nib` identifier so the TCC Accessibility grant survives, and
#   (d) launch a fresh process with stderr piped to ~/Library/Logs/nib.log
#   so the focus-tracker + arbiter logs are tailable. (The log carries
#   fragments of the user's prose, so it lives in the user's home — not
#   world-readable, symlink-attackable /tmp.)
#
#   Usage:
#       ./scripts/install-dev.sh                # uses the existing build
#       ./scripts/install-dev.sh --build        # full release rebuild (~15 min)
#       ./scripts/install-dev.sh --fast         # fast dev-release rebuild (~3 min)
#       ./scripts/install-dev.sh --build --tail # rebuild then tail the log
#
#   Codesigning note: we ad-hoc sign with `-` so the binary's identifier
#   stays `app.nib` across rebuilds (Tauri's default per-build random
#   identifier invalidates TCC grants).
#
#   --fast vs --build:
#     --fast  uses [profile.release-dev]: opt-level=1, codegen-units=16,
#             no LTO, no strip, no DMG bundling. Use during iteration.
#     --build uses [profile.release]:     opt-level=z, codegen-units=1,
#             full LTO, stripped, DMG generated. Use for ship builds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Nib.app"
BUILT_APP="$REPO_ROOT/shell/src-tauri/target/release/bundle/macos/$APP_NAME"
INSTALL_DIR="$HOME/Applications"
INSTALL_PATH="$INSTALL_DIR/$APP_NAME"
LOG_DIR="$HOME/Library/Logs"
LOG="$LOG_DIR/nib.log"
QVAC_CACHE="$HOME/.cache/qvac/qvac-fabric-llm.cpp"
QVAC_RESOURCES="$REPO_ROOT/shell/src-tauri/resources/qvac"

# One-version-only policy: every build, sweep Nib.app / Nib_*.dmg from
# everywhere they tend to accumulate so the user only ever sees the
# currently-installed copy in Spotlight / Finder. Conservative — never
# touches /Applications or user docs, only build artifacts + /tmp + the
# residual /Volumes mounts left by testing previous .dmgs.
cleanup_old_versions() {
  # 0. Kill any running Nib first — a copy running from a mounted DMG keeps
  #    the volume busy, which would make the ejects below fail.
  pkill -9 -f "$APP_NAME/Contents/MacOS/nib" 2>/dev/null || true
  sleep 1
  # 1. Eject any /Volumes/Nib* — testers leave these mounted. Best-effort:
  #    a busy volume (Finder window open, etc.) must not abort the install.
  for v in /Volumes/Nib*; do
    [[ -d "$v" ]] || continue
    diskutil eject "$v" 2>&1 | head -1 || \
      echo "[nib] warning: could not eject $v (busy?) — continuing"
  done
  # 2. Remove stale staging DMGs in /tmp (we always rebuild a fresh one).
  rm -f /tmp/Nib_*.dmg /tmp/Quill_*.dmg 2>/dev/null
  # 3. Remove the *previous* build artifact in target/<profile>/bundle/.
  rm -rf "$REPO_ROOT/shell/src-tauri/target/release-dev/bundle/macos/Nib.app" \
         "$REPO_ROOT/shell/src-tauri/target/release/bundle/macos/Nib.app" \
         "$REPO_ROOT/shell/src-tauri/target/release-dev/bundle/dmg" \
         "$REPO_ROOT/shell/src-tauri/target/release/bundle/dmg" 2>/dev/null
  # 4. Sweep any leftover Quill.app from the old name in ~/Applications.
  rm -rf "$HOME/Applications/Quill.app" 2>/dev/null
  # 5. Drop old WebKit caches from the previous bundle ID.
  rm -rf "$HOME/Library/WebKit/io.quill.app" 2>/dev/null
  # 6. Spotlight-skip marker — keep future target/ rebuilds out of Cmd+Space.
  mkdir -p "$REPO_ROOT/shell/src-tauri/target"
  touch "$REPO_ROOT/shell/src-tauri/target/.metadata_never_index"
}

# Build QVAC (BitNet + on-device LoRA training engine) once, cache the
# binaries under ~/.cache/qvac/, then copy them into the Tauri resources
# directory so they get bundled inside Nib.app/Contents/Resources/qvac/.
prepare_qvac() {
  mkdir -p "$QVAC_RESOURCES"
  local need_build=0
  for bin in llama-cli llama-finetune-lora; do
    if [[ ! -x "$QVAC_CACHE/build/bin/$bin" ]]; then
      need_build=1
    fi
  done
  if [[ "$need_build" -eq 1 ]]; then
    echo "[nib] QVAC binaries missing — cloning + building (one-time, ~5 min)…"
    if [[ ! -d "$QVAC_CACHE" ]]; then
      mkdir -p "$(dirname "$QVAC_CACHE")"
      git clone --depth 1 https://github.com/tetherto/qvac-fabric-llm.cpp \
        "$QVAC_CACHE"
    fi
    (
      cd "$QVAC_CACHE"
      cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=ON >/dev/null
      cmake --build build --config Release \
        --target llama-cli llama-finetune-lora -j "$(sysctl -n hw.ncpu)"
    )
  else
    echo "[nib] QVAC cache hit at $QVAC_CACHE"
  fi
  echo "[nib] staging QVAC binaries → $QVAC_RESOURCES"
  # Collect candidates without aborting on no-match glob expansion.
  shopt -s nullglob
  local files=(
    "$QVAC_CACHE/build/bin/llama-cli"
    "$QVAC_CACHE/build/bin/llama-finetune-lora"
    "$QVAC_CACHE/build/bin"/*.dylib
    "$QVAC_CACHE/build/bin"/*.metallib
  )
  shopt -u nullglob
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    # Skip symlinked versions — only copy concrete files.
    [[ -L "$f" ]] && continue
    cp -f "$f" "$QVAC_RESOURCES/"
  done
  echo "[nib] QVAC staged: $(ls "$QVAC_RESOURCES" | wc -l | tr -d ' ') files, \
$(du -sh "$QVAC_RESOURCES" | cut -f1)"
}

BUILD=0
FAST=0
TAIL=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    --fast)  FAST=1 ;;
    --tail)  TAIL=1 ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

if [[ "$BUILD" -eq 1 && "$FAST" -eq 1 ]]; then
  echo "[nib] --build and --fast are mutually exclusive" >&2
  exit 2
fi

if [[ "$BUILD" -eq 1 ]]; then
  echo "[nib] cleaning up old versions before build…"
  cleanup_old_versions
  prepare_qvac
  echo "[nib] building release with llm + overlay features (full opt, ~15 min)…"
  ( cd "$REPO_ROOT/shell/src-tauri" && cargo tauri build --features llm,overlay )
fi

if [[ "$FAST" -eq 1 ]]; then
  echo "[nib] cleaning up old versions before build…"
  cleanup_old_versions
  prepare_qvac
  echo "[nib] FAST build (profile=release-dev, no DMG, ~3 min)…"
  ( cd "$REPO_ROOT/shell/src-tauri" && \
    cargo tauri build --features llm,overlay --bundles app -- --profile release-dev )
  # `--profile release-dev` makes cargo put the binary at target/release-dev/
  # but Tauri's bundler looks at target/release/ for the .app it produces.
  # Tauri 2 actually emits to target/<profile>/bundle/macos/. Adjust path.
  BUILT_APP="$REPO_ROOT/shell/src-tauri/target/release-dev/bundle/macos/$APP_NAME"
fi

if [[ ! -d "$BUILT_APP" ]]; then
  echo "[nib] no built app at $BUILT_APP — run with --build" >&2
  exit 1
fi

echo "[nib] killing any running Nib processes…"
pkill -9 -f "$APP_NAME/Contents/MacOS/nib" 2>/dev/null || true
sleep 2

mkdir -p "$INSTALL_DIR"
echo "[nib] replacing $INSTALL_PATH"
rm -rf "$INSTALL_PATH"
cp -R "$BUILT_APP" "$INSTALL_PATH"
xattr -dr com.apple.quarantine "$INSTALL_PATH" 2>/dev/null || true

echo "[nib] ad-hoc codesign with stable identifier…"
codesign --force --deep --sign - "$INSTALL_PATH" 2>&1 | tail -1
codesign --display --verbose=2 "$INSTALL_PATH" 2>&1 | grep -E "Identifier|Signature" || true
# Fail fast if the identifier drifted — the TCC Accessibility grant is tied
# to app.nib, so a mismatch silently drops the grant.
if ! codesign --display --verbose=2 "$INSTALL_PATH" 2>&1 | grep -q "Identifier=app.nib"; then
  echo "[nib] ERROR: codesign Identifier mismatch (expected app.nib) — TCC grant would break" >&2
  codesign --display --verbose=2 "$INSTALL_PATH" 2>&1 >&2 || true
  exit 1
fi

mkdir -p "$LOG_DIR"
rm -f "$LOG"
( umask 077; : > "$LOG" )   # pre-create 0600 — the log carries prose fragments
echo "[nib] launching with stderr → $LOG"
"$INSTALL_PATH/Contents/MacOS/nib" > "$LOG" 2>&1 &
NIB_PID=$!
sleep 4
echo "[nib] pid=$NIB_PID  log=$LOG"

# Final sweep after install — drop the freshly-built artifact in target/
# now that it's been copied into ~/Applications. Spotlight only sees the
# canonical install path.
rm -rf "$REPO_ROOT/shell/src-tauri/target/release-dev/bundle/macos/Nib.app" \
       "$REPO_ROOT/shell/src-tauri/target/release/bundle/macos/Nib.app" 2>/dev/null

if [[ "$TAIL" -eq 1 ]]; then
  echo "[nib] tailing $LOG (Ctrl-C to stop)…"
  exec tail -f "$LOG" | grep --line-buffered -E "\[nib\]|focus-update|cursor|overlay-js|apply"
fi
