#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Local Transcription — One-shot Setup & Build
#  Works on any Apple Silicon Mac — safe to run again to rebuild.
#
#  What it does:
#    1. Installs Python 3.12 via Homebrew (if needed)
#    2. Creates a fresh .venv with all Python packages
#    3. Verifies every import
#    4. Builds the LocalTranscription.app bundle from scratch (Swift launcher)
#    5. Generates the app icon
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── Paths (all derived from this script's location — portable) ────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/transcription_app"
VENV="$APP_DIR/.venv"
APP="$SCRIPT_DIR/LocalTranscription.app"
CONTENTS="$APP/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"
BINARY="$MACOS_DIR/LocalTranscription"
VENV_PYTHON="$VENV/bin/python3"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ "$(uname -m)" != "arm64" ]; then
    echo "  ✗ This app requires an Apple Silicon (M-series) Mac."
    read -p "  Press Return to close…  " _; exit 1
fi
if [ ! -x /opt/homebrew/bin/brew ]; then
    echo "  ✗ Homebrew not found. Install it first: https://brew.sh"
    read -p "  Press Return to close…  " _; exit 1
fi
if ! xcode-select -p &>/dev/null || ! command -v swiftc &>/dev/null; then
    echo "  ✗ Xcode Command Line Tools are required (to compile the launcher)."
    echo "    Starting the installer now — re-run this script when it finishes."
    xcode-select --install 2>/dev/null || true
    read -p "  Press Return to close…  " _; exit 1
fi

eval "$(/opt/homebrew/bin/brew shellenv)"
PYTHON312="/opt/homebrew/opt/python@3.12/bin/python3.12"

echo ""
echo "  ══════════════════════════════════════════════════════════════"
echo "  🎙  Local Transcription — Setup & Build"
echo "  ══════════════════════════════════════════════════════════════"
echo ""
echo "  Project: $SCRIPT_DIR"
echo ""

# ── 1. Python 3.12 ────────────────────────────────────────────────────────────
echo "  [1/5] Checking Python 3.12…"
if ! command -v brew &>/dev/null; then
    echo "  ✗ Homebrew not found. Install it first: https://brew.sh"
    exit 1
fi
if [ ! -f "$PYTHON312" ]; then
    echo "  Installing python@3.12 (takes a minute)…"
    brew install python@3.12 python-tk@3.12
fi
"$PYTHON312" -c "import tkinter" 2>/dev/null || brew install python-tk@3.12
echo "  ✓ Python $("$PYTHON312" --version 2>&1 | awk '{print $2}')"

# ── 2. Virtual environment ────────────────────────────────────────────────────
echo ""
echo "  [2/5] Creating venv…"
rm -rf "$VENV"
"$PYTHON312" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q

echo "  Installing packages (first time: 3–5 min)…"
"$VENV/bin/pip" install -r "$APP_DIR/requirements.txt" -q
echo "  ✓ Packages installed"

# ── 3. Import verification ────────────────────────────────────────────────────
echo ""
echo "  [3/5] Verifying imports…"
"$VENV_PYTHON" - << 'PY'
failures = []
for name, stmt in [
    ("tkinter",        "import tkinter"),
    ("customtkinter",  "import customtkinter"),
    ("PIL",            "from PIL import Image"),
    ("sounddevice",    "import sounddevice"),
    ("mlx_whisper",    "import mlx_whisper"),
    ("ollama",         "import ollama"),
    ("pyannote.audio", "import pyannote.audio"),
]:
    try:
        exec(stmt); print(f"    ✓ {name}")
    except Exception as e:
        print(f"    ✗ {name}: {e}"); failures.append(name)
if failures:
    raise SystemExit(f"Failed imports: {failures}")
PY

# ── 4. Build app bundle from scratch ─────────────────────────────────────────
echo ""
echo "  [4/5] Building LocalTranscription.app…"

# Create bundle structure
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Write Info.plist
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>       <string>LocalTranscription</string>
    <key>CFBundleIdentifier</key>       <string>com.local.transcription</string>
    <key>CFBundleName</key>             <string>Transcribe</string>
    <key>CFBundleDisplayName</key>      <string>Transcribe</string>
    <key>CFBundleVersion</key>          <string>2.0</string>
    <key>CFBundleShortVersionString</key><string>2.0</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>CFBundleIconFile</key>         <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>   <string>13.0</string>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Microphone access is needed to transcribe in-person meetings.</string>
</dict>
</plist>
PLIST

# Compile Swift launcher — paths baked in at build time
SWIFT_SRC="$(mktemp /tmp/lt_launcher_XXXXXX.swift)"

cat > "$SWIFT_SRC" << SWIFT
import Foundation

let appDir      = "$APP_DIR"
let mainPy      = appDir + "/main.py"
let venvPython  = "$VENV_PYTHON"
let venvDir     = "$VENV"

// Launch Ollama if it's not already running
let check = Process()
check.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
check.arguments = ["-x", "ollama"]
try? check.run(); check.waitUntilExit()
if check.terminationStatus != 0 {
    let ol = Process()
    ol.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    ol.arguments = ["/Applications/Ollama.app"]
    try? ol.run()
    Thread.sleep(forTimeInterval: 1.5)
}

// Launch Python app
let p = Process()
p.executableURL = URL(fileURLWithPath: venvPython)
p.arguments    = [mainPy]
p.currentDirectoryURL = URL(fileURLWithPath: appDir)
p.environment = ProcessInfo.processInfo.environment.merging([
    "PATH":        "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin",
    "VIRTUAL_ENV": venvDir,
]) { \$1 }

do {
    try p.run(); p.waitUntilExit()
} catch {
    let a = Process()
    a.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    a.arguments = ["-e",
        "display alert \"Transcribe — Launch Error\" message \"\(error)\" buttons {\"OK\"}"]
    try? a.run(); a.waitUntilExit()
}
SWIFT

swiftc -O -o "$BINARY" "$SWIFT_SRC" && rm -f "$SWIFT_SRC"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
echo "  ✓ Launcher compiled"

# ── 5. Icon ───────────────────────────────────────────────────────────────────
echo ""
echo "  [5/5] Generating icon…"
"$VENV_PYTHON" "$APP_DIR/make_icon.py" "$APP" 2>/dev/null \
    && echo "  ✓ Icon generated" \
    || echo "  ⚠ Icon skipped (non-fatal)"

# ── Ollama model (optional) ───────────────────────────────────────────────────
echo ""
REC_MODEL="$(cd "$APP_DIR" && "$VENV_PYTHON" -c 'import summarizer; print(summarizer.recommended_model())')"
RAM_GB="$(( $(sysctl -n hw.memsize) / 1073741824 ))"
echo "  Recommended summary model for ${RAM_GB} GB RAM: $REC_MODEL"
if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$REC_MODEL"; then
        echo "  ✓ $REC_MODEL already installed"
    else
        read -p "  Download $REC_MODEL now? (several GB) [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            pgrep -x ollama >/dev/null || (open -a Ollama; sleep 3)
            ollama pull "$REC_MODEL" || echo "  ⚠ Pull failed — run: ollama pull $REC_MODEL"
        else
            echo "  Later, run: ollama pull $REC_MODEL"
        fi
    fi
else
    echo "  ⚠ Ollama not installed (needed for summaries): https://ollama.com"
    echo "    Then run: ollama pull $REC_MODEL"
fi

echo ""
echo "  ══════════════════════════════════════════════════════════════"
echo "  ✓  All done!"
echo ""
echo "  ➤  Double-click LocalTranscription.app to launch"
echo ""
echo "  Prerequisites (install separately if not present):"
echo "    • Ollama   — ollama.com"
echo "    • BlackHole 2ch — existential.audio/blackhole  (online meetings only)"
echo "  ══════════════════════════════════════════════════════════════"
echo ""
read -p "  Press Return to close…  " _
