# 🎙 Local Meeting Transcription

**100% on-device** · **Apple Silicon native** · **No cloud, no subscriptions**

Transcribes online and in-person meetings in real time using
[mlx-whisper](https://github.com/ml-explore/mlx-examples) (Apple MLX framework)
and generates AI summaries via [Ollama](https://ollama.com).

---

## Requirements

- **macOS 13 Ventura or later** (Apple Silicon M1–M5, recommended M2+)
- **Homebrew** — [brew.sh](https://brew.sh) (for Python install)
- **Ollama** (optional, for AI summaries) — [ollama.com](https://ollama.com)
- **BlackHole 2ch** (optional, for online meetings) — [existential.audio/blackhole](https://existential.audio/blackhole/)

---

## Quick Start

Double-click **`SETUP_AND_BUILD.command`** in the project root.

It will:
1. Install Python 3.12 via Homebrew (if needed)
2. Create a fresh `.venv` with all required packages
3. Verify every import
4. Compile the native macOS `.app` launcher
5. Generate the app icon

When it finishes, double-click **`LocalTranscription.app`** to launch.

> **First launch:** Whisper `large-v3-turbo` (~800 MB) downloads automatically the first time you start a recording.

---

## Transferring to a New Mac

Because the app runs entirely on-device with a compiled Swift launcher, you need to rebuild it on any new machine. Here's how:

**Step 1 — Copy the source folder**

Copy the entire `Local Transcription LLM App/` folder to the new Mac. You can use AirDrop, a USB drive, or any file transfer method. The `.venv/` folder inside `transcription_app/` does **not** need to be copied — it will be rebuilt automatically.

**Step 2 — Install prerequisites on the new Mac**

- Install **Homebrew**: `https://brew.sh`
- Install **Ollama**: `https://ollama.com` (then run `ollama pull llama3.2:3b`)
- Optionally install **BlackHole 2ch** for online meetings

**Step 3 — Run setup on the new Mac**

Double-click **`SETUP_AND_BUILD.command`** on the new machine. It detects the new paths automatically and rebuilds everything from scratch — Python venv, `.app` bundle, and launcher binary.

**Step 4 — Launch**

Double-click `LocalTranscription.app`. Done.

---

## Using the App

The app uses a single-window interface with a sidebar list of all your past meetings.

**Starting a new meeting:**
1. Select your audio source: **Microphone** (in-person) or **System Audio** (online calls)
2. Click **⏺ Record** — transcription starts immediately
3. Click **⏹ Stop** when done — the transcript is saved automatically

**AI features (after stopping):**
- **✨ Summarize** — generates a meeting summary with key points and action items
- **✍ Refine** — cleans up the raw transcript for readability (works for both new and past meetings)

**Viewing past meetings:**
- Click any meeting in the left sidebar to open it
- The main panel shows the transcript and any saved summary
- You can generate or re-generate a summary for any past meeting
- Click **+ New Meeting** in the topbar to return to the recording view

Transcripts are saved as Markdown (`.md`) to:

```
~/Documents/Transcriptions/
```

---

## Audio Sources

### 🎤 Microphone (in-person meetings)
Works out of the box. Just select **Microphone** before recording.

### 🖥 System Audio (online meetings — Zoom, Meet, Teams)
Requires **BlackHole 2ch** — a free virtual audio driver.

**Setup steps:**

1. **Install BlackHole 2ch**
   Download from [existential.audio/blackhole](https://existential.audio/blackhole/)
   (use the 2ch version)

2. **Open Audio MIDI Setup**
   Press `⌘ Space` → type "Audio MIDI Setup" → open it

3. **Create a Multi-Output Device**
   - Click the `+` button in the bottom-left
   - Choose **Create Multi-Output Device**
   - Check: **MacBook Pro Speakers** (or your headphones)
   - Check: **BlackHole 2ch**
   - Name it: "Meeting Output"

4. **Set system output**
   Open **System Settings → Sound → Output**
   Select **Meeting Output**
   _(Your audio will still play through your speakers/headphones)_

5. **In Zoom / Meet / Teams**
   Set your speaker output to **Meeting Output** in the app's audio settings

6. **In the transcription app**
   Select **System Audio** → Start Recording

> ℹ️ Remember to switch your system audio output back after the meeting.

---

## AI Summaries

Install [Ollama](https://ollama.com), then pull a model:

```bash
# Fast and compact (recommended)
ollama pull llama3.2:3b

# Higher quality (requires ~5 GB)
ollama pull llama3.2:8b

# Most capable (requires ~8 GB)
ollama pull mistral:7b
```

The app launches Ollama automatically if it's not already running.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| App won't open | Run `SETUP_AND_BUILD.command` again to rebuild |
| "BlackHole not found" | Install BlackHole 2ch and restart the app |
| No audio from mic | Check System Settings → Privacy → Microphone |
| Whisper not loading | First recording triggers a ~800 MB model download; wait for it |
| Ollama error | Make sure Ollama is installed: `ollama.com` |
| Blank transcript | Check your mic isn't muted; try speaking closer to the mic |

---

## Architecture

```
main.py              — CustomTkinter GUI, single-window design, event loop
audio_capture.py     — sounddevice streams, WebRTC VAD, chunk emitter
transcriber.py       — mlx-whisper inference, background worker thread
summarizer.py        — Ollama streaming chat, extractive fallback
make_icon.py         — generates the .icns app icon at build time
```

All processing happens on-device. No audio or text ever leaves your Mac.
