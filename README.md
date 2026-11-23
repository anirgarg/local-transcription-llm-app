# 🎙 Local Meeting Transcription

**100% on-device** · **Apple Silicon native** · **No cloud, no subscriptions**

Transcribes online and in-person meetings in real time using
[mlx-whisper](https://github.com/ml-explore/mlx-examples) (Apple MLX framework)
and generates AI summaries via [Ollama](https://ollama.com).

---

## Requirements

- **Apple Silicon Mac (M1 or later), macOS 13 Ventura+.** Intel Macs, Windows and Linux are not supported (the app uses Apple's MLX framework).
- **Homebrew** — [brew.sh](https://brew.sh)
- **Xcode Command Line Tools** — `xcode-select --install` (the setup script prompts if missing)
- **Ollama**, recent version (for AI summaries; Gemma 4 / Qwen3.8 need a 2026 release) — [ollama.com](https://ollama.com)
- **BlackHole 2ch** (optional, for online meetings) — [existential.audio/blackhole](https://existential.audio/blackhole/)
- **Hugging Face token** (optional, for speaker labels) — see [Speaker Diarization](#speaker-diarization)
- **Disk:** ~1.5 GB for the app environment + Whisper, plus 5–20 GB for the summary model

---

## Install

```bash
git clone https://github.com/anirgarg/local-transcription-llm-app.git
cd local-transcription-llm-app
./SETUP_AND_BUILD.command
```

(Or double-click `SETUP_AND_BUILD.command` in Finder. If you downloaded a ZIP instead of cloning, run `chmod +x SETUP_AND_BUILD.command` first.)

The script:
1. Checks for Apple Silicon, Homebrew and Xcode Command Line Tools
2. Installs Python 3.12 via Homebrew (if needed)
3. Creates a fresh `.venv` with all required packages and verifies imports
4. Compiles the native `LocalTranscription.app` launcher for **your** machine's paths
5. Generates the app icon
6. Recommends a summary model for your RAM and offers to download it

When it finishes, double-click **`LocalTranscription.app`** to launch. You can drag it to the Dock; keep the project folder where it is, since the launcher points to it. If you move the folder, re-run the setup script.

> **First launch:** Whisper `large-v3-turbo` (~800 MB) downloads automatically the first time you start a recording.

**Updating:** `git pull`, then re-run `SETUP_AND_BUILD.command`.

---

## AI Summaries — choosing a model

The app picks a summary model automatically based on your Mac's memory:

| Unified memory | Default model | Download size |
|---|---|---|
| 8 GB | `gemma4:e4b` | 6.6 GB |
| 16–24 GB | `gemma4:12b` | 8.0 GB |
| 32 GB+ | `qwen3.8:27b` | 17.7 GB |

Picks current as of November 2026: Qwen3.8-27B is the strongest dense model that fits a 32 GB Mac, and Gemma 4 12B is the usual recommendation for summarization on smaller machines. Thinking mode is turned off for summaries (much faster, no quality loss for this task), and the context window is sized to the transcript so long meetings aren't cut off.

If the recommended model isn't installed, the app uses the best one it finds from this list (or the older `qwen2.5` defaults). Install a model with, e.g.:

```bash
ollama pull gemma4:12b
```

Prefer a different model? `gemma4:31b` (20.4 GB) is a good alternative on 48 GB+ Macs.

To force a specific model (any Ollama model works), set `TRANSCRIBE_MODEL` before launching:

```bash
launchctl setenv TRANSCRIBE_MODEL llama3.1:8b
```

The app launches Ollama automatically if it isn't already running.

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

## Speaker Diarization

Optional: labels who said what, using [pyannote](https://github.com/pyannote/pyannote-audio).

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Paste it into the app's settings panel (**HuggingFace Token**)

The token is stored locally at `~/.config/transcription_app/hf_token`, never in the project folder.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| App won't open | Run `SETUP_AND_BUILD.command` again to rebuild |
| "BlackHole not found" | Install BlackHole 2ch and restart the app |
| No audio from mic | Check System Settings → Privacy → Microphone |
| Whisper not loading | First recording triggers a ~800 MB model download; wait for it |
| Ollama error | Install Ollama (`ollama.com`) and pull the model the app names in the error |
| Summaries very slow / Mac freezes | Model too large for your RAM — use a smaller one (see AI Summaries) |
| Setup fails at "Building" | Install Xcode Command Line Tools: `xcode-select --install` |
| Blank transcript | Check your mic isn't muted; try speaking closer to the mic |

---

## Architecture

Source lives in `transcription_app/`:

```
main.py              — CustomTkinter GUI, single-window design, event loop
audio_capture.py     — sounddevice streams, voice-activity detection, chunk emitter
transcriber.py       — mlx-whisper inference, background worker thread
summarizer.py        — Ollama streaming chat, RAM-based model selection
diarizer.py          — pyannote speaker diarization (optional)
calendar_sync.py     — reads today's events from macOS Calendar (AppleScript)
make_icon.py         — generates the .icns app icon at build time
```

All processing happens on-device. No audio or text ever leaves your Mac.
