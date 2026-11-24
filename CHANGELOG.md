# Changelog

## 2026-11-23

### Summaries
- Default model now chosen by unified memory: `gemma4:e4b` (8 GB), `gemma4:12b` (16–24 GB), `qwen3.8:27b` (32 GB+). Previous `qwen2.5` defaults are still used if they're the only ones installed. Override with `TRANSCRIBE_MODEL`.
- Context window now sized to the transcript. Previously Ollama's default context silently truncated long meetings.
- Thinking mode disabled for summaries (Qwen3.x / Gemma 4 otherwise reason for minutes before writing).
- Setup script recommends a model for the Mac's RAM and offers to download it.

### Speaker diarization
- Fixed: was broken on fresh installs (pyannote 4 removed the `use_auth_token` argument) and pyannote wasn't installed by older setups.
- Now uses `pyannote/speaker-diarization-community-1` with pyannote.audio 4.x. Accept the model's terms on Hugging Face before first use.
- Audio is passed in memory (no ffmpeg dependency); the non-overlapping speaker track is used to label transcript lines.

### Install / sharing
- Install via `git clone`; setup script checks for Apple Silicon, Homebrew and Xcode Command Line Tools.
- The machine-specific `LocalTranscription.app` is no longer committed; each Mac builds its own.
- `numpy` pinned below 2.5 (required by numba, used by mlx-whisper).

### Verified (MacBook Pro, 48 GB)
- Diarization: 2-voice, 4-turn test clip → 2 speakers, all turns labelled correctly, boundaries within ~0.2 s.
- Summary: `qwen3.8:27b` auto-selected, short summary returned in ~8 s.
- Not yet verified: full end-to-end recording with the speaker-naming dialog.
