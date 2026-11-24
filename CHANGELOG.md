# Changelog

## 2026-11-23

### Transcription
- Whisper now gets a clinical vocabulary prompt (English/auto-detect), editable at `~/.config/transcription_app/vocabulary.txt`. Output that just repeats the prompt (a Whisper failure mode on silence) is dropped.
- Evaluated NVIDIA Parakeet (TDT 0.6B v2 and v3, via parakeet-mlx) as a replacement and decided against it for now. Test: 6 synthetic clinical sentences (5 English accents via macOS TTS), clean and with added noise, 17 key clinical terms:

  | Engine | WER clean / noisy | Key terms | Speed (M-series, 48 GB) |
  |---|---|---|---|
  | Whisper large-v3-turbo | 9.3% / 10.3% | 10–11/17 | ~29x realtime |
  | **Whisper + vocabulary (shipped)** | **3.1% / 9.3%** | **15/17** | ~29x realtime |
  | Parakeet v3 | 9.3% / 9.3% | 8/17 | 50–80x realtime |
  | Parakeet v2 (English) | 10.3% / 10.3% | 9–10/17 | ~87x realtime |

  Parakeet is faster, but Whisper already runs ~29x faster than real time here, so speed isn't the bottleneck; drug-name accuracy is. On room-noise silence Parakeet v3 produced a Spanish phrase; Parakeet v2 output nothing. Caveat: TTS voices, not real clinic audio.

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
