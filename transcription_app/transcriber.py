"""
transcriber.py
──────────────
Wraps mlx-whisper for Apple Silicon-native transcription.
Runs in a background thread and processes AudioChunks from a queue,
emitting text segments via a callback.

Model: mlx-community/whisper-large-v3-turbo
  • ~800 MB on disk
  • Runs entirely on the M-series Neural Engine / GPU via Apple MLX
  • Real-time factor ~0.1× on M5 Pro (10× faster than real-time)
"""

import queue
import re
import threading
import time
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

try:
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Fall back to faster-whisper if mlx-whisper isn't installed
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


MODEL_ID = "mlx-community/whisper-large-v3-turbo"
FALLBACK_MODEL = "large-v3-turbo"

# Whisper hallucination filter phrases (common false positives on silence/noise)
_HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thanks for listening",
    "please subscribe", "you", ".", "..", "...", "bye", "bye bye",
    "[music]", "[applause]", "[laughter]", "(music)", "(applause)",
}


def _clean_transcript(text: str) -> str:
    """
    Filter out Whisper hallucinations and repeated-word artifacts.

    Whisper sometimes outputs looping repetitions ("Hi! Hi! Hi!...") on
    short or noisy audio, or emits common filler phrases on silence.
    This function catches both cases.
    """
    if not text:
        return ""

    text = text.strip()

    # Drop known hallucination phrases (case-insensitive, exact match)
    if text.lower().rstrip(".!?, ") in _HALLUCINATION_PHRASES:
        return ""

    # Detect repeated token loops: split on word boundaries and check if
    # the same short token repeats more than 4 times consecutively.
    # e.g. "Hi! Hi! Hi! Hi! Hi!" → drop
    tokens = re.split(r"[\s,!?.]+", text)
    tokens = [t for t in tokens if t]
    if len(tokens) >= 5:
        # Find longest run of the same token
        max_run = 1
        cur_run = 1
        for i in range(1, len(tokens)):
            if tokens[i].lower() == tokens[i - 1].lower():
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        # If more than half the tokens are the same repeated word, it's a loop
        if max_run > 4 and max_run / len(tokens) > 0.5:
            return ""

    return text


@dataclass
class TranscriptSegment:
    text: str
    source: str        # "microphone" or "system"
    timestamp: float   # wall-clock time of the audio chunk
    duration_ms: int   # approximate duration of audio transcribed
    speaker: str = ""  # assigned by diarizer, e.g. "Speaker 1"


class Transcriber:
    """
    Consumes AudioChunks from a queue, transcribes them, and fires
    `on_segment` with each TranscriptSegment.

    Thread-safe: start(), stop(), and the queue are safe to call from
    any thread.
    """

    def __init__(
        self,
        on_segment: Callable[[TranscriptSegment], None],
        on_status: Optional[Callable[[str], None]] = None,
        language: Optional[str] = None,  # None = auto-detect
    ):
        self.on_segment = on_segment
        self.on_status = on_status or (lambda s: None)
        self.language = language

        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._model = None
        self._backend: str = "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def load_model(self):
        """
        Load the Whisper model. Call this before start() — it may take a
        few seconds on first run (model download ~800 MB).
        """
        if MLX_AVAILABLE:
            self.on_status("Loading Whisper (mlx-whisper)…")
            try:
                # Pre-load by running a silent inference
                dummy = np.zeros(16000, dtype=np.float32)
                mlx_whisper.transcribe(dummy, path_or_hf_repo=MODEL_ID)
                self._backend = "mlx"
                self.on_status("Whisper ready (MLX backend)")
                return
            except Exception as e:
                self.on_status(f"MLX load failed ({e}), trying faster-whisper…")

        if FASTER_WHISPER_AVAILABLE:
            self.on_status("Loading Whisper (faster-whisper, CPU)…")
            self._model = WhisperModel(
                FALLBACK_MODEL,
                device="cpu",
                compute_type="int8",
            )
            self._backend = "faster-whisper"
            self.on_status("Whisper ready (faster-whisper backend)")
            return

        raise RuntimeError(
            "Neither mlx-whisper nor faster-whisper is installed.\n"
            "Run: pip install mlx-whisper"
        )

    def start(self):
        if self._running:
            return
        if self._backend == "none":
            self.load_model()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Unblock the worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, chunk):
        """Add an AudioChunk to the transcription queue (non-blocking)."""
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest item to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except Exception:
                pass

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                chunk = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if chunk is None:
                break

            try:
                t0 = time.time()
                text = self._transcribe(chunk.audio)
                elapsed_ms = int((time.time() - t0) * 1000)

                text = _clean_transcript(text)
                if text:
                    seg = TranscriptSegment(
                        text=text,
                        source=chunk.source.value,
                        timestamp=chunk.timestamp,
                        duration_ms=elapsed_ms,
                    )
                    self.on_segment(seg)

            except Exception as e:
                self.on_status(f"Transcription error: {e}")

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run inference on float32 mono 16 kHz audio, return transcript text."""
        if self._backend == "mlx":
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=MODEL_ID,
                language=self.language,
                # fp16 runs natively on MLX — no explicit flag needed
            )
            return result.get("text", "")

        elif self._backend == "faster-whisper":
            segments, _ = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(s.text for s in segments)

        return ""
