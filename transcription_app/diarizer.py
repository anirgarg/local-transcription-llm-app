"""
diarizer.py
───────────
Speaker diarization using pyannote.audio (optional dependency).
Runs as a post-processing step after recording stops, producing
(start_sec, end_sec, speaker_label) segments that are mapped back
onto the Whisper transcript.

Setup (one-time):
  1. pip install pyannote.audio
  2. Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1
     and https://hf.co/pyannote/segmentation-3.0
  3. Create a free HuggingFace token at https://hf.co/settings/tokens
  4. Paste the token into the app's Settings → HuggingFace Token field
"""

import threading
import wave
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

_PYANNOTE_AVAILABLE: bool | None = None  # None = not yet checked

def _check_pyannote() -> bool:
    """Lazy-check for pyannote — defers the heavy torch/torchaudio import
    until diarization is actually needed, preventing mic interference at startup."""
    global _PYANNOTE_AVAILABLE
    if _PYANNOTE_AVAILABLE is None:
        try:
            import pyannote.audio  # noqa: F401
            _PYANNOTE_AVAILABLE = True
        except ImportError:
            _PYANNOTE_AVAILABLE = False
    return _PYANNOTE_AVAILABLE

HF_TOKEN_FILE = Path.home() / ".config" / "transcription_app" / "hf_token"
SAMPLE_RATE   = 16000

# (start_seconds, end_seconds, "SPEAKER_00")
DiarizationSegment = Tuple[float, float, str]


class Diarizer:
    """
    Wraps pyannote speaker-diarization-3.1.

    Usage:
        d = Diarizer()
        wav = d.build_wav(audio_chunks, session_start)
        d.diarize(wav, on_done=my_cb, on_error=err_cb)
    """

    def __init__(self):
        self._pipeline = None
        self._lock     = threading.Lock()

    # ── HF token helpers ──────────────────────────────────────────────────────

    @staticmethod
    def get_hf_token() -> Optional[str]:
        try:
            tok = HF_TOKEN_FILE.read_text().strip()
            return tok or None
        except Exception:
            return None

    @staticmethod
    def save_hf_token(token: str):
        HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        HF_TOKEN_FILE.write_text(token.strip())

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(self.get_hf_token())  # pyannote checked lazily on first load

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self, on_status: Optional[Callable[[str], None]] = None):
        """Load the pipeline (blocking — run in a thread)."""
        with self._lock:
            if self._pipeline:
                return
            if not _check_pyannote():
                raise RuntimeError(
                    "pyannote.audio is not installed. Run: pip install pyannote.audio"
                )
            from pyannote.audio import Pipeline as _Pipeline  # lazy import
            token = self.get_hf_token()
            if not token:
                raise RuntimeError("No HuggingFace token configured.")
            if on_status:
                on_status("Loading speaker diarization model…")
            self._pipeline = _Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            # Use Apple Silicon GPU if torch + MPS is available
            try:
                import torch
                if torch.backends.mps.is_available():
                    self._pipeline = self._pipeline.to(torch.device("mps"))
            except Exception:
                pass

    def preload(self, on_status: Optional[Callable[[str], None]] = None):
        """Load the pipeline in the background (non-blocking)."""
        if not self.is_available() or self._pipeline:
            return
        threading.Thread(target=self.load, kwargs={"on_status": on_status},
                         daemon=True).start()

    # ── Audio building ────────────────────────────────────────────────────────

    @staticmethod
    def build_wav(
        chunks: list,          # [{'audio': np.ndarray, 'timestamp': float}, …]
        session_start: float,  # time.time() when recording started
    ) -> Optional[str]:
        """
        Reconstruct a properly-timed WAV from accumulated audio chunks,
        inserting silence for gaps (pauses, silence-filtered regions).
        Returns path to a temp WAV file, or None if no audio.
        """
        if not chunks:
            return None

        last         = chunks[-1]
        total_dur    = (last['timestamp'] - session_start) + len(last['audio']) / SAMPLE_RATE
        total_samples = max(SAMPLE_RATE, int(total_dur * SAMPLE_RATE))

        full = np.zeros(total_samples, dtype=np.float32)
        for c in chunks:
            offset = max(0.0, c['timestamp'] - session_start)
            s = int(offset * SAMPLE_RATE)
            audio = c['audio']
            e = min(s + len(audio), total_samples)
            full[s:e] = audio[:e - s]

        audio_i16 = (full * 32767).clip(-32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                          prefix='transcription_diar_')
        with wave.open(tmp.name, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_i16.tobytes())
        return tmp.name

    # ── Diarization ───────────────────────────────────────────────────────────

    def diarize(
        self,
        wav_path:  str,
        on_done:   Callable[[List[DiarizationSegment]], None],
        on_error:  Callable[[str], None],
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """Run diarization in a background thread."""
        threading.Thread(
            target=self._run,
            args=(wav_path, on_done, on_error, on_status),
            daemon=True,
        ).start()

    def _run(self, wav_path, on_done, on_error, on_status):
        try:
            if not self._pipeline:
                self.load(on_status=on_status)
            if on_status:
                on_status("Identifying speakers…")
            diarization = self._pipeline(wav_path)
            segments = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            on_done(segments)
        except Exception as e:
            on_error(str(e))
        finally:
            # Clean up temp WAV
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── Segment mapping ───────────────────────────────────────────────────────

    @staticmethod
    def assign_speakers(
        transcript_segments: list,   # list of TranscriptSegment
        diar_segments: List[DiarizationSegment],
        session_start: float,
    ) -> dict:
        """
        Returns a dict {segment_index: speaker_label} mapping each
        transcript segment to the speaker who was dominant during it.
        """
        mapping = {}
        for i, seg in enumerate(transcript_segments):
            offset = seg.timestamp - session_start
            best_speaker  = None
            best_overlap  = 0.0
            seg_dur       = max(0.5, getattr(seg, 'duration_ms', 500) / 1000)
            seg_end       = offset + seg_dur

            for (d_start, d_end, speaker) in diar_segments:
                overlap = max(0.0, min(seg_end, d_end) - max(offset, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            if best_speaker:
                mapping[i] = best_speaker
        return mapping
