"""
audio_capture.py
────────────────
Handles real-time audio capture from:
  • Microphone (in-person meetings)
  • System audio via BlackHole (online meetings — Zoom, Meet, Teams, etc.)
  • Both simultaneously (mixed)

Uses energy-based VAD (pure Python, no compilation required) to detect
speech and emit chunks only when someone is talking.
"""

import threading
import time
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Callable, List
from enum import Enum


# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper expects 16 kHz
CHANNELS = 1                  # Mono
FRAME_DURATION_MS = 30        # Analysis frame size in ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
CHUNK_SECONDS = 8             # Max seconds of audio per Whisper call
SILENCE_TIMEOUT = 1.5         # Seconds of silence before flushing chunk

# Energy VAD thresholds (RMS of float32 audio, range 0.0–1.0)
# Tune lower if mic is quiet; raise if picking up too much background noise
SPEECH_RMS_THRESHOLD = 0.003   # above this → speech
SILENCE_RMS_THRESHOLD = 0.0015 # below this → silence


class AudioSource(Enum):
    MICROPHONE = "microphone"
    SYSTEM = "system"        # BlackHole
    BOTH = "both"


@dataclass
class AudioChunk:
    """A chunk of speech-containing audio ready for transcription."""
    audio: np.ndarray          # float32, shape (N,), sample rate = 16000
    source: AudioSource
    timestamp: float           # time.time() when chunk was created


# ── Device helpers ────────────────────────────────────────────────────────────

def list_audio_devices() -> dict:
    """Return available input devices, highlighting BlackHole if present."""
    devices = sd.query_devices()
    result = {"microphones": [], "blackhole": None}
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"]
            info = {"index": i, "name": name, "samplerate": int(d["default_samplerate"])}
            if "BlackHole" in name or "blackhole" in name.lower():
                result["blackhole"] = info
            else:
                result["microphones"].append(info)
    return result


def get_default_mic_index() -> Optional[int]:
    """Return the index of the system default input device."""
    try:
        return sd.default.device[0]
    except Exception:
        return None


def get_blackhole_index() -> Optional[int]:
    """Return the device index of BlackHole 2ch, or None if not installed."""
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if "BlackHole" in d["name"] and d["max_input_channels"] > 0:
                return i
    except Exception:
        pass
    return None


# ── Pure-Python energy VAD ────────────────────────────────────────────────────

def _is_speech(frame: np.ndarray, threshold: float = SPEECH_RMS_THRESHOLD) -> bool:
    """Return True if the RMS energy of the frame exceeds the speech threshold."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return rms >= threshold


# ── Main capturer class ───────────────────────────────────────────────────────

class AudioCapturer:
    """
    Captures audio from one or two sources, applies energy-based VAD, and
    emits speech chunks via a callback.

    Usage:
        capturer = AudioCapturer(source=AudioSource.MICROPHONE,
                                 on_chunk=my_callback)
        capturer.start()
        ...
        capturer.stop()

    The `on_chunk` callback receives an AudioChunk and is called from a
    background thread — keep it non-blocking (put work on a queue).
    """

    def __init__(
        self,
        source: AudioSource,
        on_chunk: Callable[[AudioChunk], None],
        mic_index: Optional[int] = None,
        system_index: Optional[int] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ):
        self.source = source
        self.on_chunk = on_chunk
        self.on_level = on_level
        self.mic_index = mic_index
        self.system_index = system_index if system_index is not None else get_blackhole_index()

        self._running = False
        self._paused  = False
        self._threads: List[threading.Thread] = []
        self._streams: List[sd.InputStream] = []

        # Per-source frame buffers (list of np.ndarray, each FRAME_SIZE samples)
        self._mic_frames: List[np.ndarray] = []
        self._sys_frames: List[np.ndarray] = []
        self._mic_lock = threading.Lock()
        self._sys_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True

        if self.source in (AudioSource.MICROPHONE, AudioSource.BOTH):
            self._open_stream(
                device=self.mic_index,
                frames_list=self._mic_frames,
                lock=self._mic_lock,
                source_tag=AudioSource.MICROPHONE,
            )

        if self.source in (AudioSource.SYSTEM, AudioSource.BOTH):
            if self.system_index is None:
                raise RuntimeError(
                    "BlackHole not found. Install BlackHole 2ch from "
                    "https://existential.audio/blackhole/"
                )
            self._open_stream(
                device=self.system_index,
                frames_list=self._sys_frames,
                lock=self._sys_lock,
                source_tag=AudioSource.SYSTEM,
            )

        # Chunk emitter runs in its own thread
        t = threading.Thread(target=self._emitter_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def pause(self):
        """Pause capture — streams stay open but frames are discarded."""
        self._paused = True

    def resume(self):
        """Resume capture after a pause."""
        self._paused = False

    def stop(self):
        self._running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams.clear()
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        # Flush remaining audio
        self._flush(self._mic_frames, self._mic_lock, AudioSource.MICROPHONE)
        self._flush(self._sys_frames, self._sys_lock, AudioSource.SYSTEM)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_stream(self, device, frames_list, lock, source_tag):
        """Open a sounddevice InputStream that feeds frames_list."""
        def callback(indata, frames, time_info, status):
            if not self._running or self._paused:
                return
            mono = indata[:, 0].copy()
            if self.on_level and source_tag == AudioSource.MICROPHONE:
                rms = float(np.sqrt(np.mean(mono ** 2)))
                self.on_level(rms)
            with lock:
                frames_list.append(mono)

        try:
            stream = sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=FRAME_SIZE,
                callback=callback,
            )
            stream.start()
            self._streams.append(stream)
        except Exception as e:
            raise RuntimeError(
                f"Could not open audio device ({source_tag.value}): {e}"
            )

    def _emitter_loop(self):
        """
        Background thread: drains frame buffers, applies energy VAD, and
        emits AudioChunks when enough speech is collected or silence occurs.
        """
        silence_limit = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION_MS)
        chunk_limit   = int(CHUNK_SECONDS   * 1000 / FRAME_DURATION_MS)

        # Per-source accumulation state
        state = {
            s: {"buf": [], "silence": 0, "has_speech": False}
            for s in (AudioSource.MICROPHONE, AudioSource.SYSTEM)
        }

        pairs = [
            (AudioSource.MICROPHONE, self._mic_frames, self._mic_lock),
            (AudioSource.SYSTEM,     self._sys_frames, self._sys_lock),
        ]

        while self._running:
            time.sleep(FRAME_DURATION_MS / 1000)

            for src, raw_buf, lock in pairs:
                # Only process sources we're actually capturing
                if self.source not in (src, AudioSource.BOTH):
                    continue

                with lock:
                    incoming = raw_buf.copy()
                    raw_buf.clear()

                if not incoming:
                    continue

                st = state[src]
                for frame in incoming:
                    st["buf"].append(frame)
                    if _is_speech(frame):
                        st["has_speech"] = True
                        st["silence"] = 0
                    else:
                        st["silence"] += 1

                # Decide whether to emit
                at_chunk_limit    = len(st["buf"]) >= chunk_limit
                silence_boundary  = st["silence"] >= silence_limit and st["has_speech"]

                if (at_chunk_limit or silence_boundary) and st["has_speech"]:
                    audio = np.concatenate(st["buf"])
                    try:
                        self.on_chunk(AudioChunk(
                            audio=audio, source=src, timestamp=time.time()))
                    except Exception:
                        pass
                    st["buf"] = []
                    st["silence"] = 0
                    st["has_speech"] = False

                elif len(st["buf"]) > chunk_limit * 3:
                    # Safety valve: drop if no speech detected for a long time
                    st["buf"] = []
                    st["has_speech"] = False
                    st["silence"] = 0

    def _flush(self, frames_list, lock, source):
        """Emit any remaining buffered audio when stopping."""
        with lock:
            if frames_list:
                audio = np.concatenate(frames_list)
                frames_list.clear()
                if len(audio) > SAMPLE_RATE * 0.5:  # skip clips < 0.5 s
                    try:
                        self.on_chunk(AudioChunk(
                            audio=audio, source=source, timestamp=time.time()))
                    except Exception:
                        pass
