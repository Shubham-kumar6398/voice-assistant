"""
processors/session_recorder.py — Records one WAV per session, as alternating turns.

User mic audio is buffered per turn (committed at the turn boundary) and agent TTS
audio appended after it, with a 300 ms gap on each speaker change.
"""

from __future__ import annotations

import io
import os
import threading
import wave
from typing import Optional

import numpy as np
from loguru import logger

from config import OUT_SAMPLE_RATE, RECORDINGS_DIR, SAMPLE_RATE

_SILENCE_300MS = np.zeros(int(0.30 * OUT_SAMPLE_RATE), dtype=np.int16).tobytes()


def _resample_i16(pcm_f32: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Linear-interpolation resample float32 → int16."""
    if in_rate == out_rate:
        return (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)
    n_out = int(len(pcm_f32) * out_rate / in_rate)
    if n_out == 0:
        return np.array([], dtype=np.int16)
    result = np.interp(
        np.linspace(0, 1, n_out), np.linspace(0, 1, len(pcm_f32)), pcm_f32
    )
    return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16)


class SessionRecorder:
    """Thread-safe recorder writing a single WAV for one session."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._lock = threading.Lock()
        self._pcm_buffer = bytearray()      # committed turns, int16 @ OUT_SAMPLE_RATE
        self._pending_user = bytearray()    # current user turn, not yet committed
        self._last_writer: Optional[str] = None
        self._finalised = False
        self._filename = f"{session_id}_session.wav"
        self._filepath = os.path.join(RECORDINGS_DIR, self._filename)
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def url(self) -> str:
        return f"/recordings/{self._filename}"

    @property
    def finalised(self) -> bool:
        return self._finalised

    @property
    def has_audio(self) -> bool:
        with self._lock:
            return bool(self._pcm_buffer or self._pending_user)

    # ── Capture ───────────────────────────────────────────────────────────────

    def _write(self, pcm_bytes: bytes, writer: str) -> None:
        """Append PCM, inserting a gap when the speaker changes."""
        if not pcm_bytes:
            return
        with self._lock:
            if self._finalised:
                return
            if self._last_writer is not None and self._last_writer != writer:
                self._pcm_buffer.extend(_SILENCE_300MS)
            self._pcm_buffer.extend(pcm_bytes)
            self._last_writer = writer

    def append_user_audio(self, audio_f32: np.ndarray) -> None:
        """Buffer mic audio (float32 @ SAMPLE_RATE) for the current turn."""
        if self._finalised:
            return
        resampled = _resample_i16(audio_f32, SAMPLE_RATE, OUT_SAMPLE_RATE)
        with self._lock:
            self._pending_user.extend(resampled.tobytes())

    def commit_user_turn(self) -> None:
        """Flush the pending user audio into the session as one block."""
        with self._lock:
            if self._finalised or not self._pending_user:
                return
            pending = bytes(self._pending_user)
            self._pending_user.clear()
        self._write(pending, "user")

    def append_agent_pcm(self, pcm_int16: np.ndarray) -> None:
        """Append agent audio (int16 @ OUT_SAMPLE_RATE) that was actually played."""
        if self._finalised:
            return
        self._write(pcm_int16.tobytes(), "agent")

    # ── Finalisation ──────────────────────────────────────────────────────────

    def finalize(self) -> Optional[str]:
        """Write the session WAV and return its URL, or None if there was no audio."""
        self.commit_user_turn()   # capture a trailing turn that got no reply
        with self._lock:
            if self._finalised:
                return self.url
            if not self._pcm_buffer:
                return None
            pcm = bytes(self._pcm_buffer)
            self._finalised = True

        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(OUT_SAMPLE_RATE)
                wf.writeframes(pcm)
            with open(self._filepath, "wb") as fh:
                fh.write(buf.getvalue())
            size_mb = os.path.getsize(self._filepath) / (1024 ** 2)
            logger.info(f"[recorder] Session saved → {self._filepath} ({size_mb:.2f} MB)")
            return self.url
        except Exception as exc:
            logger.error(f"[recorder] Failed to save session: {exc}")
            return None
