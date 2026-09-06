"""
services/whisper_stt.py — whisper.cpp STT as a pipecat SegmentedSTTService.

The parent buffers audio between the VAD start/stop frames; run_stt() transcribes
that segment on a thread executor.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import wave
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.settings import STTSettings
from pipecat.utils.tracing.service_decorators import traced_stt

from config import SAMPLE_RATE, STT_N_THREADS, WHISPER_MODEL, MIN_AUDIO_SECS


class WhisperCppSTTService(SegmentedSTTService):
    """Whisper.cpp STT via pywhispercpp, yielding TranscriptionFrames."""

    def __init__(
        self,
        *,
        model_name: str = WHISPER_MODEL,
        n_threads: int = STT_N_THREADS,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate or SAMPLE_RATE,
            settings=STTSettings(model=model_name, language="en"),
            **kwargs,
        )
        self._model_name = model_name
        self._n_threads  = n_threads
        self._model      = None
        self._infer_lock = threading.Lock()
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        logger.info(f"[stt] Loading whisper.cpp: {self._model_name}")
        from pywhispercpp.model import Model

        self._model = Model(
            self._model_name,
            redirect_whispercpp_logs_to=None,
            n_threads=self._n_threads,
            language="en",
            no_context=True,
            suppress_blank=True,
            print_realtime=False,
            print_progress=False,
            print_special=False,
            print_timestamps=False,
            single_segment=False,
            no_speech_thold=0.6,
            temperature=0.0,
        )
        # Greedy decoding for faster inference
        try:
            self._model._params.beam_size = 1
            self._model._params.best_of   = 1
            logger.debug("[stt] greedy decoding enabled (beam_size=1, best_of=1)")
        except AttributeError:
            logger.debug("[stt] greedy decoding not settable on this build")

        logger.info("[stt] whisper.cpp ready")
        self._warm_up()

    def _warm_up(self) -> None:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            self._model.transcribe(silence)
            logger.debug("[stt] warm-up done")
        except Exception as exc:
            logger.warning(f"[stt] warm-up warning: {exc}")

    # ── SegmentedSTTService interface ─────────────────────────────────────────

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one complete WAV segment from the parent class."""
        if not audio:
            return

        try:
            with wave.open(io.BytesIO(audio)) as wf:
                pcm = wf.readframes(wf.getnframes())
                sr  = wf.getframerate()
        except Exception as exc:
            logger.error(f"[stt] failed to decode WAV: {exc}")
            yield ErrorFrame(error=str(exc))
            return

        audio_i16 = np.frombuffer(pcm, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        duration = len(audio_f32) / sr
        if duration < MIN_AUDIO_SECS:
            logger.debug(f"[stt] skipping short utterance ({duration:.2f}s)")
            return

        t0 = time.monotonic()
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe, audio_f32
            )
        except Exception as exc:
            logger.error(f"[stt] transcription error: {exc}")
            yield ErrorFrame(error=str(exc))
            return

        elapsed = time.monotonic() - t0
        text    = (text or "").strip()

        logger.info(f"[stt] {elapsed:.2f}s (audio {duration:.2f}s): '{text[:80]}'")

        if text:
            yield TranscriptionFrame(
                text=text,
                user_id="user",
                timestamp=str(time.time()),
                language=None,
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _transcribe(self, audio_f32: np.ndarray) -> str:
        """Run whisper on an executor thread, padded to 1 s and capped at 30 s."""
        min_samples, max_samples = SAMPLE_RATE, SAMPLE_RATE * 30
        if len(audio_f32) < min_samples:
            audio_f32 = np.concatenate(
                [audio_f32, np.zeros(min_samples - len(audio_f32), dtype=np.float32)]
            )
        audio_f32 = np.ascontiguousarray(audio_f32[-max_samples:])

        with self._infer_lock:
            segments = self._model.transcribe(audio_f32)

        return " ".join(s.text for s in segments).strip()

    def can_generate_metrics(self) -> bool:
        return True