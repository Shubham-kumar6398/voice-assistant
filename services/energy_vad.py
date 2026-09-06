"""
services/energy_vad.py — Energy-based VAD implementing pipecat's VADAnalyzer ABC.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

from config import (
    CHUNK_SIZE,
    ENERGY_SMOOTHING_ALPHA,
    ENERGY_SPEECH_SCALE,
    SAMPLE_RATE,
    VAD_CALIBRATION_CHUNKS,
    VAD_MIN_START,
    VAD_START_OFFSET,
)


class EnergyVADAnalyzer(VADAnalyzer):
    """VAD driven by smoothed mic energy against a calibrated noise floor.

    The parent class handles start/stop debouncing against a fixed 0.5 confidence,
    so voice_confidence() scales our adaptive threshold to land on that 0.5 mark.
    """

    def __init__(
        self,
        *,
        params: VADParams,
        sample_rate: Optional[int] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate or SAMPLE_RATE, params=params)
        self._smoothed_rms  = 0.0
        self._smoothed_prob = 0.0
        self.start_threshold = VAD_MIN_START
        self._cal_probs: list = []
        self._calibrated = False

    def num_frames_required(self) -> int:
        return CHUNK_SIZE

    def voice_confidence(self, buffer: bytes) -> float:
        """Smoothed energy as a 0-1 probability, normalised so >0.5 means speech."""
        audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self._smoothed_rms = (
            ENERGY_SMOOTHING_ALPHA * rms
            + (1.0 - ENERGY_SMOOTHING_ALPHA) * self._smoothed_rms
        )
        self._smoothed_prob = min(1.0, self._smoothed_rms / ENERGY_SPEECH_SCALE)

        if not self._calibrated:
            self._cal_probs.append(self._smoothed_prob)
            if len(self._cal_probs) >= VAD_CALIBRATION_CHUNKS:
                self._finalize_calibration()
        if self._calibrated and self.start_threshold > 0:
            return self._smoothed_prob / (self.start_threshold * 2.0)
        return self._smoothed_prob

    def _finalize_calibration(self) -> None:
        """Set the speech-start threshold from the measured ambient noise floor."""
        noise_floor = min(float(np.percentile(self._cal_probs, 60)), 0.50)
        self.start_threshold = max(VAD_MIN_START, noise_floor + VAD_START_OFFSET)
        self._calibrated = True
        self._cal_probs.clear()
        logger.info(
            f"[vad] calibrated — floor={noise_floor:.3f} start={self.start_threshold:.3f}"
        )
