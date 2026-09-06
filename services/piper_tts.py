"""
services/piper_tts.py — Piper TTS. Synchronous synthesis to int16 PCM; run via executor.
"""

from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config import (
    PIPER_HF_REPO,
    PIPER_MODELS_DIR,
    TTS_MODEL,
    TTS_SPEED,
)


def _get_piper_model_paths(model_name: str) -> tuple[Path, Path]:
    """Return (onnx_path, config_path), downloading from HuggingFace if needed."""
    models_dir = Path(PIPER_MODELS_DIR)
    models_dir.mkdir(exist_ok=True)

    onnx_path   = models_dir / f"{model_name}.onnx"
    config_path = models_dir / f"{model_name}.onnx.json"

    if not onnx_path.exists() or not config_path.exists():
        parts = model_name.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unsupported Piper model name: {model_name!r}")
        lang_region = parts[0]
        speaker     = parts[1]
        quality     = "-".join(parts[2:])
        lang        = lang_region.split("_")[0].lower()
        prefix      = f"{lang}/{lang_region}/{speaker}/{quality}"

        import shutil
        from huggingface_hub import hf_hub_download

        logger.info(f"[tts] Downloading Piper voice '{model_name}' from HuggingFace...")
        local_onnx = hf_hub_download(
            repo_id=PIPER_HF_REPO,
            filename=f"{prefix}/{model_name}.onnx",
            local_dir=str(models_dir),
        )
        local_json = hf_hub_download(
            repo_id=PIPER_HF_REPO,
            filename=f"{prefix}/{model_name}.onnx.json",
            local_dir=str(models_dir),
        )
        src_onnx = Path(local_onnx)
        src_json = Path(local_json)
        if src_onnx != onnx_path:
            shutil.copy2(src_onnx, onnx_path)
        if src_json != config_path:
            shutil.copy2(src_json, config_path)

    return onnx_path, config_path


class PiperTTSService:
    """Local Piper TTS. Thread-safe via a lock; callers run synthesise() in an executor."""

    def __init__(
        self,
        *,
        model_name: str = TTS_MODEL,
        speed: float = TTS_SPEED,
    ) -> None:
        self._model_name = model_name
        self._speed      = speed
        self._lock       = threading.Lock()
        self._voice      = None
        self._sample_rate: int = 22_050
        self._load_voice(model_name)

    def _load_voice(self, model_name: str) -> None:
        from piper.voice import PiperVoice
        logger.info(f"[tts] Loading Piper voice: {model_name}")
        onnx_path, config_path = _get_piper_model_paths(model_name)
        voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))
        self._sample_rate = voice.config.sample_rate
        with self._lock:
            self._voice      = voice
            self._model_name = model_name
        logger.info(f"[tts] Piper loaded: {model_name}, sr={self._sample_rate}")

    def switch_voice(self, model_name: str) -> None:
        """Hot-swap TTS voice model."""
        self._load_voice(model_name)
        logger.info(f"[tts] Voice switched to {model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def synthesise(self, text: str) -> Optional[np.ndarray]:
        """Synthesise text → int16 PCM ndarray. Blocking — run in executor."""
        if not text or not text.strip():
            return None

        from piper.config import SynthesisConfig
        syn_cfg = SynthesisConfig(length_scale=1.0 / max(self._speed, 0.1))

        with self._lock:
            voice = self._voice

        accum = bytearray()
        for chunk in voice.synthesize(text.strip(), syn_cfg):
            pcm = (np.clip(chunk.audio_float_array, -1.0, 1.0) * 32767).astype(np.int16)
            accum.extend(pcm.tobytes())

        if not accum:
            return None
        return np.frombuffer(bytes(accum), dtype=np.int16)

    def pcm_to_wav(self, pcm: np.ndarray) -> bytes:
        """Wrap int16 PCM in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
