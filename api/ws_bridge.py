"""
api/ws_bridge.py — JSON protocol bridge between the browser and the pipeline.

Routes inbound audio and control messages in, and text/audio/metrics/state events out.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from loguru import logger

from config import AGENT_SYSTEM_PROMPT, LLM_MAX_TOKENS, LLM_TEMPERATURE, WHISPER_MODEL

if TYPE_CHECKING:
    from processors.voice_orchestrator import VoiceOrchestrator
    from processors.session_recorder import SessionRecorder
    from services.mlx_llm import MLXLLMService
    from services.piper_tts import PiperTTSService


class WebSocketBridge:
    """Bidirectional JSON <-> pipeline bridge for one WebSocket client."""

    def __init__(
        self,
        ws,
        pipeline_task,
        audio_input,         # BrowserAudioInputProcessor
        orchestrator: "VoiceOrchestrator",
        llm: "MLXLLMService",
        tts: "PiperTTSService",
        recorder: "SessionRecorder",
        session_id: str,
        vad=None,
    ) -> None:
        self._ws            = ws
        self._task          = pipeline_task
        self._audio_input   = audio_input
        self._orchestrator  = orchestrator
        self._llm           = llm
        self._tts           = tts
        self._recorder      = recorder
        self._session_id    = session_id
        self._vad           = vad
        self._listening     = False
        self._audio_log_count = 0
        self._vad_task: Optional[asyncio.Task] = None

    @property
    def session_id(self) -> str:
        """The live session id — it changes on 'start' and 'new_session'."""
        return self._session_id

    # ── Outbound helpers ──────────────────────────────────────────────────────

    async def send_event(self, event: str, payload: Dict[str, Any] = {}) -> None:
        try:
            await self._ws.send_text(json.dumps({"event": event, **payload}))
        except Exception as exc:
            logger.debug(f"[bridge] send_event failed: {exc}")

    # ── VAD real-time loop ────────────────────────────────────────────────────

    def _vad_prob(self) -> float:
        return getattr(self._vad, "_smoothed_prob", 0.0) if self._vad else 0.0

    async def _vad_update_loop(self) -> None:
        """Push VAD probability to the browser every 100 ms while listening."""
        try:
            while self._listening:
                await self.send_event("vad", {"prob": round(self._vad_prob(), 3)})
                await asyncio.sleep(0.10)
        except (asyncio.CancelledError, Exception):
            pass

    async def _stop_vad_task(self) -> None:
        if self._vad_task and not self._vad_task.done():
            self._vad_task.cancel()
            try:
                await self._vad_task
            except (asyncio.CancelledError, Exception):
                pass
        self._vad_task = None

    async def cleanup(self) -> None:
        """Cancel background tasks when the connection closes."""
        await self._stop_vad_task()

    async def send_state(self) -> None:
        rec = self._recorder
        await self.send_event("state", {
            "listening":  self._listening,
            "vad":        round(self._vad_prob(), 3),
            "responding": False,
            "session_id": self._session_id,
            "recording_url": rec.url if rec and not rec.finalised else None,
            "config": {
                "model":         self._llm.model_name,
                "stt_model":     f"whisper.cpp/{WHISPER_MODEL}",
                "tts_model":     self._tts.model_name,
                "system_prompt": self._llm.system_prompt,
                "temperature":   self._llm.temperature,
                "max_tokens":    self._llm.max_tokens,
            },
        })

    async def send_audio(self, wav_bytes: bytes, index: int) -> None:
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        await self.send_event("media", {
            "mime":  "audio/wav",
            "audio": encoded,
            "index": index,
        })

    async def send_text(
        self,
        role: str,
        text: str,
        *,
        is_interim: bool = False,
        complete: bool = False,
    ) -> None:
        payload: Dict[str, Any] = {"role": role, "text": text}
        if is_interim:
            payload["is_interim"] = True
        if complete:
            payload["complete"] = True
        await self.send_event("text", payload)

    async def send_metrics(
        self,
        *,
        stt: Optional[Dict] = None,
        llm: Optional[Dict] = None,
        tts: Optional[Dict] = None,
        e2e: Optional[Dict] = None,
    ) -> None:
        metrics: Dict = {}
        if stt:
            metrics["stt"] = stt
        if llm:
            metrics["llm"] = llm
        if tts:
            metrics["tts"] = tts
        if e2e:
            metrics["e2e"] = e2e
        if metrics:
            await self.send_event("metrics", {"metrics": metrics})

    # ── Inbound message router ────────────────────────────────────────────────

    async def handle_message(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            await self._push_audio_bytes(raw)
            return

        try:
            payload = json.loads(raw)
        except Exception:
            return

        event = payload.get("event", "")

        if event == "media":
            await self._push_audio_b64(payload.get("audio", ""))

        elif event == "start":
            self._listening = True
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            await self._stop_vad_task()
            self._vad_task = asyncio.get_running_loop().create_task(
                self._vad_update_loop()
            )
            await self.send_state()

        elif event == "stop":
            if payload.get("target") == "playback":
                await self._orchestrator.cancel_response(notify_browser=True)
            else:
                self._listening = False
                await self._stop_vad_task()
                await self.send_state()

        elif event == "interrupt":
            await self._orchestrator.cancel_response(notify_browser=True)

        elif event == "wake_up":
            await self._send_greeting()

        elif event == "new_session":
            await self._new_session()

        elif event == "set_config":
            await self._set_config(payload)

        elif event == "reset_config":
            await self._set_config({
                "system_prompt": AGENT_SYSTEM_PROMPT,
                "temperature":   LLM_TEMPERATURE,
                "max_tokens":    LLM_MAX_TOKENS,
            })

        elif event == "get_config":
            await self.send_state()

    # ── Audio injection ───────────────────────────────────────────────────────

    async def _push_audio_b64(self, audio_b64: str) -> None:
        if not audio_b64 or not self._listening:
            return
        try:
            raw = base64.b64decode(audio_b64.encode("ascii"))
        except Exception:
            return
        await self._push_audio_bytes(raw)

    async def _push_audio_bytes(self, raw: bytes) -> None:
        if not raw or len(raw) % 4 != 0:
            return
        audio_f32 = np.frombuffer(raw, dtype=np.float32)

        self._audio_log_count += 1
        if self._audio_log_count % 100 == 1:
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
            logger.debug(f"[bridge] audio chunk #{self._audio_log_count}, rms={rms:.4f}")

        # Skip capture while the agent speaks, so mic echo doesn't overlap its track.
        if self._recorder and self._listening and not self._orchestrator.is_speaking:
            self._recorder.append_user_audio(audio_f32)

        if self._audio_input:
            await self._audio_input.push_audio(audio_f32)

    # ── Control actions ───────────────────────────────────────────────────────

    async def _send_greeting(self) -> None:
        """Speak a fixed greeting and record it as the agent's first turn."""
        greeting = "Hi! How can I assist you?"
        await self.send_text("assistant", greeting, complete=True)
        try:
            pcm = await asyncio.get_running_loop().run_in_executor(
                None, self._tts.synthesise, greeting
            )
        except Exception as exc:
            logger.warning(f"[bridge] greeting TTS failed: {exc}")
            return
        if pcm is None:
            return
        await self.send_audio(self._tts.pcm_to_wav(pcm), 0)
        if self._recorder:
            self._recorder.append_agent_pcm(pcm)
        self._orchestrator.add_assistant_message(greeting)

    async def _new_session(self) -> None:
        if self._recorder and self._recorder.has_audio:
            url = self._recorder.finalize()
            if url:
                # Send the OLD session_id so the frontend can match the recording.
                await self.send_event(
                    "session_recording", {"url": url, "session_id": self._session_id}
                )
        self._orchestrator.reset_conversation()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        await self.send_state()

    async def _set_config(self, payload: Dict) -> None:
        voice = payload.get("voice")
        if voice:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._tts.switch_voice, voice
                )
                await self.send_event("config_applied", {"key": "voice", "value": voice})
            except Exception as exc:
                logger.warning(f"[bridge] voice switch failed: {exc}")

        prompt = payload.get("system_prompt")
        if prompt:
            self._llm.system_prompt = prompt.strip()

        for key, cast in (("temperature", float), ("max_tokens", int)):
            value = payload.get(key)
            if value is not None:
                try:
                    setattr(self._llm, key, cast(value))
                except (ValueError, TypeError):
                    pass

        await self.send_state()
