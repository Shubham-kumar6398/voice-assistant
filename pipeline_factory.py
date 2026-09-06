"""
pipeline_factory.py — Wires one pipecat pipeline per WebSocket session.

Layout: browser audio -> VAD -> Whisper STT -> VoiceOrchestrator -> sink. The LLM and
TTS run outside the pipeline as asyncio tasks, replying via the WebSocketBridge.
"""

from __future__ import annotations

import asyncio
import gc

import numpy as np
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, Frame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from config import (
    AGENT_SYSTEM_PROMPT,
    SAMPLE_RATE,
    VAD_START_SECS,
    VAD_STOP_SECS,
)
from processors.session_recorder import SessionRecorder
from processors.voice_orchestrator import VoiceOrchestrator
from services.energy_vad import EnergyVADAnalyzer
from services.whisper_stt import WhisperCppSTTService


class BrowserAudioInputProcessor(FrameProcessor):
    """Injects audio from WebSocket into the pipecat pipeline as InputAudioRawFrames."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, **kwargs):
        super().__init__(**kwargs)
        self._sr = sample_rate

    async def push_audio(self, float32_data: np.ndarray) -> None:
        """Called by the WebSocket handler for every incoming mic chunk."""
        pcm = (np.clip(float32_data, -1.0, 1.0) * 32767).astype(np.int16)
        await self.push_frame(
            InputAudioRawFrame(audio=pcm.tobytes(), sample_rate=self._sr, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class PipelineSink(FrameProcessor):
    """Terminal processor — absorbs all frames at the end of the pipeline."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Intentionally do not push further


class PipelineSession:
    """All state and pipecat objects for one active WebSocket session."""

    def __init__(
        self,
        ws: WebSocket,
        shared: "SharedResources",
        session_id: str,
    ) -> None:
        self._ws = ws
        self._recorder = SessionRecorder(session_id)

        # Fresh VAD per session, so noise-floor calibration is per-connection.
        self._vad = EnergyVADAnalyzer(
            params=VADParams(
                confidence=0.50,
                start_secs=VAD_START_SECS,
                stop_secs=VAD_STOP_SECS,
                min_volume=0.0,
            )
        )

        # We manage WebSocket reads ourselves rather than using a pipecat transport.
        self._audio_input = BrowserAudioInputProcessor(sample_rate=SAMPLE_RATE)

        self._orchestrator = VoiceOrchestrator(
            llm_service=shared.llm,
            tts_service=shared.tts,
            rag_handler=shared.rag,
            recorder=self._recorder,
            todo_store=shared.todos,
        )

        self._task = PipelineTask(
            Pipeline([
                self._audio_input,
                VADProcessor(vad_analyzer=self._vad),
                shared.stt,
                self._orchestrator,
                PipelineSink(),
            ]),
            params=PipelineParams(allow_interruptions=True),
        )
        self._runner = PipelineRunner(handle_sigint=False)

        from api.ws_bridge import WebSocketBridge
        self._bridge = WebSocketBridge(
            ws=ws,
            pipeline_task=self._task,
            audio_input=self._audio_input,
            orchestrator=self._orchestrator,
            llm=shared.llm,
            tts=shared.tts,
            recorder=self._recorder,
            session_id=session_id,
            vad=self._vad,
        )
        self._orchestrator.set_bridge(self._bridge)

    # ── Run ───────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the pipeline and the WebSocket message loop concurrently."""
        await asyncio.gather(
            self._runner.run(self._task),
            self._run_message_loop(),
            return_exceptions=True,
        )

    async def _run_message_loop(self) -> None:
        """Receive JSON messages from the browser and route them."""
        try:
            while True:
                try:
                    msg = await self._ws.receive()
                except Exception:
                    break

                if msg["type"] == "websocket.disconnect":
                    break

                if msg["type"] == "websocket.receive":
                    data = msg.get("bytes") or msg.get("text")
                    if data:
                        await self._bridge.handle_message(data)
        except Exception as exc:
            logger.debug(f"[pipeline] message loop ended: {exc}")
        finally:
            await self._finalize_session()
            await self._task.queue_frame(EndFrame())

    async def _finalize_session(self) -> None:
        """Cancel in-flight work, then save and announce the session recording."""
        await self._bridge.cleanup()
        await self._orchestrator.cancel_response()
        if not self._recorder.has_audio:
            return
        url = self._recorder.finalize()
        if not url:
            return
        try:
            await self._bridge.send_event("session_recording", {
                "url": url,
                "session_id": self._bridge.session_id,
            })
            logger.info(f"[pipeline] Session recording saved: {url}")
        except Exception:
            pass


class SharedResources:
    """Process-level singletons loaded once at startup and shared by every session."""

    def __init__(self) -> None:
        logger.info("Initialising shared resources...")
        from processors.todo_store import TodoStore
        from services.mlx_llm import MLXLLMService
        from services.piper_tts import PiperTTSService

        self.stt = WhisperCppSTTService()
        self.llm = MLXLLMService(system_prompt=AGENT_SYSTEM_PROMPT)
        self.tts = PiperTTSService()
        self.todos = TodoStore()
        try:
            from services.rag_handler import RAGHandler
            self.rag = RAGHandler()
        except Exception as exc:
            logger.warning(f"[rag] init failed — document Q&A disabled: {exc}")
            self.rag = None
        logger.info("Shared resources ready")

    async def shutdown(self) -> None:
        """Release the STT and LLM models on server shutdown."""
        self.stt._model = None
        self.llm.shutdown()
        gc.collect()
        logger.info("[shared] Resources shut down")


def create_session(ws: WebSocket, shared: SharedResources, session_id: str) -> PipelineSession:
    return PipelineSession(ws, shared, session_id)
