"""
processors/voice_orchestrator.py — The brain: turns transcriptions into spoken replies.

Drops backchannels, dedupes, then runs one LLM tool-calling turn + TTS as async tasks
outside the pipecat pipeline (so they stay cancellable), tracking latency throughout.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

from config import (
    CHAT_SYSTEM_PROMPT,
    DEDUP_RECENT_WINDOW,
    DEDUP_SIMILARITY_THRESHOLD,
    SAMPLE_RATE,
    SUMMARY_KEEP,
    SUMMARY_TRIGGER,
    VAD_STOP_SECS,
)
from processors.ack_classifier import InterruptType, classify_interrupt, is_annotation, jaccard
from processors.agent_tools import AgentTools, tool_specs
from services.mlx_llm import _split_sentences

# INPUT router: is the user talking about their tasks? Decides only whether the tools and
# list go in the prompt — the model still chooses which tool to call. "to-do" is spelt
# loosely because Whisper renders it as "two do", "total", "to do" and so on.
_TODO = r"(?:to[-\s]?dos?|todos?|t[wo]o\s*dos?)"
_TODO_INTENT_RE = re.compile(rf"""
      \b{_TODO}\b
    | \btotal\s+list\b                                  # Whisper's "to-do list"
    | \b(?:my|the|that)\s+(?:\w+\s+)?(?:list|tasks?|reminders?)\b
    | \bremind\s+me\b
    | \b(?:add|put|keep|save|note)\b[^.?!]{{0,40}}?\b(?:list|task|reminder)\b
    | ^\s*(?:add|remember)\b                             # bare "add call the dentist"
    | \bi\s+(?:have|need)\s+to\b
    | \bmark(?:ed)?\s+(?:it|that|this|the|off|as|my|all)\b
    | \bmark\b[^.?!]{{0,40}}?\bas\s+(?:done|pending|hold|in[\s_-]?progress)\b
    | ^\s*(?:update|change|rename|reword|edit|rewrite)\b     # "update X to Y"
    | \b(?:update|rename|reword|edit|rewrite)\b[^.?!]{{0,50}}?\bto\b
    | \b(?:in[\s_-]?progress|on\s+hold)\b
    | \b(?:is|are|it|that)\s+(?:it\s+)?(?:done|updated|completed|finished)\b
    | \b(?:delete|remove|cancel|drop)\b[^.?!]{{0,40}}?\b(?:it|that|this|task|list|from)\b
    | \bwhat(?:'s|\s+is|\s+are)\b[^.?!]{{0,30}}?\b(?:list|tasks?|pending|left|active)\b
    | \bany(?:thing)?\s+(?:else\s+)?(?:pending|left|on\s+it)\b
""", re.I | re.X)

# OUTPUT check: the model describing the list or a change to it. Anything matching here
# must come from a real tool call, or it gets withheld and the turn retried — a 3B model
# will otherwise confidently report work it never did, or recite a stale list.
# Needs explicit agent-action or list-state phrasing, so advice ("you can mark it done")
# and ordinary prose never match.
_TODO_CLAIM_RE = re.compile(
    # "Added X", "Sure, I've marked X", "Done — removed X"
    r"^\W*(?:(?:sure|okay|ok|yes|yeah|alright|great|done|no problem)\b[\s,.!:-]*)?"
    r"(?:i\s*(?:'ve|'ll|have|will)?\s+)?(?:just\s+)?"
    r"(?:added|removed|deleted|updated|marked|changed|moved|created|set)\b"
    r"|\b(?:has|have)\s+been\s+(?:added|removed|deleted|updated|marked|created|set)\b"
    r"|\b(?:is|it'?s)\s+now\s+(?:marked|set)\b"
    # A bare promise ("I'll add that") strands the user just as badly — they hear it as saved.
    r"|\bi\s*(?:'ll|will|'m going to|am going to)\s+(?:just\s+)?"
    r"(?:add|remove|delete|update|mark|change|create|set)\b"
    # Reciting or summarising the list, including claiming it is empty.
    rf"|\b(?:your|the)\s+(?:current\s+)?(?:{_TODO}\s+)?list\s+(?:includes|contains|has|is)\b"
    rf"|\b(?:your|the)\s+(?:\w+\s+){{0,2}}?(?:{_TODO}|task)\s*(?:items?)?\s+(?:are|is|include)"
    rf"|\byou\s+(?:have|['’]ve\s+got)\s+(?:the\s+following|\d+|one|two|three|no)\b"
    rf"|\byou\s+(?:have|haven'?t)\s+(?:not\s+)?(?:created|added|got)\b"
    rf"|\b(?:following|these\s+are\s+your)\s+(?:{_TODO}|tasks?|items?)\b"
    rf"|\b(?:no|zero|any)\s+(?:{_TODO}|tasks?)\s+(?:yet|so\s+far|on\s+(?:your|the)\s+list)\b",
    re.I,
)


def _looks_like_json(text: str) -> bool:
    """True for machine output that must never be spoken (echoed tool results)."""
    t = text.strip()
    return bool(
        (t.startswith(("{", "[")) and ('":' in t or '" :' in t))
        or re.search(r'"(?:ok|action|todo|status|id|text)"\s*:', t)
    )


class VoiceOrchestrator(FrameProcessor):
    """Orchestration layer: STT -> this -> (runs the LLM agent loop + TTS as tasks)."""

    def __init__(
        self,
        llm_service,
        tts_service,
        rag_handler=None,
        recorder=None,
        todo_store=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._llm      = llm_service
        self._tts      = tts_service
        self._rag      = rag_handler
        self._recorder = recorder
        self._bridge   = None

        self._tools = AgentTools(todo_store)

        # Conversation state
        self._conversation: List[Dict[str, str]] = []
        self._pending_assistant: List[str]       = []
        self._sentence_buffer: List[str]         = []
        self._agent_speaking  = False
        self._user_speaking   = False
        self._last_turn_used_tools = False

        # Dedup
        self._last_processed:    Optional[str] = None
        self._last_processed_ts: float         = 0.0

        # ONNX model deciding COMPLETE vs INCOMPLETE turns from the last ~8 s of audio.
        self._turn_analyzer = LocalSmartTurnAnalyzerV3()
        self._turn_analyzer.set_sample_rate(SAMPLE_RATE)
        self._smart_turn_task: Optional[asyncio.Task] = None

        # LLM response task and cancellation
        self._response_task:   Optional[asyncio.Task]    = None
        self._llm_stop_event:  Optional[threading.Event] = None
        self._cancel_event:    Optional[asyncio.Event]   = None

        # Latency tracking
        self._t_vad_stop: Optional[float] = None
        self._tts_index = 0

    # ── Frame processing ──────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._t_vad_stop = None
            await self._cancel_smart_turn()
            if self._bridge:
                await self._bridge.send_event("speech_start")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._t_vad_stop = time.monotonic()
            if self._bridge:
                # Include VAD stop delay so frontend can compute true E2E from last spoken word
                await self._bridge.send_event("speech_end", {"vad_stop_ms": int(VAD_STOP_SECS * 1000)})
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame.text)
            return  # absorbed — don't push downstream

        # Feed raw mic audio to the turn analyser alongside the VAD speaking flag.
        if isinstance(frame, InputAudioRawFrame):
            try:
                self._turn_analyzer.append_audio(frame.audio, self._user_speaking)
            except Exception as exc:
                logger.warning(f"[orch] smart-turn append_audio failed: {exc}")
            await self.push_frame(frame, direction)
            return

        # Pass everything else downstream
        await self.push_frame(frame, direction)

    # ── Transcription orchestration ───────────────────────────────────────────

    async def _handle_transcription(self, text: str) -> None:
        text = text.strip()
        if not text or is_annotation(text):
            return

        # Filter bracketed Whisper tokens, e.g. "[BLANK_AUDIO]"
        if text.startswith("[") and text.endswith("]"):
            return

        # Send STT latency metric
        if self._t_vad_stop is not None and self._bridge:
            stt_latency = time.monotonic() - self._t_vad_stop
            await self._bridge.send_metrics(stt={"latency": round(stt_latency, 3)})

        agent_q = self._agent_last_asked_question()
        itype   = classify_interrupt(text, agent_asked_question=agent_q)

        if itype == InterruptType.ACKNOWLEDGEMENT:
            logger.info(f"[orch] ACK dropped (agent_asked_q={agent_q}): '{text}'")
            if self._bridge:
                await self._bridge.send_event("ack_dropped", {"text": text})
            return
        if itype == InterruptType.NONE:
            return

        if self._agent_speaking and itype == InterruptType.SUBSTANTIVE:
            logger.info(f"[orch] Interrupting agent for: '{text}'")
            await self.cancel_response(notify_browser=True)

        if self._should_dedup(text):
            logger.info(f"[orch] Dedup: '{text}'")
            return

        self._sentence_buffer.append(text)
        await self._cancel_smart_turn()

        if not self._user_speaking:
            self._smart_turn_task = asyncio.get_running_loop().create_task(
                self._smart_flush_check()
            )

    async def _smart_flush_check(self) -> None:
        """Flush the buffered utterance to the agent once the turn looks COMPLETE.

        On INCOMPLETE we wait for the user's next pause, with a timed fallback so a
        finished utterance can't be held forever.
        """
        SAFETY_FALLBACK_S = 2.0
        try:
            await asyncio.sleep(0.05)  # minimal gap for audio accumulation
            if not self._sentence_buffer:
                return

            state, _ = await self._turn_analyzer.analyze_end_of_turn()

            if state == EndOfTurnState.COMPLETE:
                await self._flush_to_llm()
                return

            # INCOMPLETE — wait for the next VAD stop event. Safety net: if no
            # further speech arrives, flush anyway so the turn cannot hang.
            logger.info("[orch] smart-turn INCOMPLETE — awaiting more speech")
            await asyncio.sleep(SAFETY_FALLBACK_S)
            if self._sentence_buffer and not self._user_speaking:
                logger.info("[orch] smart-turn fallback flush after silence")
                await self._flush_to_llm()
        except asyncio.CancelledError:
            pass

    async def _flush_to_llm(self) -> None:
        if not self._sentence_buffer:
            return

        merged = " ".join(self._sentence_buffer).strip()
        self._sentence_buffer.clear()
        if not merged:
            return

        self._last_processed    = merged
        self._last_processed_ts = time.monotonic()

        # Commit the user turn before the reply audio, for clean turn-taking in the WAV.
        if self._recorder:
            self._recorder.commit_user_turn()

        if self._bridge:
            await self._bridge.send_text("user", merged, complete=True)

        logger.info(f"[orch] Dispatched to agent: '{merged}'")
        await self._run_llm_response(merged)

    # ── Agent loop + TTS (runs OUTSIDE pipecat) ──────────────────────────────

    async def _run_llm_response(self, user_input: str) -> None:
        t_vad_stop = self._t_vad_stop
        await self.cancel_response()

        cancel   = asyncio.Event()
        llm_stop = threading.Event()
        self._cancel_event   = cancel
        self._llm_stop_event = llm_stop
        self._response_task  = asyncio.create_task(
            self._agent_turn(user_input, cancel, llm_stop, t_vad_stop)
        )

    async def _agent_turn(
        self,
        user_input: str,
        cancel: asyncio.Event,
        llm_stop: threading.Event,
        t_vad_stop: Optional[float] = None,
    ) -> None:
        """One user turn: the agent tool loop runs, streaming its final answer to TTS as
        it writes (low latency), calling to-do tools as needed."""
        t_llm_start = time.monotonic()
        try:
            self._agent_speaking = True
            self._pending_assistant = []
            if self._bridge:
                await self._bridge.send_event("response_start")

            self._conversation.append({"role": "user", "content": user_input})

            if len(self._conversation) >= SUMMARY_TRIGGER:
                await self._compress_history()

            # Tools only on task turns: given them on a general question, a 3B model
            # answers as a to-do bot ("the prime minister is not on your list").
            todo_turn = self._is_todo_turn(user_input)

            if todo_turn:
                # Live system prompt (UI-settable via the bridge), not the config constant.
                sys_content = self._llm.system_prompt + "\n\n" + self._tools.todos_context()
                history = [dict(m) for m in self._conversation]
            else:
                # A conversational turn gets the plain chat prompt and a history stripped of
                # tool traffic. Measured on this model: keeping either around drags general
                # answers from ~100% down to ~83% as it slips back into to-do-bot mode.
                sys_content = CHAT_SYSTEM_PROMPT
                history = self._plain_history()

            rag_ctx = await self._rag_lookup(user_input)
            if rag_ctx:
                sys_content += "\n\n[Reference — answer from these document excerpts]\n" + rag_ctx
            if self._llm.summary:
                sys_content += "\n\n[Conversation summary so far]\n" + self._llm.summary
            msgs: List[Dict] = [{"role": "system", "content": sys_content}] + history

            spoken = await self._agent_generate(
                msgs, tool_specs() if todo_turn else None,
                cancel, llm_stop, t_vad_stop, t_llm_start, user_input)

            if cancel.is_set():
                pass
            elif self._last_turn_used_tools:
                # History already has the tool_call/tool pair. Storing the confirmation
                # too would teach the model to just SAY it instead of calling the tool.
                pass
            elif spoken:
                self._conversation.append({"role": "assistant", "content": spoken})
            else:
                # Empty generation — ask for a repeat (not a canned topic answer).
                fallback = "Sorry, I didn't catch that — could you say it again?"
                await self._speak_text(fallback, cancel, t_vad_stop, t_llm_start)
                self._conversation.append({"role": "assistant", "content": fallback})

        except asyncio.CancelledError:
            logger.info("[agent] cancelled")
            if llm_stop:
                llm_stop.set()
            raise
        except Exception as e:
            logger.exception(f"[agent] error: {e}")
        finally:
            self._agent_speaking = False
            self._cancel_event   = None
            self._llm_stop_event = None
            if self._bridge:
                await self._bridge.send_state()

    @staticmethod
    def _is_todo_turn(user_input: str) -> bool:
        """Whether to put the to-do tools and list in this turn's prompt."""
        return bool(_TODO_INTENT_RE.search(user_input))

    def _plain_history(self) -> List[Dict]:
        """Conversation as plain user/assistant text, with tool traffic left out."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._conversation
            if m["role"] in ("user", "assistant") and (m.get("content") or "").strip()
        ]

    async def _agent_generate(self, msgs, tools, cancel, llm_stop, t_vad_stop,
                              t_llm_start, user_input: str = "") -> str:
        """One agent turn: stream a plain answer, or dispatch the tools it called.

        `tools` is None on a conversational turn. A described-but-uncalled to-do change is
        never spoken — the turn is retried forcing a real call.
        """
        if cancel.is_set():
            return ""
        self._llm.log_prompt(msgs)

        spoken, calls, faked = await self._stream_turn(
            msgs, tools, cancel, llm_stop, t_vad_stop, t_llm_start)

        if faked and not calls:
            # Force the real specs even on a turn routed as conversational.
            calls = await self._retry_forcing_tools(msgs, tools or tool_specs(), cancel)

        self._last_turn_used_tools = bool(calls)
        if calls:
            return await self._run_tool_calls(calls, cancel, t_vad_stop, t_llm_start)

        if faked:
            # Read the real state instead of confirming a change that never happened.
            truth = self._describe_tool_result(
                "list_todos", await self._tools.dispatch("list_todos", {}))
            await self._speak_text(truth, cancel, t_vad_stop, t_llm_start)
            return truth

        # The UI's final text is EXACTLY the spoken sentences (so screen == audio).
        final = " ".join(spoken).strip()
        if final and self._bridge and not cancel.is_set():
            await self._bridge.send_text("assistant", final, complete=True)
        return final

    async def _stream_turn(self, msgs, tools, cancel, llm_stop, t_vad_stop, t_llm_start):
        """Stream one generation, speaking sentences as they form.

        Returns (spoken, tool_calls, faked). `faked` means the prose claimed a to-do
        change; that sentence is withheld from TTS so a false confirmation is never heard.
        """
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop = llm_stop or threading.Event()
        fut = loop.run_in_executor(
            self._llm.executor,
            lambda: self._llm.stream_tool_turn(msgs, tools, q, loop, stop))

        first = True
        faked = False
        spoken: List[str] = []
        while True:
            sent = await q.get()
            if sent is None:
                break
            if cancel.is_set() or faked:
                continue  # drain the queue quietly
            sent = self._clean_for_speech(sent)
            if not sent:
                continue
            # Machine output is never speakable: the model sometimes copies a tool result
            # verbatim, and reading JSON aloud is worse than saying nothing.
            if _looks_like_json(sent):
                faked = True
                stop.set()
                continue
            # Check before speaking: a claim must come from a real tool call.
            if first and _TODO_CLAIM_RE.search(sent):
                faked = True
                stop.set()          # stop generating; we'll retry
                continue
            spoken.append(sent)
            await self._emit_sentence(sent, first, cancel, t_vad_stop, t_llm_start)
            first = False

        _, calls = await fut
        return spoken, calls, faked

    async def _retry_forcing_tools(self, msgs, tools, cancel) -> List[Dict]:
        """Re-run the turn demanding a tool call. Returns the calls, or [] if it refused."""
        if cancel.is_set():
            return []
        forced = list(msgs) + [{"role": "system", "content": (
            "You just described the to-do list without calling a tool, so nothing happened. "
            "Reply with ONLY the correct tool call (create_todo, list_todos, "
            "update_todo_status, update_todo_text or delete_todo) for the user's last message."
        )}]
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _, calls = await loop.run_in_executor(
            self._llm.executor,
            lambda: self._llm.stream_tool_turn(
                forced, tools, q, loop, threading.Event(), tools_only=True))
        return calls

    async def _run_tool_calls(self, calls, cancel, t_vad_stop, t_llm_start) -> str:
        """Dispatch each tool call and speak one deterministic confirmation per result."""
        logger.info(f"[agent] tool calls: {[c['name'] for c in calls]}")
        spoken_parts: List[str] = []
        first = True
        changed = False
        for call in calls:
            if cancel.is_set():
                break
            args = call.get("arguments") or {}
            result_json = await self._tools.dispatch(call["name"], args)
            line = self._describe_tool_result(call["name"], result_json)
            # Record the tool transcript so the model sees that actions happen by CALLING
            # tools. Never the raw JSON or a list snapshot: the model copies whatever shape
            # it finds here and reads it out, and a snapshot goes stale on the next change.
            self._conversation.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {
                    "name": call["name"], "arguments": args}}],
            })
            self._conversation.append({
                "role": "tool",
                "content": "(the list was read out to the user)"
                           if call["name"] == "list_todos" else f"({line})",
            })
            if call["name"] != "list_todos":
                try:
                    changed = changed or json.loads(result_json).get("ok", False)
                except Exception:
                    pass
            if not line:
                continue
            spoken_parts.append(line)
            await self._emit_sentence(line, first, cancel, t_vad_stop, t_llm_start)
            first = False

        if changed and self._bridge:
            await self._bridge.send_event("todos_changed")
        final = " ".join(spoken_parts).strip()
        if final and self._bridge and not cancel.is_set():
            await self._bridge.send_text("assistant", final, complete=True)
        return final

    @staticmethod
    def _describe_tool_result(name: str, result_json: str) -> str:
        """Turn one tool's JSON result into a short spoken confirmation. Wording is
        deterministic rather than model-generated — narrating structured data reliably
        is beyond what a 3B model can be trusted to do."""
        try:
            result = json.loads(result_json)
        except Exception:
            return "Sorry, something went wrong with that."
        if not result.get("ok"):
            return result.get("error") or "Sorry, something went wrong with that."

        action = result.get("action")
        if action == "created":
            return f"Added \"{result['todo']['text']}\" to your to-do list."
        if action == "updated":
            todo = result["todo"]
            return f"Marked \"{todo['text']}\" as {todo['status'].replace('_', ' ')}."
        if action == "reworded":
            return f"Changed \"{result['was']}\" to \"{result['todo']['text']}\"."
        if action == "deleted":
            return f"Removed \"{result['todo']['text']}\" from your to-do list."
        if action == "listed":
            todos = result.get("todos") or []
            if not todos:
                return "Your to-do list is empty."
            parts = [f'"{t["text"]}" ({t["status"].replace("_", " ")})' for t in todos]
            if len(parts) == 1:
                return f"You have one task: {parts[0]}."
            return f"You have {len(parts)} tasks: " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
        return "Done."

    async def _rag_lookup(self, user_input: str) -> Optional[str]:
        """Relevant document chunks for the utterance (threshold-gated), or None."""
        if not (self._rag and getattr(self._rag, "has_documents", lambda: False)()):
            return None
        try:
            chunks = await asyncio.get_running_loop().run_in_executor(
                None, self._rag.query_texts_only, user_input)
        except Exception as exc:
            logger.warning(f"[orch] RAG lookup failed: {exc}")
            return None
        if chunks and self._bridge:
            await self._bridge.send_event("rag_context", {"chunks": len(chunks)})
        return "\n\n".join(chunks) if chunks else None

    async def _speak_text(
        self,
        text: str,
        cancel: asyncio.Event,
        t_vad_stop: Optional[float],
        t_llm_start: float,
    ) -> None:
        """Speak a full answer sentence-by-sentence so the first audio lands sooner."""
        text = self._clean_for_speech(text)
        if not text or cancel.is_set():
            return
        sentences = _split_sentences(
            text, self._llm.sentence_delimiters, self._llm.min_tokens_for_tts) or [text]
        first = True
        spoken: List[str] = []
        for sentence in sentences:
            if cancel.is_set():
                break
            sentence = sentence.strip()
            if not sentence:
                continue
            spoken.append(sentence)
            await self._emit_sentence(sentence, first, cancel, t_vad_stop, t_llm_start)
            first = False
        if self._bridge and not cancel.is_set():
            # Final UI text == exactly what was spoken.
            await self._bridge.send_text("assistant", " ".join(spoken).strip() or text, complete=True)

    async def _emit_sentence(
        self,
        sentence: str,
        first: bool,
        cancel: asyncio.Event,
        t_vad_stop: Optional[float],
        t_llm_start: float,
    ) -> None:
        """Send one sentence to the transcript and TTS; the first also closes the timers."""
        if first and self._bridge:
            await self._bridge.send_metrics(
                llm={"first_token": round(time.monotonic() - t_llm_start, 3)})
        self._pending_assistant.append(sentence)
        if self._bridge:
            await self._bridge.send_text("assistant", sentence)   # partial — appended in UI
        await self._synthesise_and_send(sentence, cancel, t_vad_stop=t_vad_stop if first else None)

    async def _synthesise_and_send(
        self,
        text: str,
        cancel: asyncio.Event,
        t_vad_stop: Optional[float] = None,
    ) -> None:
        """Synthesise one sentence and send WAV to browser."""
        if cancel.is_set() or not text.strip():
            return

        t0 = time.monotonic()
        try:
            pcm = await asyncio.get_running_loop().run_in_executor(
                None, self._tts.synthesise, text
            )
        except Exception as e:
            logger.error(f"[tts] synthesis error: {e}")
            return

        if pcm is None or cancel.is_set():
            return

        t_ready = time.monotonic()
        metrics: Dict = {"tts": {"first_audio": round(t_ready - t0, 3)}}
        if t_vad_stop is not None:
            metrics["e2e"] = {"latency": round(t_ready - t_vad_stop, 3)}
        if self._bridge:
            await self._bridge.send_metrics(**metrics)
            await self._bridge.send_audio(self._tts.pcm_to_wav(pcm), self._tts_index)
        self._tts_index += 1

        if self._recorder:
            self._recorder.append_agent_pcm(pcm)

    # ── Cancellation ──────────────────────────────────────────────────────────

    async def cancel_response(self, notify_browser: bool = False) -> None:
        """Cancel any in-flight agent/TTS response."""
        if self._llm_stop_event:
            self._llm_stop_event.set()
        if self._cancel_event and not self._cancel_event.is_set():
            self._cancel_event.set()

        task = self._response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self._response_task  = None
        self._cancel_event   = None
        self._llm_stop_event = None
        self._agent_speaking = False
        self._pending_assistant = []

        if notify_browser and self._bridge:
            await self._bridge.send_event("interrupt")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _should_dedup(self, text: str) -> bool:
        """True if this repeats the previous utterance within the dedup window."""
        if not self._last_processed:
            return False
        if time.monotonic() - self._last_processed_ts >= DEDUP_RECENT_WINDOW:
            return False
        return jaccard(text, self._last_processed) >= DEDUP_SIMILARITY_THRESHOLD

    def _agent_last_asked_question(self) -> bool:
        """True if the agent just asked a question, making a bare 'yes' a real answer."""
        if self._pending_assistant:
            last = self._pending_assistant[-1].strip()
            return bool(last) and last.endswith("?")
        last_msg = self._conversation[-1] if self._conversation else None
        if last_msg and last_msg["role"] == "assistant":
            content = (last_msg.get("content") or "").strip()
            return bool(content) and content.endswith("?")
        return False

    async def _cancel_smart_turn(self) -> None:
        """Drop any pending end-of-turn check so a new utterance restarts the analysis."""
        if self._smart_turn_task and not self._smart_turn_task.done():
            self._smart_turn_task.cancel()
            try:
                await self._smart_turn_task
            except asyncio.CancelledError:
                pass
        self._smart_turn_task = None

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown, smart quotes and non-ASCII drift the TTS shouldn't read."""
        text = (text or "").strip()
        for uni, asc in (("—", " - "), ("–", "-"), ("‘", "'"),
                         ("’", "'"), ("“", '"'), ("”", '"'), ("…", "...")):
            text = text.replace(uni, asc)
        text = re.sub(r"[^\x00-\x7f]", "", text)   # drop remaining non-ASCII
        text = re.sub(r"\*\*|__|`", "", text)      # bold / code markers
        lines = [
            re.sub(r"^\s*([-*#>]+|\d+\.)\s+", "", ln).strip()   # bullets / numbering
            for ln in text.splitlines()
        ]
        return re.sub(r"\s{2,}", " ", " ".join(l for l in lines if l)).strip()

    @staticmethod
    def _describe_history_message(msg: Dict) -> str:
        """Readable one-liner for the summariser — tool-call messages have no content."""
        calls = msg.get("tool_calls")
        if calls:
            return "; ".join(
                f"called {c['function']['name']}({c['function'].get('arguments') or {}})"
                for c in calls
            )
        return msg.get("content") or ""

    async def _compress_history(self) -> None:
        """Replace older messages with a ≤6-point summary, keeping the prompt bounded."""
        if len(self._conversation) <= SUMMARY_KEEP:
            return
        split = len(self._conversation) - SUMMARY_KEEP
        # Never start the kept window on a tool result whose tool_call we dropped.
        while split < len(self._conversation) and self._conversation[split]["role"] == "tool":
            split += 1
        older  = self._conversation[:split]
        recent = self._conversation[split:]

        new_text = "\n".join(
            f"{m['role'].capitalize()}: {self._describe_history_message(m)[:400]}"
            for m in older
        ).strip()
        prior = (self._llm.summary or "").strip()
        body = (f"Summary so far:\n{prior}\n\n" if prior else "") + f"New messages:\n{new_text}"

        summary_msgs = [
            {"role": "user", "content":
             "Update the running summary of this whole conversation into 5 to 6 short "
             "third-person sentences covering the key facts and what the user wants. "
             "Keep the most important points and drop trivia. Output only the sentences, "
             "nothing else.\n\n" + body},
        ]
        try:
            bullets = await asyncio.get_running_loop().run_in_executor(
                self._llm.executor,
                self._llm.generate_text_sync,
                summary_msgs, 220, 0.0,
            )
            points = [l.strip(" -•\t") for l in (bullets or "").splitlines()]
            points = [p for p in points if p][:6]
            clean = " ".join(points)
            # A degenerate summary (the model echoing the instruction) poisons later turns.
            if clean and "bullet point" not in clean.lower():
                self._llm.summary = "\n".join(f"- {p}" for p in points)
                self._conversation = list(recent)
                logger.info(f"[orch] compressed {len(older)} msgs -> {len(points)} points")
            else:
                logger.warning(f"[orch] discarded degenerate summary: {clean[:80]!r}")
        except Exception as e:
            logger.warning(f"[orch] summary failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._agent_speaking

    def add_assistant_message(self, text: str) -> None:
        """Record an assistant turn produced outside the agent loop (e.g. the greeting)."""
        self._conversation.append({"role": "assistant", "content": text})

    def reset_conversation(self) -> None:
        self._conversation      = []
        self._pending_assistant = []
        self._sentence_buffer   = []
        self._last_processed    = None
        self._last_processed_ts = 0.0
        self._llm.summary       = None
        logger.info("[orch] Conversation reset")

    def set_bridge(self, bridge) -> None:
        self._bridge = bridge
