"""
services/mlx_llm.py — MLX/Qwen LLM service.

Runs all MLX/Metal calls on one dedicated thread (Metal command streams aren't
thread-safe). Provides: streamed tool-calling turns (the agent) and plain text
generation (summaries). Plus helpers for tool-call parsing and sentence splitting.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import List, Dict, Optional

from loguru import logger

from config import (
    LLM_LOG_PROMPT,
    LLM_MAX_SENTENCES,
    LLM_MAX_TOKENS,
    LLM_MIN_TOKENS_FOR_TTS,
    LLM_MODEL,
    LLM_SENTENCE_DELIMITERS,
    LLM_TEMPERATURE,
)

# Tool calls arrive inside <tool_call>...</tool_call> blocks. The inner payload comes in
# two shapes depending on the model:
#   • JSON (Qwen2.5):  {"name": "create_todo", "arguments": {"text": "..."}}
#   • XML  (Qwen3):    <function=create_todo><parameter=text>...</parameter></function>
# We capture the whole inner block and parse whichever shape it is.
_TOOLCALL_RE      = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_XML_FUNCTION_RE  = re.compile(r"<function(?:=|\s+name=\")([^>\"\s]+)")
_XML_PARAMETER_RE = re.compile(r"<parameter(?:=|\s+name=\")([^>\"\s]+)\"?\s*>(.*?)</parameter>", re.DOTALL)


def _parse_one_tool_call(inner: str) -> Optional[Dict]:
    """Parse one tool-call block, accepting either the JSON or the XML (Qwen3) shape."""
    inner = inner.strip()
    if not inner.startswith("{"):
        fn = _XML_FUNCTION_RE.search(inner)
        if fn:
            args = {k.strip(): v.strip() for k, v in _XML_PARAMETER_RE.findall(inner)}
            return {"name": fn.group(1).strip(), "arguments": args}
    return _loads_tool_json(inner)


def _loads_tool_json(s: str) -> Optional[Dict]:
    """Parse one tool-call JSON object, tolerating the doubled-brace quirk."""
    s = s.strip()
    candidates = [s]
    if s.startswith("{{") and s.endswith("}}"):
        candidates.append(s[1:-1])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
    # Last-resort salvage: pull name + arguments out by regex.
    nm = re.search(r'"name"\s*:\s*"([^"]+)"', s)
    if not nm:
        return None
    args: Dict = {}
    am = re.search(r'"arguments"\s*:\s*(\{.*\})', s, re.DOTALL)
    if am:
        frag = am.group(1)
        for cand in (frag, frag.rstrip("}") + "}"):
            try:
                args = json.loads(cand)
                break
            except Exception:
                continue
    return {"name": nm.group(1), "arguments": args}


def parse_tool_calls(text: str) -> List[Dict]:
    """Extract every {name, arguments} tool call from a generation. Empty if none."""
    calls = []
    for m in _TOOLCALL_RE.finditer(text):
        parsed = _parse_one_tool_call(m.group(1))
        if parsed and parsed.get("name"):
            calls.append({"name": parsed["name"], "arguments": parsed.get("arguments") or {}})
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call blocks (and any stray tags) from a generation, leaving prose."""
    cleaned = _TOOLCALL_RE.sub("", text)
    cleaned = cleaned.replace("<tool_call>", "").replace("</tool_call>", "")
    return cleaned.strip()


def extract_sentences(buffer: str, delimiters: str, min_tokens: int) -> tuple:
    """Split buffer at last sentence delimiter; return (complete_sentences, remainder)."""
    last_delim = -1
    for i, ch in enumerate(buffer):
        if ch in delimiters:
            last_delim = i

    if last_delim < 0:
        return [], buffer

    complete  = _split_sentences(buffer[:last_delim + 1], delimiters, min_tokens)
    remaining = buffer[last_delim + 1:]
    return complete, remaining


def _split_sentences(text: str, delimiters: str, min_tokens: int) -> List[str]:
    raw, buf = [], ""
    for ch in text:
        buf += ch
        if ch in delimiters:
            if buf.strip():
                raw.append(buf.strip())
            buf = ""
    if buf.strip():
        raw.append(buf.strip())
    if not raw:
        return [text]
    merged = []
    for s in raw:
        if not merged or len(s.split()) >= min_tokens:
            merged.append(s)
        else:
            merged[-1] = (merged[-1] + " " + s).strip()
    return merged


class MLXLLMService:
    """Plain MLX LLM service (not a pipecat FrameProcessor).

    All MLX/Metal calls run on a single dedicated thread via mlx_executor.
    """

    def __init__(
        self,
        *,
        model_name: str = LLM_MODEL,
        system_prompt: str = "",
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        max_sentences: int = LLM_MAX_SENTENCES,
        sentence_delimiters: str = LLM_SENTENCE_DELIMITERS,
        min_tokens_for_tts: int = LLM_MIN_TOKENS_FOR_TTS,
    ) -> None:
        self.model_name          = model_name
        # Live generation knobs — the single source of truth. The agent loop reads them
        # off the service and the WebSocket bridge writes them from the UI.
        self.system_prompt       = system_prompt
        self.max_tokens          = max_tokens
        self.temperature         = temperature
        self.max_sentences       = max_sentences
        self.sentence_delimiters = sentence_delimiters
        self.min_tokens_for_tts  = min_tokens_for_tts

        self._summary: Optional[str] = None

        # MLX single-thread executor
        self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx"
        )

        self._model = None
        self._tokenizer = None
        self._load_future = self._mlx_executor.submit(self._load_model, model_name)

    def _load_model(self, model_name: str) -> None:
        logger.info(f"[llm] Loading model: {model_name}")
        try:
            from mlx_lm import load
            self._model, self._tokenizer = load(model_name)
            logger.info("[llm] Model loaded")
        except Exception as exc:
            logger.error(f"[llm] Failed to load model '{model_name}': {exc}")
            raise

    def _ensure_loaded(self) -> None:
        # Always call .result(): waits if still loading, re-raises any load exception
        self._load_future.result()

    @property
    def executor(self):
        return self._mlx_executor

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @summary.setter
    def summary(self, value: Optional[str]):
        self._summary = value

    def generate_text_sync(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Blocking non-streaming generation. Call via executor."""
        self._ensure_loaded()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._format_prompt(messages)
        result = generate(
            self._model, self._tokenizer, prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return (result if isinstance(result, str) else str(result)).strip()

    # ── Tool-calling (single-agent loop) ──────────────────────────────────────

    def stream_tool_turn(self, messages, tools, sentence_q, loop, stop_event,
                         tools_only: bool = False) -> tuple:
        """One agent step. Pushes finished sentences to sentence_q (then None) so they can
        be spoken as they're written; a tool-call generation pushes nothing. Run via executor.

        tools_only=True makes this a cheap classifier: generation stops the moment the output
        can no longer be a tool call, so nothing is generated or spoken for a plain answer.

        Returns (text, tool_calls).
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        max_tokens, max_sentences = self.max_tokens, self.max_sentences
        prompt = self._format_prompt(messages, tools)
        sampler = make_sampler(temp=self.temperature)
        full = ""
        mode = None          # None = undecided, "tool", or "text"
        cursor = 0           # chars of `full` already emitted as sentences
        emitted = 0
        seen: set = set()
        try:
            for resp in stream_generate(self._model, self._tokenizer, prompt,
                                        max_tokens=max_tokens, sampler=sampler):
                if stop_event.is_set():
                    break
                full += resp.text
                if mode is None:
                    s = full.lstrip()
                    if "<tool_call>" in full:
                        mode = "tool"
                    elif s and not "<tool_call>".startswith(s[:11]):
                        mode = "text"
                    # else: still ambiguous (could become a tool call) — keep buffering
                if mode == "text" and tools_only:
                    break        # not a tool call; the caller answers in a separate pass
                if mode == "text":
                    complete, remainder = extract_sentences(
                        full[cursor:], self.sentence_delimiters, self.min_tokens_for_tts)
                    stop = False
                    for sent in complete:
                        st = sent.strip()
                        if not st:
                            continue
                        # An exact-repeated sentence means the model is looping — stop.
                        key = st.lower()
                        if len(key) > 4 and key in seen:
                            stop = True
                            break
                        seen.add(key)
                        loop.call_soon_threadsafe(sentence_q.put_nowait, st)
                        emitted += 1
                        if emitted >= max_sentences:
                            stop = True
                            break
                    cursor = len(full) - len(remainder)
                    if stop:
                        break
        except Exception as exc:
            logger.error(f"[llm] stream_tool_turn error: {exc}")
        finally:
            if mode == "text" and not tools_only and emitted < max_sentences:
                tail = full[cursor:].strip()
                if tail and tail.lower() not in seen:
                    loop.call_soon_threadsafe(sentence_q.put_nowait, tail)
            loop.call_soon_threadsafe(sentence_q.put_nowait, None)

        if "<tool_call>" in full:
            return "", parse_tool_calls(full)
        return strip_tool_calls(full), []

    def _format_prompt(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """Render messages via the chat template, with tool specs when `tools` is given.

        enable_thinking=False suppresses Qwen3 <think> output so none is ever spoken.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        extra = {"tools": tools} if tools else {}
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False, **extra,
        )

    def log_prompt(self, messages: List[Dict[str, str]]) -> None:
        if not LLM_LOG_PROMPT:
            return
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"[LLM PROMPT DUMP]  {len(messages)} message(s)")
        print(sep)
        for i, m in enumerate(messages):
            role    = m.get("role", "?").upper()
            content = m.get("content", "")
            print(f"\n--- [{i}] {role} ---")
            print(content)
        print(f"\n{sep}\n")

    def shutdown(self) -> None:
        self._mlx_executor.shutdown(wait=True, cancel_futures=True)
