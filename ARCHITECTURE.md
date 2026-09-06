# Voice Assistant — Architecture

An offline, on-device voice assistant for everyday use. It does two things:

1. **Conversation** — general chat, optionally grounded in documents you upload (RAG).
2. **To-do management by voice** — add, list, re-status and delete tasks.

Everything runs locally: Whisper (STT) → MLX/Qwen (LLM) → Piper (TTS), with ChromaDB + BGE for RAG.

---

## 1. Components

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (frontend/index.html)                                      │
│  mic capture · wake word · playback · transcript · to-dos · uploads  │
└──────────────┬─────────────────────────────┬────────────────────────┘
               │ WebSocket /ws: audio+events │ HTTP: /todos /rag /stats
               ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  server.py (FastAPI) + api/ws_bridge.py                             │
│  audio in/out · control events · session recording · greeting        │
└──────────────┬──────────────────────────────────────────────────────┘
               ▼   (pipecat pipeline)
  BrowserAudioInput → VAD → Whisper STT → VoiceOrchestrator → sink
                                              │ (LLM + TTS run outside)
        ┌───────────────────┬─────────────────┼──────────────────┐
        ▼                   ▼                 ▼                  ▼
 services/mlx_llm.py  processors/       processors/         services/
 (Qwen: streamed      agent_tools.py    todo_store.py       rag_handler.py
  tool calls)         (4 to-do tools)   (JSON persistence)  piper_tts.py
```

The **VoiceOrchestrator** ([processors/voice_orchestrator.py](processors/voice_orchestrator.py)) is the brain. STT hands it text; it runs the LLM + TTS itself, outside the pipecat pipeline, so a reply stays cancellable mid-sentence.

---

## 2. One agent, one prompt — no modes

A single system prompt ([config.AGENT_SYSTEM_PROMPT](config.py)) and a single tool-calling loop. Four tools, all operating on the to-do list:

| tool | when |
|------|------|
| `create_todo(text)` | "remind me to…", "add … to my list" |
| `list_todos(status?)` | "what's on my list?" |
| `update_todo_status(id, status)` | "mark X done", "put X on hold" |
| `delete_todo(id)` | "remove X" |

Statuses: `pending`, `in_progress`, `hold`, `done`.

RAG is **not** a tool — relevant chunks are auto-injected (§4). The live to-do list is likewise injected every turn, so the model can map "mark the groceries one done" onto the right id.

```
user turn
    │
    ├─ auto-inject RAG chunks + current to-do list into the system prompt
    ▼
agent tool loop ──► tool calls? ──yes──► dispatch each, speak a fixed confirmation
    │                                    per result, emit todos_changed
    └──no──► stream the answer to TTS sentence-by-sentence
```

**Why tool confirmations are fixed strings, not model-written:** a 3B model narrating structured results is unreliable (it invents statuses and miscounts). `_describe_tool_result` renders them deterministically instead.

---

## 3. To-do storage

[processors/todo_store.py](processors/todo_store.py) — a lock-guarded JSON file (`todos.json`), shared by the agent and the REST API, so the voice path and the UI never diverge.

`resolve(ref)` accepts an id *or* a unique text substring, because small models often pass the task text where an id was asked for. An ambiguous substring resolves to nothing rather than guessing.

---

## 4. RAG (auto-injected, optional)

```
user turn → BGE embed → ChromaDB top-K chunks → keep distance ≤ threshold
         → inject as a [Reference] block in the system prompt
```

- Tuning in [config.py](config.py): `RAG_TOP_K = 3` (a definition can span a chunk split), `RAG_RELEVANCE_THRESHOLD = 0.5`.
- The prompt *prefers* the Reference but still answers from general knowledge when it doesn't cover the question — so uploading documents never turns the assistant into something that refuses ordinary questions.
- Embedding is symmetric (`encode()`, not `encode_queries()`): the asymmetric path pushed borderline matches over the threshold.

---

## 5. LLM layer ([services/mlx_llm.py](services/mlx_llm.py))

All MLX/Metal work runs on one dedicated thread (Metal command streams aren't thread-safe). Two generation paths:

| method | used by | output |
|--------|---------|--------|
| `stream_tool_turn` | agent loop | streams spoken sentences; returns `(text, tool_calls)` |
| `generate_text_sync` | conversation summary | plain text |

- **Streaming = low latency**: a normal answer is spoken sentence-by-sentence as it's written; a tool-call generation stays silent (detected by the `<tool_call>` prefix).
- **Tool-call parsing** tolerates both the JSON (Qwen2.5) and XML (Qwen3) shapes, plus the doubled-brace quirk some 4-bit Qwen builds emit.
- **Degeneration guards**: an exact-repeated sentence stops the stream, and `LLM_MAX_SENTENCES` caps output.

---

## 6. Turn-taking

- **Energy VAD** ([services/energy_vad.py](services/energy_vad.py)) calibrates a noise floor over the first ~2 s, then gates speech on it. Fresh per session.
- **Smart-turn analyser** (an ONNX model) decides whether a pause is really end-of-turn before flushing to the agent, with a 2 s fallback so a finished utterance can't hang.
- **Backchannel filter** ([processors/ack_classifier.py](processors/ack_classifier.py)) drops "uh huh"/"got it". Context words like "yes" count as filler *only* when the agent wasn't asking a question.
- **Barge-in**: a substantive utterance while the agent speaks cancels the reply immediately. Browser-side echo correlation ([frontend/correlator.js](frontend/correlator.js)) keeps the agent's own voice from self-triggering that.

## 7. Spoken output = on-screen text

`_clean_for_speech` strips markdown, maps Unicode punctuation to ASCII and drops non-ASCII drift. Each cleaned sentence goes to both TTS and the UI, and the final "complete" message is exactly the concatenation of what was spoken — so the transcript matches the audio.

---

## 8. Session recording

Turn-based ([processors/session_recorder.py](processors/session_recorder.py)): mic audio is buffered per turn (and not captured while the agent speaks, to avoid echo), committed as one block at the turn boundary, with a 300 ms gap on each speaker change — so the WAV plays back as clean alternating turns.
