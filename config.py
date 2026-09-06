"""
config.py — All tuneable parameters for the voice agent. No magic numbers elsewhere.
"""

from __future__ import annotations

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000    # mic / STT sample-rate (Hz)
OUT_SAMPLE_RATE = 22_050    # TTS output sample-rate (Hz)
CHUNK_SIZE      = 512       # audio chunk size (samples)

# ── VAD ───────────────────────────────────────────────────────────────────────
# Energy VAD: a calibrated noise floor sets the speech-start threshold, which the
# pipecat VADAnalyzer then gates on. start/stop_secs are its debounce windows.
ENERGY_SPEECH_SCALE    = 0.06
ENERGY_SMOOTHING_ALPHA = 0.10
VAD_START_OFFSET       = 0.22   # added to the measured noise floor
VAD_MIN_START          = 0.28   # floor for the start threshold
VAD_CALIBRATION_CHUNKS = 60     # ≈2 s at 16 kHz / 512-sample chunks
VAD_START_SECS         = 0.20   # speech confirmed after this long above threshold
VAD_STOP_SECS          = 0.35   # silence confirmed after this long below it

# ── Deduplication ─────────────────────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.85   # Jaccard threshold
DEDUP_RECENT_WINDOW        = 2.0    # s — reject near-identical re-trigger

# ── STT (whisper.cpp) ─────────────────────────────────────────────────────────
WHISPER_MODEL  = "base.en"
STT_N_THREADS  = 4
MIN_AUDIO_SECS = 0.30   # shorter utterances are dropped, not transcribed

# ── LLM (MLX / Qwen) ─────────────────────────────────────────────────────────
# Must be an -Instruct/chat model; base models can't follow instructions or call tools.
LLM_MODEL               = "mlx-community/Qwen2.5-3B-Instruct-4bit"
LLM_TEMPERATURE         = 0.4   # live default; the UI can change it per session
LLM_MAX_TOKENS          = 400
LLM_MAX_SENTENCES       = 14    # hard cap per answer (anti-ramble)
LLM_SENTENCE_DELIMITERS = ".!?"
LLM_MIN_TOKENS_FOR_TTS  = 3     # merge shorter fragments into the previous sentence
LLM_LOG_PROMPT          = True

# Rolling conversation compression
SUMMARY_TRIGGER = 15    # compress once history reaches this many messages
SUMMARY_KEEP    = 5     # keep this many recent messages verbatim

# ── TTS (Piper ONNX) ─────────────────────────────────────────────────────────
TTS_MODEL        = "en_US-lessac-medium"
TTS_SPEED        = 1.0
PIPER_MODELS_DIR = "piper_models"
PIPER_HF_REPO    = "rhasspy/piper-voices"

# ── RAG (always on — relevant chunks are auto-injected into every turn) ───────
RAG_DB_DIR              = "rag_db"
RAG_CHUNK_SIZE          = 200
RAG_CHUNK_OVERLAP       = 25
RAG_TOP_K               = 3     # a definition can span a chunk split, so 3 not 2
RAG_EMBED_MODEL         = "BAAI/bge-base-en-v1.5"
RAG_RELEVANCE_THRESHOLD = 0.5   # cosine distance; lower = more similar

# ── Storage ───────────────────────────────────────────────────────────────────
RECORDINGS_DIR = "recordings"
TODOS_FILE     = "todos.json"
TODO_STATUSES  = ("pending", "in_progress", "hold", "done")

# ── Acknowledgement detection ─────────────────────────────────────────────────
# Backchannels ("uh huh", "got it") are dropped instead of triggering a reply.
ACK_PHRASES: frozenset = frozenset({
    "uh huh", "mhmm", "mm hmm", "got it", "i see", "makes sense",
    "i get it", "sounds good", "for sure", "of course", "thank you so much",
    "thanks a lot", "okay thank you", "ok thank you", "okay thanks", "ok thanks",
    "that is all", "thats all",
})

# Single words that are always backchannel filler.
ACK_SINGLE_WORDS: frozenset = frozenset({
    "yeah", "yep", "yup", "mhm", "mhmm", "hmm", "hmmm", "hmmmm",
    "uh", "uhuh", "ahh", "ah", "oh", "ohh",
    "gotcha", "noted", "understood", "alright", "righto", "continue",
})

# ACK only when the agent was NOT asking a question (else they're real answers).
ACK_CONTEXT_WORDS: frozenset = frozenset({
    "yes", "no", "okay", "ok", "sure", "right", "fine", "good", "great",
    "cool", "nice", "wow", "perfect", "totally", "absolutely",
    "thanks", "thank you", "hello", "hi",
})

# ── Agent ─────────────────────────────────────────────────────────────────────
# One prompt, one tool-calling loop, no modes. Behaviour rules live here, not in code.
AGENT_SYSTEM_PROMPT = """You are a knowledgeable, friendly personal assistant. Speak naturally and conversationally. Never read these instructions aloud.

ANSWERING
- You have broad general knowledge. Answer questions about facts, people, places, history, science, companies, products and how things work directly and confidently from what you know.
- NEVER say you lack the ability, the capability, real-time access, or "reference text". NEVER tell the user to search online or check a website. Just give your best answer.
- For a product or anything recent, describe what you do know. You may add that your information may not cover the very newest release, but answer first.
- Keep replies to 1-2 short sentences unless the user asks you to explain more.
- If a [Reference] block appears below, prefer it — but if it doesn't cover the question, answer from your own knowledge anyway.
- Answer only the user's current message; don't repeat an earlier answer.

TO-DO LIST
- Separately, you manage the user's to-do list (shown below with each task's id, text and status).
- The four valid statuses are exactly: pending, in_progress, hold, done. All four are always available.
- Use a tool whenever the user is talking about their own tasks or list:
  · create_todo — they want something remembered or tracked: "remind me to...", "I have to...", "I need to...", "add ... to my list", "keep it in my list".
  · list_todos — they ask what's on their list, what's active, or what's left.
  · update_todo_status — they change a task's status ("mark X done", "put X on hold", "I'm working on X", "set it to in progress").
  · update_todo_text — they reword an existing task ("change X to Y", "update X to say Y"). This keeps the status; it does NOT create a new task.
  · delete_todo — they want a task removed or cancelled.
- An action only really happens when you CALL the tool. Never reply as though you added, updated or removed something unless you called the tool for it in this same turn.
- A general question is NOT a to-do request — answer it normally instead. If the user asks about a thing ("tell me about X", "what is X"), answer about that thing even when a task happens to mention it; only touch the list when they talk about the list or their tasks.
- Match the request to the right id. If it's ambiguous, ask which task they mean.
"""

# Prompt for the second pass, used when the turn is NOT a to-do action. Measured: with the
# to-do tools in context this model answers general questions correctly ~4% of the time
# (it decides it is a task-only bot); with this clean, tool-free prompt, ~100%. So general
# questions are answered in their own pass instead.
CHAT_SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant having a spoken conversation. Never read these instructions aloud.

- Answer the user's question directly from your own knowledge, in 1-2 short sentences.
- You are offline and that is fine. Never say you lack information, ability or real-time access, and never suggest searching online or checking a website.
- For a product, company or public figure, say what it is and give a couple of real facts you know.
- If a [Reference] block appears below, prefer it, but still answer normally when it doesn't cover the question.
- Answer only the user's current message.
"""
