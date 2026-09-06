# Voice Assistant

An offline, on-device voice assistant. Talk to it in the browser — it chats, answers
questions about documents you upload (RAG), and manages a to-do list, all by voice.
No cloud APIs: speech-to-text, the language model, and text-to-speech all run locally.

```
you speak → Whisper (STT) → Qwen LLM (MLX) → Piper (TTS) → it speaks back
```

## What it does

- **Conversation** — ask it anything; it answers from its own knowledge, spoken back
  in real time (sentence-by-sentence streaming, so it starts talking before it's
  finished thinking).
- **Ask about your documents** — upload a PDF/.docx/.txt and it's chunked, embedded
  and stored in a local vector DB. Relevant chunks are pulled into context automatically
  whenever you ask something they might answer.
- **To-do list by voice** — "remind me to call the plumber", "what's on my list?",
  "mark the groceries done", "remove that" — four tools cover create/list/update/delete,
  backed by a small JSON store shared with a REST API and the web UI.
- **Natural turn-taking** — it detects when you've actually stopped talking (not just
  paused), lets you interrupt it mid-sentence, and ignores "uh huh" / "got it" style
  backchannel without treating it as a new turn.
- **Wake word** — say "Tbook" (or "hey book") and it starts listening on its own, no
  click needed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and the
reasoning behind the trickier decisions.

## Requirements

- **macOS on Apple Silicon** — the LLM runs on [MLX](https://github.com/ml-explore/mlx),
  which is Metal-only. Everything else is plain Python and would port to other
  platforms, but the LLM layer as written won't.
- Python 3.10+
- A working microphone, and a browser that supports `getUserMedia` (Chrome/Edge/Safari).
- For the wake word specifically: a browser with the Web Speech API (Chrome/Edge). Note
  that this one piece isn't on-device — Chrome's built-in speech recognition sends audio
  to Google to detect the wake word. Everything after a session starts (STT, LLM, TTS)
  stays fully local.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then just run the server:

```bash
python server.py
```

Open **http://localhost:8000** in your browser and allow microphone access when prompted.

That's it — no API keys, no config needed. On first run it downloads what it needs and
caches it for next time:

| model | size | purpose |
|---|---|---|
| `Qwen2.5-3B-Instruct-4bit` | ~2 GB | the LLM (chat + tool calling) |
| whisper `base.en` | ~140 MB | speech-to-text |
| Piper `en_US-lessac-medium` | ~60 MB | text-to-speech |
| `BAAI/bge-base-en-v1.5` | ~430 MB | embeddings for RAG |

First launch will take a few minutes while these download; after that, startup is fast.

## Using it

- Click the mic, or just say **"Tbook"** (or "hey book") to start a session hands-free —
  the page listens for the wake word in the background whenever it's idle.
- Once a session is running, just talk; speech start/end is detected automatically, no
  push-to-talk needed.
- Say things like *"add buy milk to my list"*, *"what's pending?"*, or *"mark buy milk
  as done"* to manage to-dos; anything else is treated as a normal question.
- Drag a document onto the upload area to make it part of what the assistant can
  reference when answering.
- The transcript, live to-do list, and a stats panel (CPU/RAM) are all in the same page.

## Project layout

```
server.py             FastAPI app: static files, REST routes, /ws websocket
pipeline_factory.py   Wires one pipecat pipeline per websocket session
config.py             Every tunable — models, thresholds, prompts
api/                  REST routes + the websocket <-> pipeline bridge
processors/           Agent loop, to-do storage, turn-taking helpers, session recording
services/             STT (whisper.cpp), LLM (MLX), TTS (Piper), RAG (ChromaDB)
frontend/             Single-page UI (vanilla HTML/JS, no build step)
```

## Notes

- Everything is local and stateful: to-dos live in `todos.json`, uploaded documents in
  `rag_db/`, session audio in `recordings/`. None of that is meant to be committed —
  see `.gitignore`.
