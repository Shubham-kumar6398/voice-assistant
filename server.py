"""
server.py — FastAPI entry point: static files, REST routes and the voice WebSocket.
"""

from __future__ import annotations

import gc
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from loguru import logger

from api.routes import register_rag_routes, register_todo_routes, router as rest_router
from config import RECORDINGS_DIR
from pipeline_factory import SharedResources, create_session

os.makedirs(RECORDINGS_DIR, exist_ok=True)

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")
_shared: SharedResources | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared
    _shared = SharedResources()
    yield
    if _shared:
        await _shared.shutdown()
    gc.collect()


app = FastAPI(title="Voice Agent (Pipecat)", lifespan=lifespan)
app.include_router(rest_router)


# ── Static files ──────────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    with open(os.path.join(_FRONTEND, "index.html")) as f:
        return HTMLResponse(f.read())


@app.get("/correlator.js")
async def get_correlator():
    with open(os.path.join(_FRONTEND, "correlator.js")) as f:
        return Response(f.read(), media_type="application/javascript")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"[ws] Client connected - session: {session_id}")

    session = create_session(ws, _shared, session_id)
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info(f"[ws] Client disconnected: {session_id}")
    except Exception as exc:
        logger.error(f"[ws] Session error: {exc}")
    finally:
        logger.info(f"[ws] Session ended: {session_id}")


# ── Routes needing shared state (registered before it exists, so resolved lazily) ──

class _Lazy:
    """Forwards attribute access to one field of the shared resources at call time."""

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _target(self):
        return getattr(_shared, self._attr) if _shared else None

    def __getattr__(self, name):
        target = self._target()
        return getattr(target, name) if target else None

    def __bool__(self):
        return self._target() is not None


register_rag_routes(app, _Lazy("rag"))
register_todo_routes(app, _Lazy("todos"))


if __name__ == "__main__":
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_graceful_shutdown=1,
    ))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("[server] Stopped.")
        os._exit(0)
