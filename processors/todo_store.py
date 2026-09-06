"""
processors/todo_store.py — Persistent to-do list shared by the voice agent and the REST API.

A simple JSON-file-backed store: create/list/update-status/delete, plus a compact text
rendering injected into the LLM's system prompt every turn so it can resolve "mark X as
done"-style requests to the right item.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config import TODOS_FILE, TODO_STATUSES

DEFAULT_STATUS = TODO_STATUSES[0]  # "pending"


class TodoStore:
    def __init__(self, path: str = TODOS_FILE) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._todos: List[Dict] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text())
        except Exception as exc:
            logger.warning(f"[todos] failed to load {self._path}: {exc}")
            return []

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._todos, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning(f"[todos] failed to save {self._path}: {exc}")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, text: str) -> Dict:
        text = (text or "").strip()
        with self._lock:
            todo = {"id": uuid.uuid4().hex[:8], "text": text, "status": DEFAULT_STATUS}
            self._todos.append(todo)
            self._save()
            return dict(todo)

    def list(self, status: Optional[str] = None) -> List[Dict]:
        with self._lock:
            items = [dict(t) for t in self._todos]
        if status:
            items = [t for t in items if t["status"] == status]
        return items

    def get(self, todo_id: str) -> Optional[Dict]:
        with self._lock:
            for t in self._todos:
                if t["id"] == todo_id:
                    return dict(t)
        return None

    def update_status(self, todo_id: str, status: str) -> Optional[Dict]:
        if status not in TODO_STATUSES:
            return None
        return self._patch(todo_id, "status", status)

    def update_text(self, todo_id: str, text: str) -> Optional[Dict]:
        """Reword a task, keeping its id and status."""
        text = (text or "").strip()
        return self._patch(todo_id, "text", text) if text else None

    def _patch(self, todo_id: str, field: str, value: str) -> Optional[Dict]:
        with self._lock:
            for t in self._todos:
                if t["id"] == todo_id:
                    t[field] = value
                    self._save()
                    return dict(t)
        return None

    def delete(self, todo_id: str) -> bool:
        with self._lock:
            before = len(self._todos)
            self._todos = [t for t in self._todos if t["id"] != todo_id]
            changed = len(self._todos) != before
            if changed:
                self._save()
            return changed

    def resolve(self, ref: str) -> Optional[Dict]:
        """Resolve a todo by exact id, falling back to a case-insensitive substring match
        on its text — small models sometimes pass the task text instead of its id."""
        ref = (ref or "").strip()
        if not ref:
            return None
        hit = self.get(ref)
        if hit:
            return hit
        ref_low = ref.lower()
        with self._lock:
            candidates = [dict(t) for t in self._todos if ref_low in t["text"].lower()]
        return candidates[0] if len(candidates) == 1 else None

    # ── Prompt context ───────────────────────────────────────────────────────

    def context_block(self) -> str:
        """The list as prompt context. No ids — resolve() matches on text, and ids in the
        prompt end up being read aloud."""
        items = self.list()
        header = "[The user's tasks, for reference. Use list_todos to read them back.]"
        if not items:
            return header + "\nThe list is empty."
        return header + "\n" + "\n".join(
            f'- "{t["text"]}" — {t["status"]}' for t in items
        )
