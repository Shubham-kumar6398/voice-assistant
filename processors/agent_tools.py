"""
processors/agent_tools.py — Tools the single LLM agent can call.

The agent runs one tool-calling loop with no hardcoded modes. Its tools manage the
user's to-do list: create_todo, list_todos, update_todo_status, delete_todo. Tool
results are returned as compact JSON strings; the orchestrator turns them into a short
spoken confirmation (wording is deterministic, not model-generated — a 3B model is not
reliable enough to narrate structured data itself).
"""

from __future__ import annotations

import json
from typing import Dict, List

from loguru import logger

from config import TODO_STATUSES
from processors.todo_store import TodoStore


def tool_specs() -> List[Dict]:
    """The to-do management tools the agent can call. The current list is injected into
    the system prompt every turn, so the agent can resolve requests to the right id."""
    return [
        {
            "type": "function",
            "function": {
                "name": "create_todo",
                "description": "Add a new task to the user's to-do list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "the task, in the user's own words"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_todos",
                "description": (
                    "List the user's to-do items. Call this when they ask what's on "
                    "their list, what's active, or what they still need to do."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": list(TODO_STATUSES) + ["all"],
                            "description": "filter by status, or 'all' for everything (default: all)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_todo_status",
                "description": "Change the status of one existing to-do item.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "the item's id, from the to-do list above"},
                        "status": {"type": "string", "enum": list(TODO_STATUSES)},
                    },
                    "required": ["id", "status"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_todo_text",
                "description": (
                    "Reword an existing task, keeping its status. Use this when the user "
                    "wants to change WHAT a task says rather than its status."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "the existing task, by id or its current wording"},
                        "text": {"type": "string", "description": "the new wording"},
                    },
                    "required": ["id", "text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_todo",
                "description": "Remove a to-do item entirely.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "the item's id, from the to-do list above"},
                    },
                    "required": ["id"],
                },
            },
        },
    ]


class AgentTools:
    """Executes the agent's to-do tool calls against the shared TodoStore."""

    def __init__(self, todo_store: TodoStore) -> None:
        self._todos = todo_store

    def todos_context(self) -> str:
        """Live to-do list injected into the prompt every turn."""
        return self._todos.context_block()

    # ── Dispatch ───────────────────────────────────────────────────────────────

    async def dispatch(self, name: str, arguments: Dict) -> str:
        arguments = arguments or {}
        try:
            if name == "create_todo":
                return self._create(arguments)
            if name == "list_todos":
                return self._list(arguments)
            if name == "update_todo_status":
                return self._update_status(arguments)
            if name == "update_todo_text":
                return self._update_text(arguments)
            if name == "delete_todo":
                return self._delete(arguments)
            return _err(f"Unknown tool '{name}'.")
        except Exception as exc:
            logger.exception(f"[tools] {name} failed: {exc}")
            return _err(f"Tool '{name}' failed: {exc}")

    # ── Tools ──────────────────────────────────────────────────────────────────

    def _create(self, args: Dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            return _err("No task text given.")
        todo = self._todos.create(text)
        return _json({"ok": True, "action": "created", "todo": todo})

    def _list(self, args: Dict) -> str:
        status = (args.get("status") or "").strip().lower()
        items = self._todos.list(None if status in ("", "all") else status)
        return _json({"ok": True, "action": "listed", "todos": items})

    def _update_status(self, args: Dict) -> str:
        status = (args.get("status") or "").strip().lower()
        if status not in TODO_STATUSES:
            return _err(f"Status must be one of {TODO_STATUSES}.")
        ref = (args.get("id") or args.get("text") or "").strip()
        todo = self._todos.resolve(ref)
        if not todo:
            return _err(f"No matching to-do for '{ref}'.")
        updated = self._todos.update_status(todo["id"], status)
        return _json({"ok": True, "action": "updated", "todo": updated})

    def _update_text(self, args: Dict) -> str:
        new_text = (args.get("text") or "").strip()
        if not new_text:
            return _err("No new wording given.")
        ref = (args.get("id") or "").strip()
        todo = self._todos.resolve(ref)
        if not todo:
            return _err(f"No matching to-do for '{ref}'.")
        old_text = todo["text"]
        updated = self._todos.update_text(todo["id"], new_text)
        return _json({"ok": True, "action": "reworded", "todo": updated, "was": old_text})

    def _delete(self, args: Dict) -> str:
        ref = (args.get("id") or args.get("text") or "").strip()
        todo = self._todos.resolve(ref)
        if not todo:
            return _err(f"No matching to-do for '{ref}'.")
        self._todos.delete(todo["id"])
        return _json({"ok": True, "action": "deleted", "todo": todo})


def _json(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg})
