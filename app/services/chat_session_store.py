"""On-disk store for saved Claude chat sessions.

Mirrors the saved-board / saved-script convention (``state/boards`` ->
``state/scripts`` -> ``state/chat_sessions``): each conversation is one
``<id>.json`` file holding its metadata and full Anthropic message history, so a
chat can be listed, resumed or deleted later without keeping it in memory.

Session IDs are generated server-side (``uuid4().hex``) and validated against a
strict 32-hex-char pattern so a client-supplied id can never escape the
directory via path traversal.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "state" / "chat_sessions"
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TITLE_MAX = 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_USAGE_KEYS = ("input", "output", "cache_write", "cache_read", "cost")


def _add_usage(existing: dict | None, delta: dict) -> dict:
    """Return the element-wise sum of two usage dicts (tokens and cost).

    ``cost`` may be None for an unpriced model; a None on either side contributes
    zero, so a session that mixes priced and unpriced turns reports the cost of
    only the priced ones rather than erroring.
    """
    base = existing or {}
    return {key: (base.get(key) or 0) + (delta.get(key) or 0) for key in _USAGE_KEYS}


def derive_title(message: str) -> str:
    """Build a short, single-line session title from the first user message."""
    collapsed = " ".join(str(message or "").split())
    if not collapsed:
        return "New conversation"
    if len(collapsed) > _TITLE_MAX:
        return collapsed[: _TITLE_MAX - 1].rstrip() + "…"
    return collapsed


class ChatSessionStore:
    """Persist and retrieve chat sessions as ``<id>.json`` files on disk."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._dir = Path(os.getenv("CHAT_SESSIONS_DIR", str(sessions_dir or _DEFAULT_DIR)))
        self._lock = threading.Lock()

    @staticmethod
    def sanitize_id(session_id: str) -> str:
        """Validate and return ``session_id``, raising ValueError if malformed."""
        candidate = str(session_id or "").strip().lower()
        if not _ID_RE.match(candidate):
            raise ValueError("session id must be 32 hex characters")
        return candidate

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{self.sanitize_id(session_id)}.json"

    def create(self, *, title: str, model: str | None) -> dict:
        """Create and persist a new, empty session; return its record."""
        session_id = uuid.uuid4().hex
        now = _now_iso()
        record: dict[str, Any] = {
            "id": session_id,
            "title": derive_title(title),
            "model": model,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._write(record)
        return record

    def save(
        self,
        session_id: str,
        *,
        messages: list[dict],
        title: str | None = None,
        model: str | None = None,
        usage: dict | None = None,
    ) -> dict:
        """Update a session's messages (and optionally title/model/usage) in place.

        ``usage`` is one turn's token/cost totals; it is *added* onto any usage
        already recorded so the stored figure is the running session total.
        """
        with self._lock:
            record = self._read(session_id) or {
                "id": self.sanitize_id(session_id),
                "title": derive_title(title or ""),
                "model": model,
                "created_at": _now_iso(),
            }
            record["messages"] = messages
            if title is not None:
                record["title"] = derive_title(title)
            if model is not None:
                record["model"] = model
            if usage is not None:
                record["usage"] = _add_usage(record.get("usage"), usage)
            record["updated_at"] = _now_iso()
            self._write(record, lock=False)
            return self._summary(record)

    def load(self, session_id: str) -> dict | None:
        """Return the full session record for ``session_id``, or None if absent."""
        with self._lock:
            return self._read(session_id)

    def list_summaries(self) -> list[dict]:
        """Return lightweight summaries of all sessions, newest first."""
        if not self._dir.exists():
            return []
        summaries: list[dict] = []
        with self._lock:
            for entry in self._dir.glob("*.json"):
                try:
                    record = json.loads(entry.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                summaries.append(self._summary(record))
        summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return summaries

    def rename(self, session_id: str, title: str) -> dict | None:
        """Rename a session; return the updated summary, or None if absent."""
        with self._lock:
            record = self._read(session_id)
            if record is None:
                return None
            record["title"] = derive_title(title)
            record["updated_at"] = _now_iso()
            self._write(record, lock=False)
            return self._summary(record)

    def delete(self, session_id: str) -> bool:
        """Delete the session ``session_id``; return whether it existed."""
        path = self._path(session_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _summary(record: dict) -> dict:
        """Project a full record down to list-friendly metadata (no messages)."""
        messages = record.get("messages") or []
        # Count only the human-visible user turns (string content), so tool
        # results that ride along as user-role messages don't inflate the count.
        turns = sum(
            1 for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        return {
            "id": record.get("id"),
            "title": record.get("title") or "New conversation",
            "model": record.get("model"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "message_count": turns,
            "usage": record.get("usage"),
        }

    def _read(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, record: dict, *, lock: bool = True) -> None:
        def _do() -> None:
            path = self._path(record["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)

        if lock:
            with self._lock:
                _do()
        else:
            _do()
