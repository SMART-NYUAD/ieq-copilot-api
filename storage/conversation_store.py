"""Bounded conversation context with SQLite persistence.

Thread model: WAL-mode SQLite with per-thread connections for reads;
a module-level write lock serialises INSERT/UPDATE/DELETE so each
conversation write is atomic without blocking concurrent readers.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


_WRITE_LOCK = threading.Lock()
_local = threading.local()

_DB_PATH = Path(
    os.getenv(
        "CONVERSATION_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "conv.db"),
    )
)

_MAX_TURNS_PER_CONVERSATION = max(4, int(os.getenv("CONVERSATION_MAX_TURNS", "24")))
_RECENT_TURNS_FOR_CONTEXT = max(2, int(os.getenv("CONVERSATION_CONTEXT_TURNS", "12")))
_MAX_CONTEXT_CHARS = max(400, int(os.getenv("CONVERSATION_CONTEXT_MAX_CHARS", "4000")))
_MAX_MESSAGE_CHARS = max(300, int(os.getenv("CONVERSATION_MESSAGE_MAX_CHARS", "2000")))
_MAX_CONVERSATIONS = max(50, int(os.getenv("CONVERSATION_MAX_STORED", "500")))

_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _open_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id              TEXT PRIMARY KEY,
            updated_at      TEXT NOT NULL,
            last_turn_index INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            conversation_id TEXT    NOT NULL,
            turn_index      INTEGER NOT NULL,
            ts              TEXT    NOT NULL,
            user            TEXT    NOT NULL DEFAULT '',
            assistant       TEXT    NOT NULL DEFAULT '',
            PRIMARY KEY (conversation_id, turn_index)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at)"
    )
    conn.commit()
    return conn


def _conn() -> sqlite3.Connection:
    """Return this thread's SQLite connection, opening it if needed."""
    c = getattr(_local, "conn", None)
    if c is None:
        _local.conn = _open_connection()
    return _local.conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(value: str, max_chars: int = _MAX_MESSAGE_CHARS) -> str:
    return (value or "").strip()[:max_chars]


def _sanitize_assistant_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"^\s*General explanation\s*\(not site-specific policy\)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\n?\s*Note:\s*Without measured data, this is a general educational explanation\.\s*For site-specific guidance, real-time measurements are required\.\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _evict_oldest_conversations(conn: sqlite3.Connection) -> None:
    """Delete oldest conversations (by updated_at) when over the cap."""
    count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    overflow = count - _MAX_CONVERSATIONS
    if overflow <= 0:
        return
    old_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM conversations ORDER BY updated_at ASC LIMIT ?",
            (overflow,),
        ).fetchall()
    ]
    if not old_ids:
        return
    placeholders = ",".join("?" * len(old_ids))
    conn.execute(f"DELETE FROM turns WHERE conversation_id IN ({placeholders})", old_ids)
    conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", old_ids)


def normalize_conversation_id(conversation_id: Optional[str]) -> str:
    candidate = (conversation_id or "").strip()
    if candidate and _CONVERSATION_ID_PATTERN.match(candidate):
        return candidate
    return uuid4().hex


def append_conversation_turn(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
) -> int:
    """Persist one turn and return the assigned turn_index."""
    cid = normalize_conversation_id(conversation_id)
    user_text = _trim_text(user_message)
    assistant_text = _trim_text(_sanitize_assistant_text(assistant_message))
    now = _utc_now()

    with _WRITE_LOCK:
        conn = _conn()
        row = conn.execute(
            "SELECT last_turn_index FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        last_index = row["last_turn_index"] if row else 0
        turn_index = last_index + 1

        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, updated_at, last_turn_index) VALUES (?, ?, ?)",
            (cid, now, turn_index),
        )
        conn.execute(
            "INSERT OR REPLACE INTO turns (conversation_id, turn_index, ts, user, assistant) VALUES (?, ?, ?, ?, ?)",
            (cid, turn_index, now, user_text, assistant_text),
        )
        # Trim turns that exceed the per-conversation cap
        conn.execute(
            """DELETE FROM turns
               WHERE conversation_id = ?
                 AND turn_index <= (
                     SELECT MAX(turn_index) - ? FROM turns WHERE conversation_id = ?
                 )""",
            (cid, _MAX_TURNS_PER_CONVERSATION, cid),
        )
        _evict_oldest_conversations(conn)
        conn.commit()

    return turn_index


def build_compact_context(conversation_id: Optional[str]) -> Tuple[Optional[str], str]:
    """Return normalized conversation_id and compact context block."""
    raw = (conversation_id or "").strip()
    if not raw:
        return normalize_conversation_id(None), ""
    cid = normalize_conversation_id(raw)

    conn = _conn()
    rows = conn.execute(
        """SELECT user, assistant FROM turns
           WHERE conversation_id = ?
           ORDER BY turn_index DESC
           LIMIT ?""",
        (cid, _RECENT_TURNS_FOR_CONTEXT),
    ).fetchall()

    if not rows:
        return cid, ""

    lines: List[str] = []
    for row in reversed(rows):  # oldest first
        user_text = _trim_text(str(row["user"] or ""), max_chars=320)
        assistant_text = _trim_text(
            _sanitize_assistant_text(str(row["assistant"] or "")), max_chars=320
        )
        if not user_text and not assistant_text:
            continue
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")

    if not lines:
        return cid, ""

    block = "Previous conversation context (most recent last):\n" + "\n".join(lines)
    return cid, block[:_MAX_CONTEXT_CHARS]
