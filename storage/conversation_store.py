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
from typing import List, Optional, Tuple
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

# Owner recorded for turns written while API-key auth is disabled. All unauthenticated
# callers share it, which preserves the pre-ownership behavior for local/dev setups.
ANONYMOUS_OWNER = "anonymous"

# Conversations written before ownership existed carry an empty owner. They are
# unclaimed: the next caller to write to one takes ownership, and until then any
# caller may read them. Once claimed, the owner check below is exclusive.
_UNCLAIMED_OWNER = ""


class ConversationAccessError(PermissionError):
    """Raised when a conversation_id is owned by a different caller.

    Conversation ids are client-supplied, so without this check any caller could
    read another caller's history by guessing or replaying an id.
    """


def _migrate_owner_column(conn: sqlite3.Connection) -> None:
    """Add the ``owner`` column to databases created before ownership existed.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so the column has
    to be added explicitly. Existing rows default to the unclaimed owner.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "owner" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN owner TEXT NOT NULL DEFAULT ''")


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
            last_turn_index INTEGER NOT NULL DEFAULT 0,
            owner           TEXT NOT NULL DEFAULT ''
        )
    """)
    _migrate_owner_column(conn)
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


def _stored_owner(conn: sqlite3.Connection, cid: str) -> Optional[str]:
    row = conn.execute("SELECT owner FROM conversations WHERE id = ?", (cid,)).fetchone()
    return None if row is None else str(row["owner"] or "")


def _assert_conversation_access(conn: sqlite3.Connection, cid: str, owner: str) -> None:
    """Raise :class:`ConversationAccessError` when ``owner`` may not use ``cid``."""
    stored = _stored_owner(conn, cid)
    if stored is None or stored == _UNCLAIMED_OWNER or stored == owner:
        return
    raise ConversationAccessError("conversation_id belongs to a different caller")


def normalize_conversation_id(conversation_id: Optional[str]) -> str:
    candidate = (conversation_id or "").strip()
    if candidate and _CONVERSATION_ID_PATTERN.match(candidate):
        return candidate
    return uuid4().hex


def append_conversation_turn(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    owner: str = ANONYMOUS_OWNER,
) -> int:
    """Persist one turn and return the assigned turn_index.

    Writing to a conversation claims it for ``owner``; writing to one already owned by
    someone else raises :class:`ConversationAccessError`.
    """
    cid = normalize_conversation_id(conversation_id)
    user_text = _trim_text(user_message)
    assistant_text = _trim_text(_sanitize_assistant_text(assistant_message))
    now = _utc_now()

    with _WRITE_LOCK:
        conn = _conn()
        _assert_conversation_access(conn, cid, owner)
        row = conn.execute(
            "SELECT last_turn_index FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        last_index = row["last_turn_index"] if row else 0
        turn_index = last_index + 1

        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, updated_at, last_turn_index, owner) "
            "VALUES (?, ?, ?, ?)",
            (cid, now, turn_index, owner),
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


def build_compact_context(
    conversation_id: Optional[str],
    owner: str = ANONYMOUS_OWNER,
) -> Tuple[Optional[str], str]:
    """Return normalized conversation_id and compact context block.

    Raises :class:`ConversationAccessError` when the id is owned by another caller,
    so history is never replayed to someone who merely guessed the id.
    """
    raw = (conversation_id or "").strip()
    if not raw:
        return normalize_conversation_id(None), ""
    cid = normalize_conversation_id(raw)

    conn = _conn()
    _assert_conversation_access(conn, cid, owner)
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
