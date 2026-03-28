"""Bounded conversation context with in-memory cache and JSON disk persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


_STORE_LOCK = Lock()
_MEMORY_STORE: Dict[str, Any] = {}
_STORE_LOADED = False

_STORE_FILE_PATH = Path(
    os.getenv(
        "CONVERSATION_STORE_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "conversation_turns.json"),
    )
)

_MAX_TURNS_PER_CONVERSATION = max(4, int(os.getenv("CONVERSATION_MAX_TURNS", "24")))
_RECENT_TURNS_FOR_CONTEXT = max(2, int(os.getenv("CONVERSATION_CONTEXT_TURNS", "12")))
_MAX_CONTEXT_CHARS = max(400, int(os.getenv("CONVERSATION_CONTEXT_MAX_CHARS", "4000")))
_MAX_MESSAGE_CHARS = max(300, int(os.getenv("CONVERSATION_MESSAGE_MAX_CHARS", "2000")))

_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(value: str, max_chars: int = _MAX_MESSAGE_CHARS) -> str:
    return (value or "").strip()[:max_chars]


def _sanitize_assistant_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    # Remove legacy forced labels from older prompt variants.
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


def _read_store() -> Dict[str, Any]:
    return _MEMORY_STORE


def _write_store(payload: Dict[str, Any]) -> None:
    global _MEMORY_STORE  # pylint: disable=global-statement
    _MEMORY_STORE = payload


def _load_store_from_disk() -> None:
    global _STORE_LOADED  # pylint: disable=global-statement
    if _STORE_LOADED:
        return

    _STORE_LOADED = True
    try:
        _STORE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _STORE_FILE_PATH.exists():
            _STORE_FILE_PATH.write_text("{}\n", encoding="utf-8")
            _write_store({})
            return

        raw = _STORE_FILE_PATH.read_text(encoding="utf-8").strip() or "{}"
        parsed = json.loads(raw)
        _write_store(parsed if isinstance(parsed, dict) else {})
    except Exception:
        _write_store({})


def _persist_store_to_disk(payload: Dict[str, Any]) -> None:
    _STORE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_FILE_PATH.write_text(f"{json.dumps(payload, ensure_ascii=True, indent=2)}\n", encoding="utf-8")


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

    with _STORE_LOCK:
        _load_store_from_disk()
        store = _read_store()
        bucket = store.get(cid) if isinstance(store.get(cid), dict) else {}
        turns = bucket.get("turns") if isinstance(bucket.get("turns"), list) else []
        last_turn_index = int(bucket.get("last_turn_index") or 0)
        turn_index = last_turn_index + 1

        turns.append(
            {
                "turn_index": turn_index,
                "timestamp": _utc_now(),
                "user": user_text,
                "assistant": assistant_text,
            }
        )
        if len(turns) > _MAX_TURNS_PER_CONVERSATION:
            turns = turns[-_MAX_TURNS_PER_CONVERSATION:]

        store[cid] = {
            "updated_at": _utc_now(),
            "last_turn_index": turn_index,
            "turns": turns,
        }
        _write_store(store)
        _persist_store_to_disk(store)

    return turn_index


def build_compact_context(conversation_id: Optional[str]) -> Tuple[Optional[str], str]:
    """Return normalized conversation_id and compact context block."""
    raw = (conversation_id or "").strip()
    if not raw:
        return normalize_conversation_id(None), ""
    cid = normalize_conversation_id(raw)

    with _STORE_LOCK:
        _load_store_from_disk()
        store = _read_store()

    bucket = store.get(cid)
    if not isinstance(bucket, dict):
        return cid, ""

    turns = bucket.get("turns")
    if not isinstance(turns, list) or not turns:
        return cid, ""

    recent_turns: List[Dict[str, Any]] = [item for item in turns if isinstance(item, dict)][
        -_RECENT_TURNS_FOR_CONTEXT:
    ]

    lines: List[str] = []
    for turn in recent_turns:
        user_text = _trim_text(str(turn.get("user") or ""), max_chars=320)
        assistant_text = _trim_text(str(turn.get("assistant") or ""), max_chars=320)
        assistant_text = _sanitize_assistant_text(assistant_text)
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
