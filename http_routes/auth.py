"""API-key authentication and caller identity for protected endpoints.

Authentication is opt-in: with ``RAG_API_KEYS`` unset every caller is the shared
:data:`ANONYMOUS_OWNER`, which is the behavior this API had before auth existed.
Setting any key turns enforcement on, and each key then maps to a distinct caller
id that scopes conversation history (see ``storage.conversation_store``).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

from fastapi import Header, HTTPException

from core_settings import api_keys
from storage.conversation_store import ANONYMOUS_OWNER


def _caller_id_for_key(key: str) -> str:
    """Stable, non-reversible caller id derived from the key.

    The key itself is never stored on conversation rows — a database copy must not
    leak credentials.
    """
    return "key:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _presented_key(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """Read the key from ``X-API-Key`` or an ``Authorization: Bearer`` header."""
    direct = (x_api_key or "").strip()
    if direct:
        return direct
    token = (authorization or "").strip()
    if token.lower().startswith("bearer "):
        return token[len("bearer "):].strip() or None
    return None


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """FastAPI dependency: authenticate the caller and return their caller id.

    The returned id is what conversation ownership is bound to, so an endpoint that
    touches conversation history must depend on this rather than reading headers itself.
    """
    configured = api_keys()
    if not configured:
        return ANONYMOUS_OWNER

    presented = _presented_key(x_api_key, authorization)
    if not presented:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Send X-API-Key: <key> or Authorization: Bearer <key>.",
        )
    for candidate in configured:
        if hmac.compare_digest(presented, candidate):
            return _caller_id_for_key(candidate)
    raise HTTPException(status_code=401, detail="Invalid API key.")
