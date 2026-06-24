"""Ollama request/response helpers shared by router and executors."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core_settings import ollama_thinking, router_thinking


def _chunk_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_generate_chunk(event: Dict[str, Any], *, thinking: Optional[bool] = None) -> str:
    """Incremental token from a streaming /api/generate event (must not strip)."""
    use_thinking = ollama_thinking() if thinking is None else thinking
    response = _chunk_text(event.get("response"))
    if response or not use_thinking:
        return response
    return _chunk_text(event.get("thinking"))


def extract_generate_text(event: Dict[str, Any], *, thinking: Optional[bool] = None) -> str:
    """Full text from a non-streaming /api/generate response."""
    return extract_generate_chunk(event, thinking=thinking).strip()


def extract_chat_content(message: Dict[str, Any], *, thinking: Optional[bool] = None) -> str:
    """Read visible text from an Ollama /api/chat message object."""
    use_thinking = router_thinking() if thinking is None else thinking
    content = _chunk_text(message.get("content")).strip()
    if content or not use_thinking:
        return content
    return _chunk_text(message.get("thinking")).strip()
