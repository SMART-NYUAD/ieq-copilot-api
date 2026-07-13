"""Ollama request/response helpers shared by router and executors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from core_settings import (
    ollama_base_url,
    ollama_model,
    ollama_thinking,
    ollama_timeout_seconds,
    router_thinking,
)


def _chunk_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def coerce_chunk_text(value: Any) -> str:
    """Flatten an Ollama/LangChain message ``content`` (str, list of parts, or dict) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def build_prompt_text_from_messages(messages: List[Any]) -> str:
    """Convert LangChain-style message objects to a plain prompt string."""
    prompt_parts = []
    for message in messages:
        role = getattr(message, "type", "user").upper()
        content = coerce_chunk_text(getattr(message, "content", ""))
        prompt_parts.append(f"{role}:\n{content}")
    return "\n\n".join(prompt_parts)


def generate_ollama_text(prompt_text: str, *, temperature: float) -> str:
    """Non-streaming /api/generate call against the answer model, returning the full text."""
    payload: Dict[str, Any] = {
        "model": ollama_model(),
        "prompt": prompt_text,
        "stream": False,
        "think": ollama_thinking(),
        "temperature": temperature,
    }
    with httpx.Client(timeout=ollama_timeout_seconds()) as client:
        response = client.post(f"{ollama_base_url()}/api/generate", json=payload)
        response.raise_for_status()
        event = response.json()
    return extract_generate_text(event)


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
