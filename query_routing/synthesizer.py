"""Synthesis-only context builder for generation prompts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_recent_turns(conversation_context: str, max_turns: int = 2) -> List[str]:
    user_lines: List[str] = []
    assistant_lines: List[str] = []
    for raw in str(conversation_context or "").splitlines():
        line = raw.strip()
        if line.lower().startswith("user:"):
            user_lines.append(line)
        elif line.lower().startswith("assistant:"):
            assistant_lines.append(line)

    merged: List[str] = []
    # Keep latest compact turns only; avoid flooding prompt.
    pairs = max(1, int(max_turns))
    user_tail = user_lines[-pairs:]
    assistant_tail = assistant_lines[-pairs:]
    for item in user_tail:
        merged.append(item[:280])
    for item in assistant_tail:
        merged.append(item[:280])
    return merged


def _summarize_tool_results(tool_results: Optional[Dict[str, Any]]) -> str:
    payload = dict(tool_results or {})
    if not payload:
        return ""
    parts: List[str] = []
    answer = str(payload.get("answer") or "").strip()
    if answer:
        parts.append(f"- Primary findings: {answer[:420]}")
    time_window = payload.get("time_window")
    if isinstance(time_window, dict):
        label = str(time_window.get("label") or "").strip()
        if label:
            parts.append(f"- Time window: {label}")
    resolved_lab = str(payload.get("resolved_lab_name") or "").strip()
    if resolved_lab:
        parts.append(f"- Scope: {resolved_lab}")
    return "\n".join(parts)


def build_synthesis_context(
    tool_results: Optional[Dict[str, Any]],
    conversation_context: str,
    question: str,
) -> str:
    """Build generation prompt context with facts before prior conversation."""
    base_question = str(question or "").strip()
    blocks: List[str] = [base_question]

    facts = _summarize_tool_results(tool_results)
    if facts:
        blocks.append("Tool Results:\n" + facts)

    recent_turns = _extract_recent_turns(conversation_context, max_turns=2)
    if recent_turns:
        blocks.append("Prior Conversation (compressed):\n" + "\n".join(recent_turns))

    return "\n\n".join([item for item in blocks if item])
