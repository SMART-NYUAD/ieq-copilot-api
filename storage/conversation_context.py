"""Canonical per-turn conversation context built once at the HTTP boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core_settings import default_stakeholder_role
from prompting.roles import ROLE_DEFAULT, normalize_role
from storage.conversation_store import ANONYMOUS_OWNER, build_compact_context
from storage.conversation_memory import apply_routing_memory, compute_question_signals, extract_routing_memory

_ROUTING_SNIPPET_LINES = 8   # lines fed to the router LLM (≈ 4 prior turns for follow-up resolution)
_LLM_HISTORY_LINES = 6        # lines fed to the answer LLM (≈ 3 prior turns)
_LLM_HISTORY_MAX_CHARS = 800  # budget for those lines


def extract_context_lines(raw_block: str) -> list[str]:
    """Return the conversation content lines from a compact context block, dropping the
    ``Previous conversation context…`` header and blank lines.

    Shared by the context builder and the router so the snippet fed to the LLM is derived
    one way only.
    """
    return [
        line for line in str(raw_block or "").strip().splitlines()
        if line.strip() and not line.startswith("Previous conversation context")
    ]


_MAX_TRANSCRIPT_LINE_CHARS = 320  # matches the per-message trim in conversation_store

# Tokens that would let transcript content impersonate prompt structure. Prior turns are
# replayed into both LLM prompts, and a turn's text is fully user-influenced — directly for
# "User:" lines, and indirectly for "Assistant:" lines, since a user can ask the model to
# repeat any text back. Neutralizing the markers is defence in depth alongside the
# instruction in the prompt itself; it is not a claim that injection is fully solved.
_PROMPT_STRUCTURE_TOKENS = (
    "```",
    "<<<",
    ">>>",
    "### ",
    "## ",
    "system:",
    "assistant:",
    "user:",
    "prior conversation",
    "end transcript",
)


def sanitize_transcript_line(line: str) -> str:
    """Flatten one transcript line so it cannot pose as prompt structure.

    Keeps the leading ``User:``/``Assistant:`` speaker label — the models rely on it to
    tell the turns apart — and neutralizes the same markers anywhere else in the text.
    """
    text = str(line or "")
    speaker = ""
    for label in ("User:", "Assistant:"):
        if text.startswith(label):
            speaker, text = label + " ", text[len(label):]
            break

    # Collapse newlines/tabs and other control characters into spaces.
    text = "".join(" " if ch < " " or ch == "\x7f" else ch for ch in text)
    lowered = text.lower()
    for token in _PROMPT_STRUCTURE_TOKENS:
        start = 0
        while True:
            idx = lowered.find(token, start)
            if idx < 0:
                break
            text = text[:idx] + ("·" * len(token)) + text[idx + len(token):]
            lowered = text.lower()
            start = idx + len(token)
    text = " ".join(text.split())
    if not any(ch.isalnum() for ch in text):
        # Nothing but structure markers survived; the line carries no reference to resolve.
        return ""
    return (speaker + text).strip()[:_MAX_TRANSCRIPT_LINE_CHARS]


def sanitize_transcript_lines(lines: list[str]) -> list[str]:
    """Sanitize every transcript line, dropping any that reduce to nothing."""
    return [cleaned for cleaned in (sanitize_transcript_line(line) for line in lines) if cleaned]


def fit_lines_to_budget(lines: list[str], max_chars: int) -> str:
    """Join the most recent lines that fit in ``max_chars``, dropping whole lines.

    A flat character slice cuts mid-sentence, handing the model a fragment whose meaning
    can differ from the turn it came from ("CO2 was high in the mo…"). Dropping the oldest
    line instead keeps every line the model sees intact.
    """
    kept: list[str] = []
    budget = max_chars
    for line in reversed(lines):
        # +1 for the newline joining this line to the next.
        cost = len(line) + (1 if kept else 0)
        if cost > budget:
            break
        kept.append(line)
        budget -= cost
    return "\n".join(reversed(kept))


@dataclass(frozen=True)
class ConversationContext:
    """Single canonical conversation context created once per HTTP turn.

    All downstream components (router, DB executor, answer LLM) receive the
    same object and read the view they need — nothing reconstructs context
    from the raw string independently.

    Carry-over values (``carried_metric``, ``carried_time_phrase``) are passed
    as structured data to executors via ``planner_hints``; they are never
    appended to ``effective_question`` so the LLM always sees the clean user
    question.
    """

    conversation_id: str
    original_question: str    # As typed by the user (for turn persistence)
    raw_block: str            # Full "Previous conversation context…" string
    effective_question: str   # Clean question — no appended carry-over text
    effective_lab: Optional[str]  # Lab resolved from memory or request
    routing_snippet: str      # Compact last-N-lines for the router LLM
    llm_history: str          # Compact block injected into answer LLM context
    carried_metric: Optional[str] = None       # Metric inferred from prior turns
    carried_time_phrase: Optional[str] = None  # Time window inferred from prior turns
    # Who the answer is being written for. Unlike everything above, this is not derived
    # from the question or the history — it is declared by the caller (see prompting.roles)
    # and resolved here so no downstream layer has to re-derive it.
    role: str = ROLE_DEFAULT
    role_source: str = "default"       # request | default
    role_fallback_used: bool = False   # caller sent a role we did not recognise


def _resolve_role(requested_role: Optional[str]) -> tuple[str, str, bool]:
    """Resolve the stakeholder role for this turn: the request, or the configured default.

    Deliberately stateless. An earlier design inherited the conversation's last-used role
    when the field was omitted, which protected a client that forgot to send it — but it
    made an omitted field mean something different on turn 5 than on turn 1, so the same
    request body could produce two differently-shaped answers depending on history. Role
    is a per-message choice; nothing about it is remembered.
    """
    role, fallback_used = normalize_role(requested_role)
    if role:
        return role, "request", fallback_used
    return default_stakeholder_role(), "default", fallback_used


def build_conversation_context(
    question: str,
    lab_name: Optional[str],
    conversation_id: Optional[str],
    owner: str = ANONYMOUS_OWNER,
    role: Optional[str] = None,
) -> ConversationContext:
    """Build the canonical context for one turn.

    Loads prior turns from the store, applies memory carry-over, and
    pre-computes every view needed downstream so no layer does its own
    extraction. ``owner`` is the authenticated caller id; loading history for a
    conversation owned by someone else raises ``ConversationAccessError``.

    ``role`` is the caller's declared stakeholder role for this message, or ``None`` for
    the configured default. It is not remembered between turns.
    """
    original_question = str(question or "").strip()
    cid, raw_block = build_compact_context(conversation_id, owner=owner)
    resolved_role, role_source, role_fallback_used = _resolve_role(role)

    # Sanitize once, here, so every prompt view is built from neutralized text — a caller
    # that formats its own view cannot forget to do it.
    content_lines = sanitize_transcript_lines(extract_context_lines(raw_block))
    routing_snippet = "\n".join(content_lines[-_ROUTING_SNIPPET_LINES:])
    llm_history = fit_lines_to_budget(content_lines[-_LLM_HISTORY_LINES:], _LLM_HISTORY_MAX_CHARS)

    carried_metric: Optional[str] = None
    carried_time_phrase: Optional[str] = None

    if raw_block:
        signals = compute_question_signals(original_question)
        memory = extract_routing_memory(conversation_context=raw_block, current_signals=signals)
        effective_question, effective_lab, carry_info = apply_routing_memory(
            question=original_question,
            lab_name=lab_name,
            memory=memory,
            current_signals=signals,
        )
        carried_metric = carry_info.get("carried_metric") or None
        carried_time_phrase = carry_info.get("carried_time_phrase") or None
    else:
        effective_question = original_question
        effective_lab = (lab_name or "").strip() or None

    return ConversationContext(
        conversation_id=cid,
        original_question=original_question,
        raw_block=raw_block,
        effective_question=effective_question,
        effective_lab=effective_lab,
        routing_snippet=routing_snippet,
        llm_history=llm_history,
        carried_metric=carried_metric,
        carried_time_phrase=carried_time_phrase,
        role=resolved_role,
        role_source=role_source,
        role_fallback_used=role_fallback_used,
    )
