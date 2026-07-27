"""Shared Server-Sent Event frame builders for streaming executors.

Every executor that streams an answer emits the same three terminal frames —
``token``, ``sources``, ``done`` — so the frame shape is built here once rather
than re-inlined per executor. Keeping them together is what stops the sync and
stream contracts from drifting apart (see ``metadata_builders`` for the same
rationale on the UI contract).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from evidence.citation_processor import process_answer_citations


def token_event(text: str) -> str:
    """One incremental answer token."""
    return f"data: {json.dumps({'event': 'token', 'text': text})}\n\n"


def done_event() -> str:
    """Terminal frame telling the client the answer is complete."""
    return f"data: {json.dumps({'event': 'done'})}\n\n"


def sources_event(
    citation_sources: List[Dict[str, Any]],
    footnotes: List[Dict[str, Any]],
) -> str:
    """Citations for the completed answer, emitted just before ``done``.

    The sync ``/query`` response carries ``citation_sources`` (every source
    offered to the model) and ``footnotes`` (the subset the answer actually
    cited). A stream cannot know the footnotes until the last token has been
    generated, so it sends them in this terminal frame instead — same two
    fields, same meaning, just delivered late.
    """
    return "data: " + json.dumps(
        {
            "event": "sources",
            "citation_sources": list(citation_sources or []),
            "footnotes": list(footnotes or []),
        }
    ) + "\n\n"


def sources_event_for_answer(
    emitted_tokens: Iterable[str],
    guideline_records: List[Dict[str, Any]],
    indexed_sources: List[Dict[str, Any]],
) -> str:
    """Build the terminal ``sources`` frame from the tokens already streamed.

    Resolves which ``[N]`` markers the model actually emitted using the same
    :func:`process_answer_citations` call the sync path makes, so both paths
    report an identical footnote list for an identical answer.
    """
    _, footnotes = process_answer_citations(
        answer_text="".join(emitted_tokens),
        guideline_records=list(guideline_records or []),
        indexed_sources=list(indexed_sources or []),
    )
    return sources_event(indexed_sources, footnotes)
