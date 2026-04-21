"""Citation helpers for numbered and source-key citation flows."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_CITATION_MARKER_RE = re.compile(r"\[\^([A-Z0-9_]+)\]")
_FOOTNOTE_BLOCK_RE = re.compile(
    r"\n+#{1,3}\s*(?:References|Sources|Citations|Footnotes)" r".*$",
    re.DOTALL | re.IGNORECASE,
)


def _build_source_key_index(guideline_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup from source_key to record."""
    index: Dict[str, Dict[str, Any]] = {}
    for record in guideline_records or []:
        key = str(record.get("source_key") or "").strip().upper()
        if key:
            index[key] = record
    return index


def resolve_citations(
    answer_text: str,
    guideline_records: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Resolve [^SOURCE_KEY] markers in answer text.

    Returns:
        - answer_text with [^SOURCE_KEY] replaced by [^N]
        - ordered list of footnote dicts

    Only resolves markers that match a record in guideline_records.
    Unknown markers are removed.
    """
    if not answer_text:
        return answer_text, []

    clean_answer = _FOOTNOTE_BLOCK_RE.sub("", answer_text).strip()
    source_key_index = _build_source_key_index(guideline_records)

    assigned: Dict[str, int] = {}
    footnotes_ordered: List[Dict[str, Any]] = []
    next_index = 1

    def _replace_marker(match: re.Match[str]) -> str:
        nonlocal next_index
        raw_key = match.group(1).upper()
        record = source_key_index.get(raw_key)
        if record is None:
            return ""
        if raw_key not in assigned:
            threshold_value = record.get("threshold_value")
            try:
                threshold_value = float(threshold_value) if threshold_value is not None else None
            except (TypeError, ValueError):
                threshold_value = None
            assigned[raw_key] = next_index
            footnotes_ordered.append(
                {
                    "index": next_index,
                    "source_key": raw_key,
                    "source_label": str(record.get("source_label") or ""),
                    "section_ref": record.get("section_ref"),
                    "citation_tier": str(record.get("citation_tier") or "regulatory"),
                    "source_url": record.get("source_url"),
                    "threshold_value": threshold_value,
                    "threshold_unit": record.get("threshold_unit"),
                    "caveat_text": record.get("caveat_text"),
                }
            )
            next_index += 1
        return f"[^{assigned[raw_key]}]"

    resolved_text = _CITATION_MARKER_RE.sub(_replace_marker, clean_answer)
    return resolved_text, footnotes_ordered


def append_footnote_block(
    answer_text: str,
    footnotes: List[Dict[str, Any]],
    include_in_answer: bool = False,
) -> str:
    """Optionally append markdown references for plain-text clients."""
    if not footnotes or not include_in_answer:
        return answer_text

    lines = ["\n\n---\n**References**"]
    for fn in footnotes:
        section = f", {fn['section_ref']}" if fn.get("section_ref") else ""
        url_part = f" — [{fn['source_url']}]({fn['source_url']})" if fn.get("source_url") else ""
        tier_note = (
            " *(research finding, not a regulatory standard)*"
            if fn.get("citation_tier") == "research"
            else " *(internal metric)*"
            if fn.get("citation_tier") == "internal"
            else ""
        )
        lines.append(f"[^{fn['index']}]: **{fn['source_label']}**{section}{url_part}{tier_note}")
    return answer_text + "\n".join(lines)


def build_numbered_sources_block(
    guideline_records: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Assign stable [N] numbers to guideline records before generation.

    Returns:
        - prompt_block: numbered source guidance for the LLM prompt
        - indexed_sources: structured source records with stable `index`
    """
    if not guideline_records:
        return "", []

    lines = [
        "Citation sources available for this response.",
        "Use [N] inline immediately after any claim that draws from these sources.",
        "Only use citation numbers from this list.",
        "Do not add a References section at the end.",
        "",
    ]
    indexed_sources: List[Dict[str, Any]] = []

    def _should_include_numbered_source(record: Dict[str, Any]) -> bool:
        """
        Keep citation sources focused on threshold-supporting references.

        CO2-specific guardrail:
        - ASHRAE 62.1 ventilation references must not be cited as CO2 ppm
          thresholds, so exclude them from numbered inline citation options.
        """
        metric = str(record.get("metric") or "").strip().lower()
        source_key = str(record.get("source_key") or "").strip().upper()
        if metric == "co2" and source_key.startswith("ASHRAE_62_1"):
            return False
        return True

    filtered_records = [record for record in guideline_records if _should_include_numbered_source(record)]

    for i, record in enumerate(filtered_records, start=1):
        tier_label = {
            "regulatory": "Standard",
            "research": "Research — not a regulatory standard",
            "internal": "Internal metric",
        }.get(str(record.get("citation_tier") or ""), "Source")
        section = f", {record['section_ref']}" if record.get("section_ref") else ""
        year = f" ({record['publication_year']})" if record.get("publication_year") else ""
        source_label = str(record.get("source_label") or "Unknown source")

        lines.append(f"[{i}] {source_label}{section}{year} — {tier_label}")

        threshold_value = record.get("threshold_value")
        try:
            threshold_value = float(threshold_value) if threshold_value is not None else None
        except (TypeError, ValueError):
            threshold_value = None

        indexed_sources.append(
            {
                "index": i,
                "source_key": record.get("source_key"),
                "source_label": source_label,
                "section_ref": record.get("section_ref"),
                "citation_tier": str(record.get("citation_tier") or "source"),
                "source_url": record.get("source_url"),
                "threshold_value": threshold_value,
                "threshold_unit": record.get("threshold_unit"),
                "caveat_text": record.get("caveat_text"),
            }
        )

    return "\n".join(lines), indexed_sources


def extract_citation_indices_from_answer(
    answer_text: str,
    indexed_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract cited [N] markers from final answer text.

    Returns only the indexed sources actually referenced, in appearance order.
    """
    if not answer_text or not indexed_sources:
        return []

    source_by_index = {int(source["index"]): source for source in indexed_sources if source.get("index") is not None}
    seen_indices: List[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer_text):
        idx = int(match.group(1))
        if idx in source_by_index and idx not in seen_indices:
            seen_indices.append(idx)
    return [source_by_index[idx] for idx in seen_indices]


def process_answer_citations(
    answer_text: str,
    guideline_records: List[Dict[str, Any]],
    append_block_for_plain_clients: bool = False,
    indexed_sources: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract structured [N] citations from a model answer.

    Pass ``indexed_sources`` when they were already built before generation to
    avoid a redundant ``build_numbered_sources_block`` call.
    """
    if not answer_text:
        return answer_text, []
    if _CITATION_MARKER_RE.search(answer_text):
        # Backward-compatible fallback for legacy [^SOURCE_KEY] outputs.
        resolved, footnotes = resolve_citations(
            answer_text=answer_text,
            guideline_records=guideline_records,
        )
        if append_block_for_plain_clients and footnotes:
            resolved = append_footnote_block(
                resolved,
                footnotes,
                include_in_answer=True,
            )
        return resolved, footnotes
    cleaned_answer = _FOOTNOTE_BLOCK_RE.sub("", answer_text).strip()
    if indexed_sources is None:
        _, indexed_sources = build_numbered_sources_block(guideline_records)
    footnotes = extract_citation_indices_from_answer(
        answer_text=cleaned_answer,
        indexed_sources=indexed_sources,
    )
    resolved = cleaned_answer
    if append_block_for_plain_clients and footnotes:
        resolved = append_footnote_block(
            resolved,
            footnotes,
            include_in_answer=True,
        )
    return resolved, footnotes
