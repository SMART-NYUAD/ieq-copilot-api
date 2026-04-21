"""Guideline record storage and retrieval."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    from storage.postgres_client import get_cursor
    from storage.embeddings import embed_texts
except ImportError:
    from .postgres_client import get_cursor
    from .embeddings import embed_texts


_GUIDELINE_EMBED_DIM = 1536


def _normalize_embedding_dim(embedding: List[float], target_dim: int = _GUIDELINE_EMBED_DIM) -> List[float]:
    values = list(embedding or [])
    if len(values) == target_dim:
        return values
    if len(values) > target_dim:
        return values[:target_dim]
    return values + [0.0] * (target_dim - len(values))


def get_thresholds_for_metrics(
    metrics: List[str],
    citation_tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic lookup by metric name.
    No embeddings. Used after DB queries return data rows.
    Returns all active guideline records for the given metrics.
    """
    if not metrics:
        return []
    try:
        with get_cursor(real_dict=True) as cur:
            if citation_tier:
                cur.execute(
                    """
                    SELECT id, source_key, source_label, source_url,
                           section_ref, publication_year, metric,
                           citation_tier, claim_text, threshold_value,
                           threshold_type, threshold_unit,
                           threshold_condition, caveat_text
                    FROM env_guideline_records
                    WHERE metric = ANY(%s)
                      AND citation_tier = %s
                      AND is_active = TRUE
                    ORDER BY
                        CASE citation_tier
                            WHEN 'regulatory' THEN 1
                            WHEN 'research' THEN 2
                            WHEN 'internal' THEN 3
                        END,
                        source_key
                """,
                    (metrics, citation_tier),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_key, source_label, source_url,
                           section_ref, publication_year, metric,
                           citation_tier, claim_text, threshold_value,
                           threshold_type, threshold_unit,
                           threshold_condition, caveat_text
                    FROM env_guideline_records
                    WHERE metric = ANY(%s)
                      AND is_active = TRUE
                    ORDER BY
                        CASE citation_tier
                            WHEN 'regulatory' THEN 1
                            WHEN 'research' THEN 2
                            WHEN 'internal' THEN 3
                        END,
                        source_key
                """,
                    (metrics,),
                )
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"[ERROR] get_thresholds_for_metrics failed: {e}")
        return []


def search_guideline_records(
    question: str,
    k: int = 3,
    metric_filter: Optional[List[str]] = None,
    tier_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over guideline records.
    Used when user asks about standards, compliance,
    why a threshold exists, or requests citations.
    """
    if not question:
        return []
    embeddings = embed_texts([question])
    if not embeddings:
        return []
    query_embedding = _normalize_embedding_dim(embeddings[0])
    try:
        with get_cursor(real_dict=True) as cur:
            cur.execute("SET LOCAL ivfflat.probes = 5")
            filters = ["is_active = TRUE"]
            params: List[Any] = [query_embedding, query_embedding]
            if metric_filter:
                filters.append("metric = ANY(%s)")
                params.append(metric_filter)
            if tier_filter:
                filters.append("citation_tier = %s")
                params.append(tier_filter)
            params.append(k)
            where_clause = " AND ".join(filters)
            cur.execute(
                f"""
                SELECT id, source_key, source_label, source_url,
                       section_ref, publication_year, metric,
                       citation_tier, claim_text, threshold_value,
                       threshold_type, threshold_unit,
                       threshold_condition, caveat_text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM env_guideline_records
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"[ERROR] search_guideline_records failed: {e}")
        return []


def wants_guideline_detail(question: str) -> bool:
    """
    Signal detection: does this question need guideline records?
    """
    q = str(question or "").lower()
    if not q.strip():
        return False

    # Domain/metric-first policy: always retrieve guideline sources when the
    # question is clearly about IEQ metrics or indoor-environment context.
    metric_patterns = (
        r"\bco2\b",
        r"\bpm\s*2\.?5\b",
        r"\bpm25\b",
        r"\btvoc\b",
        r"\bvoc\b",
        r"\bhumidity\b",
        r"\btemp(?:erature)?\b",
        r"\blux\b",
        r"\blight(?:ing)?\b",
        r"\bnoise\b",
        r"\bsound\b",
        r"\bieq\b",
        r"\biaq\b",
    )
    domain_patterns = (
        r"\bindoor air\b",
        r"\bair quality\b",
        r"\bventilation\b",
        r"\bmold\b",
        r"\boccupant comfort\b",
        r"\bthermal comfort\b",
    )
    if any(re.search(pattern, q) for pattern in metric_patterns + domain_patterns):
        return True

    hints = (
        "standard",
        "guideline",
        "limit",
        "threshold",
        "ashrae",
        "who ",
        "epa ",
        "reset air",
        "well building",
        "why is",
        "what level",
        "safe level",
        "acceptable level",
        "regulation",
        "compliance",
        "certified",
        "benchmark",
        "citation",
        "source",
        "reference",
        "based on",
        "according to",
        "per standard",
        "norm ",
        "requirement",
    )
    return any(hint in q for hint in hints)
