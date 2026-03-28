"""Standalone knowledge-card normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List


ALLOWED_CARD_TYPES = {"interpretation", "explanation", "rule", "caveat", "ieq_subindex"}
REQUIRED_FIELDS = {
    "card_type",
    "topic",
    "title",
    "summary",
    "content",
    "condition_json",
    "recommendation_json",
    "tags",
}

SOURCE_REGISTRY: Dict[str, Dict[str, str]] = {
    "RESET_AIR": {
        "url": "https://reset.build/standard/air",
        "label": "RESET Air Standard",
    },
    "EPA_PM25_AQI": {
        "url": "https://www.epa.gov/pm-pollution",
        "label": "EPA PM2.5 Guidance",
    },
    "EPA_HUMIDITY": {
        "url": "https://www.epa.gov/indoor-air-quality-iaq/inside-story-guide-indoor-air-quality",
        "label": "EPA Indoor Humidity Guidance",
    },
    "ASHRAE_55": {
        "url": "https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy",
        "label": "ASHRAE 55",
    },
    "WHO_NOISE": {
        "url": "https://www.who.int/publications/i/item/9789289053563",
        "label": "WHO Noise Guidance",
    },
    "INTERNAL_COMBINED_POLICY": {
        "url": "",
        "label": "Internal Combined Interpretation Policy",
    },
    "INTERNAL_GUARDRAIL": {
        "url": "",
        "label": "Internal Communication Guardrail",
    },
}


class KnowledgeCardValidationError(ValueError):
    """Raised when a knowledge card is malformed."""


def normalize_metric_name(metric_name: Any) -> Any:
    if metric_name is None:
        return None
    value = str(metric_name).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"tvoc", "tvoc_ugm3"}:
        return "tvoc"
    return lowered


def _ensure_json_object(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise KnowledgeCardValidationError(f"{field_name} must be an object")
    return value


def _ensure_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise KnowledgeCardValidationError("tags must be an array of strings")
    normalized_tags = []
    for item in value:
        if item is None:
            continue
        normalized_tags.append(str(item).strip())
    return [tag for tag in normalized_tags if tag]


def normalize_card(raw_card: Dict[str, Any]) -> Dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(raw_card.keys()))
    if missing:
        raise KnowledgeCardValidationError(f"missing required fields: {', '.join(missing)}")

    card_type = str(raw_card.get("card_type") or "").strip().lower()
    if card_type not in ALLOWED_CARD_TYPES:
        raise KnowledgeCardValidationError(f"unsupported card_type: {card_type}")

    topic = str(raw_card.get("topic") or "").strip().lower()
    title = str(raw_card.get("title") or "").strip()
    summary = str(raw_card.get("summary") or "").strip()
    content = str(raw_card.get("content") or "").strip()
    if not topic or not title or not summary or not content:
        raise KnowledgeCardValidationError("topic, title, summary, and content must be non-empty")

    original_metric_name = raw_card.get("metric_name")
    metric_name = normalize_metric_name(original_metric_name)
    source_url_key = raw_card.get("source_url_key")
    source_metadata = dict(SOURCE_REGISTRY.get(str(source_url_key or "").strip(), {}))
    if original_metric_name not in (None, ""):
        source_metadata["original_metric_name"] = str(original_metric_name)
    if source_url_key:
        source_metadata["source_url_key"] = str(source_url_key)

    return {
        "card_type": card_type,
        "topic": topic,
        "title": title,
        "summary": summary,
        "content": content,
        "audience": raw_card.get("audience"),
        "severity_level": raw_card.get("severity_level"),
        "metric_name": metric_name,
        "condition_json": _ensure_json_object(raw_card.get("condition_json"), "condition_json"),
        "recommendation_json": _ensure_json_object(raw_card.get("recommendation_json"), "recommendation_json"),
        "tags": _ensure_tags(raw_card.get("tags")),
        "source_label": raw_card.get("source_label"),
        "source_url_key": source_url_key,
        "source_metadata": source_metadata,
    }

