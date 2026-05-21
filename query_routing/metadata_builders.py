"""UI contract derivation from routing/execution context."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from query_routing.intent_classifier import IntentType
except ImportError:
    from .intent_classifier import IntentType

_ANALYTICAL_TRANSITION = "slide"
_DEFAULT_TRANSITION = "fade"


def derive_ui_contract(
    execution_intent: IntentType,
    metrics: List[str],
    visualization_type: str,
    has_floor_comparison: bool,
    clarification_required: bool,
    use_knowledge_executor: bool,
) -> Dict[str, Any]:
    primary_metric = metrics[0] if metrics else ""

    if clarification_required:
        mode, panel = "clarify", "overview"
    elif use_knowledge_executor:
        mode, panel = "conversational", "overview"
    elif execution_intent == IntentType.FORECAST_DB:
        mode, panel = "forecast", "forecast"
    elif execution_intent == IntentType.COMPARISON_DB and has_floor_comparison:
        mode, panel = "analytical", "heatmap"
    elif execution_intent in {IntentType.AGGREGATION_DB, IntentType.COMPARISON_DB}:
        if len(metrics) > 1:
            mode, panel = "analytical", "ieq_composite"
        else:
            mode, panel = "analytical", "single_metric"
    elif execution_intent == IntentType.ANOMALY_ANALYSIS_DB:
        mode, panel = "analytical", "single_metric"
    elif execution_intent in {IntentType.CURRENT_STATUS_DB, IntentType.POINT_LOOKUP_DB}:
        if len(metrics) > 1:
            mode, panel = "status", "ieq_composite"
        else:
            mode, panel = "status", "single_metric"
    else:
        mode, panel = "conversational", "overview"

    transition = _ANALYTICAL_TRANSITION if mode in {"analytical", "forecast"} else _DEFAULT_TRANSITION

    return {
        "mode": mode,
        "panel": panel,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "transition": transition,
    }
