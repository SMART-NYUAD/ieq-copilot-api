"""Typed tool wrappers used by the agentic orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from executors.db_query_executor import run_db_query
    from executors.env_query_langchain import answer_env_question_with_metadata
    from query_routing.intent_classifier import IntentType
except ImportError:
    from ..executors.db_query_executor import run_db_query
    from ..executors.env_query_langchain import answer_env_question_with_metadata
    from .intent_classifier import IntentType


AGENT_TOOL_NAMES: Tuple[str, ...] = (
    "query_db",
    "search_knowledge_cards",
    "compare_spaces",
    "forecast_metric",
    "analyze_anomaly",
)


def _coerce_intent(value: Any, default: IntentType) -> IntentType:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    try:
        return IntentType(raw)
    except ValueError:
        return default


def execute_agent_tool_call(
    *,
    tool_name: str,
    question: str,
    k: int,
    lab_name: Optional[str],
    planner_hints: Optional[Dict[str, Any]] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one named tool with normalized structured result envelope."""
    args = dict(arguments or {})
    normalized_tool = str(tool_name or "").strip().lower()
    effective_question = str(args.get("question") or question or "").strip()
    effective_lab_name = (str(args.get("lab_name") or lab_name or "").strip() or None)
    effective_hints = dict(planner_hints or {})
    if isinstance(args.get("planner_hints"), dict):
        effective_hints.update(dict(args.get("planner_hints") or {}))

    if normalized_tool == "query_db":
        intent = _coerce_intent(args.get("intent"), default=IntentType.CURRENT_STATUS_DB)
        result = run_db_query(
            question=effective_question,
            intent=intent,
            lab_name=effective_lab_name,
            planner_hints=effective_hints,
        )
        return {
            "ok": True,
            "tool_name": normalized_tool,
            "intent": intent.value,
            "result": result,
        }

    if normalized_tool == "compare_spaces":
        result = run_db_query(
            question=effective_question,
            intent=IntentType.COMPARISON_DB,
            lab_name=effective_lab_name,
            planner_hints=effective_hints,
        )
        return {
            "ok": True,
            "tool_name": normalized_tool,
            "intent": IntentType.COMPARISON_DB.value,
            "result": result,
        }

    if normalized_tool == "forecast_metric":
        result = run_db_query(
            question=effective_question,
            intent=IntentType.FORECAST_DB,
            lab_name=effective_lab_name,
            planner_hints=effective_hints,
        )
        return {
            "ok": True,
            "tool_name": normalized_tool,
            "intent": IntentType.FORECAST_DB.value,
            "result": result,
        }

    if normalized_tool == "analyze_anomaly":
        result = run_db_query(
            question=effective_question,
            intent=IntentType.ANOMALY_ANALYSIS_DB,
            lab_name=effective_lab_name,
            planner_hints=effective_hints,
        )
        return {
            "ok": True,
            "tool_name": normalized_tool,
            "intent": IntentType.ANOMALY_ANALYSIS_DB.value,
            "result": result,
        }

    if normalized_tool == "search_knowledge_cards":
        result = answer_env_question_with_metadata(
            user_question=effective_question,
            k=max(1, min(int(args.get("k") or k or 5), 8)),
            space=effective_lab_name,
        )
        return {
            "ok": True,
            "tool_name": normalized_tool,
            "intent": IntentType.DEFINITION_EXPLANATION.value,
            "result": result,
        }

    return {
        "ok": False,
        "tool_name": normalized_tool,
        "error": "unknown_tool",
        "result": None,
    }


def summarize_tool_observation(tool_output: Dict[str, Any]) -> str:
    """Create a compact observation string that can be fed back to planning."""
    if not bool(tool_output.get("ok")):
        return f"tool={tool_output.get('tool_name')} failed error={tool_output.get('error')}"
    payload = dict(tool_output.get("result") or {})
    answer = str(payload.get("answer") or "").strip().replace("\n", " ")
    if len(answer) > 280:
        answer = answer[:277] + "..."
    cards = int(payload.get("cards_retrieved") or 0)
    timescale = str(payload.get("timescale") or "none")
    return (
        f"tool={tool_output.get('tool_name')} ok intent={tool_output.get('intent')} "
        f"timescale={timescale} cards={cards} answer={answer}"
    )
