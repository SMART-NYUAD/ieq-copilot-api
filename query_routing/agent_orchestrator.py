"""Agentic query loop (plan -> tool call -> observe -> finalize)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    from query_routing.agent_tools import execute_agent_tool_call, summarize_tool_observation
    from query_routing.router_types import AgentAction, RouteExecutor
except ImportError:
    from .agent_tools import execute_agent_tool_call, summarize_tool_observation
    from .router_types import AgentAction, RouteExecutor


def _attach_agent_metadata(
    *,
    payload: Dict[str, Any],
    steps: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    finish_reason: str,
    agent_enabled: bool = True,
) -> Dict[str, Any]:
    merged = dict(payload or {})
    metadata = dict(merged.get("metadata") or {})
    metadata["agent_mode"] = "enabled" if agent_enabled else "disabled"
    metadata["agent_steps"] = int(len(steps))
    metadata["agent_step_trace"] = list(steps)
    metadata["tools_called"] = [str(item.get("tool_name") or "") for item in tool_calls]
    metadata["agent_tool_calls"] = list(tool_calls)
    metadata["agent_finish_reason"] = str(finish_reason or "unknown")
    successful_tools = [item for item in tool_calls if bool(item.get("ok"))]
    metadata["agent_verifier_status"] = "pass" if successful_tools or not tool_calls else "warn"
    metadata["agent_verifier_issues"] = [] if successful_tools or not tool_calls else ["no_successful_tool_calls"]
    merged["metadata"] = metadata
    return merged


_TOOL_BUDGET_BY_INTENT: Dict[str, int] = {
    "comparison_db": 1,
    "aggregation_db": 1,
    "current_status_db": 1,
    "point_lookup_db": 1,
    "forecast_db": 1,
    "anomaly_analysis_db": 2,
}
_ALLOWED_AGENT_GOALS = {"compare", "explain", "recommend"}
_COMPARE_HINTS = ("compare", "comparison", "vs", "versus", "difference")
_EXPLAIN_HINTS = ("why", "explain", "reason", "because", "difference happened")
_RECOMMEND_HINTS = ("recommend", "action", "what to take", "what should", "next step", "improve")
_EXPLAIN_COVERAGE_MARKERS = ("because", "due to", "likely", "reason", "explains")
_RECOMMEND_COVERAGE_MARKERS = ("recommend", "should", "action", "improve", "consider")


def _tool_budget_for_intent(intent_value: str) -> int:
    return max(1, int(_TOOL_BUDGET_BY_INTENT.get(str(intent_value or "").strip().lower(), 1)))


def _derive_required_goals(question: str) -> set[str]:
    q = str(question or "").strip().lower()
    goals: set[str] = set()
    if any(token in q for token in _COMPARE_HINTS):
        goals.add("compare")
    if any(token in q for token in _EXPLAIN_HINTS):
        goals.add("explain")
    if any(token in q for token in _RECOMMEND_HINTS):
        goals.add("recommend")
    return goals


def _normalize_declared_goal_coverage(route_plan: Any) -> set[str]:
    raw = getattr(route_plan, "goal_coverage", ())
    if not isinstance(raw, (list, tuple)):
        return set()
    out: set[str] = set()
    for item in raw:
        goal = str(item or "").strip().lower()
        if goal in _ALLOWED_AGENT_GOALS:
            out.add(goal)
    return out


def _derive_goals_from_tool_result(*, route_type: str, tool_name: Optional[str], answer_text: str) -> set[str]:
    goals: set[str] = set()
    route = str(route_type or "").strip().lower()
    tool = str(tool_name or "").strip().lower()
    answer = str(answer_text or "").strip().lower()
    if route == "comparison_db" or tool == "compare_spaces":
        goals.add("compare")
    if any(marker in answer for marker in _EXPLAIN_COVERAGE_MARKERS):
        goals.add("explain")
    if any(marker in answer for marker in _RECOMMEND_COVERAGE_MARKERS):
        goals.add("recommend")
    return goals


def run_agentic_query_loop(
    *,
    question: str,
    generation_question: str,
    k: int,
    lab_name: Optional[str],
    allow_clarify: bool,
    max_steps: int,
    max_consecutive_failures: int,
    stall_threshold: int,
    get_route_decision_contract_fn: Callable[..., Any],
    execute_with_contract_fn: Callable[..., Dict[str, Any]],
    build_clarify_result_fn: Callable[..., Dict[str, Any]],
    build_clarify_prompt_fn: Callable[[Any], str],
    record_agent_run_fn: Optional[Callable[..., None]] = None,
    record_agent_step_fn: Optional[Callable[..., None]] = None,
    record_agent_tool_call_fn: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Execute bounded agent loop and return a standard query payload."""
    planning_question = str(question or "").strip()
    observations: List[str] = []
    steps: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    consecutive_failures = 0
    repeated_action_count = 0
    last_action_signature = ""
    final_contract = None
    finish_reason = "max_steps"
    required_goals = _derive_required_goals(question)
    covered_goals: set[str] = set()
    tool_calls_by_intent: Dict[str, int] = {}

    for step_index in range(1, max(1, int(max_steps)) + 1):
        route_contract = get_route_decision_contract_fn(
            planning_question,
            lab_name,
            False,
        )
        final_contract = route_contract
        route_plan = route_contract.route_plan
        decision = route_plan.decision
        action = getattr(route_plan, "agent_action", AgentAction.FINALIZE)
        action_value = str(getattr(action, "value", action) or AgentAction.FINALIZE.value)
        tool_name = str(getattr(route_plan, "tool_name", "") or "").strip().lower() or None
        if record_agent_step_fn is not None:
            record_agent_step_fn(action=action_value)
        step_entry = {
            "step": step_index,
            "action": action_value,
            "route_type": decision.intent.value,
            "executor": route_contract.executor.value,
            "confidence": float(decision.confidence),
            "tool_name": tool_name,
            "required_goals": sorted(list(required_goals)),
            "covered_goals": sorted(list(covered_goals)),
        }
        steps.append(step_entry)
        action_signature = f"{action_value}:{tool_name or '-'}:{decision.intent.value}"
        if action_signature == last_action_signature:
            repeated_action_count += 1
        else:
            repeated_action_count = 0
            last_action_signature = action_signature

        if allow_clarify and (
            action_value == AgentAction.CLARIFY.value or route_contract.executor == RouteExecutor.CLARIFY_GATE
        ):
            finish_reason = "clarify"
            clarify_payload = build_clarify_result_fn(
                route_plan=route_plan,
                decision=decision,
                k=k,
                lab_name=lab_name,
                clarify_threshold=0.0,
                clarify_text=build_clarify_prompt_fn(route_plan),
            )
            if record_agent_run_fn is not None:
                record_agent_run_fn(
                    finish_reason=finish_reason,
                    step_count=len(steps),
                    tool_calls=len(tool_calls),
                )
            return _attach_agent_metadata(
                payload=clarify_payload,
                steps=steps,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        if action_value == AgentAction.TOOL_CALL.value and tool_name:
            route_type = str(decision.intent.value or "").strip().lower()
            intent_budget = _tool_budget_for_intent(route_type)
            calls_for_intent = int(tool_calls_by_intent.get(route_type, 0))
            if calls_for_intent >= intent_budget:
                step_entry["blocked_repeat"] = True
                step_entry["blocked_reason"] = "tool_budget_exhausted"
                step_entry["intent_tool_budget"] = intent_budget
                finish_reason = "tool_budget_exhausted"
                break
            tool_output = execute_agent_tool_call(
                tool_name=tool_name,
                question=planning_question,
                k=k,
                lab_name=lab_name,
                planner_hints=route_plan.planner_parameters,
                arguments=getattr(route_plan, "tool_arguments", {}),
            )
            tool_calls_by_intent[route_type] = calls_for_intent + 1
            tool_calls.append(
                {
                    "step": step_index,
                    "tool_name": tool_name,
                    "ok": bool(tool_output.get("ok")),
                    "intent": tool_output.get("intent"),
                }
            )
            if record_agent_tool_call_fn is not None:
                record_agent_tool_call_fn(success=bool(tool_output.get("ok")))
            if not bool(tool_output.get("ok")):
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            observation = summarize_tool_observation(tool_output)
            observations.append(observation)
            if bool(tool_output.get("ok")):
                result_payload = dict(tool_output.get("result") or {})
                answer_text = str(result_payload.get("answer") or observation)
                covered_goals.update(
                    _derive_goals_from_tool_result(
                        route_type=route_type,
                        tool_name=tool_name,
                        answer_text=answer_text,
                    )
                )
                covered_goals.update(_normalize_declared_goal_coverage(route_plan))
                step_entry["covered_goals"] = sorted(list(covered_goals))
                all_goals_met = not required_goals or required_goals.issubset(covered_goals)
                enough_evidence = bool(getattr(route_plan, "enough_evidence", False))
                if enough_evidence and all_goals_met:
                    finish_reason = "evidence_sufficient"
                    break
                if required_goals and all_goals_met:
                    finish_reason = "goal_coverage_complete"
                    break
            planning_question = (
                f"{question}\n\nAgent tool observations:\n"
                + "\n".join(f"- {item}" for item in observations[-4:])
                + (
                    f"\n\nCoverage status:\n- required_goals: {sorted(list(required_goals))}\n"
                    f"- covered_goals: {sorted(list(covered_goals))}\n"
                    f"- remaining_goals: {sorted(list(required_goals - covered_goals))}\n"
                    "If remaining_goals is empty, choose action=finalize."
                    if required_goals
                    else ""
                )
            )
            if (
                consecutive_failures >= max(1, int(max_consecutive_failures))
                or repeated_action_count >= max(1, int(stall_threshold))
            ):
                finish_reason = "tool_loop_stalled"
                break
            continue

        finish_reason = "finalize"
        final_payload = execute_with_contract_fn(
            route_contract=route_contract,
            question=question,
            generation_question=generation_question,
            k=k,
            lab_name=lab_name,
        )
        if record_agent_run_fn is not None:
            record_agent_run_fn(
                finish_reason=finish_reason,
                step_count=len(steps),
                tool_calls=len(tool_calls),
            )
        return _attach_agent_metadata(
            payload=final_payload,
            steps=steps,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    if final_contract is None:
        final_contract = get_route_decision_contract_fn(question, lab_name, False)
    final_payload = execute_with_contract_fn(
        route_contract=final_contract,
        question=question,
        generation_question=generation_question,
        k=k,
        lab_name=lab_name,
    )
    if record_agent_run_fn is not None:
        record_agent_run_fn(
            finish_reason=finish_reason,
            step_count=len(steps),
            tool_calls=len(tool_calls),
        )
    return _attach_agent_metadata(
        payload=final_payload,
        steps=steps,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
