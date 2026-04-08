#!/usr/bin/env python3
"""Break down where wall-clock time goes for one routed query (router vs DB vs Ollama).

Usage (from repo root RAG_API_SERVER):
  python scripts/benchmark_query_latency.py
  python scripts/benchmark_query_latency.py -q "Average CO2 in smart_lab last 24 hours"

Requires live Postgres (for DB path) and Ollama at OLLAMA_BASE_URL / OLLAMA_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * part / total:.1f}%"


def _run_db_breakdown(
    question: str,
    lab_name: str | None,
    allow_clarify: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    from executors.db_query_executor import _render_db_answer_with_llm, prepare_db_query
    from query_routing.query_orchestrator import get_route_decision_contract, resolve_execution_intent
    from query_routing.router_types import RouteExecutor

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    contract = get_route_decision_contract(question=question, lab_name=lab_name, allow_clarify=allow_clarify)
    timings["router_contract_ms"] = (time.perf_counter() - t0) * 1000.0

    executor = contract.executor
    meta: Dict[str, Any] = {
        "executor": executor.value,
        "execution_intent": contract.execution_intent.value,
    }

    if executor == RouteExecutor.CLARIFY_GATE:
        return timings, meta

    if executor == RouteExecutor.KNOWLEDGE_QA:
        from executors.env_query_langchain import (
            _build_knowledge_context,
            _build_prompt_text_from_messages,
            _generate_ollama_text,
        )
        from query_routing.synthesizer import build_synthesis_context

        route_plan = contract.route_plan
        signals = (route_plan.planner_parameters or {}).get("query_signals") or {}
        suppress = bool(signals.get("is_hypothetical_conditional")) and not bool(
            signals.get("requests_current_measured_data")
        )
        k = 5
        effective_space = None if suppress else lab_name
        knowledge_q = (
            question
            if suppress
            else build_synthesis_context(
                tool_results=None,
                conversation_context="",
                question=question,
            )
        )

        t1 = time.perf_counter()
        ctx = _build_knowledge_context(user_question=knowledge_q, k=k, space=effective_space)
        timings["knowledge_retrieval_ms"] = (time.perf_counter() - t1) * 1000.0

        from executors.env_query_langchain import get_qa_prompt

        t2 = time.perf_counter()
        messages = get_qa_prompt().format_messages(
            question=knowledge_q,
            context_label="Measured room facts with knowledge grounding",
            context_data=ctx["grounded_context"],
        )
        prompt_text = _build_prompt_text_from_messages(messages)
        _generate_ollama_text(prompt_text, temperature=0.4, think=False)
        timings["ollama_generate_ms"] = (time.perf_counter() - t2) * 1000.0

        return timings, meta

    # DB path
    route_plan = contract.route_plan
    intent = resolve_execution_intent(route_plan.decision.intent)

    t3 = time.perf_counter()
    db_ctx = prepare_db_query(
        question=question,
        intent=intent,
        lab_name=lab_name,
        planner_hints=route_plan.planner_parameters,
    )
    timings["db_prepare_and_query_ms"] = (time.perf_counter() - t3) * 1000.0

    if db_ctx.get("invariant_violation"):
        meta["invariant_violation"] = True
        return timings, meta

    t4 = time.perf_counter()
    _render_db_answer_with_llm(
        question=question,
        intent=intent,
        metric_alias=db_ctx["metric_alias"],
        window_label=db_ctx["window_label"],
        rows=db_ctx["rows"],
        fallback_answer=db_ctx["fallback_answer"],
        time_window=db_ctx.get("time_window"),
        forecast=db_ctx.get("forecast"),
        correlation=db_ctx.get("correlation"),
        knowledge_cards=db_ctx.get("knowledge_cards"),
    )
    timings["ollama_generate_ms"] = (time.perf_counter() - t4) * 1000.0

    meta["row_count"] = len(db_ctx.get("rows") or [])
    return timings, meta


def _full_execute_ms(question: str, lab_name: str | None, k: int, allow_clarify: bool) -> float:
    from query_routing.query_orchestrator import execute_query

    t0 = time.perf_counter()
    execute_query(
        question=question,
        k=k,
        lab_name=lab_name,
        allow_clarify=allow_clarify,
        endpoint_key="benchmark_sync",
        conversation_context="",
    )
    return (time.perf_counter() - t0) * 1000.0


async def _stream_collect_ms(question: str, lab_name: str | None, k: int) -> Tuple[float, Dict[str, Any]]:
    from executors.db_query_executor import prepare_db_query, stream_db_query
    from executors.env_query_langchain import stream_answer_env_question
    from http_routes.query_runtime import fetch_db_context, fetch_knowledge_stats, resolve_agent_stream_runtime
    from http_routes.stream_shared import prepare_stream_execution_context
    from query_routing.metadata_builders import (
        build_stream_clarify_metadata,
        build_stream_db_metadata,
        build_stream_knowledge_metadata,
    )
    from query_routing.query_orchestrator import (
        get_route_plan,
        query_scope_class,
        resolve_execution_intent,
        should_clarify,
        should_use_knowledge_executor,
    )

    t_prep = time.perf_counter()
    stream_ctx = await prepare_stream_execution_context(
        latest_user_question=question.strip(),
        k=k,
        lab_name=lab_name,
        allow_clarify=True,
        conversation_context="",
        endpoint_key="benchmark_stream",
        resolve_agent_stream_runtime_fn=resolve_agent_stream_runtime,
        get_route_plan_fn=get_route_plan,
        query_scope_class_fn=query_scope_class,
        should_clarify_fn=should_clarify,
        should_use_knowledge_executor_fn=should_use_knowledge_executor,
        resolve_execution_intent_fn=resolve_execution_intent,
        fetch_knowledge_stats_fn=fetch_knowledge_stats,
        fetch_db_context_fn=fetch_db_context,
        prepare_db_query_fn=prepare_db_query,
        build_stream_clarify_metadata_fn=build_stream_clarify_metadata,
        build_stream_knowledge_metadata_fn=build_stream_knowledge_metadata,
        build_stream_db_metadata_fn=build_stream_db_metadata,
    )
    prep_ms = (time.perf_counter() - t_prep) * 1000.0

    gen_meta: Dict[str, Any] = {"mode": stream_ctx.mode, "prep_ms": prep_ms}

    t_gen = time.perf_counter()
    if stream_ctx.mode == "knowledge":
        async for _ in stream_answer_env_question(
            user_question=str(stream_ctx.knowledge_question or ""),
            k=max(1, min(k, 8)),
            space=stream_ctx.knowledge_lab_name,
        ):
            pass
    elif stream_ctx.mode == "db":
        async for _ in stream_db_query(
            question=str(stream_ctx.effective_question or ""),
            intent=stream_ctx.execution_intent,
            lab_name=stream_ctx.effective_lab_name,
            planner_hints=stream_ctx.route_plan.planner_parameters,
            query_context=stream_ctx.db_context or {},
        ):
            pass
    else:
        gen_meta["skipped_stream_body"] = stream_ctx.mode

    gen_ms = (time.perf_counter() - t_gen) * 1000.0
    return prep_ms + gen_ms, {**gen_meta, "stream_generate_ms": gen_ms, "stream_total_ms": prep_ms + gen_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark query latency breakdown.")
    parser.add_argument(
        "-q",
        "--question",
        default="Average CO2 in smart_lab last 24 hours",
        help="User question to benchmark",
    )
    parser.add_argument("--lab", default="smart_lab", help="lab_name parameter (empty for none)")
    parser.add_argument("-k", type=int, default=5, help="k for knowledge retrieval")
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Pass allow_clarify=False to match low-confidence paths without clarify gate",
    )
    parser.add_argument("--stream", action="store_true", help="Also time HTTP-stream-equivalent path (async)")
    args = parser.parse_args()

    import core_settings  # noqa: F401 — loads .env

    from core_settings import load_settings

    settings = load_settings()
    lab = (args.lab or "").strip() or None
    allow_clarify = not args.no_clarify

    print("=== RAG_API_SERVER query latency benchmark ===")
    print(f"question: {args.question!r}")
    print(f"lab_name: {lab!r}")
    print(f"agentic_mode: {settings.agentic_mode}")
    print(f"allow_clarify: {allow_clarify}")
    print()

    try:
        timings, meta = _run_db_breakdown(args.question, lab, allow_clarify)
    except Exception as exc:
        print(f"Breakdown failed: {exc}")
        raise

    print("--- Step breakdown (one routing pass; not the agent multi-step total) ---")
    for key in sorted(timings.keys()):
        print(f"  {key}: {timings[key]:.1f} ms")
    print("  meta:", meta)

    accounted = sum(timings.values())
    print(f"  sum(steps): {accounted:.1f} ms")
    print()

    try:
        full_ms = _full_execute_ms(args.question, lab, args.k, allow_clarify)
        print(f"--- Full execute_query() wall time: {full_ms:.1f} ms ---")
        if settings.agentic_mode:
            print(
                "  (agentic_mode is on: this includes the agent loop when applicable; "
                "step breakdown above is a single contract + one executor slice)"
            )
        if full_ms > 0 and accounted > 0:
            print(f"  breakdown sum / full execute: {_pct(accounted, full_ms)} of full (approximate)")
    except Exception as exc:
        print(f"Full execute_query failed: {exc}")

    if args.stream:
        print()
        try:
            total_ms, smeta = asyncio.run(_stream_collect_ms(args.question, lab, args.k))
            print(f"--- Stream path (prepare + consume tokens): {total_ms:.1f} ms ---")
            print(f"  {smeta}")
        except Exception as exc:
            print(f"Stream benchmark failed: {exc}")


if __name__ == "__main__":
    main()
