#!/usr/bin/env python3
"""Break down where wall-clock time goes for one routed query (router vs retrieval vs Ollama).

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
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * part / total:.1f}%"


def _run_breakdown(
    question: str,
    lab_name: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    from query_routing.llm_router_planner import plan_route
    from query_routing.query_orchestrator import _choose_executor
    from query_routing.router_types import RouteExecutor

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    route = plan_route(question, lab_name)
    timings["router_ms"] = (time.perf_counter() - t0) * 1000.0

    executor = _choose_executor(route)
    meta: Dict[str, Any] = {
        "executor": executor.value,
        "intent": route.intent.value,
        "lab": route.lab_name,
        "metrics": route.metrics,
        "confidence": route.confidence,
        "fallback_used": route.fallback_used,
    }

    if executor == RouteExecutor.KNOWLEDGE_QA:
        from core_settings import ollama_temperature
        from executors.knowledge_executor import (
            _build_knowledge_context,
            _build_knowledge_prompt,
            _generate_ollama_text,
        )

        t1 = time.perf_counter()
        ctx = _build_knowledge_context(user_question=question, k=5, space=lab_name)
        timings["knowledge_retrieval_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        prompt_text = _build_knowledge_prompt(question, ctx["grounded_context"])
        _generate_ollama_text(prompt_text, temperature=ollama_temperature())
        timings["ollama_generate_ms"] = (time.perf_counter() - t2) * 1000.0

    else:
        from executors.db_query_executor import _render_db_answer_with_llm, prepare_db_query

        t3 = time.perf_counter()
        db_ctx = prepare_db_query(
            question=question,
            intent=route.intent,
            lab_name=lab_name,
            planner_hints={
                "metrics_priority": list(route.metrics),
                "needs_cards": False,
                "card_topics": [],
                "max_cards": 2,
                "second_lab_name": route.second_lab_name,
            },
        )
        timings["db_prepare_ms"] = (time.perf_counter() - t3) * 1000.0

        if db_ctx.get("invariant_violation"):
            meta["invariant_violation"] = True
            return timings, meta

        t4 = time.perf_counter()
        _render_db_answer_with_llm(
            question=question,
            intent=route.intent,
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


def _full_execute_ms(question: str, lab_name: Optional[str], k: int) -> float:
    from query_routing.query_orchestrator import execute_query

    t0 = time.perf_counter()
    execute_query(
        question=question,
        k=k,
        lab_name=lab_name,
        endpoint_key="benchmark_sync",
        conversation_context="",
    )
    return (time.perf_counter() - t0) * 1000.0


async def _stream_collect_ms(question: str, lab_name: Optional[str], k: int) -> float:
    from query_routing.query_orchestrator import stream_query

    t0 = time.perf_counter()
    async for _ in stream_query(
        question=question,
        k=k,
        lab_name=lab_name,
        endpoint_key="benchmark_stream",
        conversation_context="",
    ):
        pass
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark query latency breakdown.")
    parser.add_argument("-q", "--question", default="Average CO2 in smart_lab last 24 hours")
    parser.add_argument("--lab", default="smart_lab")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    import core_settings  # noqa: F401 — loads .env

    lab = (args.lab or "").strip() or None

    print("=== RAG_API_SERVER query latency benchmark ===")
    print(f"question: {args.question!r}")
    print(f"lab_name: {lab!r}")
    print()

    try:
        timings, meta = _run_breakdown(args.question, lab)
    except Exception as exc:
        print(f"Breakdown failed: {exc}")
        raise

    print("--- Step breakdown ---")
    for key in sorted(timings.keys()):
        print(f"  {key}: {timings[key]:.1f} ms")
    print("  meta:", meta)
    accounted = sum(timings.values())
    print(f"  sum(steps): {accounted:.1f} ms")
    print()

    try:
        full_ms = _full_execute_ms(args.question, lab, args.k)
        print(f"--- Full execute_query() wall time: {full_ms:.1f} ms ---")
        if full_ms > 0 and accounted > 0:
            print(f"  breakdown / full: {_pct(accounted, full_ms)}")
    except Exception as exc:
        print(f"Full execute_query failed: {exc}")

    if args.stream:
        print()
        try:
            total_ms = asyncio.run(_stream_collect_ms(args.question, lab, args.k))
            print(f"--- Stream path: {total_ms:.1f} ms ---")
        except Exception as exc:
            print(f"Stream benchmark failed: {exc}")


if __name__ == "__main__":
    main()
