"""Router prompt eval — scores the live LLM router against golden cases.

`_SYSTEM_PROMPT` is the most-edited, highest-risk artifact in the routing layer, and the
unittest suite is deliberately hermetic, so nothing in it exercises the prompt against a
real model. This runner closes that gap: it gives prompt changes a number to move instead
of a hand-sampled impression.

Two eval surfaces, one scorecard:

  * **router cases** (``tests/router_eval_cases.json``) — one ``plan_route`` call each,
    scored on intent, slots, UI-control parameters, ``resolved_question``, and the clarify
    decision. Prior context is supplied inline, so coreference is probed without running a
    conversation. Only the router LLM needs to be up.
  * **golden conversation** (``tests/conversation_harness.py``) — the same scripted
    multi-turn thread driven through the real pipeline, which catches what a single call
    cannot: state that only goes wrong once a turn is persisted and replayed. Needs the
    router, the answer model, and the sensor API.

This is NOT part of ``unittest discover`` — it talks to the network by design, and
``scripts/check_tests_hermetic.py`` only scans ``tests/test_*.py``, so the name is
deliberately not ``test_``-prefixed.

    python tests/router_eval.py                      # both surfaces
    python tests/router_eval.py --router-only        # prompt work: fast loop
    python tests/router_eval.py --group context      # one capability
    python tests/router_eval.py --repeat 3           # flakiness at temperature 0
    python tests/router_eval.py --verbose            # print every case, not just failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if TEST_DIR not in sys.path:  # so `conversation_harness` imports when run from anywhere
    sys.path.insert(0, TEST_DIR)

CASES_PATH = Path(TEST_DIR) / "router_eval_cases.json"

_GREEN, _RED, _YELLOW, _DIM, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _c(text: str, color: str) -> str:
    return text if not sys.stdout.isatty() else f"{color}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Scoring one case
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    group: str
    question: str
    plan_summary: str
    fallback_used: bool
    failures: List[str] = field(default_factory=list)
    checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _score(expect: Dict[str, Any], plan) -> Tuple[List[str], int]:
    """Return (failure descriptions, number of expectations checked)."""
    failures: List[str] = []
    checked = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal checked
        checked += 1
        if not ok:
            failures.append(f"{name}: {detail}")

    resolved = _lower(plan.resolved_question)
    metrics = [_lower(m) for m in (plan.metrics or [])]

    if "intent" in expect:
        want = expect["intent"]
        check("intent", plan.intent.value == want, f"want={want} got={plan.intent.value}")
    if "intent_in" in expect:
        want = expect["intent_in"]
        check("intent_in", plan.intent.value in want, f"want one of {want} got={plan.intent.value}")
    if "lab" in expect:
        check("lab", plan.lab_name == expect["lab"], f"want={expect['lab']} got={plan.lab_name}")
    if "metrics_include" in expect:
        missing = [m for m in expect["metrics_include"] if m not in metrics]
        check("metrics_include", not missing, f"missing={missing} got={metrics}")
    if "metrics_exclude" in expect:
        present = [m for m in expect["metrics_exclude"] if m in metrics]
        check("metrics_exclude", not present, f"unexpected={present} got={metrics}")
    if "time_phrase_contains" in expect:
        want = _lower(expect["time_phrase_contains"])
        check("time_phrase_contains", want in _lower(plan.time_phrase),
              f"want~{want!r} got={plan.time_phrase!r}")
    if "needs_clarification" in expect:
        want = bool(expect["needs_clarification"])
        check("needs_clarification", plan.needs_clarification == want,
              f"want={want} got={plan.needs_clarification} q={plan.clarification_question!r}")
    if "analysis_mode" in expect:
        check("analysis_mode", plan.analysis_mode == expect["analysis_mode"],
              f"want={expect['analysis_mode']} got={plan.analysis_mode}")

    if "resolved_contains_all" in expect:
        missing = [t for t in expect["resolved_contains_all"] if _lower(t) not in resolved]
        check("resolved_contains_all", not missing,
              f"missing={missing} resolved={plan.resolved_question!r}")
    if "resolved_contains_any" in expect:
        hit = [t for t in expect["resolved_contains_any"] if _lower(t) in resolved]
        check("resolved_contains_any", bool(hit),
              f"want any of {expect['resolved_contains_any']} resolved={plan.resolved_question!r}")
    if "resolved_excludes" in expect:
        present = [t for t in expect["resolved_excludes"] if _lower(t) in resolved]
        check("resolved_excludes", not present,
              f"unexpected={present} resolved={plan.resolved_question!r}")

    for attr in ("viewer_type", "heatmap_action", "heatmap_metric",
                 "download_format", "download_metric", "download_interval"):
        if attr in expect:
            got = getattr(plan, attr)
            check(attr, got == expect[attr], f"want={expect[attr]} got={got}")

    return failures, checked


def _context_block(lines: Optional[List[str]]) -> str:
    """Wrap prior transcript lines the way the conversation context builder does.

    The router strips this header itself via ``extract_context_lines``; building it the
    same way here keeps a fixture case byte-identical to what a real turn would send.
    """
    if not lines:
        return ""
    return "Previous conversation context (most recent last):\n" + "\n".join(lines)


def run_router_cases(cases: List[dict], *, verbose: bool = False) -> List[CaseResult]:
    from query_routing.llm_router_planner import plan_route

    results: List[CaseResult] = []
    for case in cases:
        question = case["question"]
        plan = plan_route(question, case.get("lab"), _context_block(case.get("context")))
        failures, checked = _score(case.get("expect", {}), plan)
        summary = (
            f"intent={plan.intent.value} conf={plan.confidence} "
            f"metrics={plan.metrics} time={plan.time_phrase!r} "
            f"clarify={plan.needs_clarification} resolved={plan.resolved_question!r}"
        )
        tr = CaseResult(
            case_id=case["id"],
            group=case.get("group", "ungrouped"),
            question=question,
            plan_summary=summary,
            fallback_used=plan.fallback_used,
            failures=failures,
            checked=checked,
        )
        results.append(tr)

        mark = _c("PASS", _GREEN) if tr.passed else _c("FAIL", _RED)
        if verbose or not tr.passed:
            print(f"[{mark}] {tr.group}/{tr.case_id}: {question!r}")
            print(_c(f"        {summary}", _DIM))
            for f in failures:
                print(_c(f"        ✗ {f}", _RED))
        else:
            print(f"[{mark}] {tr.group}/{tr.case_id}")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _scorecard(title: str, rows: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    print()
    print(_c("═" * 72, _DIM))
    print(title)
    passed = total = 0
    for name in sorted(rows):
        p, t = rows[name]
        passed += p
        total += t
        print(f"  {name:14} {_c(f'{p}/{t}', _GREEN if p == t else _RED)}")
    color = _GREEN if passed == total else _YELLOW
    print(f"  {'TOTAL':14} {_c(f'{passed}/{total}', color)}")
    print(_c("═" * 72, _DIM))
    return passed, total


def _by_group(results: List[CaseResult]) -> Dict[str, Tuple[int, int]]:
    rows: Dict[str, List[int]] = {}
    for r in results:
        agg = rows.setdefault(r.group, [0, 0])
        agg[1] += 1
        if r.passed:
            agg[0] += 1
    return {k: (v[0], v[1]) for k, v in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Router prompt eval")
    parser.add_argument("--router-only", action="store_true",
                        help="skip the multi-turn conversation harness")
    parser.add_argument("--conversation-only", action="store_true",
                        help="run only the multi-turn conversation harness")
    parser.add_argument("--group", action="append",
                        help="restrict to a case group (repeatable): intent, context, "
                             "clarify, ui-control, diagnostic, injection")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the router cases N times to expose non-determinism")
    parser.add_argument("--verbose", action="store_true", help="print passing cases in full")
    args = parser.parse_args()

    exit_code = 0

    if not args.conversation_only:
        cases = json.loads(CASES_PATH.read_text())["cases"]
        if args.group:
            wanted = set(args.group)
            cases = [c for c in cases if c.get("group") in wanted]
        if not cases:
            print(_c("No cases matched the requested group(s).", _RED))
            return 1

        all_results: List[CaseResult] = []
        for run in range(args.repeat):
            if args.repeat > 1:
                print(_c(f"\n── router cases, run {run + 1}/{args.repeat} " + "─" * 30, _DIM))
            all_results.extend(run_router_cases(cases, verbose=args.verbose))

        if any(r.fallback_used for r in all_results):
            print(_c("\nWARNING: at least one case hit the regex fallback — the router LLM "
                     "was unreachable, so these numbers do not describe the prompt.", _RED))
            exit_code = 1

        passed, total = _scorecard("ROUTER CASES", _by_group(all_results))
        if passed != total:
            exit_code = 1

    if not args.router_only:
        from conversation_harness import GOLDEN_CONVERSATION, print_transcript, run_conversation

        print(_c("\n── golden conversation (live pipeline) " + "─" * 32, _DIM))
        results = run_conversation(GOLDEN_CONVERSATION)
        print_transcript(results)

        rows: Dict[str, List[int]] = {}
        for tr in results:
            for chk in tr.checks:
                agg = rows.setdefault(chk.phase, [0, 0])
                agg[1] += 1
                if chk.passed:
                    agg[0] += 1
        if any(tr.fallback_used for tr in results):
            print(_c("\nWARNING: a turn hit the regex fallback — the router LLM was "
                     "unreachable during the conversation.", _RED))
            exit_code = 1
        passed, total = _scorecard("GOLDEN CONVERSATION", {k: (v[0], v[1]) for k, v in rows.items()})
        if passed != total:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
