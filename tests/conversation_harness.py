"""Multi-turn conversation eval harness.

Unlike the per-question unit tests, this drives a *whole conversation* through the
real request pipeline — the same ``build_query_context`` → ``execute_query`` →
``persist_turn`` path the HTTP endpoints use — with a single shared
``conversation_id`` so each turn sees the persisted history of the ones before it.
That makes it possible to exercise the behaviours that only show up across turns:

  * carry-over      — a follow-up that only names a new metric/time keeps the rest
  * topic switch    — a new topic must NOT drag along the prior metric
  * date ranges     — "May 1-7" must resolve to a 7-day window, not a single day
  * ambiguity       — an unresolvable reference should trigger a clarify, not a guess
  * units           — "give me that in cm" should honour the requested unit
  * verbosity       — a simple factual question should get a short answer

Each turn carries an :class:`Expect` describing the *target* behaviour and the phase
of the improvement plan that delivers it (BASE = should already work today). Running
this on ``main`` therefore doubles as a gap report: checks tagged P0–P4 that fail are
exactly the things the plan fixes.

Run it live against the configured Ollama router/answer models:

    python tests/conversation_harness.py              # run the golden conversation
    python tests/conversation_harness.py --interactive # free-form REPL with memory

It uses an isolated temp conversation DB, so it never touches real ``data/conv.db``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)


# ---------------------------------------------------------------------------
# Scenario description
# ---------------------------------------------------------------------------

@dataclass
class Expect:
    """Target behaviour for one turn. Every field left ``None`` is not checked.

    Assertions are deliberately outcome-oriented (intent, resolved lab, resolved
    time window, answer content) rather than tied to the internal carry-over
    mechanism, so the same golden conversation stays meaningful after Phase 1
    replaces regex carry-over with an LLM ``resolved_question``.
    """

    phase: str = "BASE"                     # BASE | P0 | P1 | P2 | P3 | P4 | P5
    note: str = ""                          # what this turn is probing
    intent: Optional[str] = None            # exact metadata intent value
    resolved_lab: Optional[str] = None      # metadata.resolved_lab_name
    window_contains: Optional[str] = None   # substring of the resolved window label
    span_days_min: Optional[float] = None   # resolved window must span >= this many days
    clarify: Optional[bool] = None          # True => expect a clarify turn
    executor: Optional[str] = None          # metadata.executor
    answer_contains_any: Optional[List[str]] = None
    answer_contains_all: Optional[List[str]] = None
    answer_excludes: Optional[List[str]] = None
    max_answer_words: Optional[int] = None


@dataclass
class Turn:
    user: str
    lab: Optional[str] = None
    expect: Expect = field(default_factory=Expect)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    phase: str


@dataclass
class TurnResult:
    index: int
    user: str
    answer: str
    executor: Optional[str]
    intent: Optional[str]
    confidence: Optional[float]
    fallback_used: bool
    resolved_question: Optional[str]
    resolved_lab: Optional[str]
    window_label: Optional[str]
    window_start: Optional[str]
    window_end: Optional[str]
    timescale: Optional[str]
    carried_metric: Optional[str]
    carried_time_phrase: Optional[str]
    effective_lab: Optional[str]
    checks: List[Check] = field(default_factory=list)

    @property
    def answer_words(self) -> int:
        return len(self.answer.split())

    @property
    def is_clarify(self) -> bool:
        return self.timescale == "clarify" or self.executor == "clarify_gate"


# ---------------------------------------------------------------------------
# Isolated conversation store
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def isolated_conversation_db():
    """Point the conversation store at a throwaway DB for the duration.

    Rebinds ``_DB_PATH`` and resets the thread-local connection so the harness
    never reads or writes the real ``data/conv.db``.
    """
    import storage.conversation_store as store

    tmp = Path(tempfile.mkdtemp(prefix="conv_harness_")) / "conv.db"
    original_path = store._DB_PATH

    def _reset_conn():
        conn = getattr(store._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
            store._local.conn = None

    _reset_conn()
    store._DB_PATH = tmp
    try:
        yield tmp
    finally:
        _reset_conn()
        store._DB_PATH = original_path


# ---------------------------------------------------------------------------
# Driving a conversation
# ---------------------------------------------------------------------------

def _span_days(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[float]:
    if not start_iso or not end_iso:
        return None
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        return None
    return (end - start).total_seconds() / 86400.0


def _evaluate(expect: Expect, tr: TurnResult) -> List[Check]:
    checks: List[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name=name, passed=passed, detail=detail, phase=expect.phase))

    if expect.intent is not None:
        add("intent", tr.intent == expect.intent, f"want={expect.intent} got={tr.intent}")
    if expect.executor is not None:
        add("executor", tr.executor == expect.executor, f"want={expect.executor} got={tr.executor}")
    if expect.resolved_lab is not None:
        add("resolved_lab", tr.resolved_lab == expect.resolved_lab,
            f"want={expect.resolved_lab} got={tr.resolved_lab}")
    if expect.window_contains is not None:
        label = (tr.window_label or "").lower()
        add("window_contains", expect.window_contains.lower() in label,
            f"want~{expect.window_contains!r} got={tr.window_label!r}")
    if expect.span_days_min is not None:
        span = _span_days(tr.window_start, tr.window_end)
        add("span_days_min", span is not None and span >= expect.span_days_min,
            f"want>={expect.span_days_min}d got={None if span is None else round(span, 2)}d")
    if expect.clarify is not None:
        add("clarify", tr.is_clarify == expect.clarify,
            f"want_clarify={expect.clarify} got={tr.is_clarify} (timescale={tr.timescale})")
    if expect.answer_contains_any is not None:
        low = tr.answer.lower()
        hit = [t for t in expect.answer_contains_any if t.lower() in low]
        add("answer_contains_any", bool(hit),
            f"want_any={expect.answer_contains_any} matched={hit}")
    if expect.answer_contains_all is not None:
        low = tr.answer.lower()
        missing = [t for t in expect.answer_contains_all if t.lower() not in low]
        add("answer_contains_all", not missing, f"missing={missing}")
    if expect.answer_excludes is not None:
        low = tr.answer.lower()
        present = [t for t in expect.answer_excludes if t.lower() in low]
        add("answer_excludes", not present, f"unexpectedly_present={present}")
    if expect.max_answer_words is not None:
        add("max_answer_words", tr.answer_words <= expect.max_answer_words,
            f"want<={expect.max_answer_words} got={tr.answer_words}")
    return checks


def run_conversation(turns: List[Turn], *, k: int = 5) -> List[TurnResult]:
    """Drive a scripted conversation end-to-end through the real pipeline.

    Each turn is routed + executed with the shared conversation_id and then
    persisted, so the next turn's ``ConversationContext`` includes it. Returns the
    per-turn results with evaluated checks.
    """
    from http_routes.route_helpers import build_query_context, persist_turn
    from query_routing.query_orchestrator import execute_query

    results: List[TurnResult] = []
    with isolated_conversation_db():
        cid: Optional[str] = None
        for i, turn in enumerate(turns, start=1):
            ctx = build_query_context(turn.user, turn.lab, cid)
            cid = ctx.conversation_id
            res = execute_query(ctx, k=k)
            md = res.get("metadata", {}) or {}
            tw = md.get("time_window") or {}
            tr = TurnResult(
                index=i,
                user=turn.user,
                answer=str(res.get("answer") or ""),
                executor=md.get("executor"),
                intent=md.get("intent"),
                confidence=md.get("route_confidence"),
                fallback_used=bool(md.get("fallback_used")),
                resolved_question=md.get("resolved_question"),
                resolved_lab=md.get("resolved_lab_name"),
                window_label=tw.get("label"),
                window_start=tw.get("start"),
                window_end=tw.get("end"),
                timescale=res.get("timescale"),
                carried_metric=ctx.carried_metric,
                carried_time_phrase=ctx.carried_time_phrase,
                effective_lab=ctx.effective_lab,
            )
            tr.checks = _evaluate(turn.expect, tr)
            results.append(tr)
            persist_turn(cid, turn.user, tr.answer)
    return results


# ---------------------------------------------------------------------------
# The golden conversation
# ---------------------------------------------------------------------------

GOLDEN_CONVERSATION: List[Turn] = [
    Turn(
        user="How was the temperature in smart_lab in the first week of May 2026?",
        expect=Expect(
            phase="BASE", note="establishes metric+time+lab for the whole thread",
            intent="aggregation_db", resolved_lab="smart_lab",
            window_contains="May", answer_contains_any=["°c", "temperature", "degc"],
        ),
    ),
    Turn(
        user="What about the humidity?",
        expect=Expect(
            phase="P1", note="metric-only follow-up: keep lab + the May window",
            resolved_lab="smart_lab", window_contains="May",
            answer_contains_any=["humidity", "rh", "%"],
        ),
    ),
    Turn(
        user="And how did it trend last week?",
        expect=Expect(
            phase="P1", note="time-only follow-up: keep humidity, switch window to last week",
            resolved_lab="smart_lab", window_contains="last week",
            answer_contains_any=["humidity", "rh", "%"],
        ),
    ),
    Turn(
        user="Ok, what was the average CO2 in smart_lab for May 1-7?",
        expect=Expect(
            phase="P0", note="date RANGE without 'from/between' must span ~7 days, not 1",
            intent="aggregation_db", resolved_lab="smart_lab",
            span_days_min=6.0, answer_contains_any=["co2", "ppm"],
        ),
    ),
    Turn(
        user="How's the air quality right now?",
        expect=Expect(
            phase="P1", note="topic switch: must NOT carry the May window or CO2-only scope",
            resolved_lab="smart_lab", answer_excludes=["first week of may"],
            answer_contains_any=["air", "ieq", "iaq", "co2", "quality"],
        ),
    ),
    Turn(
        user="What about over there?",
        expect=Expect(
            phase="P2",
            note="genuine ambiguity: 'over there' names no resolvable lab -> should clarify, "
                 "not silently reuse smart_lab",
            clarify=True,
        ),
    ),
    Turn(
        user="How wide is the main door?",
        expect=Expect(
            phase="BASE", note="IFC model question",
            executor="ifc_qa",
        ),
    ),
    Turn(
        user="Give me that in centimeters.",
        expect=Expect(
            phase="P3", note="unit follow-up: the answer must be stated in cm",
            executor="ifc_qa",
            # Only the presence of cm is checked. Excluding "mm" was too strict: showing the
            # conversion it came from ("2200 mm → 220 cm") is good grounding, not a failure,
            # and an answer left in millimetres fails this check anyway — it would contain
            # neither "cm" nor "centimet".
            answer_contains_any=["cm", "centimet"],
        ),
    ),
    Turn(
        user="Is the CO2 level ok?",
        expect=Expect(
            phase="P4", note="simple factual question -> short answer",
            intent="current_status_db", max_answer_words=90,
            answer_contains_any=["co2", "ppm"],
        ),
    ),
    # P5 — the question asked is the thing that must be answered. These turns ask
    # what to DO; a status report that happens to mention the metric is a failure,
    # which is why the action vocabulary is asserted and the two stock non-answers
    # ("no action needed" / "no immediate action") are excluded.
    Turn(
        user="How can I improve the air quality?",
        expect=Expect(
            phase="P5", note="advice question -> concrete actions, not a status dump",
            intent="current_status_db",
            answer_contains_any=[
                "ventilat", "window", "filter", "purif", "increase",
                "reduce", "adjust", "clean",
            ],
            answer_excludes=["no action needed", "no immediate action"],
        ),
    ),
    Turn(
        user="What would you recommend to improve VOC?",
        expect=Expect(
            phase="P5",
            note="advice on an in-range metric: still owes actions, not 'nothing to do'",
            answer_contains_any=[
                "ventilat", "window", "filter", "purif", "increase",
                "reduce", "adjust", "clean", "source",
            ],
            answer_excludes=["no action needed", "no immediate action"],
        ),
    ),
    Turn(
        user="What is VOC?",
        expect=Expect(
            phase="P5",
            note="concept question after advice turns: answer what it IS, no reading, "
                 "and the article rule still holds deep in a conversation",
            # The point of this turn is the intent. The cap is loose on purpose — it is
            # here to catch a definition that turns into a full report, not to police
            # length; P4 is the strict brevity check (90 words on a factual question).
            intent="definition_explanation", max_answer_words=180,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_GREEN, _RED, _YELLOW, _DIM, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _c(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{_RESET}"


def print_transcript(results: List[TurnResult]) -> None:
    for tr in results:
        print()
        print(_c(f"── Turn {tr.index} " + "─" * 60, _DIM))
        print(_c("You:", _YELLOW), tr.user)
        routing = (
            f"intent={tr.intent} conf={tr.confidence} executor={tr.executor} "
            f"lab={tr.resolved_lab} window={tr.window_label!r} timescale={tr.timescale}"
        )
        if tr.fallback_used:
            routing += _c("  [REGEX FALLBACK — LLM router unreachable]", _RED)
        print(_c("Routing:", _DIM), _c(routing, _DIM))
        print(_c("Memory:", _DIM),
              _c(f"carried_metric={tr.carried_metric} carried_time={tr.carried_time_phrase} "
                 f"effective_lab={tr.effective_lab}", _DIM))
        ans = tr.answer if len(tr.answer) <= 500 else tr.answer[:500] + " …"
        print(_c("Bot:", _GREEN), ans.replace("\n", "\n     "))
        for chk in tr.checks:
            mark = _c("PASS", _GREEN) if chk.passed else _c("FAIL", _RED)
            print(f"   [{mark}] ({chk.phase}) {chk.name}: {chk.detail}")


def summarize(results: List[TurnResult]) -> Tuple[int, int]:
    all_checks = [c for tr in results for c in tr.checks]
    passed = sum(1 for c in all_checks if c.passed)
    total = len(all_checks)

    by_phase: dict = {}
    for c in all_checks:
        agg = by_phase.setdefault(c.phase, [0, 0])
        agg[1] += 1
        if c.passed:
            agg[0] += 1

    print()
    print(_c("═" * 72, _DIM))
    print("SCORECARD")
    for phase in sorted(by_phase):
        p, t = by_phase[phase]
        color = _GREEN if p == t else _RED
        print(f"  {phase:5} {_c(f'{p}/{t}', color)} checks")
    fallback = any(tr.fallback_used for tr in results)
    if fallback:
        print(_c("  WARNING: at least one turn fell back to regex (router LLM unreachable) — "
                 "this was NOT a real LLM conversation.", _RED))
    overall_color = _GREEN if passed == total else _YELLOW
    print(_c(f"  TOTAL {passed}/{total}", overall_color))
    print(_c("═" * 72, _DIM))
    return passed, total


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive() -> None:
    """Free-form multi-turn chat against the live pipeline (memory preserved)."""
    from http_routes.route_helpers import build_query_context, persist_turn
    from query_routing.query_orchestrator import execute_query

    print(_c("Interactive conversation — type 'exit' to quit, 'reset' for a new thread.", _DIM))
    with isolated_conversation_db():
        cid: Optional[str] = None
        while True:
            try:
                user = input(_c("\nYou: ", _YELLOW)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user.lower() == "reset":
                cid = None
                print(_c("(started a new conversation thread)", _DIM))
                continue
            ctx = build_query_context(user, None, cid)
            cid = ctx.conversation_id
            res = execute_query(ctx, k=5)
            md = res.get("metadata", {}) or {}
            tw = md.get("time_window") or {}
            routing = (f"intent={md.get('intent')} lab={md.get('resolved_lab_name')} "
                       f"window={tw.get('label')!r} timescale={res.get('timescale')}")
            if md.get("fallback_used"):
                routing += "  [REGEX FALLBACK]"
            print(_c(f"[{routing}]", _DIM))
            print(_c("Bot:", _GREEN), str(res.get("answer") or ""))
            persist_turn(cid, user, str(res.get("answer") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-turn conversation eval harness")
    parser.add_argument("--interactive", action="store_true", help="free-form chat with memory")
    args = parser.parse_args()

    if args.interactive:
        interactive()
        return 0

    results = run_conversation(GOLDEN_CONVERSATION)
    print_transcript(results)
    passed, total = summarize(results)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
