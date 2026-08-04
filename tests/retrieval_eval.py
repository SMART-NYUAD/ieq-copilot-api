#!/usr/bin/env python3
"""Score knowledge-card retrieval against the live embedding store.

Network by design, like ``tests/router_eval.py`` — it needs Postgres and the embedding
model, so it is deliberately NOT part of ``unittest discover`` (the suite is hermetic and
the card store is on a non-loopback host). This is the only thing that measures retrieval.

    python tests/retrieval_eval.py                # score the golden cases
    python tests/retrieval_eval.py --group definition
    python tests/retrieval_eval.py --verbose      # show the ranked cards per case
    python tests/retrieval_eval.py --probe-sweep  # ivfflat recall diagnostic
    python tests/retrieval_eval.py --ablate       # with/without the bge query prefix

Cases live in ``tests/retrieval_eval_cases.json``; a retrieval bug belongs there as a case
first. A case passes when the TOP-1 card satisfies every ``expect_*`` key it declares;
``recall@3`` additionally reports whether an acceptable card appeared in the top three,
which is what actually reaches the answer model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.knowledge_executor import search_knowledge_cards  # noqa: E402
from storage.embeddings import embed_query, embed_texts  # noqa: E402
from storage.postgres_client import get_cursor  # noqa: E402

CASES_PATH = os.path.join(TEST_DIR, "retrieval_eval_cases.json")


@dataclass
class CaseResult:
    case_id: str
    group: str
    question: str
    top1_ok: bool
    recall3_ok: bool
    cards: List[Dict[str, Any]]
    detail: str


def _card_matches(card: Dict[str, Any], expect: Dict[str, Any]) -> bool:
    """Whether one card satisfies every expectation a case declares."""
    if "expect_metric" in expect:
        if str(card.get("metric_name") or "").lower() != str(expect["expect_metric"]).lower():
            return False
    if "expect_type" in expect:
        if str(card.get("card_type") or "") != expect["expect_type"]:
            return False
    if "expect_title_contains" in expect:
        needle = str(expect["expect_title_contains"]).lower()
        haystack = f"{card.get('title') or ''} {card.get('summary') or ''}".lower()
        if needle not in haystack:
            return False
    return True


def run_cases(cases: List[Dict[str, Any]], k: int = 3, verbose: bool = False) -> List[CaseResult]:
    results: List[CaseResult] = []
    for case in cases:
        question = case["question"]
        cards = search_knowledge_cards(question, k=k) or []
        expect = {key: val for key, val in case.items() if key.startswith("expect_")}
        top1_ok = bool(cards) and _card_matches(cards[0], expect)
        recall3_ok = any(_card_matches(card, expect) for card in cards)
        want = ", ".join(f"{key.removeprefix('expect_')}={val}" for key, val in expect.items())
        got = (
            f"{cards[0].get('card_type')}/{cards[0].get('metric_name')}/"
            f"{str(cards[0].get('title'))[:40]}"
            if cards
            else "NO CARDS RETURNED"
        )
        results.append(
            CaseResult(
                case_id=case["id"],
                group=case.get("group", "ungrouped"),
                question=question,
                top1_ok=top1_ok,
                recall3_ok=recall3_ok,
                cards=cards,
                detail=f"want[{want}] got[{got}]",
            )
        )
        status = "PASS" if top1_ok else ("top3" if recall3_ok else "FAIL")
        print(f"  [{status:4}] {case.get('group','')}/{case['id']}")
        if verbose or not top1_ok:
            print(f"         {question!r}")
            print(f"         {results[-1].detail}")
            for rank, card in enumerate(cards, 1):
                print(
                    f"           {rank}. [{card.get('card_type'):<14}] "
                    f"{str(card.get('metric_name')):<10} {str(card.get('title'))[:52]}"
                )
    return results


def _scorecard(results: List[CaseResult]) -> None:
    groups: Dict[str, List[CaseResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)
    width = 72
    print("\n" + "=" * width)
    print("RETRIEVAL — top-1 correct / recall@3 / total")
    for group in sorted(groups):
        rows = groups[group]
        print(
            f"  {group:<16} {sum(r.top1_ok for r in rows):>3} / "
            f"{sum(r.recall3_ok for r in rows):>3} / {len(rows):<3}"
        )
    print(
        f"  {'TOTAL':<16} {sum(r.top1_ok for r in results):>3} / "
        f"{sum(r.recall3_ok for r in results):>3} / {len(results):<3}"
    )
    print("=" * width)


def probe_sweep() -> None:
    """Diagnostic: how much of the corpus each ivfflat probe setting can actually reach.

    An ivfflat index partitions the vectors into `lists` buckets and scans only `probes` of
    them. When `lists` is large relative to the row count most buckets are empty, so a low
    probe count returns fewer rows than the LIMIT asks for — silently, as reduced recall
    rather than an error. This sweep is what exposed that.
    """
    emb = embed_query("what is CO2?")
    sql = (
        "SELECT c.title, 1-(e.embedding <=> %s::vector) AS sim "
        "FROM env_knowledge_card_embeddings e JOIN env_knowledge_cards c "
        "ON c.knowledge_card_id = e.knowledge_card_id "
        "ORDER BY e.embedding <=> %s::vector LIMIT 10"
    )
    with get_cursor(real_dict=True) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM env_knowledge_card_embeddings")
        total = cur.fetchone()["n"]
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename='env_knowledge_card_embeddings'"
        )
        idx = [r["indexdef"] for r in cur.fetchall()]
        print(f"\ncorpus: {total} embedded cards")
        for line in idx:
            print(f"index:  {line}")
        print("\n  probes   rows returned (LIMIT 10)   top similarity")
        for probes in (1, 2, 5, 10, 25, 50, 100):
            cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
            cur.execute(sql, (emb, emb))
            rows = cur.fetchall()
            top = f"{rows[0]['sim']:.4f} {str(rows[0]['title'])[:34]}" if rows else "-"
            print(f"  {probes:>6}   {len(rows):>6}                    {top}")


def ablate_prefix() -> None:
    """Measure the bge query-instruction prefix, query side only.

    bge-*-en-v1.5 is trained with an asymmetric retrieval objective: queries carry an
    instruction prefix, passages do not. Embedding both the same way is the documented
    misuse, and it costs ranking quality on exactly the short questions this system gets.
    """
    sql = (
        "SELECT c.title, c.card_type, c.metric_name, 1-(e.embedding <=> %s::vector) AS sim "
        "FROM env_knowledge_card_embeddings e JOIN env_knowledge_cards c "
        "ON c.knowledge_card_id = e.knowledge_card_id "
        "ORDER BY e.embedding <=> %s::vector LIMIT 3"
    )
    questions = ["what is CO2?", "what is VOC?", "what is PM2.5?", "is this air dangerous?"]
    with get_cursor(real_dict=True) as cur:
        cur.execute("SET LOCAL ivfflat.probes = 1000")
        for question in questions:
            print(f"\n  {question!r}")
            for label, emb in (
                ("raw   ", embed_texts([question])[0]),
                ("prefix", embed_query(question)),
            ):
                cur.execute(sql, (emb, emb))
                rows = cur.fetchall()
                shown = " | ".join(
                    f"{r['sim']:.3f} {str(r['title'])[:26]}" for r in rows
                )
                print(f"    {label}: {shown}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="score only one group")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--probe-sweep", action="store_true", help="ivfflat recall diagnostic")
    parser.add_argument("--ablate", action="store_true", help="bge query-prefix ablation")
    args = parser.parse_args()

    if args.probe_sweep:
        probe_sweep()
        return 0
    if args.ablate:
        ablate_prefix()
        return 0

    with open(CASES_PATH, "r", encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    if args.group:
        cases = [c for c in cases if c.get("group") == args.group]
        if not cases:
            print(f"no cases in group {args.group!r}")
            return 2

    results = run_cases(cases, verbose=args.verbose)
    _scorecard(results)
    return 0 if all(r.top1_ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
