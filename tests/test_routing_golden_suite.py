import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from query_routing.intent_classifier import IntentType, RouteDecision
from query_routing.route_policy_engine import build_route_decision_contract
from query_routing.router_signals import extract_query_signals
from query_routing.router_types import IntentCategory, RouteExecutor, RoutePlan


def _intent_category(intent: IntentType) -> IntentCategory:
    if intent in {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}:
        return IntentCategory.SEMANTIC_EXPLANATORY
    if intent == IntentType.FORECAST_DB:
        return IntentCategory.PREDICTION
    if intent in {IntentType.COMPARISON_DB, IntentType.ANOMALY_ANALYSIS_DB}:
        return IntentCategory.ANALYTICAL_VISUALIZATION
    return IntentCategory.STRUCTURED_FACTUAL_DB


def _golden_plan(
    *,
    question: str,
    intent: IntentType,
    confidence: float,
    response_mode: str,
    signal_overrides: dict | None = None,
) -> RoutePlan:
    signals = extract_query_signals(question=question, lab_name=None)
    if signal_overrides:
        signals.update(signal_overrides)
    return RoutePlan(
        decision=RouteDecision(intent=intent, confidence=confidence, reason="golden_test"),
        intent_category=_intent_category(intent),
        route_source="golden_test",
        planner_model="none",
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw={},
        planner_parameters={
            "response_mode": response_mode,
            "query_signals": signals,
        },
    )


class RoutingGoldenSuiteTests(unittest.TestCase):
    GOLDEN_CASES = [
        # Conceptual knowledge-only prompts.
        {"q": "How should I interpret PM2.5, TVOC, and humidity together?", "intent": "definition_explanation", "conf": 0.93, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "conceptual interpretation without explicit measured scope"},
        {"q": "What is the difference between a warning trend and an anomaly?", "intent": "definition_explanation", "conf": 0.92, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "domain terms but no lab/time request"},
        {"q": "What does CO2 mean in IEQ terms?", "intent": "definition_explanation", "conf": 0.91, "mode": "db", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "semantic question must ignore planner db drift"},
        {"q": "Define PM2.5 and why people track it indoors.", "intent": "definition_explanation", "conf": 0.9, "mode": "db", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "definition prompt"},
        {"q": "Explain why TVOC can change through the day.", "intent": "definition_explanation", "conf": 0.91, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "causal explanation"},
        {"q": "What is an IEQ index and how is it interpreted?", "intent": "definition_explanation", "conf": 0.94, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "concept-only request"},
        {"q": "How do humidity and comfort relate in offices?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "interpretive guidance without measured scope"},
        {"q": "What is considered high CO2 for indoor spaces?", "intent": "definition_explanation", "conf": 0.92, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "threshold explanation"},
        {"q": "When should I treat PM2.5 changes as noise vs signal?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "decision framing prompt"},
        {"q": "Explain anomaly detection in simple IEQ terms.", "intent": "definition_explanation", "conf": 0.89, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "conceptual anomaly question"},
        {"q": "What does ventilation quality usually affect first?", "intent": "definition_explanation", "conf": 0.89, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "guidance request"},
        {"q": "How can I explain IEQ to non-technical staff?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "communication guidance"},
        {"q": "What is the difference between stale air and high CO2?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "concept distinction"},
        {"q": "Explain how PM2.5 differs from TVOC in health impact.", "intent": "definition_explanation", "conf": 0.91, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "cross-metric conceptual ask"},
        {"q": "What does a stable IEQ trend usually indicate?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "trend concept without scoped data"},
        {"q": "How should occupants react to a short PM2.5 spike in principle?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "general response guidance"},
        {"q": "Give me a plain-language explanation of humidity comfort bands.", "intent": "definition_explanation", "conf": 0.92, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "definition-like ask"},
        {"q": "What should I infer from persistent moderate CO2 levels?", "intent": "definition_explanation", "conf": 0.91, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "interpretation without explicit scope"},
        {"q": "How do warning trends differ from immediate alerts?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "conceptual alert semantics"},
        {"q": "Explain what 'good air quality' usually means.", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "high-level definition"},
        {"q": "What is the role of noise in IEQ comfort?", "intent": "definition_explanation", "conf": 0.88, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "conceptual metric relation"},
        {"q": "Could you summarize IEQ caveats before acting on sensors?", "intent": "definition_explanation", "conf": 0.9, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "conceptual_semantic_forces_knowledge", "rationale": "knowledge-card style conceptual ask"},
        {"q": "If humidity is persistently above 70%, what risk should be flagged?", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "knowledge_qa", "rule": "hypothetical_without_live_scope_forces_knowledge", "rationale": "hypothetical conditional risk framing should stay semantic", "override": {"is_hypothetical_conditional": True, "requests_current_measured_data": False, "asks_for_db_facts": False, "query_scope_class": "ambiguous"}},
        # Scoped/measured prompts -> DB.
        {"q": "Average CO2 in smart_lab this week", "intent": "aggregation_db", "conf": 0.93, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "explicit metric + lab + time"},
        {"q": "Compare PM2.5 in smart_lab vs concrete_lab last week", "intent": "comparison_db", "conf": 0.94, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "explicit comparison scope"},
        {"q": "Is it comfortable in smart_lab right now?", "intent": "current_status_db", "conf": 0.9, "mode": "knowledge_only", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "comfort status needs measured data"},
        {"q": "Show humidity trend in smart_lab over the last 3 days", "intent": "aggregation_db", "conf": 0.93, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "time series scope"},
        {"q": "What was IEQ in smart_lab yesterday?", "intent": "point_lookup_db", "conf": 0.9, "mode": "knowledge_only", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "point lookup with time"},
        {"q": "Give me the latest CO2 for smart_lab", "intent": "current_status_db", "conf": 0.91, "mode": "knowledge_only", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "latest measured value"},
        {"q": "Trend of TVOC in smart_lab this month", "intent": "aggregation_db", "conf": 0.92, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "metric+lab+month scope"},
        {"q": "Which lab had lower PM2.5 today, smart_lab or shores_office?", "intent": "comparison_db", "conf": 0.92, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "today comparison"},
        {"q": "CO2 values per hour in smart_lab last 24 hours", "intent": "aggregation_db", "conf": 0.93, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "hourly series request"},
        {"q": "Forecast PM2.5 in smart_lab for next day", "intent": "forecast_db", "conf": 0.91, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "forecast intent"},
        {"q": "Any anomalies in CO2 for smart_lab this week?", "intent": "anomaly_analysis_db", "conf": 0.92, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "anomaly + scoped window"},
        {"q": "Average humidity in concrete_lab between 2026-03-01 and 2026-03-07", "intent": "aggregation_db", "conf": 0.93, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "explicit date window"},
        {"q": "Current PM2.5 and CO2 in smart_lab", "intent": "point_lookup_db", "conf": 0.89, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "current measured facts"},
        {"q": "Was smart_lab more humid than concrete_lab yesterday?", "intent": "comparison_db", "conf": 0.91, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "comparative past scope"},
        {"q": "Show temperature trend for shores_office past 7 days", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "past-N-days trend"},
        {"q": "What is the max CO2 in smart_lab this month?", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "aggregation operator"},
        {"q": "Give weekly IEQ average for smart_lab and concrete_lab", "intent": "comparison_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "cross-lab weekly aggregation"},
        {"q": "How was air quality in smart_lab on March 5?", "intent": "point_lookup_db", "conf": 0.88, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "day-specific scope"},
        {"q": "PM2.5 in smart_lab vs shores_office over last month", "intent": "comparison_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "explicit compare period"},
        {"q": "List anomalies for temperature in concrete_lab yesterday", "intent": "anomaly_analysis_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "anomaly query scoped"},
        {"q": "Give me PM2.5 chart for smart_lab this week", "intent": "aggregation_db", "conf": 0.91, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "chart over time requires measured data"},
        {"q": "What are the latest TVOC readings in shores_office?", "intent": "current_status_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "latest readings"},
        {"q": "Compare humidity and CO2 in smart_lab this week", "intent": "comparison_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "multi-metric scoped compare"},
        {"q": "Did PM2.5 spike in smart_lab last 48 hours?", "intent": "anomaly_analysis_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "spike + explicit window"},
        {"q": "Average sound level in smart_lab yesterday", "intent": "aggregation_db", "conf": 0.89, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "non-air metric with scope"},
        {"q": "How did IEQ change in smart_lab over past 14 days?", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "time-window trend"},
        {"q": "Predict CO2 in smart_lab next 12 hours", "intent": "forecast_db", "conf": 0.91, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "prediction scope"},
        {"q": "What is IEQ in smart_lab now?", "intent": "current_status_db", "conf": 0.9, "mode": "knowledge_only", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "status with lab"},
        {"q": "Compare light levels in smart_lab and concrete_lab today", "intent": "comparison_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "measured_scope_forces_db", "rationale": "compare two labs and day window"},
        {"q": "Compare humidity in concrete_lab against its baseline for this morning", "intent": "comparison_db", "conf": 0.9, "mode": "db", "exec": "db_query", "rule": "single_lab_baseline_forces_db", "rationale": "single-lab baseline reference comparison should remain DB", "override": {"is_baseline_reference_query": True, "single_explicit_lab_with_baseline_reference": True, "has_explicit_second_space": False, "asks_for_db_facts": True, "query_scope_class": "domain"}},
        # Ambiguous low confidence prompts -> clarify.
        {"q": "How is it over there?", "intent": "aggregation_db", "conf": 0.25, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "underspecified phrase without IEQ anchors should stay non-domain"},
        {"q": "Can you check that trend?", "intent": "aggregation_db", "conf": 0.25, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "underspecified trend phrase without IEQ anchors"},
        {"q": "Is this okay?", "intent": "unknown_fallback", "conf": 0.25, "mode": "knowledge_only", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "underspecified follow-up outside explicit IEQ scope", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "What about the other one?", "intent": "comparison_db", "conf": 0.25, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "implicit comparison phrase without scoped entities"},
        {"q": "Give me numbers.", "intent": "aggregation_db", "conf": 0.25, "mode": "db", "exec": "clarify_gate", "rule": "clarify_gate_confidence_or_strategy", "rationale": "scope absent"},
        {"q": "Can you explain that with data?", "intent": "aggregation_db", "conf": 0.25, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "non-domain ambiguous follow-up with no IEQ anchor"},
        {"q": "Show me the issue.", "intent": "anomaly_analysis_db", "conf": 0.25, "mode": "db", "exec": "clarify_gate", "rule": "clarify_gate_confidence_or_strategy", "rationale": "anomaly target missing"},
        {"q": "What changed?", "intent": "aggregation_db", "conf": 0.25, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "broad non-domain change question without IEQ anchors"},
        # Non-domain prompts -> knowledge with scope guardrail path.
        {"q": "What day is today?", "intent": "aggregation_db", "conf": 0.92, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "outside IEQ domain", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "Who won the football match yesterday?", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "non-domain sports question", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "What is the capital of Japan?", "intent": "current_status_db", "conf": 0.93, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "general geography question", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "Write a Python function to reverse a list.", "intent": "comparison_db", "conf": 0.91, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "coding question non-domain", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "How does compound interest work?", "intent": "aggregation_db", "conf": 0.9, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "finance non-domain", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
        {"q": "What's the weather tomorrow?", "intent": "forecast_db", "conf": 0.9, "mode": "db", "exec": "knowledge_qa", "rule": "non_domain_scope_forces_knowledge", "rationale": "external weather query", "override": {"query_scope_class": "non_domain", "asks_for_db_facts": False}},
    ]

    def test_golden_prompt_executor_expectations_and_rationale_snapshots(self):
        self.assertGreaterEqual(len(self.GOLDEN_CASES), 50)
        self.assertLessEqual(len(self.GOLDEN_CASES), 100)
        for case in self.GOLDEN_CASES:
            question = case["q"]
            intent = IntentType(case["intent"])
            overrides = dict(case.get("override") or {})
            if case["exec"] == "knowledge_qa" and intent in {
                IntentType.DEFINITION_EXPLANATION,
                IntentType.UNKNOWN_FALLBACK,
            } and "query_scope_class" not in overrides:
                overrides.update(
                    {
                        "query_scope_class": "ambiguous",
                        "asks_for_db_facts": False,
                        "is_general_knowledge_question": True,
                        "has_lab_reference": False,
                        "has_time_window_hint": False,
                        "has_db_scope_phrase": False,
                    }
                )
            plan = _golden_plan(
                question=question,
                intent=intent,
                confidence=float(case["conf"]),
                response_mode=str(case["mode"]),
                signal_overrides=overrides or None,
            )
            contract = build_route_decision_contract(
                latest_user_question=question,
                route_plan=plan,
                allow_clarify=True,
            )
            expected_executor = RouteExecutor(str(case["exec"]))
            expected_rule = str(case["rule"])
            self.assertEqual(
                contract.executor,
                expected_executor,
                msg=f"question={question} rationale={case['rationale']}",
            )
            self.assertGreater(len(contract.rule_trace), 0, msg=f"question={question}")
            # Snapshot-style rationale lock: first policy rule should stay stable.
            self.assertEqual(
                contract.rule_trace[0],
                expected_rule,
                msg=f"question={question} rationale={case['rationale']}",
            )

    def test_history_contamination_is_ignored_for_scope_signals(self):
        question = (
            "How should I interpret PM2.5, TVOC, and humidity together?\n\n"
            "Previous conversation context (most recent last):\n"
            "User: show PM2.5 in shores_office for the last 24 hours\n"
            "Assistant: PM2.5 was low."
        )
        signals = extract_query_signals(question=question, lab_name=None)
        self.assertFalse(bool(signals.get("asks_for_db_facts")))
        plan = _golden_plan(
            question=question,
            intent=IntentType.DEFINITION_EXPLANATION,
            confidence=0.9,
            response_mode="knowledge_only",
        )
        contract = build_route_decision_contract(
            latest_user_question=question,
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertEqual(contract.executor, RouteExecutor.KNOWLEDGE_QA)


if __name__ == "__main__":
    unittest.main()
