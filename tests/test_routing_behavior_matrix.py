import os
import sys
import unittest
from typing import Iterable, Optional, Sequence, Set


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
from query_routing.router_types import AnswerStrategy, IntentCategory, RouteExecutor, RoutePlan


def _intent_category(intent: IntentType) -> IntentCategory:
    if intent in {IntentType.DEFINITION_EXPLANATION, IntentType.UNKNOWN_FALLBACK}:
        return IntentCategory.SEMANTIC_EXPLANATORY
    if intent == IntentType.FORECAST_DB:
        return IntentCategory.PREDICTION
    if intent in {IntentType.COMPARISON_DB, IntentType.ANOMALY_ANALYSIS_DB}:
        return IntentCategory.ANALYTICAL_VISUALIZATION
    return IntentCategory.STRUCTURED_FACTUAL_DB


def _build_plan(
    *,
    question: str,
    intent: IntentType,
    response_mode: str,
    needs_measured_data: bool,
    confidence: float = 0.92,
    answer_strategy: AnswerStrategy = AnswerStrategy.DIRECT,
    clarify_reason: Optional[str] = None,
) -> RoutePlan:
    signals = extract_query_signals(question=question, lab_name=None)
    signals["needs_measured_data"] = needs_measured_data
    return RoutePlan(
        decision=RouteDecision(intent=intent, confidence=confidence, reason="matrix_test"),
        intent_category=_intent_category(intent),
        route_source="matrix_test",
        planner_model="none",
        planner_fallback_used=False,
        planner_fallback_reason=None,
        planner_raw={},
        planner_parameters={
            "response_mode": response_mode,
            "query_signals": signals,
            "needs_measured_data": needs_measured_data,
            "clarify_reason": clarify_reason,
        },
        answer_strategy=answer_strategy,
        clarify_reason=clarify_reason,
    )


class RoutingBehaviorMatrixTests(unittest.TestCase):
    def _assert_executor(
        self,
        *,
        question: str,
        intent: IntentType,
        response_mode: str,
        needs_measured_data: bool,
        expected: Set[RouteExecutor],
        confidence: float = 0.92,
        answer_strategy: AnswerStrategy = AnswerStrategy.DIRECT,
        clarify_reason: Optional[str] = None,
    ) -> None:
        plan = _build_plan(
            question=question,
            intent=intent,
            response_mode=response_mode,
            needs_measured_data=needs_measured_data,
            confidence=confidence,
            answer_strategy=answer_strategy,
            clarify_reason=clarify_reason,
        )
        contract = build_route_decision_contract(
            latest_user_question=question,
            route_plan=plan,
            allow_clarify=True,
        )
        self.assertIn(contract.executor, expected, msg=f"question={question}")
        self.assertGreater(len(contract.rule_trace), 0, msg=f"question={question}")

    def _assert_group(
        self,
        *,
        questions: Sequence[str],
        intent: IntentType,
        response_mode: str,
        needs_measured_data: bool,
        expected: Set[RouteExecutor],
        confidence: float = 0.92,
        answer_strategy: AnswerStrategy = AnswerStrategy.DIRECT,
        clarify_reason: Optional[str] = None,
    ) -> None:
        for q in questions:
            self._assert_executor(
                question=q,
                intent=intent,
                response_mode=response_mode,
                needs_measured_data=needs_measured_data,
                expected=expected,
                confidence=confidence,
                answer_strategy=answer_strategy,
                clarify_reason=clarify_reason,
            )

    def test_general_knowledge_definition_queries_route_to_knowledge(self):
        questions = [
            "I keep hearing about PM2.5 but nobody explains what makes it different from regular dust, can you break that down?",
            "What's the relationship between CO2 buildup and how tired people feel in a room?",
            "If humidity and temperature both affect comfort, which one matters more and why?",
            "Someone told me TVOC is dangerous but I don't really understand what volatile means in this context",
            "How does the IEQ index actually get calculated, like what goes into that number?",
            "Is 600 ppm CO2 considered good or is that still too high for a classroom setting?",
            "Why would PM2.5 spike indoors if all the windows are closed?",
            "What's the difference between TVOC and CO2 as indicators of air quality problems?",
            "Does high humidity make CO2 feel worse or are they completely independent factors?",
            "At what point does noise level actually become harmful versus just annoying?",
        ]
        self._assert_group(
            questions=questions,
            intent=IntentType.DEFINITION_EXPLANATION,
            response_mode="knowledge_only",
            needs_measured_data=False,
            expected={RouteExecutor.KNOWLEDGE_QA},
        )

    def test_scoped_lab_data_queries_route_to_db_without_clarify(self):
        questions = [
            "Has CO2 been climbing steadily in smart_lab since Monday or did it spike and drop?",
            "What were the worst three hours for PM2.5 in concrete_lab last week?",
            "Give me a full picture of air quality in shores_office right now - everything you have",
            "Is smart_lab currently comfortable or should people working late tonight expect issues?",
            "How did IEQ behave in concrete_lab during the afternoon hours yesterday?",
            "Was there anything unusual about humidity in smart_lab over the past 48 hours?",
            "What's the CO2 situation in shores_office looking like compared to this time yesterday?",
            "Show me TVOC readings for smart_lab last Tuesday, I want to see if there was a pattern",
            "Did temperature in concrete_lab stay within comfortable range all of last week?",
            "What's the current light level in smart_lab and is it adequate for focused work?",
        ]
        self._assert_group(
            questions=questions,
            intent=IntentType.AGGREGATION_DB,
            response_mode="db",
            needs_measured_data=True,
            expected={RouteExecutor.DB_QUERY},
        )

    def test_multi_space_comparisons_route_to_comparison_db(self):
        questions = [
            "Which space tends to have better air quality overall, smart_lab or concrete_lab?",
            "I work between smart_lab and shores_office - which one should I prefer if I care about CO2?",
            "How does PM2.5 differ between concrete_lab and shores_office on typical weekday mornings?",
            "Are the humidity conditions meaningfully different between smart_lab and concrete_lab or roughly the same?",
            "If I had to pick one lab for someone with respiratory sensitivity, smart_lab or shores_office, based on recent air quality?",
            "Compare how IEQ scores trend through the day in smart_lab versus concrete_lab this week",
            "Which lab had more CO2 anomalies last month, smart_lab or concrete_lab?",
            "Is shores_office consistently noisier than smart_lab or does it depend on the time of day?",
            "Between smart_lab and concrete_lab, which recovered faster after a CO2 spike last week?",
            "How do temperature swings compare between the two labs - is one more stable than the other?",
        ]
        for q in questions:
            self._assert_executor(
                question=q,
                intent=IntentType.COMPARISON_DB,
                response_mode="db",
                needs_measured_data=True,
                expected={RouteExecutor.DB_QUERY},
            )

    def test_metric_only_comparisons_without_lab_route_to_knowledge(self):
        questions = [
            "Which is a more reliable indicator of poor ventilation, CO2 or TVOC?",
            "Between PM2.5 and TVOC, which one is harder to control once it starts rising?",
            "Does CO2 rise faster than humidity when a room fills up with people?",
            "Which metric tends to drop first when you open a window, CO2 or TVOC?",
            "How does PM2.5 compare to CO2 in terms of how quickly occupants feel the effects?",
            "Is IEQ more sensitive to temperature changes or humidity changes generally?",
            "What's more concerning for health at elevated levels - high CO2 or high PM2.5?",
        ]
        self._assert_group(
            questions=questions,
            intent=IntentType.DEFINITION_EXPLANATION,
            response_mode="knowledge_only",
            needs_measured_data=False,
            expected={RouteExecutor.KNOWLEDGE_QA},
        )

    def test_ambiguous_queries_route_to_clarify_or_knowledge(self):
        clarify_or_knowledge = {RouteExecutor.CLARIFY_GATE, RouteExecutor.KNOWLEDGE_QA}
        questions = [
            "Is the air quality okay?",
            "What does CO2 look like right now?",
            "Is it comfortable in there?",
            "Has it been getting worse lately?",
            "What happened to the IEQ score?",
            "Is PM2.5 high?",
            "How's the air in the lab?",
            "Is CO2 normal for this time of day?",
            "Should I be worried about the readings?",
            "What was it like earlier today?",
        ]
        for q in questions:
            self._assert_executor(
                question=q,
                intent=IntentType.CURRENT_STATUS_DB,
                response_mode="db",
                needs_measured_data=True,
                expected=clarify_or_knowledge,
                confidence=0.45,
                answer_strategy=AnswerStrategy.CLARIFY,
                clarify_reason="ambiguous_intent",
            )

    def test_diagnostic_root_cause_queries_route_to_db(self):
        questions = [
            "IEQ has been dropping in smart_lab every afternoon this week - what's driving that?",
            "Something is causing CO2 to spike in concrete_lab around 2pm, can you figure out what?",
            "Why does smart_lab feel stuffy even when CO2 readings look acceptable?",
            "The IEQ score in shores_office tanked last Thursday - what metrics were responsible?",
            "CO2 looks fine in smart_lab but people are still complaining about air quality - what else could explain that?",
            "Is there a pattern between when TVOC rises and when IEQ drops in concrete_lab?",
            "What's the most likely cause of the humidity anomaly in smart_lab last weekend?",
            "Why is the IEQ score in concrete_lab always lower in the morning than the afternoon?",
        ]
        self._assert_group(
            questions=questions,
            intent=IntentType.AGGREGATION_DB,
            response_mode="db",
            needs_measured_data=True,
            expected={RouteExecutor.DB_QUERY},
        )

    def test_anomaly_detection_queries_route_to_anomaly_db(self):
        questions = [
            "Were there any unusual CO2 spikes in smart_lab over the past week that shouldn't be there?",
            "Did anything abnormal happen to air quality in shores_office last month?",
            "Flag any outlier PM2.5 readings in concrete_lab in the last 30 days",
            "Are there any anomalies in the IEQ trend for smart_lab this week that I should know about?",
            "Did temperature behave unusually in concrete_lab during the weekend?",
            "Has TVOC shown any spikes in smart_lab that don't match the normal pattern?",
        ]
        self._assert_group(
            questions=questions,
            intent=IntentType.ANOMALY_ANALYSIS_DB,
            response_mode="db",
            needs_measured_data=True,
            expected={RouteExecutor.DB_QUERY},
        )

    def test_adversarial_queries_follow_expected_routing_contract(self):
        knowledge_only = {
            "Compare CO2 in smart_lab to what's considered healthy",
            "What does CO2 mean and what is it right now in smart_lab?",
            "Is PM2.5 dangerous and how high is it in concrete_lab today?",
            "What's a normal CO2 level and are we within that range in shores_office?",
        }
        clarify_expected = {
            "Tell me everything about air quality",
            "Is it better now than before?",
            "Compare this week to last week",
            "Which lab has the problem?",
        }
        db_expected = {
            "How does smart_lab compare to itself over the last two weeks?",
            "Should we open a window in smart_lab?",
        }

        for q in sorted(knowledge_only):
            self._assert_executor(
                question=q,
                intent=IntentType.DEFINITION_EXPLANATION,
                response_mode="knowledge_only",
                needs_measured_data=False,
                expected={RouteExecutor.KNOWLEDGE_QA},
            )

        for q in sorted(clarify_expected):
            self._assert_executor(
                question=q,
                intent=IntentType.CURRENT_STATUS_DB,
                response_mode="db",
                needs_measured_data=True,
                expected={RouteExecutor.CLARIFY_GATE},
                confidence=0.35,
                answer_strategy=AnswerStrategy.CLARIFY,
                clarify_reason="ambiguous_intent",
            )

        for q in sorted(db_expected):
            self._assert_executor(
                question=q,
                intent=IntentType.AGGREGATION_DB,
                response_mode="db",
                needs_measured_data=True,
                expected={RouteExecutor.DB_QUERY},
            )


if __name__ == "__main__":
    unittest.main()
