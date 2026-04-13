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

from query_routing.intent_classifier import IntentType, classify_intent
from query_routing.router_signals import extract_query_signals


class IntentClassifierLightQueryTests(unittest.TestCase):
    def test_how_is_light_in_lab_maps_to_current_status_db(self):
        decision = classify_intent("How is the light in the smart lab?")
        self.assertEqual(decision.intent, IntentType.CURRENT_STATUS_DB)

    def test_how_is_pm25_in_lab_maps_to_current_status_db(self):
        decision = classify_intent("How is the pm2.5 in the smart lab?")
        self.assertEqual(decision.intent, IntentType.CURRENT_STATUS_DB)

    def test_classifier_exposes_ranked_intent_candidates(self):
        decision = classify_intent("Compare smart_lab vs concrete_lab this week")
        ranked = decision.ranked_intents
        self.assertTrue(len(ranked) >= 1)
        self.assertEqual(ranked[0][0], decision.intent)
        self.assertGreaterEqual(ranked[0][1], 0.0)

    def test_router_signals_expose_weighted_strengths(self):
        strong = extract_query_signals(
            "What is the CO2 level in smart_lab right now?",
            lab_name=None,
        )
        weak = extract_query_signals(
            "How is the air?",
            lab_name=None,
        )
        for key in ("metric_signal_strength", "scope_signal_strength", "diagnostic_signal_strength"):
            self.assertIn(key, strong)
            self.assertGreaterEqual(float(strong[key]), 0.0)
            self.assertLessEqual(float(strong[key]), 1.0)
        self.assertGreater(float(strong["metric_signal_strength"]), float(weak["metric_signal_strength"]))
        self.assertGreater(float(strong["scope_signal_strength"]), float(weak["scope_signal_strength"]))

    def test_unscoped_metric_comparison_routes_to_definition(self):
        decision = classify_intent("how does PM2.5 compare with CO2 or humidity")
        self.assertEqual(decision.intent, IntentType.DEFINITION_EXPLANATION)

    def test_unscoped_metric_comparison_not_marked_db_facts(self):
        signals = extract_query_signals("how does PM2.5 compare with CO2 or humidity", lab_name=None)
        self.assertFalse(bool(signals.get("asks_for_db_facts")))

    def test_scoped_lab_comparison_stays_comparison_db(self):
        decision = classify_intent("compare PM2.5 in smart_lab vs concrete_lab")
        self.assertEqual(decision.intent, IntentType.COMPARISON_DB)

    def test_time_scoped_comparison_keeps_db_facts_signal(self):
        signals = extract_query_signals("how does PM2.5 compare this week", lab_name=None)
        self.assertTrue(bool(signals.get("asks_for_db_facts")))

    def test_unscoped_metric_comparison_is_not_domain_scope(self):
        signals = extract_query_signals("how does PM2.5 compare with CO2 or humidity", lab_name=None)
        self.assertFalse(bool(signals.get("is_comfort_assessment_phrase")))
        self.assertFalse(bool(signals.get("asks_for_db_facts")))
        self.assertEqual(str(signals.get("query_scope_class") or ""), "ambiguous")

    def test_complex_wording_routes_to_aggregation_db(self):
        decision = classify_intent(
            "Can you walk me through how indoor air quality has been shifting throughout today in smart lab?"
        )
        self.assertEqual(decision.intent, IntentType.AGGREGATION_DB)

    def test_natural_metric_alias_is_detected_as_metric_reference(self):
        signals = extract_query_signals(
            "How has carbon dioxide been varying lately in smart lab?",
            lab_name=None,
        )
        self.assertTrue(bool(signals.get("has_metric_reference")))
        self.assertTrue(bool(signals.get("has_time_window_hint")))
        self.assertTrue(bool(signals.get("asks_for_db_facts")))

if __name__ == "__main__":
    unittest.main()
