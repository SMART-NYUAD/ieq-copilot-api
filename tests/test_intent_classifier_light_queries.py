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

if __name__ == "__main__":
    unittest.main()
