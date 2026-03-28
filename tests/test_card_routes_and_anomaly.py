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

from query_routing.intent_classifier import IntentType
from query_routing.query_orchestrator import resolve_execution_intent
from executors.db_query_executor import _detect_anomaly_points


class CardRoutesAndAnomalyTests(unittest.TestCase):
    def test_execution_intent_uses_router_task_intents_directly(self):
        self.assertEqual(
            resolve_execution_intent(IntentType.DEFINITION_EXPLANATION),
            IntentType.DEFINITION_EXPLANATION,
        )
        self.assertEqual(
            resolve_execution_intent(IntentType.CURRENT_STATUS_DB),
            IntentType.CURRENT_STATUS_DB,
        )
        self.assertEqual(
            resolve_execution_intent(IntentType.ANOMALY_ANALYSIS_DB),
            IntentType.ANOMALY_ANALYSIS_DB,
        )
        self.assertEqual(
            resolve_execution_intent(IntentType.UNKNOWN_FALLBACK),
            IntentType.UNKNOWN_FALLBACK,
        )

    def test_db_intents_are_not_remapped(self):
        self.assertEqual(
            resolve_execution_intent(IntentType.POINT_LOOKUP_DB),
            IntentType.POINT_LOOKUP_DB,
        )
        self.assertEqual(
            resolve_execution_intent(IntentType.COMPARISON_DB),
            IntentType.COMPARISON_DB,
        )

    def test_backend_anomaly_detector_flags_spike(self):
        rows = [
            {"lab_space": "smart_lab", "bucket": f"2026-03-10T{hour:02d}:00:00+00:00", "value": 20.0}
            for hour in range(7)
        ] + [
            {"lab_space": "smart_lab", "bucket": "2026-03-10T07:00:00+00:00", "value": 120.0}
        ] + [
            {"lab_space": "smart_lab", "bucket": f"2026-03-10T{hour:02d}:00:00+00:00", "value": 21.0}
            for hour in range(8, 12)
        ]
        anomalies = _detect_anomaly_points(rows, z_threshold=2.0)
        self.assertGreaterEqual(len(anomalies), 1)
        self.assertTrue(any(float(item.get("value") or 0.0) >= 120.0 for item in anomalies))

if __name__ == "__main__":
    unittest.main()
