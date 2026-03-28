import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from executors.db_query_executor import _resolve_lab_alias
from query_routing.llm_router_planner import extract_query_signals


class LabAliasResolutionTests(unittest.TestCase):
    @patch("executors.db_query_executor._build_lab_alias_map")
    def test_resolve_lab_alias_accepts_natural_and_short_forms(self, mock_alias_map):
        mock_alias_map.return_value = {
            "smart_lab": "smart_lab",
            "smart lab": "smart_lab",
            "smart": "smart_lab",
            "concrete_lab": "concrete_lab",
            "concrete lab": "concrete_lab",
            "concrete": "concrete_lab",
        }

        self.assertEqual(_resolve_lab_alias("smart_lab"), "smart_lab")
        self.assertEqual(_resolve_lab_alias("smart lab"), "smart_lab")
        self.assertEqual(_resolve_lab_alias("smart"), "smart_lab")
        self.assertEqual(_resolve_lab_alias("concrete lab"), "concrete_lab")

    def test_router_extracts_lab_candidates_from_natural_language(self):
        signals = extract_query_signals(
            question="Compare smart lab vs concrete lab this week",
            lab_name=None,
        )
        self.assertTrue(signals.get("has_lab_reference"))
        candidates = signals.get("lab_candidates") or []
        self.assertIn("smart_lab", candidates)
        self.assertIn("concrete_lab", candidates)

    def test_router_extracts_candidates_from_one_word_compare(self):
        signals = extract_query_signals(
            question="compare smart and concrete for this week air quality",
            lab_name=None,
        )
        candidates = signals.get("lab_candidates") or []
        self.assertIn("smart_lab", candidates)
        self.assertIn("concrete_lab", candidates)


if __name__ == "__main__":
    unittest.main()
