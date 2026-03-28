import os
import sys
import unittest
from unittest.mock import patch


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_query_executor import _fetch_knowledge_cards, _planner_card_controls


class DbCardControlsTests(unittest.TestCase):
    def test_planner_card_controls_defaults(self):
        needs_cards, topics, max_cards = _planner_card_controls(None)
        self.assertFalse(needs_cards)
        self.assertEqual(topics, [])
        self.assertEqual(max_cards, 2)

    def test_planner_card_controls_clamps_and_normalizes(self):
        needs_cards, topics, max_cards = _planner_card_controls(
            {
                "needs_cards": True,
                "card_topics": ["metric explanations", "caveats", "metric_explanations"],
                "max_cards": 99,
            }
        )
        self.assertTrue(needs_cards)
        self.assertEqual(topics, ["metric_explanations", "caveats"])
        self.assertEqual(max_cards, 4)

    @patch("executors.db_query_executor.search_knowledge_cards")
    def test_fetch_knowledge_cards_filters_by_topics(self, mock_search):
        mock_search.return_value = [
            {
                "card_type": "rule",
                "topic": "ieq",
                "title": "Ventilation recommendations",
                "summary": "summary",
                "content": "content",
                "severity_level": "medium",
                "source_label": "guide",
            },
            {
                "card_type": "caveat",
                "topic": "ieq",
                "title": "Health caveat",
                "summary": "summary",
                "content": "content",
                "severity_level": "high",
                "source_label": "guide",
            },
        ]
        cards = _fetch_knowledge_cards(
            question="How can I improve comfort?",
            limit=2,
            card_topics=["recommendations"],
        )
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].get("card_type"), "rule")


if __name__ == "__main__":
    unittest.main()
