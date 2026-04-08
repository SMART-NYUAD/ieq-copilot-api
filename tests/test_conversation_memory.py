import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from query_routing.router_signals import extract_query_signals
from storage.conversation_memory import apply_routing_memory, extract_routing_memory


class ConversationMemoryTests(unittest.TestCase):
    def test_followup_what_about_yesterday_carries_lab_and_metric(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: Compare co2 in smart_lab this week\n"
            "Assistant: smart_lab was stable this week."
        )
        current_question = "what about yesterday?"
        current_signals = extract_query_signals(question=current_question, lab_name=None)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertIn("yesterday", effective_question.lower())
        self.assertIn("co2", effective_question.lower())
        self.assertTrue(bool(details.get("applied")))

    def test_definitional_followup_skips_time_phrase_but_keeps_lab(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What's the CO2 right now in smart_lab?\n"
            "Assistant: CO2 is 710 ppm right now."
        )
        current_question = "What does CO2 mean?"
        current_signals = extract_query_signals(question=current_question, lab_name=None)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertNotIn("right now", effective_question.lower())
        self.assertNotIn("(now)", effective_question.lower())
        self.assertIn("co2", effective_question.lower())
        self.assertTrue(bool(details.get("applied")))
        self.assertIsNone(details.get("carried_time_phrase"))

    def test_non_definitional_followup_still_carries_time_phrase(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What's the CO2 right now in smart_lab?\n"
            "Assistant: CO2 is 710 ppm right now."
        )
        current_question = "What about PM2.5?"
        current_signals = extract_query_signals(question=current_question, lab_name=None)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertIn("right now", effective_question.lower())
        self.assertTrue(bool(details.get("applied")))
        self.assertEqual(details.get("carried_time_phrase"), "right now")


if __name__ == "__main__":
    unittest.main()
