import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from storage.conversation_memory import (
    apply_routing_memory,
    compute_question_signals,
    extract_routing_memory,
)


class ConversationMemoryTests(unittest.TestCase):
    def test_followup_what_about_yesterday_carries_lab_and_metric(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: Compare co2 in smart_lab this week\n"
            "Assistant: smart_lab was stable this week."
        )
        current_question = "what about yesterday?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        # effective_question is clean — carry-over lives in details, not the string
        self.assertEqual(effective_question, current_question)
        self.assertEqual(details.get("carried_metric"), "co2")
        self.assertTrue(bool(details.get("applied")))

    def test_definitional_followup_skips_time_phrase_but_keeps_lab(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What's the CO2 right now in smart_lab?\n"
            "Assistant: CO2 is 710 ppm right now."
        )
        current_question = "What does CO2 mean?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        # effective_question is always clean — no carry-over text appended
        self.assertEqual(effective_question, current_question)
        self.assertIsNone(details.get("carried_time_phrase"))
        # Lab is carried even for definitional questions
        self.assertTrue(bool(details.get("applied")))

    def test_non_definitional_followup_still_carries_time_phrase(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What's the CO2 right now in smart_lab?\n"
            "Assistant: CO2 is 710 ppm right now."
        )
        current_question = "What about PM2.5?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        # effective_question is clean — carry-over lives in details, not the string
        self.assertEqual(effective_question, current_question)
        self.assertTrue(bool(details.get("applied")))
        self.assertEqual(details.get("carried_time_phrase"), "right now")

    def test_air_quality_question_does_not_carry_prior_temperature(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: How is the temperature today in smart_lab?\n"
            "Assistant: Temperature is 24.7C in smart_lab."
        )
        current_question = "How is the air quality today?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertNotIn("temperature", effective_question.lower())
        self.assertNotIn("(temperature)", effective_question.lower())
        self.assertEqual(details.get("carried_metric"), None)

    def test_issues_with_this_carries_humidity_and_today(self):
        context = (
            "Previous conversation context (most recent last):\n"
            "User: How is the humidity today?\n"
            "Assistant: Humidity in smart_lab is stable at 52%."
        )
        current_question = "Can you find any issues with this?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        effective_question, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        # effective_question is clean — carry-over lives in details, not the string
        self.assertEqual(effective_question, current_question)
        self.assertEqual(details.get("carried_metric"), "humidity")
        self.assertEqual(details.get("carried_time_phrase"), "today")
        self.assertTrue(bool(details.get("applied")))


    def test_new_metric_followup_carries_specific_date_day(self):
        # "another metric on that day" — prior turn used an explicit calendar
        # date, so the follow-up that only swaps the metric must keep the day.
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What was the average temperature in smart_lab on June 2?\n"
            "Assistant: It averaged 23.4C."
        )
        current_question = "what about humidity?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        _, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertEqual(details.get("carried_time_phrase"), "june 2")
        # The new metric is named in this turn, so nothing is carried for it.
        self.assertIsNone(details.get("carried_metric"))

    def test_new_time_followup_carries_prior_metric(self):
        # "another time on that metric" — prior turn established the metric, the
        # follow-up changes only the time window, so the metric must carry over.
        context = (
            "Previous conversation context (most recent last):\n"
            "User: What was the average temperature in smart_lab yesterday?\n"
            "Assistant: It averaged 23.4C."
        )
        current_question = "what about last week?"
        current_signals = compute_question_signals(current_question)
        memory = extract_routing_memory(context, current_signals)
        _, effective_lab, details = apply_routing_memory(
            question=current_question,
            lab_name=None,
            memory=memory,
            current_signals=current_signals,
        )
        self.assertEqual(effective_lab, "smart_lab")
        self.assertEqual(details.get("carried_metric"), "temperature")
        # The follow-up names its own time window, so the prior time is not carried.
        self.assertIsNone(details.get("carried_time_phrase"))


if __name__ == "__main__":
    unittest.main()
