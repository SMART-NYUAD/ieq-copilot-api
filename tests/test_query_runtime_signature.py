import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from http_routes.query_runtime import _tool_call_signature


class QueryRuntimeSignatureTests(unittest.TestCase):
    def test_signature_ignores_volatile_question_and_observation_fields(self):
        sig_a = _tool_call_signature(
            tool_name="compare_spaces",
            arguments={
                "question": "Compare smart_lab vs concrete_lab for last 7 days",
                "observation": "first pass summary",
                "metric": "co2",
                "window": "last_7_days",
            },
            route_type="comparison_db",
            lab_name=None,
        )
        sig_b = _tool_call_signature(
            tool_name="compare_spaces",
            arguments={
                "question": "Compare again with tool observations appended",
                "observation": "second pass summary",
                "metric": "co2",
                "window": "last_7_days",
            },
            route_type="comparison_db",
            lab_name=None,
        )
        self.assertEqual(sig_a, sig_b)

    def test_signature_changes_when_semantic_arguments_change(self):
        sig_a = _tool_call_signature(
            tool_name="compare_spaces",
            arguments={"metric": "co2", "window": "last_7_days"},
            route_type="comparison_db",
            lab_name=None,
        )
        sig_b = _tool_call_signature(
            tool_name="compare_spaces",
            arguments={"metric": "pm25", "window": "last_7_days"},
            route_type="comparison_db",
            lab_name=None,
        )
        self.assertNotEqual(sig_a, sig_b)


if __name__ == "__main__":
    unittest.main()
