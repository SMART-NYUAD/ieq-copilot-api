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


class IntentClassifierLightQueryTests(unittest.TestCase):
    def test_how_is_light_in_lab_maps_to_current_status_db(self):
        decision = classify_intent("How is the light in the smart lab?")
        self.assertEqual(decision.intent, IntentType.CURRENT_STATUS_DB)

    def test_how_is_pm25_in_lab_maps_to_current_status_db(self):
        decision = classify_intent("How is the pm2.5 in the smart lab?")
        self.assertEqual(decision.intent, IntentType.CURRENT_STATUS_DB)


if __name__ == "__main__":
    unittest.main()
