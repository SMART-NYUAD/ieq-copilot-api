import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from storage.conversation_store import _sanitize_assistant_text


class ConversationStoreTests(unittest.TestCase):
    def test_sanitize_legacy_general_explanation_prefix(self):
        raw = (
            "General explanation (not site-specific policy): PM2.5 is tiny particulate matter."
        )
        cleaned = _sanitize_assistant_text(raw)
        self.assertEqual(cleaned, "PM2.5 is tiny particulate matter.")

    def test_sanitize_legacy_general_explanation_suffix_note(self):
        raw = (
            "PM2.5 refers to fine particles.\n"
            "Note: Without measured data, this is a general educational explanation. "
            "For site-specific guidance, real-time measurements are required."
        )
        cleaned = _sanitize_assistant_text(raw)
        self.assertEqual(cleaned, "PM2.5 refers to fine particles.")


if __name__ == "__main__":
    unittest.main()
