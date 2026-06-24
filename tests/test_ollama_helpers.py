import unittest

from ollama_helpers import extract_chat_content, extract_generate_chunk, extract_generate_text


class TestOllamaHelpers(unittest.TestCase):
    def test_extract_generate_text_prefers_response(self):
        event = {"response": "hello", "thinking": "internal"}
        self.assertEqual(extract_generate_text(event, thinking=False), "hello")

    def test_extract_generate_text_uses_thinking_when_enabled(self):
        event = {"response": "", "thinking": "internal"}
        self.assertEqual(extract_generate_text(event, thinking=True), "internal")

    def test_extract_generate_text_empty_when_thinking_disabled(self):
        event = {"response": "", "thinking": "internal"}
        self.assertEqual(extract_generate_text(event, thinking=False), "")

    def test_extract_generate_chunk_preserves_whitespace_only_tokens(self):
        event = {"response": " ", "thinking": ""}
        self.assertEqual(extract_generate_chunk(event, thinking=False), " ")

    def test_extract_chat_content_prefers_content(self):
        message = {"content": "route json", "thinking": "internal"}
        self.assertEqual(extract_chat_content(message, thinking=False), "route json")

    def test_extract_chat_content_uses_thinking_when_enabled(self):
        message = {"content": "", "thinking": "internal"}
        self.assertEqual(extract_chat_content(message, thinking=True), "internal")


if __name__ == "__main__":
    unittest.main()
