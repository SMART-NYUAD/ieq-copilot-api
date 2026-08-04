"""Knowledge-card retrieval: the parts that can be tested without the network.

Retrieval quality itself is measured by ``tests/retrieval_eval.py``, which needs Postgres
and the embedding model and is therefore not part of ``unittest discover``. What IS testable
hermetically is the logic that surrounds the vector search, and every rule below came from a
measured failure:

* the corpus scored **3/17** top-1 before this work and **16/17** after;
* an ivfflat index built ``WITH (lists = 100)`` over 55 rows meant a probe scan could only
  ever see a fraction of the corpus — "what is CO2?" returned zero rows at the Postgres
  default and the wrong cards at the setting the code used;
* the card-type nudges reached ±0.7 against a real similarity spread of about 0.2, so
  ``card_type`` decided ranking outright and the embedding was decoration;
* ``_is_explanation_query`` missed "explain" and "what are", so definition questions were
  routed to the branch that PENALISES explanation cards;
* ``_is_guardrail_query`` only knew health-risk phrasing, so "does CO2 tell me everything
  about air quality?" penalised the caveat card written to answer exactly that.
"""

import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.knowledge_executor import (  # noqa: E402
    _MAX_PRIORITY_NUDGE,
    _is_explanation_query,
    _is_guardrail_query,
    _knowledge_card_priority,
)
from storage import embeddings  # noqa: E402

_CARD_TYPES = ("explanation", "interpretation", "rule", "caveat", "ieq_subindex")


class QueryPassageAsymmetryTests(unittest.TestCase):
    """bge-*-en-v1.5 wants an instruction prefix on queries and none on passages."""

    def test_query_side_applies_the_instruction_prefix(self):
        captured = {}

        def _fake(texts, model_name=embeddings._DEFAULT_MODEL):
            captured["texts"] = list(texts)
            return [[0.0]]

        with patch.object(embeddings, "embed_texts", side_effect=_fake):
            embeddings.embed_query("what is CO2?")
        self.assertEqual(len(captured["texts"]), 1)
        self.assertTrue(captured["texts"][0].startswith(embeddings._BGE_QUERY_INSTRUCTION))
        self.assertTrue(captured["texts"][0].endswith("what is CO2?"))

    def test_passage_side_is_left_alone(self):
        # Stored vectors must NOT carry the prefix. If this ever changes, everything already
        # embedded silently becomes inconsistent with newly embedded text.
        captured = {}

        def _fake(texts, model_name=embeddings._DEFAULT_MODEL):
            captured["texts"] = list(texts)
            return [[0.0] for _ in texts]

        with patch.object(embeddings, "embed_texts", side_effect=_fake):
            embeddings.embed_documents(["CO2 is a proxy for ventilation."])
        self.assertNotIn(embeddings._BGE_QUERY_INSTRUCTION, captured["texts"][0])

    def test_instruction_is_model_scoped(self):
        # A non-bge model must not receive bge's prefix.
        self.assertEqual(embeddings._query_instruction("BAAI/bge-large-en-v1.5"),
                         embeddings._BGE_QUERY_INSTRUCTION)
        self.assertEqual(embeddings._query_instruction("sentence-transformers/all-MiniLM-L6-v2"), "")


class RerankNudgeTests(unittest.TestCase):
    """The nudge breaks near-ties. It must never outweigh the similarity itself."""

    _QUESTIONS = (
        "what is CO2?",
        "explain PM2.5",
        "is this air dangerous?",
        "how is the air quality today?",
        "does CO2 tell me everything about air quality?",
    )

    def test_no_nudge_exceeds_the_declared_ceiling(self):
        # Real cosine similarities on this corpus span roughly 0.45-0.65. A nudge larger
        # than that spread stops being a tie-breaker and becomes the ranking.
        for question in self._QUESTIONS:
            for card_type in _CARD_TYPES:
                nudge = _knowledge_card_priority(question, card_type)
                self.assertLessEqual(abs(nudge), _MAX_PRIORITY_NUDGE, f"{question}/{card_type}")

    def test_ceiling_stays_below_the_observed_similarity_spread(self):
        self.assertLessEqual(_MAX_PRIORITY_NUDGE, 0.15)


class QueryClassificationTests(unittest.TestCase):
    def test_definition_phrasings_are_recognised(self):
        for question in (
            "what is CO2?",
            "what are volatile organic compounds?",
            "explain PM2.5 to me",
            "tell me about VOCs",
            "define relative humidity",
            "what does IEQ stand for?",
            "what does the thermal comfort sub-index mean?",
        ):
            self.assertTrue(_is_explanation_query(question), question)

    def test_status_questions_are_not_definitions(self):
        for question in (
            "how is the air quality today?",
            "what was the average CO2 last week?",
            "is the humidity high right now?",
        ):
            self.assertFalse(_is_explanation_query(question), question)

    def test_definition_queries_favour_explanation_over_status_cards(self):
        # The specific inversion that made "explain PM2.5" return a status card.
        for question in ("explain PM2.5", "what are VOCs?", "tell me about humidity"):
            explanation = _knowledge_card_priority(question, "explanation")
            interpretation = _knowledge_card_priority(question, "interpretation")
            self.assertGreater(explanation, interpretation, question)

    def test_guardrail_covers_health_risk_and_metric_limitations(self):
        for question in (
            "is this air dangerous to my health?",
            "is it safe to work in here all day?",
            "does CO2 tell me everything about air quality?",
            "is CO2 the whole story for air quality?",
            "what are the limitations of this reading?",
        ):
            self.assertTrue(_is_guardrail_query(question), question)
            self.assertGreater(
                _knowledge_card_priority(question, "caveat"),
                _knowledge_card_priority(question, "interpretation"),
                question,
            )

    def test_caveats_are_still_held_back_from_ordinary_status_questions(self):
        # The penalty exists so caveats do not crowd out an answer to "how is it now?".
        # Removing it is not the fix for the guardrail gap above.
        self.assertLess(_knowledge_card_priority("how is the air quality today?", "caveat"), 0)


class EmbeddingDimensionGuardTests(unittest.TestCase):
    def test_padding_a_narrower_vector_is_allowed(self):
        from storage.seed_guidelines import _GUIDELINE_EMBED_DIM, _normalize_embedding_dim

        padded = _normalize_embedding_dim([0.5] * 1024)
        self.assertEqual(len(padded), _GUIDELINE_EMBED_DIM)
        # Zero padding preserves cosine similarity exactly, which is why it is safe.
        self.assertEqual(set(padded[1024:]), {0.0})

    def test_a_wider_vector_raises_instead_of_being_truncated(self):
        from storage.seed_guidelines import _normalize_embedding_dim

        with self.assertRaises(ValueError):
            _normalize_embedding_dim([0.5] * 2048)


if __name__ == "__main__":
    unittest.main()
