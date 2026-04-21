import unittest

from evidence.citation_processor import process_answer_citations, resolve_citations


MOCK_RECORDS = [
    {
        "source_key": "RESET_AIR_V2",
        "source_label": "RESET Air Standard v2.1",
        "section_ref": "Section 4",
        "citation_tier": "regulatory",
        "source_url": "https://reset.build/standard/air",
        "threshold_value": 1000,
        "threshold_unit": "ppm",
        "caveat_text": None,
    },
    {
        "source_key": "ALLEN_ET_AL_2016",
        "source_label": "Allen et al. 2016",
        "section_ref": "Table 3",
        "citation_tier": "research",
        "source_url": "https://doi.org/10.1289/ehp.1510037",
        "threshold_value": None,
        "threshold_unit": None,
        "caveat_text": "Research finding, not a standard.",
    },
]


class TestCitationProcessor(unittest.TestCase):
    def test_resolves_known_marker(self):
        answer = "CO2 exceeds Grade A [^RESET_AIR_V2]."
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertIn("[^1]", resolved)
        self.assertNotIn("[^RESET_AIR_V2]", resolved)
        self.assertEqual(len(footnotes), 1)
        self.assertEqual(footnotes[0]["source_key"], "RESET_AIR_V2")
        self.assertEqual(footnotes[0]["index"], 1)

    def test_removes_unknown_marker(self):
        answer = "Some claim [^MADE_UP_SOURCE]."
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertNotIn("[^MADE_UP_SOURCE]", resolved)
        self.assertEqual(len(footnotes), 0)

    def test_sequential_numbering(self):
        answer = (
            "First claim [^RESET_AIR_V2]. "
            "Second claim [^ALLEN_ET_AL_2016]."
        )
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertIn("[^1]", resolved)
        self.assertIn("[^2]", resolved)
        self.assertEqual(len(footnotes), 2)

    def test_same_source_reused(self):
        answer = (
            "Claim one [^RESET_AIR_V2]. "
            "Claim two also [^RESET_AIR_V2]."
        )
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertEqual(resolved.count("[^1]"), 2)
        self.assertEqual(len(footnotes), 1)

    def test_strips_llm_references_block(self):
        answer = (
            "CO2 is high [^RESET_AIR_V2].\n\n"
            "## References\n"
            "[^RESET_AIR_V2]: Some invented text"
        )
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertNotIn("## References", resolved)
        self.assertNotIn("Some invented text", resolved)
        self.assertEqual(len(footnotes), 1)

    def test_no_markers_returns_unchanged(self):
        answer = "CO2 is 800 ppm which is fine."
        resolved, footnotes = resolve_citations(answer, MOCK_RECORDS)
        self.assertEqual(resolved, answer)
        self.assertEqual(footnotes, [])

    def test_empty_records_removes_markers(self):
        answer = "CO2 exceeds Grade A [^RESET_AIR_V2]."
        resolved, footnotes = resolve_citations(answer, [])
        self.assertNotIn("[^RESET_AIR_V2]", resolved)
        self.assertEqual(len(footnotes), 0)

    def test_process_answer_citations_round_trip(self):
        answer = "CO2 exceeds Grade A [^RESET_AIR_V2]."
        resolved, footnotes = process_answer_citations(answer, MOCK_RECORDS)
        self.assertIn("[^1]", resolved)
        self.assertEqual(len(footnotes), 1)


if __name__ == "__main__":
    unittest.main()
