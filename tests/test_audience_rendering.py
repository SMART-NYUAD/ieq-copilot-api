"""Audience-scoped rendering of the evidence, not just of the instructions.

Reported: occupant answers were "bringing the guidelines again", naming standards bodies and
threshold figures and listing every metric, rather than saying what a person would notice.

Prompting could not fix it, and three rounds of trying is the evidence. The occupant block
already said, in the first bullets of the system prompt: name as few metrics as possible,
never use an index acronym, never name a standards body. Meanwhile the model was handed a
section headed "COMPUTED ... authoritative: state them as given" containing

    - VOC = 0.06 ppm — is approaching the strictest applicable limit of 0.061 ppm
      (WHO Indoor Air Quality Guidelines: Selected Pollutants 2010 (ppm equivalent)) [14].
    - CO2 = 454 ppm — is within the strictest applicable limit of 1000 ppm (RESET Air ...)
    ... one such line per metric, including the five that were fine ...

plus `## Measured Room Facts` serialized verbatim with `"iaq": 93.6` as a copyable key. An
instruction cannot outrank the data it is pointing at, and it should not have to: the fix is
to render the evidence for the audience, exactly as `_restrict_rows_to_metrics` narrows the
rows rather than asking the model to ignore columns.

Verdicts are NOT audience-dependent. The same computation runs for everyone; only its
description changes, and a metric that is not fine keeps its own line for every reader.
"""

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_query_executor import _plainify_payload  # noqa: E402
from executors.db_support import threshold_assessment  # noqa: E402
from prompting.roles import (  # noqa: E402
    ROLE_EXECUTIVE,
    ROLE_FACILITY_MANAGER,
    ROLE_OCCUPANT,
    ROLE_RESEARCHER,
    role_wants_compliance_detail,
)

_READINGS = {"co2": 454.0, "pm25": 7.2, "voc": 0.06, "humidity": 51.0, "ieq": 79.0, "iaq": 94.0}

_SOURCES = [
    {"index": 14, "metric": "voc", "source_key": "WHO_VOC_PPM",
     "source_label": "WHO Indoor Air Quality Guidelines", "citation_tier": "regulatory",
     "threshold_value": 0.061, "threshold_unit": "ppm", "threshold_type": "max"},
    {"index": 4, "metric": "co2", "source_key": "RESET_AIR_V2",
     "source_label": "RESET Air Standard v2.1", "citation_tier": "regulatory",
     "threshold_value": 1000, "threshold_unit": "ppm", "threshold_type": "max"},
    {"index": 10, "metric": "pm25", "source_key": "WHO_AQG_2021",
     "source_label": "WHO Global Air Quality Guidelines 2021", "citation_tier": "regulatory",
     "threshold_value": 15.0, "threshold_unit": "µg/m³", "threshold_type": "max"},
]


def _plain():
    return threshold_assessment.build_assessment_section(_READINGS, _SOURCES, compliance_detail=False)


def _full():
    return threshold_assessment.build_assessment_section(_READINGS, _SOURCES, compliance_detail=True)


class RoleDetailMappingTests(unittest.TestCase):
    def test_operational_roles_get_compliance_detail(self):
        for role in (ROLE_RESEARCHER, ROLE_FACILITY_MANAGER):
            self.assertTrue(role_wants_compliance_detail(role), role)

    def test_plain_language_roles_do_not(self):
        for role in (ROLE_OCCUPANT, ROLE_EXECUTIVE):
            self.assertFalse(role_wants_compliance_detail(role), role)

    def test_unknown_role_degrades_to_the_default(self):
        # occupant is the default, so an unrecognised role gets plain language rather than
        # accidentally receiving compliance detail.
        self.assertFalse(role_wants_compliance_detail("cto"))
        self.assertFalse(role_wants_compliance_detail(None))


class PlainAssessmentTests(unittest.TestCase):
    def test_no_threshold_figures_or_standards_bodies(self):
        text = _plain()
        for leaked in ("0.061", "1000", "15.0", "WHO", "RESET", "ASHRAE", "EPA"):
            self.assertNotIn(leaked, text, f"{leaked} leaked into the plain rendering")

    def test_no_index_acronyms(self):
        text = _plain()
        for acronym in ("IAQ", "IEQ", "ITC", "IAC", "IIL", "Sub-index"):
            self.assertNotIn(acronym, text, acronym)

    def test_a_metric_that_is_not_fine_keeps_its_own_line_with_value_and_unit(self):
        # The completeness boundary. Brevity may collapse what is fine, never what is not.
        line = [l for l in _plain().splitlines() if l.startswith("- VOC")]
        self.assertEqual(len(line), 1)
        self.assertIn("0.06 ppm", line[0])
        self.assertIn("NEAR", line[0])

    def test_metrics_within_range_collapse_into_one_line(self):
        collapsed = [l for l in _plain().splitlines() if "Everything else measured" in l]
        self.assertEqual(len(collapsed), 1)
        # And that claim must be true of every metric it names.
        for fine in ("CO2", "PM2.5"):
            self.assertIn(fine, collapsed[0])

    def test_an_unrated_metric_is_not_collapsed_away(self):
        # Humidity has no threshold in this fixture, so it is `not rated` — a flagged status.
        # Collapsing it into "everything else is within range" would be the omission the
        # completeness rules exist to prevent, stated as a claim that is not true.
        text = _plain()
        collapsed = [l for l in text.splitlines() if "Everything else measured" in l][0]
        self.assertNotIn("Humidity", collapsed)
        own_line = [l for l in text.splitlines() if l.startswith("- Humidity")]
        self.assertEqual(len(own_line), 1)
        self.assertIn("could not be checked", own_line[0])

    def test_the_plain_rendering_is_materially_shorter(self):
        self.assertLess(len(_plain().splitlines()), len(_full().splitlines()))

    def test_citations_survive_so_the_source_is_still_reachable(self):
        # The [N] marker is what replaces naming the standard inline; losing it would make
        # the answer unciteable rather than merely plainer.
        self.assertIn("[14]", _plain())

    def test_the_verdict_itself_is_unchanged(self):
        # Only the description is audience-scoped. Status must be identical.
        for status in ("NEAR",):
            self.assertIn(status, _plain())
            self.assertIn(status, _full())
        self.assertIn("OVERALL", _plain())


class ComplianceAssessmentTests(unittest.TestCase):
    def test_operational_readers_still_get_numbers_and_sources(self):
        text = _full()
        self.assertIn("0.061", text)
        self.assertIn("WHO Indoor Air Quality Guidelines", text)
        self.assertIn("IAQ Sub-index", text)

    def test_every_metric_keeps_its_own_line(self):
        lines = [l for l in _full().splitlines() if l.startswith("- ")]
        self.assertEqual(len(lines), len(_READINGS))


class PayloadRelabellingTests(unittest.TestCase):
    _PAYLOAD = {"metric": "co2", "rows": [{"bucket": "t", "co2": 451.0, "iaq": 93.6, "ieq": 79.0}]}

    def test_index_keys_are_relabelled_not_removed(self):
        # `"iaq": 93.6` is serialized verbatim into the prompt and was being copied as
        # "the IAQ score is 93.6". Renaming removes the copyable token; removing the key
        # would break an IEQ question, which legitimately needs the scores.
        row = _plainify_payload(self._PAYLOAD)["rows"][0]
        self.assertNotIn("iaq", row)
        self.assertIn("air_quality_score", row)
        self.assertEqual(row["air_quality_score"], 93.6)
        self.assertIn("overall_comfort_score", row)

    def test_non_index_metrics_are_untouched(self):
        row = _plainify_payload(self._PAYLOAD)["rows"][0]
        self.assertEqual(row["co2"], 451.0)
        self.assertEqual(row["bucket"], "t")

    def test_payload_metadata_survives(self):
        self.assertEqual(_plainify_payload(self._PAYLOAD)["metric"], "co2")

    def test_a_payload_without_rows_is_returned_unchanged(self):
        payload = {"metric": "co2", "window": "today"}
        self.assertEqual(_plainify_payload(payload), payload)


if __name__ == "__main__":
    unittest.main()
