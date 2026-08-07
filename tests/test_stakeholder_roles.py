"""Stakeholder roles change who an answer is written for — and nothing else.

A role is declared by the caller, not inferred from the question, so unlike `intent` or
`analysis_mode` there is no router eval that can catch a regression here. These tests carry
the whole guarantee, in three parts.

**Role is per-message and stateless.** Sending a different role on consecutive turns of one
conversation is the normal case, and omitting it always means the configured default —
never "whatever the last turn used". Nothing about a role is persisted.

**Role may never hide a problem.** The completeness rules in db_prompts and the computed
threshold verdicts were each won back from a specific wrong answer (see
test_air_quality_completeness and test_threshold_assessment), and two roles now actively
ask for fewer metrics: `occupant` wants as few as possible, `executive` is capped at 60
words. Every block therefore scopes that permission to metrics the Threshold Assessment
reports as within range — a metric flagged EXCEEDS, NEAR or not-rated appears in every
answer for every audience.

**Role may widen the data fetched, never narrow it.** The metric pack a role resolves to
can only ever grow, asserted as subset containment across every role × scope.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from executors.db_support.metric_planning import (
    SCOPE_AIR_QUALITY,
    SCOPE_COMFORT,
    SCOPE_DIAGNOSTIC,
    SCOPE_FULL,
    SCOPE_IEQ_INDEX,
    SCOPE_NAMED,
    VALID_METRIC_SCOPES,
    plan_metrics,
)
from prompting.db_prompts import (
    DB_TOOL_RESPONSE_DIRECTIVE,
    DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
)
from prompting.role_prompts import role_addendum, role_style_block
from prompting.roles import (
    ROLE_DEFAULT,
    ROLE_EXECUTIVE,
    ROLE_FACILITY_MANAGER,
    ROLE_OCCUPANT,
    ROLE_RESEARCHER,
    VALID_ROLES,
    coerce_role,
    normalize_role,
    role_catalog,
)
from prompting.shared_prompts import SHARED_SYSTEM_PROMPT, shared_system_prompt
from query_routing.intent_classifier import IntentType


# The audience wording that was hardcoded in SHARED_SYSTEM_PROMPT before roles existed.
# Pinned as a literal rather than read back from the module so that editing the module
# cannot quietly satisfy this test.
_LEGACY_AUDIENCE_LINES = (
    "- Prefer natural, compassionate phrasing over clinical/policy-heavy wording unless the "
    "user explicitly asks for formal compliance language.\n"
    "- Write for non-technical occupants: plain language, no jargon, focus on what people "
    "would actually notice or feel."
)


class RoleNormalizationTests(unittest.TestCase):
    def test_canonical_names_pass_through(self):
        for role in VALID_ROLES:
            self.assertEqual(normalize_role(role), (role, False))

    def test_aliases_fold(self):
        cases = {
            "fm": ROLE_FACILITY_MANAGER,
            "Operator": ROLE_FACILITY_MANAGER,
            "facility manager": ROLE_FACILITY_MANAGER,
            "facility-manager": ROLE_FACILITY_MANAGER,
            "analyst": ROLE_RESEARCHER,
            "EXEC": ROLE_EXECUTIVE,
            "general": ROLE_OCCUPANT,
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_role(raw), (expected, False), raw)

    def test_absent_role_is_not_a_fallback(self):
        # None/"" means "the caller did not choose", which is how the resolution order
        # tells "not sent" apart from "sent something wrong".
        for empty in (None, "", "   "):
            self.assertEqual(normalize_role(empty), (None, False))

    def test_unknown_role_falls_back_and_is_flagged(self):
        role, fallback = normalize_role("cto")
        self.assertIsNone(role)
        self.assertTrue(fallback)

    def test_coerce_never_raises(self):
        self.assertEqual(coerce_role("nonsense"), ROLE_DEFAULT)
        self.assertEqual(coerce_role(None), ROLE_DEFAULT)

    def test_catalog_covers_vocabulary_exactly(self):
        catalog = role_catalog()
        self.assertEqual([entry["id"] for entry in catalog], list(VALID_ROLES))
        self.assertEqual([e["id"] for e in catalog if e["default"]], [ROLE_DEFAULT])


class DefaultRoleTests(unittest.TestCase):
    def test_default_is_the_occupant_role(self):
        self.assertEqual(ROLE_DEFAULT, ROLE_OCCUPANT)

    def test_module_constant_is_the_default_rendering(self):
        self.assertEqual(shared_system_prompt(ROLE_OCCUPANT), SHARED_SYSTEM_PROMPT)

    def test_unknown_role_renders_the_default_prompt(self):
        self.assertEqual(shared_system_prompt("cto"), SHARED_SYSTEM_PROMPT)

    def test_occupant_deliberately_replaced_the_legacy_wording(self):
        # The occupant block began as this wording verbatim, which made the feature a
        # provable no-op. It is not any more, on purpose: "no jargon" constrained
        # vocabulary and said nothing about volume, so an occupant still got every fetched
        # pollutant listed — the researcher's answer in friendlier words. If this assertion
        # ever starts failing because the old lines came back, the occupant rundown came
        # back with them.
        self.assertNotIn(_LEGACY_AUDIENCE_LINES, SHARED_SYSTEM_PROMPT)
        self.assertIn("as FEW metrics as the question allows", role_style_block(ROLE_OCCUPANT))

    def test_every_role_including_the_default_has_an_addendum(self):
        # The occupant block now says something the hand-written IFC/sensor prompts do not,
        # so returning "" for it would apply the default voice on the DB path only.
        for role in VALID_ROLES:
            self.assertTrue(role_addendum(role).strip(), role)


class RoleMayNotHideAProblemTests(unittest.TestCase):
    """Two roles now ask for fewer metrics. This is the line they may not cross."""

    def test_every_role_carries_the_invariant_clause(self):
        for role in VALID_ROLES:
            block = role_style_block(role).lower()
            self.assertIn("exceeds, near, or not rated", block, role)
            self.assertIn("every answer, for every audience", block, role)
            self.assertIn("reporting the problem wins", block, role)

    def test_the_permission_is_scoped_to_metrics_within_range(self):
        # The failure this guards against is a role block that reads as a general licence
        # to say less. Every block must tie "fewer metrics" to the within-range case.
        for role in VALID_ROLES:
            self.assertIn("within range", role_style_block(role).lower(), role)

    def test_role_does_not_touch_the_assessment_directives(self):
        # Role is spliced into the system prompt only. If a future change starts appending
        # it to the directives too, the completeness block gains a competing voice — which
        # is the failure mode CLAUDE.md records for the advisory bug.
        for directive in (
            DB_TOOL_RESPONSE_DIRECTIVE,
            DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        ):
            for role in VALID_ROLES:
                self.assertNotIn(role_style_block(role), directive)

    def test_the_shortest_role_still_reports_a_failing_metric(self):
        # Executive is capped at 60 words and is where the PM2.5 omission would return.
        block = role_style_block(ROLE_EXECUTIVE).lower()
        self.assertIn("must appear in the answer even if nothing else does", block)
        self.assertIn("never hiding a problem", block)

    def test_completeness_rule_still_mandates_the_metrics_that_failed(self):
        # The relaxation that lets brief roles stop reciting in-range readings must not
        # have weakened the rule for the metrics the original bug dropped: PM2.5 above the
        # WHO guideline and VOC with no comparable threshold were EXCEEDS and not-rated.
        for directive in (
            DB_TOOL_RESPONSE_DIRECTIVE,
            DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        ):
            # These are wrapped prompt constants; assert on meaning, not line breaks.
            text = " ".join(directive.lower().split())
            self.assertIn('exceeds, near, or "not rated"', text)
            self.assertIn("must appear in the answer with its value and unit", text)
            self.assertIn("no word cap, audience, or brevity instruction permits omitting one", text)
            # An all-clear about the metrics that were not named is still a claim.
            self.assertIn("never imply an all-clear you have not earned", text)


class RolesActuallyDifferTests(unittest.TestCase):
    """Each role must ask for something the others do not.

    The first version of these blocks differed mostly in adjectives, which produced four
    answers that read alike. What separates them is the *question each reader is really
    asking*: do I need to intervene (FM), what do the numbers and guidelines say
    (researcher), is anything wrong (executive), does this affect me (occupant).
    """

    def test_facility_manager_answers_whether_to_intervene(self):
        block = role_style_block(ROLE_FACILITY_MANAGER).lower()
        self.assertIn("do i need to do something about it", block)
        self.assertIn("always give the action", block)
        self.assertIn("a status report with no action is a failed answer", block)

    def test_facility_manager_is_people_first_and_preventative(self):
        block = role_style_block(ROLE_FACILITY_MANAGER).lower()
        self.assertIn("people first", block)
        self.assertIn("occupant comfort, health and complaints are the job", block)
        self.assertIn("be preventative, not just reactive", block)
        self.assertIn("before* it crosses", block)

    def test_facility_manager_covers_sla_and_warranty_exposure(self):
        block = role_style_block(ROLE_FACILITY_MANAGER).lower()
        self.assertIn("flag sla and compliance exposure", block)
        self.assertIn("service agreement", block)
        self.assertIn("respect warranties and service contracts", block)
        self.assertIn("void a warranty", block)
        # In-house vs contractor is the distinction that makes the warranty rule usable.
        self.assertIn("filter swap", block)
        self.assertIn("route it to the contractor", block)

    def test_facility_manager_names_the_physical_checks(self):
        block = role_style_block(ROLE_FACILITY_MANAGER).lower()
        for check in ("filtration", "ventilation rate", "damper", "humidifier", "lamp failure"):
            self.assertIn(check, block, check)
        # Grounded in the model/context — the DB directive already forbids inventing HVAC,
        # and a role that invites naming equipment has to repeat that boundary.
        self.assertIn("never invent equipment", block)

    def test_researcher_asks_for_trends_and_the_provenance_of_each_verdict(self):
        block = role_style_block(ROLE_RESEARCHER).lower()
        self.assertIn("report the trend", block)
        self.assertIn("peak and trough with their timestamps", block)
        # Provenance of the comparison, not a survey of the literature. This block used to
        # ask for EVERY applicable guideline "not just the governing one" — which became
        # unsatisfiable once Citation Sources was narrowed to the thresholds actually
        # applied, and an unsatisfiable instruction is an invitation to fabricate the rest.
        self.assertIn("report the governing threshold", block)
        self.assertIn("its averaging basis", block)
        self.assertIn("may not name or quote", block)
        self.assertNotIn("not just the governing one", block)

    def test_executive_leads_with_all_clear_or_alarm_and_is_warm(self):
        block = role_style_block(ROLE_EXECUTIVE).lower()
        self.assertIn("is everything alright", block)
        self.assertIn("warm and human", block)
        # Names who should act rather than what to do technically.
        self.assertIn("facilities team", block)
        self.assertIn("this reader does not want a rundown", block)

    def test_occupant_is_warm_and_avoids_index_acronyms(self):
        block = role_style_block(ROLE_OCCUPANT).lower()
        self.assertIn("never use an index acronym", block)
        self.assertIn("what it means for them", block)
        self.assertIn("warm", block)

    def test_the_two_brief_roles_both_ask_for_fewer_metrics(self):
        for role in (ROLE_OCCUPANT, ROLE_EXECUTIVE):
            block = role_style_block(role).lower()
            self.assertTrue(
                "few metrics" in block or "one or two metrics" in block
                or "at most the one or two metrics" in block,
                role,
            )

    def test_nothing_in_the_assembled_prompt_contradicts_the_action_policy(self):
        """The reported bug: an FM answer that closed "No action needed unless the user asks
        for recommendations" — the model reciting an instruction instead of answering.

        Five separate clauses told it to withhold action guidance (two of them *after* the
        role block, since PRESENTATION_STYLE_PROMPT is embedded in both the system prompt
        and every directive suffix) against one line in the role block saying to give it.
        The role lost, exactly as the advisory bug did. The clauses are gone; the policy now
        lives once, in the role block, and each role states its own.
        """
        from executors.db_query_executor import _build_db_prompt_text

        banned = (
            'do not say "no action needed"',
            "if they did not ask, omit recommendations",
            'do not add a "recommendations" section',
            "did not ask for recommendations, omit that section",
            'do not add a "next steps" or "recommendations" section unless asked',
        )
        for role in VALID_ROLES:
            prompt = " ".join(
                _build_db_prompt_text(
                    question="how is the room today?",
                    intent=IntentType.CURRENT_STATUS_DB,
                    context_data="CTX",
                    role=role,
                ).lower().split()
            )
            for clause in banned:
                self.assertNotIn(clause, prompt, f"{role}: {clause}")

    def test_nothing_in_the_assembled_prompt_mandates_a_metric_rundown(self):
        """The second reported bug: an occupant answer that listed CO2, VOC, PM2.5, humidity,
        IAQ and IIL with RESET/WHO/ASHRAE threshold numbers — against a role block saying
        name as few metrics as possible, never use an index acronym, no standards language.

        Same shape as the action-guidance bug: five volume mandates outvoted the audience.
        Two forced the sub-index acronyms specifically ("do not omit them to stay brief:
        include each available sub-index"), one lived in the air-quality directive itself,
        and PRESENTATION_STYLE_PROMPT's "report every pollutant" landed twice because it is
        embedded in both the system prompt and every directive suffix.

        What replaced them keeps the floor: a metric the Threshold Assessment FLAGS is still
        mandatory everywhere (see RoleMayNotHideAProblemTests). Only the obligation to
        recite the metrics that are *fine* is now the audience's call.
        """
        from executors.db_query_executor import _build_db_prompt_text

        banned = (
            "do not omit them to stay brief",
            "include each available sub-index once",
            "for metric-by-metric air-quality assessments",
            "report every pollutant, combining them into one bullet if needed",
            "every available pollutant",
        )
        for role in VALID_ROLES:
            prompt = " ".join(
                _build_db_prompt_text(
                    question="how is the air quality today?",
                    intent=IntentType.CURRENT_STATUS_DB,
                    context_data="CTX",
                    role=role,
                ).lower().split()
            )
            for clause in banned:
                self.assertNotIn(clause, prompt, f"{role}: {clause}")

    def test_an_ieq_question_still_reports_every_sub_index(self):
        # The mandate was removed from the air-quality directive, where the indices are
        # supporting context. _BASE_IEQ keeps it: there they are the subject, and dropping
        # one would be the inverse failure.
        from prompting.db_prompts import DB_TOOL_RESPONSE_DIRECTIVE_IEQ

        self.assertIn(
            "report every available sub-index explicitly", DB_TOOL_RESPONSE_DIRECTIVE_IEQ
        )

    def test_sub_index_semantics_survived_the_removal(self):
        # The scale semantics are a correctness rule from a real inversion bug (a high ITC
        # described as hot/stuffy). Only the "include every one" volume mandate was cut.
        prompt = shared_system_prompt(ROLE_OCCUPANT)
        self.assertIn("high score (e.g. 90+) = occupants are thermally COMFORTABLE", prompt)
        self.assertIn("higher is always better", prompt)
        self.assertIn("use its correct meaning", prompt)

    def test_each_role_states_its_own_action_policy_exactly_once(self):
        # One owner for the decision. If a second voice reappears anywhere, the role stops
        # being what decides, which is how this broke the first time.
        for role in VALID_ROLES:
            block = role_style_block(role)
            marker = "ALWAYS GIVE THE ACTION" if role == ROLE_FACILITY_MANAGER else "ACTION GUIDANCE:"
            self.assertEqual(block.count(marker), 1, role)

    def test_only_the_facility_manager_volunteers_actions(self):
        for role in (ROLE_OCCUPANT, ROLE_RESEARCHER):
            self.assertIn(
                "only when they ask for it", role_style_block(role).lower(), role
            )
        # Executive is the exception among the restrained roles: it always names who should
        # look into a problem, because "is there an alarm?" is unanswered without it. What
        # it withholds is the technical fix, not the escalation.
        executive = role_style_block(ROLE_EXECUTIVE).lower()
        self.assertIn("name who should look into it", executive)
        self.assertIn("never the technical fix", executive)

    def test_the_stock_non_answer_is_banned_once_for_every_audience(self):
        """"No action needed" is what both reported answers closed on.

        It is the same rule for every role, so unlike the *whether to volunteer an action*
        policy it does not belong in the role blocks — four copies of an identical rule is
        the duplication this file exists to avoid. It lives once in PRESENTATION_STYLE_PROMPT,
        which is embedded in both the system prompt and every directive suffix.
        """
        from prompting.shared_prompts import PRESENTATION_STYLE_PROMPT

        banned = PRESENTATION_STYLE_PROMPT.lower()
        self.assertIn("banned closer, every audience", banned)
        for variant in ("no action needed", "no immediate action", "no action\nis required"):
            self.assertIn(variant.replace("\n", " "), " ".join(banned.split()))
        # And not duplicated back into the role blocks.
        for role in VALID_ROLES:
            self.assertNotIn("no action needed", role_style_block(role).lower(), role)

    def test_no_two_role_blocks_are_alike(self):
        blocks = {role: role_style_block(role) for role in VALID_ROLES}
        self.assertEqual(len(set(blocks.values())), len(VALID_ROLES))
        # Beyond mere inequality: the shared invariant clause is the only text every block
        # has in common, so strip it and the remainder must still be disjoint enough that
        # no block is a substring of another.
        for a in VALID_ROLES:
            for b in VALID_ROLES:
                if a != b:
                    self.assertNotIn(blocks[a], blocks[b], f"{a} inside {b}")


class RoleWidensButNeverNarrowsTests(unittest.TestCase):
    def _plan(self, scope, role=None, question="how is the air quality?"):
        return plan_metrics(
            question=question,
            explicit_metrics=[],
            hinted_metrics=[],
            intent=IntentType.CURRENT_STATUS_DB,
            metric_scope=scope,
            role=role,
        )

    def test_no_role_ever_returns_fewer_metrics_than_the_default(self):
        for scope in sorted(VALID_METRIC_SCOPES):
            baseline = self._plan(scope)
            for role in VALID_ROLES:
                roled = self._plan(scope, role)
                self.assertGreaterEqual(
                    len(roled.selected), len(baseline.selected), f"{role}/{scope}"
                )
                # Stronger than a count: nothing the default fetched may disappear.
                self.assertTrue(
                    set(baseline.selected).issubset(set(roled.selected)), f"{role}/{scope}"
                )

    def test_researcher_widening_does_not_change_the_subject(self):
        # Widening gives a rigorous reader more of the SAME subject. Unioning the air pack
        # with SCOPE_FULL gave them a different one: "how is the air quality today?" came
        # back reporting sound and light, because the pack had been widened across domains
        # behind the question. An air pack is already the complete air answer.
        for scope in (SCOPE_AIR_QUALITY, SCOPE_IEQ_INDEX):
            baseline = self._plan(scope).selected
            widened = self._plan(scope, ROLE_RESEARCHER).selected
            self.assertEqual(baseline, widened, scope)
            for off_topic in ("sound", "light"):
                self.assertNotIn(off_topic, widened, f"{scope}/{off_topic}")

    def test_researcher_still_widens_the_cross_domain_scopes(self):
        # The widening is narrowed, not removed: a scope that already spans every dimension
        # has nothing off-topic to gain from the full pack.
        baseline = self._plan(SCOPE_DIAGNOSTIC).selected
        widened = self._plan(SCOPE_DIAGNOSTIC, ROLE_RESEARCHER).selected
        self.assertTrue(set(baseline).issubset(set(widened)))

    def test_researcher_widening_keeps_the_asked_about_metrics_first(self):
        widened = self._plan(SCOPE_IEQ_INDEX, ROLE_RESEARCHER).selected
        self.assertEqual(widened[:5], self._plan(SCOPE_IEQ_INDEX).selected)

    def test_comfort_pack_is_not_swapped_for_the_smaller_full_pack(self):
        # SCOPE_COMFORT is ten metrics and SCOPE_FULL is eight. A promote-to-full
        # implementation would silently drop itc/iaq here — the sub-indices that make it a
        # comfort assessment — the same shape of bug as the comfort comparison that lost
        # `sound` and `light`.
        baseline = self._plan(SCOPE_COMFORT).selected
        widened = self._plan(SCOPE_COMFORT, ROLE_RESEARCHER).selected
        self.assertEqual(sorted(baseline), sorted(widened))
        for sub in ("itc", "iaq"):
            self.assertNotIn(sub, self._plan(SCOPE_FULL).selected)
            self.assertIn(sub, widened)

    def test_a_named_metric_stays_named_for_every_role(self):
        for role in VALID_ROLES:
            plan = plan_metrics(
                question="what is the co2?",
                explicit_metrics=["co2"],
                hinted_metrics=[],
                intent=IntentType.POINT_LOOKUP_DB,
                metric_scope=SCOPE_NAMED,
                role=role,
            )
            self.assertEqual(plan.selected, ["co2"], role)

    def test_non_researcher_roles_leave_the_pack_alone(self):
        for role in (ROLE_OCCUPANT, ROLE_FACILITY_MANAGER, ROLE_EXECUTIVE):
            for scope in (SCOPE_AIR_QUALITY, SCOPE_COMFORT, SCOPE_FULL, SCOPE_DIAGNOSTIC):
                self.assertEqual(
                    self._plan(scope, role).selected, self._plan(scope).selected, f"{role}/{scope}"
                )


class NoClaimsAboutWhatPeopleAreDoingTests(unittest.TestCase):
    """A sensor reading cannot tell you what anyone is doing.

    Real answers closed on "The building operations team is monitoring the situation and
    will investigate potential sources" and "The system is monitoring it, and no immediate
    action is needed". Nothing in the evidence said either. Both are the kind of claim a
    reader acts on by NOT acting -- a reassurance-shaped gap being filled, because the
    answer has just reported a problem and reaches for the sentence that resolves it.

    Stated once in the shared invariant rather than per role: every role produced it, and
    repeating one rule across four prompts is the failure these blocks exist to avoid.
    """

    def test_every_role_forbids_asserting_someone_is_acting(self):
        for role in VALID_ROLES:
            block = role_style_block(role).lower()
            with self.subTest(role=role):
                self.assertIn("you report what the sensors measured", block)
                self.assertIn("never that anyone already is", block)

    def test_the_observed_fabrications_are_named_verbatim(self):
        # Naming the exact sentences that were produced, not a paraphrase of the category.
        block = role_style_block(ROLE_EXECUTIVE).lower()
        for phrase in ("is investigating", "has been logged", "we are monitoring this"):
            self.assertIn(phrase, block)

    def test_recommending_an_action_is_still_permitted(self):
        # The rule must not silence the executive block, whose whole job is naming who
        # should look at something.
        block = role_style_block(ROLE_EXECUTIVE).lower()
        self.assertIn("recommending that someone look into something", block)
        self.assertIn("is a suggestion you can support", block)
        self.assertIn("facilities team", block)

    def test_the_rule_lives_in_one_place_only(self):
        # The advisory bug started with one instruction restated in four prompts until the
        # model followed none of them. This must not be duplicated into the DB directives.
        from prompting import db_prompts

        for name in dir(db_prompts):
            value = getattr(db_prompts, name)
            if isinstance(value, str):
                self.assertNotIn(
                    "you report what the sensors measured", value.lower(), name
                )



class _StoreTestCase(unittest.TestCase):
    """Each test gets its own SQLite file; the store caches a connection per thread."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        import storage.conversation_store as store

        self.store = store
        self._patch = patch.object(
            store, "_DB_PATH", __import__("pathlib").Path(self._tmp.name) / "conv.db"
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        if hasattr(store._local, "conn"):
            store._local.conn.close()
            del store._local.conn
        self.addCleanup(self._reset_conn)

    def _reset_conn(self):
        if hasattr(self.store._local, "conn"):
            self.store._local.conn.close()
            del self.store._local.conn


class RoleIsNotRememberedTests(_StoreTestCase):
    """Role is per-message. Nothing about it survives the turn.

    An earlier design stored the last-used role on the conversation and inherited it when
    the field was omitted. It protected a client that forgot to send the field, at the cost
    of making an omitted field mean something different on turn 5 than on turn 1 — the same
    request body producing two differently-shaped answers depending on history, which is
    exactly what makes a role impossible to experiment with.
    """

    def test_the_store_holds_no_role_column(self):
        conn = self.store._conn()
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
        self.assertNotIn("role", columns)

    def test_append_turn_takes_no_role(self):
        import inspect

        self.assertNotIn(
            "role", inspect.signature(self.store.append_conversation_turn).parameters
        )

    def test_omitting_the_field_gives_the_default_regardless_of_history(self):
        from storage.conversation_context import build_conversation_context

        cid = "conv-stateless"
        # Turn 1 explicitly picks a role and is persisted.
        first = build_conversation_context(
            question="how is the air quality?", lab_name=None,
            conversation_id=cid, owner="o", role=ROLE_EXECUTIVE,
        )
        self.assertEqual(first.role, ROLE_EXECUTIVE)
        self.store.append_conversation_turn(cid, "q", "a", owner="o")

        # Turn 2 omits it — and gets the default, not the previous turn's choice.
        second = build_conversation_context(
            question="and now?", lab_name=None, conversation_id=cid, owner="o", role=None,
        )
        self.assertEqual((second.role, second.role_source), (ROLE_DEFAULT, "default"))


class RoleResolutionOrderTests(_StoreTestCase):
    def _build(self, conversation_id, role=None, owner="o"):
        from storage.conversation_context import build_conversation_context

        return build_conversation_context(
            question="how is the air quality?",
            lab_name="smart_lab",
            conversation_id=conversation_id,
            owner=owner,
            role=role,
        )

    def test_request_role_wins(self):
        ctx = self._build("conv-order-1", role=ROLE_RESEARCHER)
        self.assertEqual((ctx.role, ctx.role_source), (ROLE_RESEARCHER, "request"))

    def test_every_role_is_honoured_within_one_conversation(self):
        # The point of making this per-message: four consecutive turns on one
        # conversation_id, four different roles, no new conversation needed.
        cid = "conv-order-switching"
        for role in VALID_ROLES:
            ctx = self._build(cid, role=role)
            self.assertEqual((ctx.role, ctx.role_source), (role, "request"), role)
            self.store.append_conversation_turn(cid, "q", "a", owner="o")

    def test_falls_back_to_the_configured_default(self):
        ctx = self._build("conv-order-3")
        self.assertEqual((ctx.role, ctx.role_source), (ROLE_DEFAULT, "default"))
        self.assertFalse(ctx.role_fallback_used)

    def test_configured_default_is_honoured(self):
        with patch.dict(os.environ, {"DEFAULT_STAKEHOLDER_ROLE": ROLE_EXECUTIVE}):
            ctx = self._build("conv-order-4")
        self.assertEqual((ctx.role, ctx.role_source), (ROLE_EXECUTIVE, "default"))

    def test_unrecognised_request_role_degrades_but_stays_visible(self):
        ctx = self._build("conv-order-5", role="cto")
        self.assertEqual(ctx.role, ROLE_DEFAULT)
        self.assertTrue(ctx.role_fallback_used)


class RoleMetadataEchoTests(unittest.TestCase):
    def test_sync_and_stream_report_the_same_role(self):
        import json
        import asyncio

        from query_routing import query_orchestrator as qo
        from query_routing.router_types import RoutePlan
        from storage.conversation_context import ConversationContext

        ctx = ConversationContext(
            conversation_id="c",
            original_question="open the ifc view",
            raw_block="",
            effective_question="open the ifc view",
            effective_lab=None,
            routing_snippet="",
            llm_history="",
            role=ROLE_EXECUTIVE,
            role_source="request",
        )
        route = RoutePlan(
            intent=IntentType.VIEWER_CONTROL,
            confidence=0.9,
            lab_name=None,
            time_phrase=None,
            model="test",
            fallback_used=False,
            metrics=[],
            viewer_type="ifc",
        )

        async def _fake_plan(*a, **kw):
            return route

        with patch.object(qo, "plan_route", return_value=route):
            sync_meta = qo.execute_query(ctx, k=5)["metadata"]

        async def _collect():
            out = []
            with patch.object(qo, "plan_route_async", side_effect=_fake_plan):
                async for chunk in qo.stream_query(ctx, k=5):
                    raw = chunk.removeprefix("data: ").strip()
                    if raw:
                        out.append(json.loads(raw))
            return out

        events = asyncio.new_event_loop().run_until_complete(_collect())
        meta_frame = [e for e in events if e.get("event") == "meta"][0]

        for meta in (sync_meta, meta_frame):
            self.assertEqual(meta["role"], ROLE_EXECUTIVE)
            self.assertEqual(meta["role_source"], "request")
            self.assertFalse(meta["role_fallback_used"])


if __name__ == "__main__":
    unittest.main()
