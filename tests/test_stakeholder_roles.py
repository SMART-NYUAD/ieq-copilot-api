"""Stakeholder roles change who an answer is written for — and nothing else.

A role is declared by the caller, not inferred from the question, so unlike `intent` or
`analysis_mode` there is no router eval that can catch a regression here. These tests carry
the whole guarantee, and it has two halves.

The first half is that the default is a no-op. Before roles existed the assistant had
exactly one persona, hardcoded in SHARED_SYSTEM_PROMPT as "Write for non-technical
occupants". `occupant` is that wording verbatim, so `shared_system_prompt(ROLE_OCCUPANT)`
must be byte-identical to the constant it replaced. That is the strongest available
statement that a caller who sends no role is unaffected by any of this.

The second half is that a role may never subtract. The completeness rules in db_prompts and
the computed threshold verdicts were each won back from a specific wrong answer (see
test_air_quality_completeness and test_threshold_assessment), and the new roles push
directly against them: `executive` is capped at 60 words and `facility_manager` wants
operational brevity. Every non-default role therefore carries an explicit clause saying
completeness outranks it, and the metric pack a role resolves to can only ever grow.
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


class DefaultRoleIsANoOpTests(unittest.TestCase):
    """The whole feature must be invisible to a caller who does not use it."""

    def test_default_is_the_occupant_role(self):
        self.assertEqual(ROLE_DEFAULT, ROLE_OCCUPANT)

    def test_occupant_prompt_is_byte_identical_to_the_module_constant(self):
        self.assertEqual(shared_system_prompt(ROLE_OCCUPANT), SHARED_SYSTEM_PROMPT)

    def test_occupant_block_is_the_previous_hardcoded_wording(self):
        self.assertEqual(role_style_block(ROLE_OCCUPANT), _LEGACY_AUDIENCE_LINES)
        self.assertIn(_LEGACY_AUDIENCE_LINES, SHARED_SYSTEM_PROMPT)

    def test_unknown_role_renders_the_default_prompt(self):
        self.assertEqual(shared_system_prompt("cto"), SHARED_SYSTEM_PROMPT)

    def test_addendum_is_empty_for_the_default_role(self):
        # IFC/sensor prompts already read as occupant-facing, so the default adds nothing
        # and those two prompts stay byte-identical to their previous form.
        self.assertEqual(role_addendum(ROLE_OCCUPANT), "")
        for role in VALID_ROLES:
            if role != ROLE_OCCUPANT:
                self.assertTrue(role_addendum(role).strip(), role)


class RoleNeverSubtractsTests(unittest.TestCase):
    _NON_DEFAULT = (ROLE_FACILITY_MANAGER, ROLE_RESEARCHER, ROLE_EXECUTIVE)

    def test_every_new_role_carries_the_completeness_clause(self):
        for role in self._NON_DEFAULT:
            block = role_style_block(role).lower()
            self.assertIn("never permit dropping a metric", block, role)
            self.assertIn("threshold assessment", block, role)
            self.assertIn("completeness wins", block, role)

    def test_role_does_not_touch_the_assessment_directives(self):
        # Role is spliced into the system prompt only. If a future change starts appending
        # it to the directives too, the completeness block gains a competing voice — which
        # is the failure mode CLAUDE.md records for the advisory bug.
        for directive in (
            DB_TOOL_RESPONSE_DIRECTIVE,
            DB_TOOL_RESPONSE_DIRECTIVE_AIR_QUALITY_POINT_LOOKUP,
        ):
            self.assertIn("Report EVERY", directive.replace("report every", "Report EVERY"))
            for role in VALID_ROLES:
                self.assertNotIn(role_style_block(role), directive)

    def test_brevity_roles_still_forbid_omitting_a_failing_metric(self):
        # The executive role is the shortest and therefore the most likely to drop the
        # metric that spoils the verdict — the exact PM2.5 regression from
        # test_air_quality_completeness, in a new place.
        block = role_style_block(ROLE_EXECUTIVE).lower()
        self.assertIn("must appear in the answer", block)
        self.assertIn("fewer words about the same facts, not fewer facts", block)


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

    def test_researcher_widens_air_quality_to_every_metric(self):
        baseline = self._plan(SCOPE_AIR_QUALITY).selected
        widened = self._plan(SCOPE_AIR_QUALITY, ROLE_RESEARCHER).selected
        self.assertNotEqual(baseline, widened)
        for extra in ("temperature", "sound", "light"):
            self.assertNotIn(extra, baseline)
            self.assertIn(extra, widened)

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


class RoleStickinessTests(_StoreTestCase):
    def test_role_persists_and_is_read_back(self):
        self.store.append_conversation_turn("conv-role-1", "q", "a", owner="o", role=ROLE_RESEARCHER)
        self.assertEqual(self.store.get_conversation_role("conv-role-1", owner="o"), ROLE_RESEARCHER)

    def test_a_turn_without_a_role_does_not_erase_the_stored_one(self):
        # The row is written with INSERT OR REPLACE, so the previous value has to be
        # carried forward explicitly or a follow-up turn would silently reset the voice.
        self.store.append_conversation_turn("conv-role-2", "q1", "a1", owner="o", role=ROLE_EXECUTIVE)
        self.store.append_conversation_turn("conv-role-2", "q2", "a2", owner="o", role=None)
        self.assertEqual(self.store.get_conversation_role("conv-role-2", owner="o"), ROLE_EXECUTIVE)

    def test_an_explicit_role_replaces_the_stored_one(self):
        self.store.append_conversation_turn("conv-role-3", "q1", "a1", owner="o", role=ROLE_EXECUTIVE)
        self.store.append_conversation_turn("conv-role-3", "q2", "a2", owner="o", role=ROLE_RESEARCHER)
        self.assertEqual(self.store.get_conversation_role("conv-role-3", owner="o"), ROLE_RESEARCHER)

    def test_unknown_conversation_has_no_role(self):
        self.assertIsNone(self.store.get_conversation_role("conv-role-absent", owner="o"))
        self.assertIsNone(self.store.get_conversation_role(None, owner="o"))

    def test_reading_another_callers_role_is_refused(self):
        # A role is a preference attached to someone else's conversation, so it is subject
        # to the same ownership rule as the history itself.
        self.store.append_conversation_turn("conv-role-4", "q", "a", owner="owner-a", role=ROLE_RESEARCHER)
        with self.assertRaises(self.store.ConversationAccessError):
            self.store.get_conversation_role("conv-role-4", owner="owner-b")

    def test_migration_adds_the_column_to_a_pre_existing_database(self):
        path = self.store._DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy = sqlite3.connect(str(path))
        legacy.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, "
            "last_turn_index INTEGER NOT NULL DEFAULT 0)"
        )
        legacy.execute("INSERT INTO conversations VALUES ('legacy-conv', '2020-01-01', 3)")
        legacy.commit()
        legacy.close()

        self.store.append_conversation_turn("legacy-conv", "q", "a", owner="o", role=ROLE_EXECUTIVE)
        self.assertEqual(self.store.get_conversation_role("legacy-conv", owner="o"), ROLE_EXECUTIVE)


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
        self.store.append_conversation_turn("conv-order-1", "q", "a", owner="o", role=ROLE_EXECUTIVE)
        ctx = self._build("conv-order-1", role=ROLE_RESEARCHER)
        self.assertEqual((ctx.role, ctx.role_source), (ROLE_RESEARCHER, "request"))

    def test_conversation_role_is_inherited_when_the_request_omits_one(self):
        self.store.append_conversation_turn("conv-order-2", "q", "a", owner="o", role=ROLE_FACILITY_MANAGER)
        ctx = self._build("conv-order-2")
        self.assertEqual((ctx.role, ctx.role_source), (ROLE_FACILITY_MANAGER, "conversation"))

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
