"""API-key authentication and conversation ownership.

``conversation_id`` is client-supplied, so before ownership existed any caller could
read another caller's history by replaying an id. Conversations are now bound to the
authenticated caller; with no keys configured every caller is the shared anonymous
owner, which preserves the previous single-tenant behavior.
"""

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from fastapi import HTTPException

from http_routes import auth
from storage import conversation_store as cs


class ApiKeyDependencyTests(unittest.TestCase):
    def test_no_keys_configured_leaves_the_api_open_as_anonymous(self):
        with patch.object(auth, "api_keys", return_value=[]):
            self.assertEqual(auth.require_api_key(None, None), cs.ANONYMOUS_OWNER)

    def test_missing_key_is_rejected_when_keys_are_configured(self):
        with patch.object(auth, "api_keys", return_value=["secret-a"]):
            with self.assertRaises(HTTPException) as ctx:
                auth.require_api_key(None, None)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_invalid_key_is_rejected(self):
        with patch.object(auth, "api_keys", return_value=["secret-a"]):
            with self.assertRaises(HTTPException) as ctx:
                auth.require_api_key("wrong", None)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_valid_key_via_header_or_bearer_yields_the_same_caller_id(self):
        with patch.object(auth, "api_keys", return_value=["secret-a"]):
            from_header = auth.require_api_key("secret-a", None)
            from_bearer = auth.require_api_key(None, "Bearer secret-a")
        self.assertEqual(from_header, from_bearer)
        self.assertTrue(from_header.startswith("key:"))

    def test_caller_id_never_contains_the_key(self):
        with patch.object(auth, "api_keys", return_value=["super-secret"]):
            caller_id = auth.require_api_key("super-secret", None)
        self.assertNotIn("super-secret", caller_id)

    def test_distinct_keys_yield_distinct_callers(self):
        with patch.object(auth, "api_keys", return_value=["a-key", "b-key"]):
            self.assertNotEqual(
                auth.require_api_key("a-key", None), auth.require_api_key("b-key", None)
            )


class ConversationOwnershipTests(unittest.TestCase):
    """Each test runs against an isolated SQLite file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_path = cs._DB_PATH
        self._original_local = cs._local
        cs._DB_PATH = Path(self._tmp.name) / "conv.db"
        cs._local = threading.local()

    def tearDown(self):
        cs._DB_PATH = self._original_path
        cs._local = self._original_local
        self._tmp.cleanup()

    def test_owner_can_read_back_their_own_history(self):
        cs.append_conversation_turn("conv-alice-1", "hi", "hello", owner="key:alice")
        cid, block = cs.build_compact_context("conv-alice-1", owner="key:alice")
        self.assertEqual(cid, "conv-alice-1")
        self.assertIn("hi", block)

    def test_other_caller_cannot_read_history(self):
        cs.append_conversation_turn("conv-alice-1", "hi", "hello", owner="key:alice")
        with self.assertRaises(cs.ConversationAccessError):
            cs.build_compact_context("conv-alice-1", owner="key:mallory")

    def test_other_caller_cannot_append_to_the_conversation(self):
        cs.append_conversation_turn("conv-alice-1", "hi", "hello", owner="key:alice")
        with self.assertRaises(cs.ConversationAccessError):
            cs.append_conversation_turn("conv-alice-1", "sneak", "in", owner="key:mallory")

    def test_unknown_id_is_not_an_access_error(self):
        cid, block = cs.build_compact_context("conv-brand-new", owner="key:alice")
        self.assertEqual(cid, "conv-brand-new")
        self.assertEqual(block, "")

    def test_anonymous_callers_share_one_namespace(self):
        cs.append_conversation_turn("conv-shared-1", "hi", "hello")
        _, block = cs.build_compact_context("conv-shared-1")
        self.assertIn("hi", block)

    def test_legacy_unowned_conversation_is_claimable_then_exclusive(self):
        # Simulate a row written before the owner column existed.
        cs.append_conversation_turn("conv-legacy-1", "hi", "hello", owner=cs._UNCLAIMED_OWNER)
        # Any caller may still read it...
        _, block = cs.build_compact_context("conv-legacy-1", owner="key:alice")
        self.assertIn("hi", block)
        # ...and the next writer claims it.
        cs.append_conversation_turn("conv-legacy-1", "again", "sure", owner="key:alice")
        with self.assertRaises(cs.ConversationAccessError):
            cs.build_compact_context("conv-legacy-1", owner="key:mallory")

    def test_owner_column_is_added_to_a_pre_existing_database(self):
        import sqlite3

        # Build the pre-ownership schema by hand, then let the store open it.
        legacy = sqlite3.connect(str(cs._DB_PATH))
        legacy.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, "
            "last_turn_index INTEGER NOT NULL DEFAULT 0)"
        )
        legacy.commit()
        legacy.close()

        cs.append_conversation_turn("conv-migrated-1", "hi", "hello", owner="key:alice")
        _, block = cs.build_compact_context("conv-migrated-1", owner="key:alice")
        self.assertIn("hi", block)
        with self.assertRaises(cs.ConversationAccessError):
            cs.build_compact_context("conv-migrated-1", owner="key:mallory")


class HttpEndpointAuthTests(unittest.TestCase):
    """The dependency must actually be attached to the routes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_path = cs._DB_PATH
        self._original_local = cs._local
        cs._DB_PATH = Path(self._tmp.name) / "conv.db"
        cs._local = threading.local()

        from fastapi.testclient import TestClient
        from app_bootstrap import app

        self.client = TestClient(app)

    def tearDown(self):
        cs._DB_PATH = self._original_path
        cs._local = self._original_local
        self._tmp.cleanup()

    _RESULT = {
        "answer": "CO2 is 600 ppm.",
        "timescale": "1hour",
        "cards_retrieved": 0,
        "recent_card": False,
        "metadata": {},
        "footnotes": [],
        "citation_sources": [],
    }

    def _post(self, body, headers=None):
        # Patch the name the route resolves (it imports execute_query directly), so no
        # router LLM or sensor API is touched.
        from http_routes import query_routes

        with patch.dict(os.environ, {"RAG_API_KEYS": "test-key"}), \
             patch.object(query_routes, "execute_query", return_value=dict(self._RESULT)):
            return self.client.post("/query", json=body, headers=headers or {})

    def test_health_stays_open_for_probes(self):
        with patch.dict(os.environ, {"RAG_API_KEYS": "test-key"}):
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_query_without_a_key_is_401(self):
        response = self._post({"question": "what is the co2?"})
        self.assertEqual(response.status_code, 401)

    def test_query_with_a_valid_key_succeeds(self):
        response = self._post(
            {"question": "what is the co2?"}, headers={"X-API-Key": "test-key"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "CO2 is 600 ppm.")

    def test_foreign_conversation_id_is_403_not_a_history_leak(self):
        cs.append_conversation_turn("conv-alice-1", "secret question", "secret answer", owner="key:alice")
        response = self._post(
            {"question": "what did I ask?", "conversation_id": "conv-alice-1"},
            headers={"X-API-Key": "test-key"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("secret", response.text)


class ContextBuilderOwnershipTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_path = cs._DB_PATH
        self._original_local = cs._local
        cs._DB_PATH = Path(self._tmp.name) / "conv.db"
        cs._local = threading.local()

    def tearDown(self):
        cs._DB_PATH = self._original_path
        cs._local = self._original_local
        self._tmp.cleanup()

    def test_route_helper_propagates_ownership(self):
        from http_routes.route_helpers import build_query_context, persist_turn

        persist_turn("conv-owned-1", "what is co2?", "CO2 is …", owner="key:alice")
        ctx = build_query_context("and humidity?", None, "conv-owned-1", owner="key:alice")
        self.assertIn("what is co2?", ctx.raw_block)

        with self.assertRaises(cs.ConversationAccessError):
            build_query_context("and humidity?", None, "conv-owned-1", owner="key:mallory")


if __name__ == "__main__":
    unittest.main()
