"""Tests for core_settings config helpers and CORS safety.

These cover the configurability/hardening changes: the sensor/predictions API base URLs
are env-driven, the display timezone is configurable, space slugs resolve consistently,
and the CORS loader refuses the spec-invalid wildcard-origin + credentials combination.
"""

import os
import sys
import unittest
from datetime import timedelta, timezone
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import core_settings as cs


class TestApiBaseUrls(unittest.TestCase):
    def test_sensor_api_base_url_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SENSOR_API_BASE_URL", None)
            self.assertTrue(cs.sensor_api_base_url().startswith("http"))

    def test_sensor_api_base_url_override(self):
        with patch.dict(os.environ, {"SENSOR_API_BASE_URL": "http://example.test:9000/"}):
            self.assertEqual(cs.sensor_api_base_url(), "http://example.test:9000")

    def test_predictions_api_base_url_override(self):
        with patch.dict(os.environ, {"PREDICTIONS_API_BASE_URL": "https://pred.test/"}):
            self.assertEqual(cs.predictions_api_base_url(), "https://pred.test")


class TestDisplayTimezone(unittest.TestCase):
    def test_default_offset_is_plus_four(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY_UTC_OFFSET_HOURS", None)
            self.assertEqual(cs.display_timezone(), timezone(timedelta(hours=4)))

    def test_offset_override(self):
        with patch.dict(os.environ, {"DISPLAY_UTC_OFFSET_HOURS": "0"}):
            self.assertEqual(cs.display_timezone(), timezone(timedelta(hours=0)))


class TestSlugifySpace(unittest.TestCase):
    def test_none_and_blank_fall_back_to_default(self):
        self.assertEqual(cs.slugify_space(None), cs.download_space_slug())
        self.assertEqual(cs.slugify_space("   "), cs.download_space_slug())

    def test_name_is_slugified(self):
        self.assertEqual(cs.slugify_space("Smart Lab"), "smart_lab")
        self.assertEqual(cs.slugify_space("Room 2B!"), "room_2b")


class TestCorsSafety(unittest.TestCase):
    def test_wildcard_with_credentials_disables_credentials(self):
        with patch.dict(os.environ, {
            "RAG_API_CORS_ALLOW_ORIGINS": "*",
            "RAG_API_CORS_ALLOW_CREDENTIALS": "true",
        }):
            settings = cs.load_settings()
        self.assertEqual(settings.cors_allow_origins, ["*"])
        self.assertFalse(settings.cors_allow_credentials)

    def test_explicit_origins_keep_credentials(self):
        with patch.dict(os.environ, {
            "RAG_API_CORS_ALLOW_ORIGINS": "https://app.example.com",
            "RAG_API_CORS_ALLOW_CREDENTIALS": "true",
        }):
            settings = cs.load_settings()
        self.assertEqual(settings.cors_allow_origins, ["https://app.example.com"])
        self.assertTrue(settings.cors_allow_credentials)


if __name__ == "__main__":
    unittest.main()
