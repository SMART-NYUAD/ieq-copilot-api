import os
import sys
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient


TEST_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
REPO_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from http_routes.health_routes import router as health_router


class HealthRouterRolloutTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(health_router)
        self.client = TestClient(app)

    def test_router_health_includes_rollout_and_thresholds(self):
        response = self.client.get("/health/router")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "healthy")
        self.assertIn("router_rollout", payload)
        self.assertIn("metrics", payload)
        self.assertIn("thresholds", payload)
        self.assertIn("slo", payload)
        self.assertIn("shadow_diff_rate_target", payload["thresholds"])
        self.assertIn("sync_stream_flip_rate_max", payload["thresholds"])
        self.assertIn("shadow_max_ok", payload["slo"])
        self.assertIn("parity_max_ok", payload["slo"])


if __name__ == "__main__":
    unittest.main()
