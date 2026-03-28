#!/usr/bin/env python3
"""Compatibility entrypoint for the reorganized RAG API server."""

import os
import sys
import logging

import uvicorn

# Ensure imports work when launched from either repo root or RAG_API_SERVER.
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(CURRENT_DIR)
for path in (CURRENT_DIR, REPO_ROOT):
    if path and path not in sys.path:
        sys.path.insert(0, path)

try:
    from app_bootstrap import app
    from core_settings import load_settings
except ImportError:
    from RAG_API_SERVER.app_bootstrap import app
    from RAG_API_SERVER.core_settings import load_settings


if __name__ == "__main__":
    # Keep logging setup local to runtime entrypoint to avoid side effects on imports/tests.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else settings.server_port
    host = sys.argv[2] if len(sys.argv) > 2 else settings.server_host

    print(f"Starting RAG API server on {host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level="info")
