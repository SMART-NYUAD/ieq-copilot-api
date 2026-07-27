#!/usr/bin/env python3
"""Fail if any test module reaches the network.

A test that talks to the live sensor API or Ollama fails off-network and silently changes
meaning when the upstream data changes. This runs each test module in a subprocess with
outbound (non-loopback) connections blocked and reports the ones that break.

    python scripts/check_tests_hermetic.py

Use ``tests/fake_sensor_api.FakeSensorApiMixin`` to stub the sensor API in new tests.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GUARD = """
import socket as _socket
_real_connect = _socket.socket.connect


def _guarded_connect(self, address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else str(address)
    if str(host) not in ("127.0.0.1", "::1", "localhost"):
        raise AssertionError("NETWORK ACCESS to %s" % (address,))
    return _real_connect(self, address, *args, **kwargs)


_socket.socket.connect = _guarded_connect
"""


def _run_module(module: str) -> tuple[bool, str]:
    code = _GUARD + f"""
import sys, unittest
suite = unittest.TestLoader().discover("tests", pattern="{module}")
result = unittest.TextTestRunner(verbosity=0).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=900,
    )
    return proc.returncode == 0, proc.stderr


def main() -> int:
    failures = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "tests", "test_*.py"))):
        module = os.path.basename(path)
        ok, stderr = _run_module(module)
        reached_network = "NETWORK ACCESS" in stderr
        if ok and not reached_network:
            print(f"  ok       {module}")
            continue
        failures.append(module)
        reason = "reached the network" if reached_network else "failed with the network blocked"
        print(f"  FAIL     {module}  ({reason})")

    if failures:
        print(f"\n{len(failures)} test module(s) are not hermetic: {', '.join(failures)}")
        return 1
    print("\nAll test modules run without network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
