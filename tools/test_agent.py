"""Run mandatory agent integration checks; missing runtime is an error, never a skip."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rsikit import EvaluationContext  # noqa: F401

    from mass.agent import TeamPolicy  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        f"Install the compatible rsikit source and requirements-agent.txt: {exc}"
    ) from None

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ["tests.test_agent", "tests.test_agent_cli"]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
