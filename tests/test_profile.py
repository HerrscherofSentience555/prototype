from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.profile import profile_span  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_profile_span_writes_request_and_runner_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = PrototypeConfig(
                profile={
                    "enabled": True,
                    "start_step": 1,
                    "end_step": 2,
                    "record_shapes": False,
                    "profile_memory": True,
                }
            )
            with profile_span(config, run_dir):
                time.sleep(0.01)

            request = json.loads((run_dir / "profiles" / "profile-request.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "profiles" / "runner-profile.json").read_text(encoding="utf-8"))

        self.assertEqual(request["start_step"], 1)
        self.assertFalse(request["record_shapes"])
        self.assertIn(request["status"], {"torch_profiler_active", "runner_wall_time_only"})
        self.assertIn("profile_trace_supported", summary)
        self.assertIn("trace_path", summary)
        self.assertGreater(summary["duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
