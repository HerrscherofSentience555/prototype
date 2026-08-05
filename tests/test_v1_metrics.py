from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import ProfileConfig  # noqa: E402
from prototype.runner.v1_metrics import append_v1_step_metrics, is_profile_window_active  # noqa: E402


class V1MetricsTests(unittest.TestCase):
    def test_profile_window_active_uses_inclusive_step_range(self) -> None:
        profile = ProfileConfig(enabled=True, start_step=2, end_step=3)

        self.assertFalse(is_profile_window_active(profile, 1))
        self.assertTrue(is_profile_window_active(profile, 2))
        self.assertTrue(is_profile_window_active(profile, 3))
        self.assertFalse(is_profile_window_active(profile, 4))

    def test_append_v1_step_metrics_writes_throughput_and_stage_timings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            append_v1_step_metrics(
                path,
                step=2,
                step_time_seconds=0.5,
                global_batch_size=16,
                profile=ProfileConfig(enabled=True, start_step=2, end_step=2),
            )
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        metric_names = [record["metric"] for record in records]
        self.assertIn("samples_per_second", metric_names)
        self.assertIn("profile_window_active", metric_names)
        self.assertIn("embedding_lookup_seconds", metric_names)
        self.assertEqual(
            next(record["value"] for record in records if record["metric"] == "profile_window_active"),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
