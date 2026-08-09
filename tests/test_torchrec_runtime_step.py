from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.runner.torchrec_runner.runtime_step import (  # noqa: E402
    write_runtime_step_report,
    write_runtime_step_summary,
)


class TorchRecRuntimeStepTests(unittest.TestCase):
    def test_write_runtime_step_summary_aggregates_rank_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_runtime_step_report(
                run_dir,
                {
                    "rank": 0,
                    "runtime_step_executed": True,
                    "runtime_loop_completed": True,
                    "requested_steps": 2,
                    "completed_steps": 2,
                    "train_loss": 0.5,
                    "accuracy": 1.0,
                    "runtime_kjt_batch": 1.0,
                    "dmp_wrapped": True,
                    "error": None,
                },
            )
            write_runtime_step_report(
                run_dir,
                {
                    "rank": 1,
                    "runtime_step_executed": True,
                    "runtime_loop_completed": True,
                    "requested_steps": 2,
                    "completed_steps": 2,
                    "train_loss": 0.4,
                    "accuracy": 1.0,
                    "runtime_kjt_batch": 1.0,
                    "dmp_wrapped": True,
                    "error": None,
                },
            )

            summary = write_runtime_step_summary(run_dir, world_size=2)
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-runtime-step-summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(summary["all_ranks_reported"])
        self.assertTrue(summary["all_ranks_executed_runtime_step"])
        self.assertTrue(summary["all_ranks_completed_runtime_loop"])
        self.assertTrue(summary["all_ranks_used_dmp"])
        self.assertEqual(summary["min_completed_steps"], 2)
        self.assertEqual(summary["max_requested_steps"], 2)
        self.assertEqual(saved["schema"], "torchrec-v1-runtime-step-summary")

    def test_write_runtime_step_summary_reports_missing_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_runtime_step_report(
                run_dir,
                {
                    "rank": 0,
                    "runtime_step_executed": True,
                    "runtime_loop_completed": True,
                    "requested_steps": 2,
                    "completed_steps": 2,
                    "dmp_wrapped": True,
                    "error": None,
                },
            )

            summary = write_runtime_step_summary(run_dir, world_size=2)

        self.assertFalse(summary["all_ranks_reported"])
        self.assertFalse(summary["all_ranks_executed_runtime_step"])
        self.assertEqual(summary["ranks"][1]["error"], "missing rank runtime step report")


if __name__ == "__main__":
    unittest.main()
