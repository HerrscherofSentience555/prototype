from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.dmp import (  # noqa: E402
    build_dmp_report,
    try_wrap_distributed_model_parallel,
    write_dmp_report,
    write_dmp_summary,
)


class TorchRecDmpWrapTests(unittest.TestCase):
    def test_build_dmp_report_does_not_attempt_without_process_group(self) -> None:
        report = build_dmp_report(
            PrototypeConfig(backend={"name": "torchrec_v1"}, nproc_per_node=1),
            {"process_group_initialized": False, "world_size": 1, "rank": 0, "device": "cpu"},
        )

        self.assertFalse(report["can_attempt_wrap"])
        self.assertFalse(report["wrapped"])

    def test_try_wrap_dmp_writes_artifact(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            wrapped, report = try_wrap_distributed_model_parallel(
                model,
                PrototypeConfig(backend={"name": "torchrec_v1"}, nproc_per_node=1),
                {"process_group_initialized": False, "world_size": 1, "rank": 0, "device": "cpu"},
                run_dir,
            )
            saved = json.loads((run_dir / "artifacts" / "torchrec-dmp-wrap.json").read_text(encoding="utf-8"))

        self.assertIs(wrapped, model)
        self.assertFalse(report["wrapped"])
        self.assertEqual(saved["schema"], "torchrec-v1-dmp-wrap")

    def test_write_dmp_summary_aggregates_rank_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_dmp_report(run_dir, {"rank": 0, "can_attempt_wrap": True, "wrapped": True, "error": None})
            write_dmp_report(run_dir, {"rank": 1, "can_attempt_wrap": True, "wrapped": True, "error": None})

            summary = write_dmp_summary(run_dir, world_size=2)
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-dmp-wrap-summary.json").read_text(encoding="utf-8")
            )

        self.assertTrue(summary["all_ranks_reported"])
        self.assertTrue(summary["all_ranks_wrapped"])
        self.assertEqual(saved["schema"], "torchrec-v1-dmp-wrap-summary")
        self.assertEqual([rank["rank"] for rank in saved["ranks"]], [0, 1])

    def test_write_dmp_summary_reports_missing_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_dmp_report(run_dir, {"rank": 0, "can_attempt_wrap": True, "wrapped": True, "error": None})

            summary = write_dmp_summary(run_dir, world_size=2)

        self.assertFalse(summary["all_ranks_reported"])
        self.assertFalse(summary["all_ranks_wrapped"])
        self.assertEqual(summary["ranks"][1]["error"], "missing rank DMP report")


if __name__ == "__main__":
    unittest.main()
