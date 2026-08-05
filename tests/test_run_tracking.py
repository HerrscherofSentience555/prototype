from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.runner.metrics import append_metric  # noqa: E402
from prototype.runner.run_tracking import compare_runs, write_run_summaries  # noqa: E402


class RunTrackingTests(unittest.TestCase):
    def test_write_run_summaries_and_compare_best_auc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            for job_id, auc in [("job-a", 0.7), ("job-b", 0.8)]:
                run_dir = runs_dir / job_id
                run_dir.mkdir()
                (run_dir / "resolved-config.yaml").write_text(
                    "mode: COLD_START\nbackend:\n  name: stub\ndata:\n  format: random\n",
                    encoding="utf-8",
                )
                (run_dir / "state.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "job_name": job_id,
                            "status": "SUCCEEDED",
                            "backend": "stub",
                        }
                    ),
                    encoding="utf-8",
                )
                append_metric(run_dir / "metrics.jsonl", step=1, metric="auc", value=auc)
                write_run_summaries(run_dir)

            comparison = compare_runs(runs_dir, ["job-a", "job-b"], "auc")

        self.assertEqual(comparison["best_run"]["job_id"], "job-b")
        self.assertEqual(comparison["runs"][0]["auc"], 0.7)


if __name__ == "__main__":
    unittest.main()
