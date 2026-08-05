from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.runner.torchrec_runner.runtime import write_runtime_smoke  # noqa: E402


class TorchRecRuntimeSmokeTests(unittest.TestCase):
    def test_runtime_smoke_reports_ready_when_core_objects_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report = write_runtime_smoke(
                run_dir,
                availability={"torch_available": True, "torchrec_available": True},
                batch_materialization={
                    "torchrec": {"keyed_jagged_tensor_created": True},
                    "torch": {"dense_tensor_created": True, "labels_tensor_created": True},
                },
                embedding_report={"runtime": {"torchrec_embedding_config_available": True}},
            )
            saved = json.loads((run_dir / "artifacts" / "torchrec-runtime-smoke.json").read_text(encoding="utf-8"))

        self.assertTrue(report["ready_for_dmp_smoke"])
        self.assertTrue(saved["ready_for_dmp_smoke"])

    def test_runtime_smoke_collects_fallback_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = write_runtime_smoke(
                Path(tmpdir),
                availability={
                    "torch_available": False,
                    "torchrec_available": False,
                    "torch_error": "torch missing",
                    "torchrec_error": "torchrec missing",
                },
                batch_materialization={"torch": {"error": "dense failed"}},
                embedding_report={"runtime": {"error": "embedding failed"}},
            )

        self.assertFalse(report["ready_for_dmp_smoke"])
        self.assertIn("torch missing", report["fallback_reasons"])
        self.assertIn("embedding failed", report["fallback_reasons"])
        self.assertIn("EmbeddingShardingPlanner.collective_plan can run against that model", report["next_required_for_dmp_smoke"])


if __name__ == "__main__":
    unittest.main()
