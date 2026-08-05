from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.capability import build_v1_capability_report, write_v1_capability_report  # noqa: E402


class CapabilityReportTests(unittest.TestCase):
    def test_torchrec_v1_report_marks_runner_scaffold_and_pending_dmp(self) -> None:
        report = build_v1_capability_report(
            PrototypeConfig(
                backend={"name": "torchrec_v1"},
                device={"embedding_placement": "MANAGED_CACHING"},
                precision={"dense_compute": "BF16"},
            )
        )

        self.assertTrue(report["mapped"]["model_contract_validation"])
        self.assertIn("managed_caching_runtime", report["not_yet_mapped"])
        self.assertIn("distributed_model_parallel", report["not_yet_mapped"])
        self.assertIn("non_fp32_precision_runtime", report["not_yet_mapped"])

    def test_write_v1_capability_report_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report = write_v1_capability_report(PrototypeConfig(), run_dir)
            saved = json.loads((run_dir / "artifacts" / "v1-capability-report.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["schema"], "torchrec-prototype-v1-capability")
        self.assertEqual(saved["backend"], report["backend"])


if __name__ == "__main__":
    unittest.main()
