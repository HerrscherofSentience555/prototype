from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.telemetry import ResourceTelemetry  # noqa: E402


class TelemetryTests(unittest.TestCase):
    def test_resource_telemetry_writes_metrics_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            telemetry = ResourceTelemetry(run_dir, config=PrototypeConfig(), interval_seconds=0.05)
            telemetry.start()
            time.sleep(0.12)
            telemetry.stop()
            records = [
                json.loads(line)
                for line in (run_dir / "resource-metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads((run_dir / "artifacts" / "resource-summary.json").read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(records), 1)
        self.assertGreaterEqual(summary["record_count"], 1)
        self.assertIn("max_run_dir_disk_bytes", summary)
        self.assertIn("gpu_telemetry_available", records[0])
        self.assertIn("wsl_telemetry_available", records[0])
        self.assertIn("max_gpu_utilization_percent", summary)
        self.assertIn("max_wsl_torch_cpu_percent", summary)


if __name__ == "__main__":
    unittest.main()
