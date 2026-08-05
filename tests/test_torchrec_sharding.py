from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.sharding import (  # noqa: E402
    build_sharding_planner_readiness,
    write_sharding_planner_readiness,
)


class TorchRecShardingReadinessTests(unittest.TestCase):
    def test_sharding_readiness_never_claims_collective_plan_without_model(self) -> None:
        report = build_sharding_planner_readiness(
            PrototypeConfig(backend={"name": "torchrec_v1"}, nproc_per_node=1),
            {"count": 26},
        )

        self.assertFalse(report["collective_plan_created"])
        self.assertIn("fallback_reasons", report)
        self.assertEqual(report["requested_topology"]["local_world_size"], 1)

    def test_write_sharding_readiness_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_sharding_planner_readiness(
                PrototypeConfig(backend={"name": "torchrec_v1"}),
                run_dir,
                {"count": 26},
            )
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-sharding-plan-readiness.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved["schema"], "torchrec-v1-sharding-plan-readiness")


if __name__ == "__main__":
    unittest.main()
