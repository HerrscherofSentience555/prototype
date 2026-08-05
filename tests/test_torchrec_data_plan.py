from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.data import build_data_plan, write_data_plan  # noqa: E402


class TorchRecDataPlanTests(unittest.TestCase):
    def test_random_data_plan_uses_criteo_like_feature_schema(self) -> None:
        plan = build_data_plan(
            PrototypeConfig(
                backend={"name": "torchrec_v1"},
                data={"format": "random", "batch_size": 8},
                nproc_per_node=2,
                device={"gpu_ids": [0, 1]},
            ),
            Path("unused"),
        )

        self.assertEqual(plan["global_batch_size"], 16)
        self.assertEqual(len(plan["feature_schema"]["dense_features"]), 13)
        self.assertEqual(len(plan["feature_schema"]["sparse_features"]), 26)
        self.assertEqual(plan["batch_contract"]["sparse_features"]["logical_type"], "TorchRec KeyedJaggedTensor")

    def test_criteo_binary_data_plan_records_expected_numpy_files(self) -> None:
        plan = build_data_plan(
            PrototypeConfig(
                backend={"name": "torchrec_v1"},
                data={
                    "format": "criteo_binary",
                    "criteo_binary_path": r"C:\data\criteo_npy",
                    "batch_size": 4,
                },
            ),
            Path("unused"),
        )

        self.assertIn("train_dense.npy", plan["converted_numpy_expected_files"]["train"][0])

    def test_write_data_plan_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_data_plan(PrototypeConfig(backend={"name": "torchrec_v1"}), run_dir)
            saved = json.loads((run_dir / "artifacts" / "torchrec-data-plan.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["schema"], "torchrec-v1-data-plan")


if __name__ == "__main__":
    unittest.main()
