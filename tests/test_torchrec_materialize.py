from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.materialize import (  # noqa: E402
    build_runtime_batch,
    materialize_batch_preview,
    write_batch_materialization,
    write_runtime_batch_report,
    write_runtime_batch_summary,
)


class TorchRecMaterializeTests(unittest.TestCase):
    def test_random_materialization_report_has_batch_contract_shapes(self) -> None:
        report = materialize_batch_preview(
            PrototypeConfig(backend={"name": "torchrec_v1"}, data={"format": "random", "batch_size": 3}),
            Path("unused"),
        )

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["dense_shape"], [3, 13])
        self.assertEqual(report["sparse_shape"], [3, 26])
        self.assertIn("torch", report)
        self.assertIn("torchrec", report)

    def test_criteo_binary_materialization_reads_numpy_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "train_dense.npy", np.ones((2, 13), dtype=np.float32))
            np.save(root / "train_sparse.npy", np.ones((2, 26), dtype=np.int64))
            np.save(root / "train_labels.npy", np.ones((2, 1), dtype=np.float32))
            report = materialize_batch_preview(
                PrototypeConfig(
                    backend={"name": "torchrec_v1"},
                    data={
                        "format": "criteo_binary",
                        "criteo_binary_path": str(root),
                        "batch_size": 4,
                    },
                ),
                root,
            )

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["source"], str(root))

    def test_write_batch_materialization_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_batch_materialization(PrototypeConfig(backend={"name": "torchrec_v1"}), run_dir)
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-batch-materialization.json").read_text(encoding="utf-8")
            )

        self.assertEqual(saved["schema"], "torchrec-v1-batch-materialization")

    def test_write_runtime_batch_report_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report = write_runtime_batch_report(
                PrototypeConfig(backend={"name": "torchrec_v1"}, data={"batch_size": 2}),
                run_dir,
            )
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-runtime-batch.json").read_text(encoding="utf-8")
            )

        self.assertEqual(saved["schema"], "torchrec-v1-runtime-batch")
        self.assertEqual(saved["runtime_batch_created"], report["runtime_batch_created"])

    def test_runtime_batch_creates_keyed_jagged_tensor_when_runtime_is_available(self) -> None:
        try:
            import torch  # noqa: F401
            import torchrec  # noqa: F401
        except Exception:
            self.skipTest("torch and torchrec are not installed in this Python environment")

        batch = build_runtime_batch(
            PrototypeConfig(backend={"name": "torchrec_v1"}, data={"batch_size": 2}),
            split="train",
        )

        self.assertEqual(list(batch.dense_features.shape), [2, 13])
        self.assertEqual(list(batch.labels.shape), [2, 1])
        self.assertIsNotNone(batch.sparse_features)
        self.assertTrue(batch.report["keyed_jagged_tensor_created"])

    def test_runtime_batch_can_be_rank_sharded(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not installed in this Python environment")

        config = PrototypeConfig(backend={"name": "torchrec_v1"}, data={"batch_size": 2})
        rank0 = build_runtime_batch(config, split="train", rank=0, world_size=2)
        rank1 = build_runtime_batch(config, split="train", rank=1, world_size=2)

        self.assertTrue(rank0.report["rank_sharded"])
        self.assertEqual(rank0.report["selected_indices_preview"], [0, 2])
        self.assertEqual(rank1.report["selected_indices_preview"], [1, 3])
        self.assertEqual(list(rank0.dense_features.shape), [2, 13])
        self.assertEqual(list(rank1.dense_features.shape), [2, 13])

    def test_write_runtime_batch_summary_aggregates_rank_reports(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_runtime_batch_report(
                PrototypeConfig(backend={"name": "torchrec_v1"}, data={"batch_size": 2}),
                run_dir,
                rank=0,
                world_size=2,
            )
            write_runtime_batch_report(
                PrototypeConfig(backend={"name": "torchrec_v1"}, data={"batch_size": 2}),
                run_dir,
                rank=1,
                world_size=2,
            )

            summary = write_runtime_batch_summary(run_dir, world_size=2)
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-runtime-batch-summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(summary["all_ranks_reported"])
        self.assertTrue(summary["all_ranks_created_runtime_batch"])
        self.assertEqual(saved["schema"], "torchrec-v1-runtime-batch-summary")
        self.assertEqual(saved["ranks"][0]["selected_indices_preview"], [0, 2])
        self.assertEqual(saved["ranks"][1]["selected_indices_preview"], [1, 3])


if __name__ == "__main__":
    unittest.main()
