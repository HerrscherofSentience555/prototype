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
    materialize_batch_preview,
    write_batch_materialization,
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


if __name__ == "__main__":
    unittest.main()
