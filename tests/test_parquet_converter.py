from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.parquet_converter import convert_parquet_to_criteo_numpy  # noqa: E402


class ParquetConverterTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed")
    def test_convert_parquet_to_criteo_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {
                    "clicked": [0, 1, 0],
                    "age": [1.0, 2.0, None],
                    "price": [3.0, 4.0, 5.0],
                    "user_id": ["u1", "u2", "u1"],
                    "item_id": ["i1", "i2", None],
                }
            )
            for split in ["train", "validation", "test"]:
                frame.to_parquet(root / f"{split}.parquet")
            schema_path = root / "schema.yaml"
            schema_path.write_text(
                "\n".join(
                    [
                        "label:",
                        "  name: clicked",
                        "dense_features:",
                        "  - name: age",
                        "  - name: price",
                        "sparse_features:",
                        "  - name: user_id",
                        "  - name: item_id",
                    ]
                ),
                encoding="utf-8",
            )
            config = PrototypeConfig(
                data={
                    "format": "parquet",
                    "train_path": str(root / "train.parquet"),
                    "validation_path": str(root / "validation.parquet"),
                    "test_path": str(root / "test.parquet"),
                    "schema_path": str(schema_path),
                }
            )
            output_dir = root / "out"
            manifest = convert_parquet_to_criteo_numpy(config, output_dir)
            saved_manifest = json.loads((output_dir / "conversion-manifest.json").read_text(encoding="utf-8"))

            dense = np.load(output_dir / "train_dense.npy")
            sparse = np.load(output_dir / "train_sparse.npy")
            labels = np.load(output_dir / "train_labels.npy")

        self.assertEqual(manifest["splits"]["train"]["rows"], 3)
        self.assertEqual(saved_manifest["format"], "criteo_numpy")
        self.assertEqual(dense.shape, (3, 2))
        self.assertEqual(sparse.shape, (3, 2))
        self.assertEqual(labels.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
