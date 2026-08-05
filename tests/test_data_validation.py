from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.data_validation import DataValidationError, validate_dataset_if_needed  # noqa: E402


class DataValidationTests(unittest.TestCase):
    def test_parquet_schema_path_must_exist(self) -> None:
        config = PrototypeConfig(
            data={
                "format": "parquet",
                "train_path": "train.parquet",
                "validation_path": "validation.parquet",
                "test_path": "test.parquet",
                "schema_path": "missing.yaml",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(DataValidationError):
                validate_dataset_if_needed(config, Path(tmpdir))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed")
    def test_valid_parquet_writes_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {
                    "clicked": [0, 1, 0],
                    "age": [20.0, 31.5, 42.0],
                    "user_id": ["u1", "u2", "u1"],
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
                        "    dtype: float",
                        "sparse_features:",
                        "  - name: user_id",
                        "    dtype: categorical",
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

            validate_dataset_if_needed(config, root)
            profile = json.loads((root / "artifacts" / "data-profile.json").read_text(encoding="utf-8"))

        self.assertEqual(profile["format"], "parquet")
        self.assertEqual(profile["splits"]["train"]["sample_rows"], 3)
        self.assertEqual(profile["splits"]["train"]["sparse_cardinality_sample"]["user_id"], 2)


if __name__ == "__main__":
    unittest.main()
