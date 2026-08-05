from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode  # noqa: E402


class PrototypeConfigTests(unittest.TestCase):
    def test_default_config_yaml_round_trip(self) -> None:
        config = PrototypeConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "resolved-config.yaml"
            path.write_text(config.to_yaml(), encoding="utf-8")
            loaded = PrototypeConfig.from_yaml_file(path)

        self.assertEqual(loaded.backend.name.value, "stub")
        self.assertEqual(loaded.backend.dlrm_root, "/mnt/c/Users/han/Desktop/dlrm")
        self.assertEqual(loaded.training.learning_rate, 0.01)

    def test_invalid_cache_load_factor_fails(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(device={"cache_load_factor": 1.5})

    def test_invalid_profile_step_range_fails(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(profile={"start_step": 10, "end_step": 9})

    def test_evaluate_without_checkpoint_fails(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(mode=RunMode.EVALUATE)

    def test_resume_without_checkpoint_fails(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(mode=RunMode.RESUME)

    def test_evaluate_with_checkpoint_passes(self) -> None:
        config = PrototypeConfig(
            mode=RunMode.EVALUATE,
            checkpoint={"load_path": "checkpoint-path"},
        )
        self.assertEqual(config.checkpoint.load_path, "checkpoint-path")

    def test_positive_numeric_guards(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(nproc_per_node=0)
        with self.assertRaises(ValidationError):
            PrototypeConfig(data={"batch_size": 0})
        with self.assertRaises(ValidationError):
            PrototypeConfig(training={"epochs": 0})

    def test_gpu_ids_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(device={"gpu_ids": []})
        with self.assertRaises(ValidationError):
            PrototypeConfig(device={"gpu_ids": [0, 0]})
        with self.assertRaises(ValidationError):
            PrototypeConfig(device={"gpu_ids": [-1]})

    def test_dlrm_nproc_must_fit_configured_gpu_ids(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(backend={"name": "dlrm"}, nproc_per_node=2, device={"gpu_ids": [0]})

        config = PrototypeConfig(
            backend={"name": "dlrm"},
            nproc_per_node=2,
            device={"gpu_ids": [0, 1]},
        )
        self.assertEqual(config.device.gpu_ids, [0, 1])

    def test_real_data_formats_require_paths(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(data={"format": "criteo_binary"})
        with self.assertRaises(ValidationError):
            PrototypeConfig(data={"format": "synthetic_multihot"})

        config = PrototypeConfig(
            data={
                "format": "criteo_binary",
                "criteo_binary_path": "/mnt/c/Users/han/Desktop/prototype/data/criteo_npy",
                "dataset_name": "criteo_kaggle",
            }
        )
        self.assertEqual(config.data.dataset_name, "criteo_kaggle")

    def test_parquet_format_requires_schema_and_splits(self) -> None:
        with self.assertRaises(ValidationError):
            PrototypeConfig(data={"format": "parquet", "train_path": "train.parquet"})


if __name__ == "__main__":
    unittest.main()
