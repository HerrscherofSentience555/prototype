from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prototype.config import PrototypeConfig


CRITEO_DENSE_FEATURES = [f"I{i}" for i in range(1, 14)]
CRITEO_SPARSE_FEATURES = [f"C{i}" for i in range(1, 27)]


def build_data_plan(config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    plan = {
        "schema": "torchrec-v1-data-plan",
        "format": config.data.format,
        "batch_size_per_rank": config.data.batch_size,
        "nproc_per_node": config.nproc_per_node,
        "global_batch_size": config.data.batch_size * config.nproc_per_node,
        "num_workers": config.data.num_workers,
        "prefetch_factor": config.data.prefetch_factor,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.persistent_workers,
        "splits": _split_paths(config),
        "batch_contract": {
            "dense_features": {
                "logical_type": "float32 tensor",
                "shape": ["batch_size", "num_dense_features"],
            },
            "sparse_features": {
                "logical_type": "TorchRec KeyedJaggedTensor",
                "keys": _sparse_feature_names(config),
            },
            "labels": {
                "logical_type": "float32/int tensor",
                "shape": ["batch_size", 1],
            },
        },
        "feature_schema": {
            "dense_features": _dense_feature_names(config),
            "sparse_features": _sparse_feature_names(config),
        },
        "runtime": {
            "torch_tensor_materialization": "pending_real_torch_runtime",
            "keyed_jagged_tensor_materialization": "pending_real_torchrec_runtime",
        },
    }
    if config.data.format == "criteo_binary" and config.data.criteo_binary_path:
        plan["converted_numpy_expected_files"] = _expected_criteo_numpy_files(config.data.criteo_binary_path)
    return plan


def write_data_plan(config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    plan = build_data_plan(config, run_dir)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-data-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plan


def _split_paths(config: PrototypeConfig) -> dict[str, str | None]:
    if config.data.format == "criteo_binary":
        return {
            "train": config.data.criteo_binary_path,
            "validation": config.data.criteo_binary_path,
            "test": config.data.criteo_binary_path,
        }
    return {
        "train": config.data.train_path,
        "validation": config.data.validation_path,
        "test": config.data.test_path,
    }


def _dense_feature_names(config: PrototypeConfig) -> list[str]:
    if config.data.format in {"criteo_binary", "random"}:
        return CRITEO_DENSE_FEATURES
    return ["configured_by_schema"]


def _sparse_feature_names(config: PrototypeConfig) -> list[str]:
    if config.data.format in {"criteo_binary", "random"}:
        return CRITEO_SPARSE_FEATURES
    return ["configured_by_schema"]


def _expected_criteo_numpy_files(root: str) -> dict[str, list[str]]:
    base = Path(root)
    return {
        split: [
            str(base / f"{split}_dense.npy"),
            str(base / f"{split}_sparse.npy"),
            str(base / f"{split}_labels.npy"),
        ]
        for split in ["train", "validation", "test"]
    }
