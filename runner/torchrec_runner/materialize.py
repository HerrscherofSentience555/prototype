from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prototype.config import PrototypeConfig
from prototype.runner.torchrec_runner.data import CRITEO_DENSE_FEATURES, CRITEO_SPARSE_FEATURES


def materialize_batch_preview(config: PrototypeConfig, run_dir: Path, split: str = "train") -> dict[str, Any]:
    dense, sparse, labels, source = _load_arrays(config, split)
    preview_rows = min(config.data.batch_size, len(labels))
    dense = dense[:preview_rows]
    sparse = sparse[:preview_rows]
    labels = labels[:preview_rows]
    report: dict[str, Any] = {
        "schema": "torchrec-v1-batch-materialization",
        "split": split,
        "source": source,
        "rows": int(preview_rows),
        "dense_shape": list(dense.shape),
        "sparse_shape": list(sparse.shape),
        "labels_shape": list(labels.shape),
        "dense_feature_names": CRITEO_DENSE_FEATURES[: dense.shape[1]],
        "sparse_feature_names": CRITEO_SPARSE_FEATURES[: sparse.shape[1]],
        "torch": {"available": False, "dense_tensor_created": False, "labels_tensor_created": False},
        "torchrec": {"available": False, "keyed_jagged_tensor_created": False},
    }
    _try_materialize_torch_objects(report, dense, sparse, labels)
    return report


def write_batch_materialization(config: PrototypeConfig, run_dir: Path, split: str = "train") -> dict[str, Any]:
    report = materialize_batch_preview(config, run_dir, split=split)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-batch-materialization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _load_arrays(config: PrototypeConfig, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if config.data.format == "criteo_binary" and config.data.criteo_binary_path:
        root = Path(config.data.criteo_binary_path)
        dense_path = root / f"{split}_dense.npy"
        sparse_path = root / f"{split}_sparse.npy"
        labels_path = root / f"{split}_labels.npy"
        if dense_path.exists() and sparse_path.exists() and labels_path.exists():
            return (
                np.load(dense_path),
                np.load(sparse_path),
                np.load(labels_path),
                str(root),
            )
    batch_size = max(config.data.batch_size, 1)
    dense = np.zeros((batch_size, len(CRITEO_DENSE_FEATURES)), dtype=np.float32)
    sparse = np.tile(np.arange(len(CRITEO_SPARSE_FEATURES), dtype=np.int64), (batch_size, 1))
    labels = np.zeros((batch_size, 1), dtype=np.float32)
    return dense, sparse, labels, "synthetic_preview"


def _try_materialize_torch_objects(
    report: dict[str, Any],
    dense: np.ndarray,
    sparse: np.ndarray,
    labels: np.ndarray,
) -> None:
    try:
        import torch

        dense_tensor = torch.as_tensor(dense, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
        report["torch"] = {
            "available": True,
            "dense_tensor_created": True,
            "labels_tensor_created": True,
            "dense_tensor_shape": list(dense_tensor.shape),
            "labels_tensor_shape": list(labels_tensor.shape),
        }
    except Exception as exc:
        report["torch"]["error"] = f"{type(exc).__name__}: {exc}"
        return

    try:
        import torch
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        keys = CRITEO_SPARSE_FEATURES[: sparse.shape[1]]
        values = torch.as_tensor(sparse.reshape(-1), dtype=torch.int64)
        lengths = torch.ones((sparse.shape[0] * sparse.shape[1],), dtype=torch.int32)
        kjt = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        report["torchrec"] = {
            "available": True,
            "keyed_jagged_tensor_created": True,
            "keys": list(kjt.keys()),
            "values_count": int(values.numel()),
            "lengths_count": int(lengths.numel()),
        }
    except Exception as exc:
        report["torchrec"]["error"] = f"{type(exc).__name__}: {exc}"
