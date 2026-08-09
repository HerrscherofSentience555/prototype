from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prototype.config import PrototypeConfig
from prototype.runner.torchrec_runner.data import CRITEO_DENSE_FEATURES, CRITEO_SPARSE_FEATURES


@dataclass
class RuntimeBatch:
    dense_features: Any
    sparse_features: Any
    labels: Any
    sparse_tensor: Any
    report: dict[str, Any]


def write_runtime_batch_summary(run_dir: Path, world_size: int) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    ranks = []
    for rank in range(world_size):
        path = artifacts_dir / f"torchrec-runtime-batch-rank{rank}.json"
        report = None
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report = {
                    "rank": rank,
                    "runtime_batch_created": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        ranks.append(
            {
                "rank": rank,
                "reported": report is not None,
                "runtime_batch_created": report.get("runtime_batch_created") if report else None,
                "keyed_jagged_tensor_created": report.get("keyed_jagged_tensor_created")
                if report
                else None,
                "rows": report.get("rows") if report else None,
                "selected_indices_preview": report.get("selected_indices_preview") if report else [],
                "error": report.get("error") if report else "missing rank runtime batch report",
            }
        )
    summary = {
        "schema": "torchrec-v1-runtime-batch-summary",
        "world_size": world_size,
        "all_ranks_reported": all(rank["reported"] for rank in ranks),
        "all_ranks_created_runtime_batch": all(rank["runtime_batch_created"] for rank in ranks),
        "all_ranks_created_keyed_jagged_tensor": all(
            rank["keyed_jagged_tensor_created"] for rank in ranks
        ),
        "ranks": ranks,
    }
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-runtime-batch-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def materialize_batch_preview(config: PrototypeConfig, run_dir: Path, split: str = "train") -> dict[str, Any]:
    dense, sparse, labels, source = load_batch_arrays(config, split)
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


def build_runtime_batch(
    config: PrototypeConfig,
    split: str = "train",
    device: str = "cpu",
    rank: int = 0,
    world_size: int = 1,
) -> RuntimeBatch:
    dense, sparse, labels, source = load_batch_arrays(config, split, min_rows=config.data.batch_size * world_size)
    global_rows_available = len(labels)
    rank = max(rank, 0)
    world_size = max(world_size, 1)
    selected_indices = np.arange(rank, len(labels), world_size, dtype=np.int64)[: config.data.batch_size]
    dense = dense[selected_indices]
    sparse = sparse[selected_indices]
    labels = labels[selected_indices]
    rows = len(labels)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Runtime batch requires torch: {type(exc).__name__}: {exc}") from exc

    dense_tensor = torch.as_tensor(dense, dtype=torch.float32, device=device)
    sparse_tensor = torch.as_tensor(sparse, dtype=torch.long, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    sparse_features = None
    torchrec_error = None
    try:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        keys = CRITEO_SPARSE_FEATURES[: sparse.shape[1]]
        values = sparse_tensor.reshape(-1)
        lengths = torch.ones((sparse.shape[0] * sparse.shape[1],), dtype=torch.int32, device=device)
        sparse_features = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
    except Exception as exc:
        torchrec_error = f"{type(exc).__name__}: {exc}"
    report = {
        "schema": "torchrec-v1-runtime-batch",
        "split": split,
        "source": source,
        "rows": int(rows),
        "rank": rank,
        "world_size": world_size,
        "rank_sharded": world_size > 1,
        "global_rows_available": int(global_rows_available),
        "selected_indices_preview": [int(index) for index in selected_indices[:10]],
        "device": device,
        "dense_tensor_shape": list(dense_tensor.shape),
        "labels_tensor_shape": list(labels_tensor.shape),
        "sparse_tensor_shape": list(sparse_tensor.shape),
        "keyed_jagged_tensor_created": sparse_features is not None,
        "runtime_batch_created": True,
        "sparse_feature_keys": CRITEO_SPARSE_FEATURES[: sparse.shape[1]]
        if sparse_features is not None
        else [],
        "torchrec_error": torchrec_error,
    }
    return RuntimeBatch(
        dense_features=dense_tensor,
        sparse_features=sparse_features,
        labels=labels_tensor,
        sparse_tensor=sparse_tensor,
        report=report,
    )


def write_batch_materialization(config: PrototypeConfig, run_dir: Path, split: str = "train") -> dict[str, Any]:
    report = materialize_batch_preview(config, run_dir, split=split)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-batch-materialization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def write_runtime_batch_report(
    config: PrototypeConfig,
    run_dir: Path,
    split: str = "train",
    device: str = "cpu",
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    try:
        report = build_runtime_batch(
            config,
            split=split,
            device=device,
            rank=rank,
            world_size=world_size,
        ).report
    except Exception as exc:
        report = {
            "schema": "torchrec-v1-runtime-batch",
            "split": split,
            "rank": rank,
            "world_size": world_size,
            "runtime_batch_created": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / f"torchrec-runtime-batch-rank{rank}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if int(rank) == 0:
        (artifacts_dir / "torchrec-runtime-batch.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def load_batch_arrays(
    config: PrototypeConfig,
    split: str,
    min_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
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
    batch_size = max(min_rows or config.data.batch_size, 1)
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
