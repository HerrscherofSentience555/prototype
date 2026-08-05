from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from prototype.runner.data_validation import DataValidationError, validate_parquet_dataset
from prototype.config import PrototypeConfig


def convert_parquet_to_criteo_numpy(config: PrototypeConfig, output_dir: Path) -> dict[str, Any]:
    if config.data.format != "parquet":
        raise DataValidationError("convert_parquet_to_criteo_numpy requires data.format=parquet")
    profile = validate_parquet_dataset(config)
    schema = yaml.safe_load(Path(config.data.schema_path).read_text(encoding="utf-8")) or {}
    label_name = schema["label"]["name"]
    dense_features = [feature["name"] for feature in schema.get("dense_features", [])]
    sparse_features = [feature["name"] for feature in schema.get("sparse_features", [])]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": Path(config.data.train_path),
        "validation": Path(config.data.validation_path),
        "test": Path(config.data.test_path),
    }
    manifest = {
        "format": "criteo_numpy",
        "output_dir": str(output_dir),
        "label": label_name,
        "dense_features": dense_features,
        "sparse_features": sparse_features,
        "splits": {},
        "profile": profile,
    }
    for split, path in split_paths.items():
        frame = pd.read_parquet(path, columns=[label_name, *dense_features, *sparse_features])
        dense = _dense_array(frame, dense_features)
        sparse = _sparse_array(frame, sparse_features)
        labels = frame[label_name].fillna(0).astype("int32").to_numpy().reshape(-1, 1)
        np.save(output_dir / f"{split}_dense.npy", dense)
        np.save(output_dir / f"{split}_sparse.npy", sparse)
        np.save(output_dir / f"{split}_labels.npy", labels)
        manifest["splits"][split] = {
            "rows": int(len(frame)),
            "dense_shape": list(dense.shape),
            "sparse_shape": list(sparse.shape),
            "labels_shape": list(labels.shape),
        }
    (output_dir / "conversion-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _dense_array(frame: pd.DataFrame, dense_features: list[str]) -> np.ndarray:
    if not dense_features:
        return np.zeros((len(frame), 0), dtype=np.float32)
    return frame[dense_features].fillna(0).astype("float32").to_numpy()


def _sparse_array(frame: pd.DataFrame, sparse_features: list[str]) -> np.ndarray:
    if not sparse_features:
        return np.zeros((len(frame), 0), dtype=np.int64)
    columns = []
    for feature in sparse_features:
        codes, _uniques = pd.factorize(frame[feature].fillna("__MISSING__"), sort=True)
        columns.append(codes.astype("int64"))
    return np.stack(columns, axis=1)
