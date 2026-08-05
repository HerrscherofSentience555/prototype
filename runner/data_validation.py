from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from prototype.config import PrototypeConfig


PROFILE_SAMPLE_ROWS = 5000


class DataValidationError(ValueError):
    pass


def validate_dataset_if_needed(config: PrototypeConfig, run_dir: Path) -> None:
    if config.data.format != "parquet":
        return
    profile = validate_parquet_dataset(config)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "data-profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_parquet_dataset(config: PrototypeConfig) -> dict[str, Any]:
    schema_path = _required_existing_path(config.data.schema_path, "data.schema_path")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    label = schema.get("label", {})
    dense_features = schema.get("dense_features", [])
    sparse_features = schema.get("sparse_features", [])
    label_name = label.get("name")
    if not label_name:
        raise DataValidationError("schema.label.name is required")
    if not isinstance(dense_features, list) or not isinstance(sparse_features, list):
        raise DataValidationError("schema.dense_features and schema.sparse_features must be lists")

    split_paths = {
        "train": _required_existing_path(config.data.train_path, "data.train_path"),
        "validation": _required_existing_path(config.data.validation_path, "data.validation_path"),
        "test": _required_existing_path(config.data.test_path, "data.test_path"),
    }
    required_columns = [label_name]
    required_columns.extend(_feature_name(feature, "dense_features") for feature in dense_features)
    required_columns.extend(_feature_name(feature, "sparse_features") for feature in sparse_features)

    profile: dict[str, Any] = {
        "format": "parquet",
        "schema_path": str(schema_path),
        "label": label,
        "dense_features": dense_features,
        "sparse_features": sparse_features,
        "splits": {},
    }
    for split, path in split_paths.items():
        frame = _read_parquet_sample(path, required_columns)
        _validate_columns(frame, required_columns, split)
        _validate_label(frame[label_name], split, label_name)
        for feature in dense_features:
            name = _feature_name(feature, "dense_features")
            if not pd.api.types.is_numeric_dtype(frame[name]):
                raise DataValidationError(f"{split}.{name} must be numeric")
        split_profile = {
            "path": str(path),
            "sample_rows": int(len(frame)),
            "columns": list(frame.columns),
            "null_rates": {
                column: float(frame[column].isna().mean()) for column in required_columns
            },
            "sparse_cardinality_sample": {
                _feature_name(feature, "sparse_features"): int(
                    frame[_feature_name(feature, "sparse_features")].nunique(dropna=True)
                )
                for feature in sparse_features
            },
        }
        profile["splits"][split] = split_profile
    return profile


def _required_existing_path(path_value: str | None, label: str) -> Path:
    if not path_value:
        raise DataValidationError(f"{label} is required")
    path = Path(path_value)
    if not path.exists():
        raise DataValidationError(f"{label} does not exist: {path_value}")
    return path


def _feature_name(feature: dict[str, Any], section: str) -> str:
    name = feature.get("name") if isinstance(feature, dict) else None
    if not name:
        raise DataValidationError(f"Every {section} entry requires a name")
    return str(name)


def _read_parquet_sample(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise DataValidationError(
            "Reading parquet requires pyarrow or fastparquet. Install project requirements first."
        ) from exc
    except Exception as exc:
        raise DataValidationError(f"Failed to read parquet file {path}: {exc}") from exc
    if len(frame) > PROFILE_SAMPLE_ROWS:
        return frame.head(PROFILE_SAMPLE_ROWS)
    return frame


def _validate_columns(frame: pd.DataFrame, required_columns: list[str], split: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{split} split is missing columns: {', '.join(missing)}")


def _validate_label(series: pd.Series, split: str, label_name: str) -> None:
    values = set(series.dropna().unique().tolist())
    if not values.issubset({0, 1, False, True}):
        raise DataValidationError(f"{split}.{label_name} must contain only binary labels")
