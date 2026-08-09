from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SETTINGS_PATH = PROJECT_ROOT / "local_settings.yaml"
EXAMPLE_SETTINGS_PATH = PROJECT_ROOT / "local_settings.example.yaml"


class RuntimeSettings(BaseModel):
    platform: str = "windows_wsl"
    wsl_distribution: str = "Ubuntu-22.04"
    python_env: str = "~/venvs/torchrec17"


class PathSettings(BaseModel):
    dlrm_root: str = "/mnt/c/Users/<your-name>/Desktop/dlrm"
    default_model_file: str = "examples/models/torchrec_v1_model.py"
    criteo_binary_path: str = "data/criteo_kaggle_sample_npy"
    synthetic_multi_hot_path: str = ""
    parquet_train_path: str = "data/business_ctr/train.parquet"
    parquet_validation_path: str = "data/business_ctr/validation.parquet"
    parquet_test_path: str = "data/business_ctr/test.parquet"
    parquet_schema_path: str = "examples/schemas/parquet-smoke-schema.yaml"
    parquet_conversion_output: str = "data/converted_criteo_npy"


class DefaultJobSettings(BaseModel):
    backend: str = "dlrm"
    data_format: str = "criteo_binary"
    dataset_name: str = "criteo_kaggle"
    batch_size: int = 16
    test_batch_size: int = 16
    max_steps: int = 100
    learning_rate: float = 0.01
    nproc_per_node: int = 1
    gpu_ids: str = "0"


class LocalSettings(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    defaults: DefaultJobSettings = Field(default_factory=DefaultJobSettings)
    source: str = "built-in defaults"

    def project_path(self, value: str | None) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str(PROJECT_ROOT / path)


def load_local_settings() -> LocalSettings:
    path = LOCAL_SETTINGS_PATH if LOCAL_SETTINGS_PATH.exists() else EXAMPLE_SETTINGS_PATH
    if not path.exists():
        return LocalSettings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = LocalSettings.model_validate(data)
    return settings.model_copy(update={"source": str(path)})


def write_local_settings_template(path: Path = LOCAL_SETTINGS_PATH) -> Path:
    if path.exists():
        return path
    source = EXAMPLE_SETTINGS_PATH
    if not source.exists():
        path.write_text(yaml.safe_dump(LocalSettings().model_dump(mode="json")), encoding="utf-8")
        return path
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def local_settings_status() -> dict[str, Any]:
    settings = load_local_settings()
    return {
        "settings_source": settings.source,
        "local_settings_exists": LOCAL_SETTINGS_PATH.exists(),
        "local_settings_path": str(LOCAL_SETTINGS_PATH),
        "example_settings_path": str(EXAMPLE_SETTINGS_PATH),
        "runtime": settings.runtime.model_dump(mode="json"),
        "paths": settings.paths.model_dump(mode="json"),
        "defaults": settings.defaults.model_dump(mode="json"),
    }
