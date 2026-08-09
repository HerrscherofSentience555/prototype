from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class RunMode(str, Enum):
    COLD_START = "COLD_START"
    RESUME = "RESUME"
    EVALUATE = "EVALUATE"


class EmbeddingPlacement(str, Enum):
    DEVICE = "DEVICE"
    MANAGED_CACHING = "MANAGED_CACHING"


class PrecisionMode(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"


class CheckpointLoadMode(str, Enum):
    FULL = "FULL"
    MODEL_ONLY = "MODEL_ONLY"


class BackendName(str, Enum):
    STUB = "stub"
    DLRM = "dlrm"
    CUSTOM = "custom"
    TORCHREC_V1 = "torchrec_v1"


class RuntimePlatform(str, Enum):
    WINDOWS_WSL = "windows_wsl"
    LINUX_NATIVE = "linux_native"


class BackendConfig(BaseModel):
    name: BackendName = BackendName.STUB
    runtime_platform: RuntimePlatform = RuntimePlatform.WINDOWS_WSL
    dlrm_root: str = "/mnt/c/Users/<your-name>/Desktop/dlrm"
    python_env: str = "~/venvs/torchrec17"
    wsl_distribution: str = "Ubuntu-22.04"


class DeviceConfig(BaseModel):
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    embedding_placement: EmbeddingPlacement = EmbeddingPlacement.DEVICE
    cache_load_factor: float = 0.2

    @field_validator("gpu_ids")
    @classmethod
    def validate_gpu_ids(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("device.gpu_ids must contain at least one GPU id")
        if any(gpu_id < 0 for gpu_id in value):
            raise ValueError("device.gpu_ids must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("device.gpu_ids must not contain duplicates")
        return value

    @field_validator("cache_load_factor")
    @classmethod
    def validate_cache_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("cache_load_factor must be between 0.0 and 1.0")
        return value


class ModelConfig(BaseModel):
    file: str = "./model.py"
    config_file: Optional[str] = None
    num_embeddings: Optional[int] = None
    embedding_dim: Optional[int] = None
    dense_arch_layer_sizes: Optional[str] = None
    over_arch_layer_sizes: Optional[str] = None

    @field_validator("num_embeddings", "embedding_dim")
    @classmethod
    def validate_optional_positive_model_ints(cls, value: Optional[int], info) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError(f"model.{info.field_name} must be greater than 0 when provided")
        return value


class PrecisionConfig(BaseModel):
    embedding: PrecisionMode = PrecisionMode.FP32
    dense_compute: PrecisionMode = PrecisionMode.FP32
    comm_forward: PrecisionMode = PrecisionMode.FP32
    comm_backward: PrecisionMode = PrecisionMode.FP32


class DataConfig(BaseModel):
    train_path: Optional[str] = "./data/train"
    validation_path: Optional[str] = "./data/validation"
    test_path: Optional[str] = "./data/test"
    format: str = "random"
    criteo_binary_path: Optional[str] = None
    synthetic_multi_hot_path: Optional[str] = None
    schema_path: Optional[str] = None
    dataset_name: str = "criteo_1t"
    batch_size: int = 32
    test_batch_size: Optional[int] = None
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = False
    mmap_mode: bool = False
    persistent_workers: bool = False

    @field_validator("batch_size", "test_batch_size", "num_workers", "prefetch_factor")
    @classmethod
    def validate_non_negative_loader_values(cls, value: Optional[int], info) -> Optional[int]:
        if value is None:
            return value
        if info.field_name == "batch_size" and value <= 0:
            raise ValueError("data.batch_size must be greater than 0")
        if info.field_name == "test_batch_size" and value <= 0:
            raise ValueError("data.test_batch_size must be greater than 0 when provided")
        if info.field_name != "batch_size" and value < 0:
            raise ValueError(f"data.{info.field_name} must be >= 0")
        return value

    @field_validator("format")
    @classmethod
    def validate_data_format(cls, value: str) -> str:
        allowed = {"random", "criteo_binary", "synthetic_multihot", "parquet"}
        if value not in allowed:
            raise ValueError(f"data.format must be one of {sorted(allowed)}")
        return value

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        allowed = {"criteo_1t", "criteo_kaggle"}
        if value not in allowed:
            raise ValueError(f"data.dataset_name must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_format_paths(self) -> "DataConfig":
        if self.format == "criteo_binary" and not self.criteo_binary_path:
            raise ValueError("data.criteo_binary_path is required when data.format=criteo_binary")
        if self.format == "synthetic_multihot" and not self.synthetic_multi_hot_path:
            raise ValueError(
                "data.synthetic_multi_hot_path is required when data.format=synthetic_multihot"
            )
        if self.format == "parquet":
            missing = [
                name
                for name, value in {
                    "train_path": self.train_path,
                    "validation_path": self.validation_path,
                    "test_path": self.test_path,
                    "schema_path": self.schema_path,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(
                    "data.format=parquet requires " + ", ".join(f"data.{name}" for name in missing)
                )
        return self


class TrainingConfig(BaseModel):
    epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 0.01
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 50

    @field_validator("epochs", "log_every_n_steps", "eval_every_n_steps")
    @classmethod
    def validate_positive_ints(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"training.{info.field_name} must be greater than 0")
        return value

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("training.max_steps must be greater than 0 when provided")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("training.learning_rate must be greater than 0")
        return value


class CheckpointConfig(BaseModel):
    enabled: bool = True
    save_dir: str = ""
    save_every_n_steps: int = 100
    keep_last: int = 3
    save_optimizer: bool = True
    load_path: Optional[str] = None
    load_mode: CheckpointLoadMode = CheckpointLoadMode.FULL

    @field_validator("save_every_n_steps", "keep_last")
    @classmethod
    def validate_positive_checkpoint_values(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"checkpoint.{info.field_name} must be greater than 0")
        return value


class ProfileConfig(BaseModel):
    enabled: bool = False
    start_step: int = 100
    end_step: int = 120
    record_shapes: bool = True
    profile_memory: bool = True

    @field_validator("end_step")
    @classmethod
    def validate_step_range(cls, value: int, info) -> int:
        start_step = info.data.get("start_step", 0)
        if value < start_step:
            raise ValueError("profile.end_step must be >= profile.start_step")
        return value

    @field_validator("start_step")
    @classmethod
    def validate_start_step(cls, value: int) -> int:
        if value < 0:
            raise ValueError("profile.start_step must be >= 0")
        return value


class PrototypeConfig(BaseModel):
    job_name: str = "torchrec-job"
    mode: RunMode = RunMode.COLD_START
    backend: BackendConfig = Field(default_factory=BackendConfig)
    nproc_per_node: int = 1
    model: ModelConfig = Field(default_factory=ModelConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)

    @field_validator("nproc_per_node")
    @classmethod
    def validate_nproc_per_node(cls, value: int) -> int:
        if value < 1:
            raise ValueError("nproc_per_node must be >= 1")
        return value

    @model_validator(mode="after")
    def validate_mode_checkpoint_relationship(self) -> "PrototypeConfig":
        if self.mode in {RunMode.RESUME, RunMode.EVALUATE} and not self.checkpoint.load_path:
            raise ValueError(f"{self.mode.value} mode requires checkpoint.load_path")
        if self.backend.name in {BackendName.DLRM, BackendName.TORCHREC_V1} and self.nproc_per_node > len(
            self.device.gpu_ids
        ):
            raise ValueError(
                "nproc_per_node must be <= number of configured device.gpu_ids "
                "for the local prototype launcher"
            )
        return self

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml_file(cls, path: Path) -> "PrototypeConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
