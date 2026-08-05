from __future__ import annotations

import json
from pathlib import Path

from prototype.config import BackendName, EmbeddingPlacement, PrecisionMode, PrototypeConfig


def build_v1_capability_report(config: PrototypeConfig) -> dict:
    backend = config.backend.name
    return {
        "schema": "torchrec-prototype-v1-capability",
        "backend": backend.value,
        "requested": {
            "gpu_ids": config.device.gpu_ids,
            "nproc_per_node": config.nproc_per_node,
            "embedding_placement": config.device.embedding_placement.value,
            "cache_load_factor": config.device.cache_load_factor,
            "precision": config.precision.model_dump(mode="json"),
        },
        "mapped": _mapped_capabilities(config),
        "not_yet_mapped": _unmapped_capabilities(config),
        "v1_acceptance_note": (
            "This report makes GPU placement/cache/precision behavior explicit for the V1 prototype. "
            "Fields listed under not_yet_mapped are preserved in config but do not yet alter the "
            "selected backend's TorchRec execution behavior."
        ),
    }


def write_v1_capability_report(config: PrototypeConfig, run_dir: Path) -> dict:
    report = build_v1_capability_report(config)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "v1-capability-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _mapped_capabilities(config: PrototypeConfig) -> dict:
    backend = config.backend.name
    mapped = {
        "cuda_visible_devices": backend in {BackendName.DLRM, BackendName.TORCHREC_V1},
        "single_machine_torchrun": backend in {BackendName.DLRM, BackendName.TORCHREC_V1},
        "nproc_per_node": backend in {BackendName.DLRM, BackendName.TORCHREC_V1},
        "checkpoint_save_load": backend in {BackendName.STUB, BackendName.CUSTOM, BackendName.DLRM, BackendName.TORCHREC_V1},
        "profile_request": True,
        "profile_step_window_metric": backend in {BackendName.STUB, BackendName.CUSTOM, BackendName.TORCHREC_V1},
    }
    if backend == BackendName.TORCHREC_V1:
        mapped["model_contract_validation"] = True
        mapped["minimal_training_loop"] = True
    return mapped


def _unmapped_capabilities(config: PrototypeConfig) -> dict:
    backend = config.backend.name
    unmapped = {}
    if config.device.embedding_placement == EmbeddingPlacement.MANAGED_CACHING:
        unmapped["managed_caching_runtime"] = (
            "FBGEMM managed caching is recorded but not yet applied in the selected backend."
        )
    if config.device.embedding_placement == EmbeddingPlacement.DEVICE and backend in {
        BackendName.STUB,
        BackendName.CUSTOM,
        BackendName.TORCHREC_V1,
    }:
        unmapped["device_embedding_runtime"] = (
            "DEVICE placement is recorded; a full TorchRec DMP model construction path is required "
            "before this changes embedding storage behavior."
        )
    precision = config.precision
    if any(
        value != PrecisionMode.FP32
        for value in [
            precision.embedding,
            precision.dense_compute,
            precision.comm_forward,
            precision.comm_backward,
        ]
    ):
        unmapped["non_fp32_precision_runtime"] = (
            "Non-FP32 precision is recorded but not yet applied to model construction or "
            "TorchRec communication settings."
        )
    if backend != BackendName.TORCHREC_V1:
        unmapped["internal_torchrec_runner"] = (
            "Use backend=torchrec_v1 to exercise the internal V1 runner scaffold."
        )
    if backend == BackendName.TORCHREC_V1:
        unmapped["distributed_model_parallel"] = (
            "The backend validates the V1 model contract and runs a minimal loop; full "
            "DistributedModelParallel is still pending."
        )
        unmapped["train_pipeline_sparse_dist"] = "TrainPipelineSparseDist integration is pending."
    return unmapped
