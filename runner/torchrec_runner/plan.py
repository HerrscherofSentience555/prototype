from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prototype.config import EmbeddingPlacement, PrecisionMode, PrototypeConfig


def build_training_plan(
    config: PrototypeConfig,
    contract_report: dict[str, Any],
    data_plan: dict[str, Any],
    sharding_report: dict[str, Any] | None = None,
) -> dict:
    return {
        "schema": "torchrec-v1-training-plan",
        "mode": config.mode.value,
        "backend": config.backend.name.value,
        "steps": [
            _step("load_model_py", "implemented", "Load user model.py and validate V1 contract."),
            _step("build_model", _function_status(contract_report, "build_model"), "Call model.build_model(config)."),
            _step(
                "build_embedding_configs",
                _function_status(contract_report, "build_embedding_configs"),
                "Call model.build_embedding_configs(config) and create EmbeddingBagConfig objects.",
            ),
            _step("build_dataloader", _function_status(contract_report, "build_dataloader"), "Create split DataLoaders."),
            _step(
                "materialize_batch_preview",
                "implemented_fallback_or_runtime",
                "Create dense/label tensors and KeyedJaggedTensor when torch/torchrec are available.",
            ),
            _step("apply_embedding_placement", _placement_status(config), "Apply DEVICE or MANAGED_CACHING placement."),
            _step(
                "materialize_embedding_configs",
                "implemented_fallback_or_runtime",
                "Create or describe EmbeddingBagConfig objects.",
            ),
            _step("apply_precision", _precision_status(config), "Apply dense/embedding/communication precision."),
            _step(
                "plan_sharding",
                _sharding_status(sharding_report),
                "Create TorchRec EmbeddingShardingPlanner and sharding plan.",
            ),
            _step("wrap_dmp", "planned", "Wrap model with DistributedModelParallel."),
            _step("build_optimizer", _function_status(contract_report, "build_optimizer"), "Create fused sparse and dense optimizers."),
            _step("load_checkpoint", _checkpoint_load_status(config), "Load checkpoint for RESUME/EVALUATE."),
            _step("train_pipeline_sparse_dist", "planned", "Run TrainPipelineSparseDist training/evaluation loop."),
            _step("save_checkpoint", _checkpoint_save_status(config), "Save checkpoint and _SUCCESS marker."),
            _step("profile", _profile_status(config), "Record profiler traces within configured step window."),
        ],
        "data_plan_summary": {
            "format": data_plan["format"],
            "global_batch_size": data_plan["global_batch_size"],
            "dense_feature_count": len(data_plan["feature_schema"]["dense_features"]),
            "sparse_feature_count": len(data_plan["feature_schema"]["sparse_features"]),
        },
        "acceptance_boundary": (
            "The current V1 runner executes a minimal metrics/checkpoint loop. Steps marked planned "
            "are explicit remaining work before this becomes a full TorchRec DMP runner."
        ),
    }


def write_training_plan(
    config: PrototypeConfig,
    run_dir: Path,
    contract_report: dict[str, Any],
    data_plan: dict[str, Any],
    sharding_report: dict[str, Any] | None = None,
) -> dict:
    plan = build_training_plan(config, contract_report, data_plan, sharding_report=sharding_report)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-training-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plan


def _step(name: str, status: str, description: str) -> dict[str, str]:
    return {"name": name, "status": status, "description": description}


def _function_status(contract_report: dict[str, Any], function_name: str) -> str:
    available = contract_report["functions"].get(function_name, {}).get("available")
    if function_name in contract_report["required_functions"]:
        return "contract_available" if available else "missing"
    return "optional_available" if available else "optional_missing"


def _placement_status(config: PrototypeConfig) -> str:
    if config.device.embedding_placement == EmbeddingPlacement.DEVICE:
        return "recorded_pending_dmp"
    return "recorded_pending_fbgemm_managed_caching"


def _precision_status(config: PrototypeConfig) -> str:
    values = [
        config.precision.embedding,
        config.precision.dense_compute,
        config.precision.comm_forward,
        config.precision.comm_backward,
    ]
    if all(value == PrecisionMode.FP32 for value in values):
        return "default_fp32_recorded"
    return "non_fp32_recorded_pending_runtime_mapping"


def _checkpoint_load_status(config: PrototypeConfig) -> str:
    return "configured" if config.checkpoint.load_path else "not_requested"


def _checkpoint_save_status(config: PrototypeConfig) -> str:
    return "minimal_loop_implemented" if config.checkpoint.enabled else "disabled"


def _profile_status(config: PrototypeConfig) -> str:
    return "window_metric_implemented_trace_pending" if config.profile.enabled else "disabled"


def _sharding_status(sharding_report: dict[str, Any] | None) -> str:
    if not sharding_report:
        return "planned"
    if sharding_report.get("collective_plan_created"):
        return "collective_plan_created"
    if sharding_report.get("topology_created"):
        return "topology_created_collective_plan_pending"
    if sharding_report.get("imports", {}).get("planner_components_available"):
        return "planner_imported_topology_pending"
    return "planned_import_unavailable"
