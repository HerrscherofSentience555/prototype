from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prototype.config import PrototypeConfig


def write_sharding_planner_readiness(
    config: PrototypeConfig,
    run_dir: Path,
    embedding_report: dict[str, Any] | None,
) -> dict[str, Any]:
    report = build_sharding_planner_readiness(config, embedding_report)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-sharding-plan-readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_sharding_planner_readiness(
    config: PrototypeConfig,
    embedding_report: dict[str, Any] | None,
) -> dict[str, Any]:
    imports = _import_planner_components()
    report = {
        "schema": "torchrec-v1-sharding-plan-readiness",
        "requested_topology": {
            "local_world_size": config.nproc_per_node,
            "world_size": config.nproc_per_node,
            "compute_device": "cuda" if config.device.gpu_ids else "cpu",
            "gpu_ids": config.device.gpu_ids,
            "batch_size": config.data.batch_size,
        },
        "embedding_config_count": embedding_report.get("count") if embedding_report else None,
        "imports": imports,
        "topology_created": False,
        "collective_plan_created": False,
        "ready_for_collective_plan": False,
        "fallback_reasons": [],
    }
    if not imports["planner_components_available"]:
        report["fallback_reasons"].append(imports["error"])
        return report

    try:
        topology_class = imports["_Topology"]
        topology = topology_class(
            local_world_size=config.nproc_per_node,
            world_size=config.nproc_per_node,
            compute_device=report["requested_topology"]["compute_device"],
        )
        report["topology_created"] = True
        report["topology_repr"] = repr(topology)
    except Exception as exc:
        report["fallback_reasons"].append(f"Topology creation failed: {type(exc).__name__}: {exc}")
        return _strip_internal_imports(report)

    report["fallback_reasons"].append(
        "collective_plan requires a real torch.nn.Module, get_default_sharders(), and initialized "
        "torch.distributed process group; this readiness check intentionally does not fake them."
    )
    return _strip_internal_imports(report)


def _import_planner_components() -> dict[str, Any]:
    try:
        from torchrec.distributed.model_parallel import get_default_sharders  # noqa: F401
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation  # noqa: F401

        return {
            "planner_components_available": True,
            "EmbeddingShardingPlanner": True,
            "Topology": True,
            "HeuristicalStorageReservation": True,
            "get_default_sharders": True,
            "_Topology": Topology,
            "_EmbeddingShardingPlanner": EmbeddingShardingPlanner,
        }
    except Exception as exc:
        return {
            "planner_components_available": False,
            "EmbeddingShardingPlanner": False,
            "Topology": False,
            "HeuristicalStorageReservation": False,
            "get_default_sharders": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _strip_internal_imports(report: dict[str, Any]) -> dict[str, Any]:
    imports = report.get("imports", {})
    imports.pop("_Topology", None)
    imports.pop("_EmbeddingShardingPlanner", None)
    return report
