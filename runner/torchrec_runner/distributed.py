from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from prototype.config import PrototypeConfig


def initialize_distributed_environment(config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    report = build_distributed_environment_report(config)
    if report["torch_available"] and report["torchrun_environment"]:
        _try_initialize_process_group(report)
    _try_set_device(report, config)
    write_distributed_environment_report(run_dir, report)
    return report


def build_distributed_environment_report(config: PrototypeConfig) -> dict[str, Any]:
    env = {
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
    }
    rank = _optional_int(env["RANK"], 0)
    local_rank = _optional_int(env["LOCAL_RANK"], rank)
    world_size = _optional_int(env["WORLD_SIZE"], 1)
    torch_available = False
    distributed_available = False
    distributed_initialized_before = False
    cuda_available = False
    torch_error = None
    try:
        import torch

        torch_available = True
        distributed_available = bool(getattr(torch, "distributed", None) and torch.distributed.is_available())
        distributed_initialized_before = bool(
            distributed_available and torch.distributed.is_initialized()
        )
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        torch_error = f"{type(exc).__name__}: {exc}"
    return {
        "schema": "torchrec-v1-distributed-environment",
        "requested_nproc_per_node": config.nproc_per_node,
        "requested_gpu_ids": config.device.gpu_ids,
        "env": env,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "torchrun_environment": bool(env["RANK"] is not None and env["WORLD_SIZE"] is not None),
        "torch_available": torch_available,
        "torch_error": torch_error,
        "torch_distributed_available": distributed_available,
        "process_group_initialized_before": distributed_initialized_before,
        "process_group_initialized": distributed_initialized_before,
        "process_group_backend": None,
        "process_group_error": None,
        "cuda_available": cuda_available,
        "device": "cpu",
        "device_error": None,
        "ready_for_dmp": False,
    }


def write_distributed_environment_report(run_dir: Path, report: dict[str, Any]) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    rank = report.get("rank", 0)
    (artifacts_dir / f"torchrec-distributed-environment-rank{rank}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if int(rank) == 0:
        (artifacts_dir / "torchrec-distributed-environment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def destroy_distributed_environment(report: dict[str, Any]) -> None:
    if not report.get("process_group_initialized"):
        return
    if report.get("process_group_initialized_before"):
        return
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        return


def _try_initialize_process_group(report: dict[str, Any]) -> None:
    if not report["torch_distributed_available"]:
        report["process_group_error"] = "torch.distributed is not available"
        return
    if report["process_group_initialized_before"]:
        report["ready_for_dmp"] = True
        return
    try:
        import torch

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        report["process_group_initialized"] = True
        report["process_group_backend"] = backend
        report["ready_for_dmp"] = True
    except Exception as exc:
        report["process_group_error"] = f"{type(exc).__name__}: {exc}"


def _try_set_device(report: dict[str, Any], config: PrototypeConfig) -> None:
    if not report["torch_available"]:
        return
    try:
        import torch

        if torch.cuda.is_available() and config.device.gpu_ids:
            visible_index = report["local_rank"]
            if visible_index >= torch.cuda.device_count():
                visible_index = 0
            torch.cuda.set_device(visible_index)
            report["device"] = f"cuda:{visible_index}"
        else:
            report["device"] = "cpu"
    except Exception as exc:
        report["device_error"] = f"{type(exc).__name__}: {exc}"


def _optional_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
