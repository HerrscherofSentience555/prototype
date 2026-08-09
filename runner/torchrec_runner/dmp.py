from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prototype.config import PrototypeConfig


def try_wrap_distributed_model_parallel(
    model,
    config: PrototypeConfig,
    distributed_report: dict[str, Any],
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    report = build_dmp_report(config, distributed_report)
    wrapped_model = model
    if report["can_attempt_wrap"]:
        try:
            import torch
            from torchrec.distributed.model_parallel import (
                DistributedModelParallel,
                get_default_sharders,
            )

            device = torch.device(distributed_report.get("device") or "cpu")
            wrapped_model = DistributedModelParallel(
                model,
                device=device,
                sharders=get_default_sharders(),
                init_data_parallel=True,
            )
            report["wrapped"] = True
            report["wrapper_type"] = type(wrapped_model).__name__
            report["model_type_after_wrap"] = type(getattr(wrapped_model, "module", wrapped_model)).__name__
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
    write_dmp_report(run_dir, report)
    return wrapped_model, report


def build_dmp_report(config: PrototypeConfig, distributed_report: dict[str, Any]) -> dict[str, Any]:
    torchrec_available = False
    dmp_available = False
    import_error = None
    default_sharder_count = None
    try:
        from torchrec.distributed.model_parallel import (  # noqa: F401
            DistributedModelParallel,
            get_default_sharders,
        )

        torchrec_available = True
        dmp_available = True
        default_sharder_count = len(get_default_sharders())
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    can_attempt = bool(
        dmp_available
        and distributed_report.get("process_group_initialized")
        and config.nproc_per_node == int(distributed_report.get("world_size", 1))
    )
    return {
        "schema": "torchrec-v1-dmp-wrap",
        "requested_nproc_per_node": config.nproc_per_node,
        "world_size": distributed_report.get("world_size"),
        "rank": distributed_report.get("rank"),
        "device": distributed_report.get("device"),
        "torchrec_available": torchrec_available,
        "distributed_model_parallel_available": dmp_available,
        "default_sharder_count": default_sharder_count,
        "process_group_initialized": distributed_report.get("process_group_initialized"),
        "can_attempt_wrap": can_attempt,
        "wrapped": False,
        "wrapper_type": None,
        "model_type_after_wrap": None,
        "import_error": import_error,
        "error": None,
    }


def write_dmp_report(run_dir: Path, report: dict[str, Any]) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    rank = report.get("rank", 0)
    (artifacts_dir / f"torchrec-dmp-wrap-rank{rank}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if int(rank) == 0:
        (artifacts_dir / "torchrec-dmp-wrap.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_dmp_summary(run_dir: Path, world_size: int) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    ranks = []
    for rank in range(world_size):
        path = artifacts_dir / f"torchrec-dmp-wrap-rank{rank}.json"
        report = None
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report = {"rank": rank, "wrapped": False, "error": f"{type(exc).__name__}: {exc}"}
        ranks.append(
            {
                "rank": rank,
                "reported": report is not None,
                "can_attempt_wrap": report.get("can_attempt_wrap") if report else None,
                "wrapped": report.get("wrapped") if report else None,
                "wrapper_type": report.get("wrapper_type") if report else None,
                "error": report.get("error") if report else "missing rank DMP report",
            }
        )
    summary = {
        "schema": "torchrec-v1-dmp-wrap-summary",
        "world_size": world_size,
        "all_ranks_reported": all(rank["reported"] for rank in ranks),
        "all_ranks_wrapped": all(rank["wrapped"] for rank in ranks),
        "ranks": ranks,
    }
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-dmp-wrap-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
