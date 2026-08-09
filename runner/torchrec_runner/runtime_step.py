from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_runtime_step_report(run_dir: Path, report: dict[str, Any]) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    rank = int(report.get("rank", 0))
    (artifacts_dir / f"torchrec-runtime-step-rank{rank}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rank == 0:
        (artifacts_dir / "torchrec-runtime-step.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_runtime_step_summary(run_dir: Path, world_size: int) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    ranks = []
    for rank in range(world_size):
        path = artifacts_dir / f"torchrec-runtime-step-rank{rank}.json"
        report = None
        if path.exists():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                report = {
                    "rank": rank,
                    "runtime_step_executed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        ranks.append(
            {
                "rank": rank,
                "reported": report is not None,
                "runtime_step_executed": report.get("runtime_step_executed") if report else None,
                "runtime_loop_completed": report.get("runtime_loop_completed") if report else None,
                "requested_steps": report.get("requested_steps") if report else None,
                "completed_steps": report.get("completed_steps") if report else None,
                "train_loss": report.get("train_loss") if report else None,
                "accuracy": report.get("accuracy") if report else None,
                "runtime_kjt_batch": report.get("runtime_kjt_batch") if report else None,
                "dmp_wrapped": report.get("dmp_wrapped") if report else None,
                "error": report.get("error") if report else "missing rank runtime step report",
            }
        )
    summary = {
        "schema": "torchrec-v1-runtime-step-summary",
        "world_size": world_size,
        "all_ranks_reported": all(rank["reported"] for rank in ranks),
        "all_ranks_executed_runtime_step": all(rank["runtime_step_executed"] for rank in ranks),
        "all_ranks_completed_runtime_loop": all(rank["runtime_loop_completed"] for rank in ranks),
        "all_ranks_used_dmp": all(rank["dmp_wrapped"] for rank in ranks),
        "min_completed_steps": min(
            (rank["completed_steps"] or 0 for rank in ranks),
            default=0,
        ),
        "max_requested_steps": max(
            (rank["requested_steps"] or 0 for rank in ranks),
            default=0,
        ),
        "ranks": ranks,
    }
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-runtime-step-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
