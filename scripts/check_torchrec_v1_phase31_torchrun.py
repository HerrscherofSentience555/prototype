from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="TorchRec V1 torchrun/DMP smoke")
    parser.add_argument("--run-root", default="prototype/runs")
    parser.add_argument("--model-file", default="prototype/examples/models/torchrec_v1_model.py")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_root) / f"phase31-wsl-torchrun-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    config = PrototypeConfig(
        backend={"name": "torchrec_v1"},
        model={"file": args.model_file},
        nproc_per_node=args.nproc_per_node,
        device={"gpu_ids": list(range(args.nproc_per_node))},
        training={"max_steps": args.max_steps},
        data={"format": "random", "batch_size": 4},
    )
    config_path = run_dir / "resolved-config.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")
    command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "prototype.runner.torchrec_runner.entry",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (run_dir / "torchrun-smoke.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        return completed.returncode

    distributed = json.loads(
        (run_dir / "artifacts" / "torchrec-distributed-environment.json").read_text(encoding="utf-8")
    )
    runtime_batch = json.loads(
        (run_dir / "artifacts" / "torchrec-runtime-batch.json").read_text(encoding="utf-8")
    )
    runtime_batch_summary_path = run_dir / "artifacts" / "torchrec-runtime-batch-summary.json"
    runtime_batch_summary = (
        json.loads(runtime_batch_summary_path.read_text(encoding="utf-8"))
        if runtime_batch_summary_path.exists()
        else None
    )
    status = json.loads(
        (run_dir / "artifacts" / "torchrec-runner-status.json").read_text(encoding="utf-8")
    )
    dmp = json.loads(
        (run_dir / "artifacts" / "torchrec-dmp-wrap.json").read_text(encoding="utf-8")
    )
    dmp_summary_path = run_dir / "artifacts" / "torchrec-dmp-wrap-summary.json"
    dmp_summary = (
        json.loads(dmp_summary_path.read_text(encoding="utf-8"))
        if dmp_summary_path.exists()
        else None
    )
    runtime_step_path = run_dir / "artifacts" / "torchrec-runtime-step.json"
    runtime_step = (
        json.loads(runtime_step_path.read_text(encoding="utf-8"))
        if runtime_step_path.exists()
        else None
    )
    runtime_step_summary_path = run_dir / "artifacts" / "torchrec-runtime-step-summary.json"
    runtime_step_summary = (
        json.loads(runtime_step_summary_path.read_text(encoding="utf-8"))
        if runtime_step_summary_path.exists()
        else None
    )
    rank_dmp_reports = []
    for rank in range(args.nproc_per_node):
        path = run_dir / "artifacts" / f"torchrec-dmp-wrap-rank{rank}.json"
        rank_dmp_reports.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
    result = {
        "run_dir": str(run_dir),
        "rank": distributed["rank"],
        "local_rank": distributed["local_rank"],
        "world_size": distributed["world_size"],
        "torchrun_environment": distributed["torchrun_environment"],
        "process_group_initialized": distributed["process_group_initialized"],
        "process_group_backend": distributed["process_group_backend"],
        "device": distributed["device"],
        "runtime_batch_created": runtime_batch.get("runtime_batch_created"),
        "keyed_jagged_tensor_created": runtime_batch.get("keyed_jagged_tensor_created"),
        "runtime_batch_rank_sharded": runtime_batch.get("rank_sharded"),
        "runtime_batch_selected_indices": runtime_batch.get("selected_indices_preview"),
        "all_ranks_created_runtime_batch": runtime_batch_summary.get(
            "all_ranks_created_runtime_batch"
        )
        if runtime_batch_summary
        else None,
        "dmp_can_attempt_wrap": dmp.get("can_attempt_wrap"),
        "dmp_wrapped": dmp.get("wrapped"),
        "dmp_error": dmp.get("error"),
        "rank_dmp_wrapped": [
            report.get("wrapped") if report else None for report in rank_dmp_reports
        ],
        "all_ranks_reported": dmp_summary.get("all_ranks_reported") if dmp_summary else None,
        "all_ranks_wrapped": dmp_summary.get("all_ranks_wrapped") if dmp_summary else None,
        "runtime_step_executed": runtime_step.get("runtime_step_executed") if runtime_step else None,
        "runtime_step_train_loss": runtime_step.get("train_loss") if runtime_step else None,
        "runtime_step_completed_steps": runtime_step.get("completed_steps") if runtime_step else None,
        "all_ranks_executed_runtime_step": runtime_step_summary.get(
            "all_ranks_executed_runtime_step"
        )
        if runtime_step_summary
        else None,
        "all_ranks_completed_runtime_loop": runtime_step_summary.get(
            "all_ranks_completed_runtime_loop"
        )
        if runtime_step_summary
        else None,
        "min_completed_steps": runtime_step_summary.get("min_completed_steps")
        if runtime_step_summary
        else None,
        "max_requested_steps": runtime_step_summary.get("max_requested_steps")
        if runtime_step_summary
        else None,
        "status": status["status"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["torchrun_environment"]:
        return 2
    if not result["process_group_initialized"]:
        return 3
    if result["world_size"] != args.nproc_per_node:
        return 4
    if not result["runtime_batch_created"]:
        return 5
    if not result["keyed_jagged_tensor_created"]:
        return 6
    if args.nproc_per_node > 1 and not result["all_ranks_created_runtime_batch"]:
        return 7
    if args.nproc_per_node > 1 and not all(result["rank_dmp_wrapped"]):
        return 8
    if args.nproc_per_node > 1 and not result["all_ranks_wrapped"]:
        return 9
    if args.nproc_per_node > 1 and not result["all_ranks_executed_runtime_step"]:
        return 10
    if args.nproc_per_node > 1 and not result["all_ranks_completed_runtime_loop"]:
        return 11
    if args.nproc_per_node > 1 and result["min_completed_steps"] != args.max_steps:
        return 12
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
