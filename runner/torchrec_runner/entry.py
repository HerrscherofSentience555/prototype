from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from prototype.config import PrototypeConfig, RunMode
from prototype.runner.checkpoints import read_checkpoint, write_checkpoint
from prototype.runner.metrics import append_metric
from prototype.runner.v1_metrics import append_v1_step_metrics
from prototype.runner.torchrec_runner.contract import (
    load_model_module,
    validate_model_contract,
    write_contract_report,
)
from prototype.runner.torchrec_runner.data import write_data_plan
from prototype.runner.torchrec_runner.embedding import materialize_embedding_configs
from prototype.runner.torchrec_runner.materialize import write_batch_materialization
from prototype.runner.torchrec_runner.plan import write_training_plan
from prototype.runner.torchrec_runner.runtime import write_runtime_smoke
from prototype.runner.torchrec_runner.sharding import write_sharding_planner_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Internal TorchRec V1 runner scaffold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = PrototypeConfig.from_yaml_file(Path(args.config))
    run_dir = Path(args.run_dir)
    return run(config, run_dir)


def run(config: PrototypeConfig, run_dir: Path) -> int:
    rank = int(os.environ.get("RANK", "0"))
    log_path = run_dir / "train-rank0.log"
    if rank != 0:
        log_path = run_dir / f"train-rank{rank}.log"
    module = load_model_module(config.model.file, project_root=Path(__file__).resolve().parents[3])
    report = validate_model_contract(module)
    if rank == 0:
        write_contract_report(module, run_dir)

    availability = _runtime_availability()
    data_plan = write_data_plan(config, run_dir) if rank == 0 else None
    batch_materialization = write_batch_materialization(config, run_dir) if rank == 0 else None
    embedding_report = materialize_embedding_configs(module, config, run_dir) if rank == 0 else None
    sharding_report = (
        write_sharding_planner_readiness(config, run_dir, embedding_report) if rank == 0 else None
    )
    runtime_smoke = (
        write_runtime_smoke(
            run_dir,
            availability=availability,
            batch_materialization=batch_materialization,
            embedding_report=embedding_report,
        )
        if rank == 0
        else None
    )
    training_plan = (
        write_training_plan(config, run_dir, report, data_plan, sharding_report=sharding_report)
        if rank == 0 and data_plan
        else None
    )
    if rank == 0:
        _write_status(
            run_dir,
            report,
            availability,
            "RUNNING_MINIMAL_LOOP",
            data_plan=data_plan,
            batch_materialization=batch_materialization,
            embedding_report=embedding_report,
            sharding_report=sharding_report,
            runtime_smoke=runtime_smoke,
            training_plan=training_plan,
        )
    with log_path.open("a", encoding="utf-8") as log:
        log.write("TorchRec V1 runner scaffold validated model.py contract.\n")
        log.write(f"torch_available={availability['torch_available']}\n")
        log.write(f"torchrec_available={availability['torchrec_available']}\n")
        log.write(f"rank={rank}\n")
        log.flush()
        if rank == 0:
            if config.mode == RunMode.EVALUATE:
                _run_evaluation(module, config, run_dir, log)
            else:
                _run_training(module, config, run_dir, log)
            _write_status(
                run_dir,
                report,
                availability,
                "SUCCEEDED_MINIMAL_LOOP",
                data_plan=data_plan,
                batch_materialization=batch_materialization,
                embedding_report=embedding_report,
                sharding_report=sharding_report,
                runtime_smoke=runtime_smoke,
                training_plan=training_plan,
            )
    return 0


def _run_training(module, config: PrototypeConfig, run_dir: Path, log) -> None:
    checkpoint_payload = None
    start_step = 0
    if config.mode == RunMode.RESUME and config.checkpoint.load_path:
        checkpoint_payload = read_checkpoint(config.checkpoint.load_path)
        start_step = int(checkpoint_payload.get("step", 0))
        log.write(f"Loaded checkpoint from: {config.checkpoint.load_path}\n")

    metrics_path = run_dir / "metrics.jsonl"
    steps = config.training.max_steps or max(1, config.training.epochs)
    latest_metrics = {}
    for step in range(start_step + 1, start_step + steps + 1):
        step_started = time.monotonic()
        if hasattr(module, "train_step"):
            metrics = module.train_step(step, config.model_dump(mode="json"))
            if not isinstance(metrics, dict):
                raise ValueError("TorchRec V1 model train_step must return a metric dict when provided")
        else:
            metrics = {
                "train_loss": max(0.05, 1.0 / step),
                "auc": min(0.5 + step * 0.02, 0.95),
            }
        latest_metrics = metrics
        step_time_seconds = max(time.monotonic() - step_started, 1e-9)
        global_batch_size = config.data.batch_size * config.nproc_per_node
        samples_per_second = global_batch_size / step_time_seconds
        batches_per_second = 1.0 / step_time_seconds
        log.write(
            "step={step} metrics={metrics} step_time_seconds={step_time_seconds:.6f} "
            "samples_per_second={samples_per_second:.2f} batches_per_second={batches_per_second:.2f}\n".format(
                step=step,
                metrics=json.dumps(metrics, ensure_ascii=False),
                step_time_seconds=step_time_seconds,
                samples_per_second=samples_per_second,
                batches_per_second=batches_per_second,
            )
        )
        log.flush()
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                append_metric(metrics_path, step=step, metric=str(metric), value=float(value))
        append_v1_step_metrics(
            metrics_path,
            step=step,
            step_time_seconds=step_time_seconds,
            global_batch_size=global_batch_size,
            profile=config.profile,
        )

    if config.checkpoint.enabled:
        checkpoint_dir = write_checkpoint(
            run_dir,
            step=start_step + steps,
            backend="torchrec_v1",
            payload={
                "backend": "torchrec_v1",
                "mode": config.mode.value,
                "step": start_step + steps,
                "metrics": latest_metrics,
                "resumed_from": config.checkpoint.load_path,
                "minimal_loop": True,
                "checkpoint_payload": checkpoint_payload,
            },
            optimizer={"learning_rate": config.training.learning_rate}
            if config.checkpoint.save_optimizer
            else None,
            keep_last=config.checkpoint.keep_last,
        )
        log.write(f"Checkpoint saved: {checkpoint_dir}\n")


def _run_evaluation(module, config: PrototypeConfig, run_dir: Path, log) -> None:
    checkpoint_payload = read_checkpoint(config.checkpoint.load_path) if config.checkpoint.load_path else {}
    if hasattr(module, "evaluate"):
        metrics = module.evaluate(config.model_dump(mode="json"), checkpoint_payload)
        if not isinstance(metrics, dict):
            raise ValueError("TorchRec V1 model evaluate must return a metric dict when provided")
    else:
        metrics = checkpoint_payload.get("metrics", {"auc": 0.5, "log_loss": 0.6931})
    evaluation = {
        "backend": "torchrec_v1",
        "mode": config.mode.value,
        "source_checkpoint": config.checkpoint.load_path,
        "checkpoint_load_supported": True,
        "minimal_loop": True,
        "metrics": metrics,
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            append_metric(metrics_path, step=None, metric=f"eval_{metric}", value=float(value))
    log.write("Evaluation completed.\n")


def _write_status(
    run_dir: Path,
    report: dict,
    availability: dict,
    status: str,
    data_plan: dict | None = None,
    batch_materialization: dict | None = None,
    embedding_report: dict | None = None,
    sharding_report: dict | None = None,
    runtime_smoke: dict | None = None,
    training_plan: dict | None = None,
) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-runner-status.json").write_text(
        json.dumps(
            {
                "runner": "prototype.runner.torchrec_runner.entry",
                "status": status,
                "contract": report,
                "runtime": availability,
                "data_plan_summary": {
                    "format": data_plan.get("format") if data_plan else None,
                    "global_batch_size": data_plan.get("global_batch_size") if data_plan else None,
                    "dense_feature_count": len(data_plan["feature_schema"]["dense_features"])
                    if data_plan
                    else None,
                    "sparse_feature_count": len(data_plan["feature_schema"]["sparse_features"])
                    if data_plan
                    else None,
                },
                "batch_materialization_summary": {
                    "rows": batch_materialization.get("rows") if batch_materialization else None,
                    "torch_dense_tensor_created": batch_materialization.get("torch", {}).get(
                        "dense_tensor_created"
                    )
                    if batch_materialization
                    else None,
                    "torchrec_kjt_created": batch_materialization.get("torchrec", {}).get(
                        "keyed_jagged_tensor_created"
                    )
                    if batch_materialization
                    else None,
                },
                "embedding_config_summary": {
                    "count": embedding_report.get("count") if embedding_report else None,
                    "source": embedding_report.get("source") if embedding_report else None,
                    "torchrec_embedding_config_available": embedding_report.get("runtime", {}).get(
                        "torchrec_embedding_config_available"
                    )
                    if embedding_report
                    else None,
                },
                "runtime_smoke_summary": {
                    "ready_for_dmp_smoke": runtime_smoke.get("ready_for_dmp_smoke")
                    if runtime_smoke
                    else None,
                    "fallback_reasons": runtime_smoke.get("fallback_reasons") if runtime_smoke else [],
                },
                "sharding_readiness_summary": {
                    "planner_components_available": sharding_report.get("imports", {}).get(
                        "planner_components_available"
                    )
                    if sharding_report
                    else None,
                    "topology_created": sharding_report.get("topology_created") if sharding_report else None,
                    "collective_plan_created": sharding_report.get("collective_plan_created")
                    if sharding_report
                    else None,
                    "fallback_reasons": sharding_report.get("fallback_reasons") if sharding_report else [],
                },
                "training_plan_summary": {
                    "step_count": len(training_plan["steps"]) if training_plan else None,
                    "planned_steps": [
                        step["name"]
                        for step in training_plan["steps"]
                        if step["status"] == "planned"
                    ]
                    if training_plan
                    else [],
                },
                "minimal_training_loop": True,
                "next_step": (
                    "Replace the minimal loop with distributed model construction, DataLoader "
                    "creation, TrainPipelineSparseDist, and Torch Distributed Checkpoint."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _runtime_availability() -> dict:
    result = {
        "torch_available": False,
        "torchrec_available": False,
        "torch_error": None,
        "torchrec_error": None,
    }
    try:
        import torch  # noqa: F401

        result["torch_available"] = True
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torchrec  # noqa: F401

        result["torchrec_available"] = True
    except Exception as exc:
        result["torchrec_error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
