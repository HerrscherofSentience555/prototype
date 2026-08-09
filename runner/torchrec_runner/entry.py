from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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
from prototype.runner.torchrec_runner.distributed import (
    destroy_distributed_environment,
    initialize_distributed_environment,
)
from prototype.runner.torchrec_runner.dmp import (
    try_wrap_distributed_model_parallel,
    write_dmp_summary,
)
from prototype.runner.torchrec_runner.embedding import materialize_embedding_configs
from prototype.runner.torchrec_runner.materialize import (
    build_runtime_batch,
    write_batch_materialization,
    write_runtime_batch_report,
    write_runtime_batch_summary,
)
from prototype.runner.torchrec_runner.plan import write_training_plan
from prototype.runner.torchrec_runner.runtime import write_runtime_smoke
from prototype.runner.torchrec_runner.runtime_step import (
    write_runtime_step_report,
    write_runtime_step_summary,
)
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
    distributed_report = initialize_distributed_environment(config, run_dir)
    rank = int(distributed_report["rank"])
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
    runtime_batch_report = (
        write_runtime_batch_report(config, run_dir, device=distributed_report["device"])
        if rank == 0 and int(distributed_report["world_size"]) == 1
        else None
    )
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
        log.write(
            "distributed world_size={world_size} local_rank={local_rank} "
            "device={device} process_group_initialized={initialized} backend={backend}\n".format(
                world_size=distributed_report["world_size"],
                local_rank=distributed_report["local_rank"],
                device=distributed_report["device"],
                initialized=distributed_report["process_group_initialized"],
                backend=distributed_report["process_group_backend"],
            )
        )
        log.flush()
        try:
            if int(distributed_report["world_size"]) > 1:
                final_status = _run_multi_rank_dmp_smoke(
                    module,
                    config,
                    run_dir,
                    log,
                    distributed_report,
                )
                if rank == 0:
                    _write_status(
                        run_dir,
                        report,
                        availability,
                        final_status,
                        data_plan=data_plan,
                        batch_materialization=batch_materialization,
                        runtime_batch_report=runtime_batch_report,
                        embedding_report=embedding_report,
                        sharding_report=sharding_report,
                        runtime_smoke=runtime_smoke,
                        training_plan=training_plan,
                    )
            elif rank == 0:
                if config.mode == RunMode.EVALUATE:
                    final_status = _run_evaluation(module, config, run_dir, log, distributed_report)
                else:
                    final_status = _run_training(module, config, run_dir, log, distributed_report)
                _write_status(
                    run_dir,
                    report,
                    availability,
                    final_status,
                    data_plan=data_plan,
                    batch_materialization=batch_materialization,
                    runtime_batch_report=runtime_batch_report,
                    embedding_report=embedding_report,
                    sharding_report=sharding_report,
                    runtime_smoke=runtime_smoke,
                    training_plan=training_plan,
                )
        finally:
            destroy_distributed_environment(distributed_report)
    return 0


def _run_multi_rank_dmp_smoke(
    module,
    config: PrototypeConfig,
    run_dir: Path,
    log,
    distributed_report: dict[str, Any],
) -> str:
    runtime_context = _prepare_single_card_runtime(
        module,
        config,
        log,
        run_dir=run_dir,
        distributed_report=distributed_report,
    )
    runtime_batch_report = write_runtime_batch_report(
        config,
        run_dir,
        device=runtime_context["device"],
        rank=int(distributed_report.get("rank", 0)),
        world_size=int(distributed_report.get("world_size", 1)),
    )
    dmp_report = runtime_context.get("dmp_report") or {}
    runtime_step_report = _run_multi_rank_runtime_loop(
        runtime_context,
        config,
        log,
        rank=int(distributed_report.get("rank", 0)),
        world_size=int(distributed_report.get("world_size", 1)),
    )
    write_runtime_step_report(run_dir, runtime_step_report)
    log.write(
        (
            "multi_rank_dmp_smoke rank={rank} world_size={world_size} "
            "runtime_batch_rows={rows} selected_indices={indices} "
            "runtime_step_executed={step_executed} dmp_wrapped={wrapped} error={error}\n"
        ).format(
            rank=distributed_report.get("rank"),
            world_size=distributed_report.get("world_size"),
            rows=runtime_batch_report.get("rows"),
            indices=runtime_batch_report.get("selected_indices_preview"),
            step_executed=runtime_step_report.get("runtime_step_executed"),
            wrapped=dmp_report.get("wrapped"),
            error=runtime_step_report.get("error") or dmp_report.get("error"),
        )
    )
    log.flush()
    _distributed_barrier(distributed_report, log)
    dmp_summary = None
    runtime_batch_summary = None
    runtime_step_summary = None
    if int(distributed_report.get("rank", 0)) == 0:
        runtime_batch_summary = write_runtime_batch_summary(
            run_dir,
            int(distributed_report.get("world_size", 1)),
        )
        runtime_step_summary = write_runtime_step_summary(
            run_dir,
            int(distributed_report.get("world_size", 1)),
        )
        dmp_summary = write_dmp_summary(run_dir, int(distributed_report.get("world_size", 1)))
    if runtime_batch_summary and not runtime_batch_summary.get("all_ranks_created_runtime_batch"):
        return "SUCCEEDED_MULTI_RANK_DMP_FALLBACK"
    if runtime_step_summary and runtime_step_summary.get("all_ranks_completed_runtime_loop"):
        return "SUCCEEDED_MULTI_RANK_RUNTIME_LOOP_SMOKE"
    if runtime_step_summary and runtime_step_summary.get("all_ranks_executed_runtime_step"):
        return "SUCCEEDED_MULTI_RANK_RUNTIME_STEP_SMOKE"
    if runtime_step_summary:
        return "SUCCEEDED_MULTI_RANK_DMP_FALLBACK"
    if dmp_summary:
        if dmp_summary.get("all_ranks_wrapped"):
            return "SUCCEEDED_MULTI_RANK_DMP_SMOKE"
        return "SUCCEEDED_MULTI_RANK_DMP_FALLBACK"
    if dmp_report.get("wrapped"):
        return "SUCCEEDED_MULTI_RANK_DMP_SMOKE"
    return "SUCCEEDED_MULTI_RANK_DMP_FALLBACK"


def _run_training(
    module,
    config: PrototypeConfig,
    run_dir: Path,
    log,
    distributed_report: dict[str, Any],
) -> str:
    checkpoint_payload = None
    start_step = 0
    if config.mode == RunMode.RESUME and config.checkpoint.load_path:
        checkpoint_payload = read_checkpoint(config.checkpoint.load_path)
        start_step = int(checkpoint_payload.get("step", 0))
        log.write(f"Loaded checkpoint from: {config.checkpoint.load_path}\n")

    metrics_path = run_dir / "metrics.jsonl"
    steps = config.training.max_steps or max(1, config.training.epochs)
    latest_metrics = {}
    runtime_context = _prepare_single_card_runtime(
        module,
        config,
        log,
        run_dir=run_dir,
        distributed_report=distributed_report,
        checkpoint_path=config.checkpoint.load_path if config.mode == RunMode.RESUME else None,
    )
    for step in range(start_step + 1, start_step + steps + 1):
        step_started = time.monotonic()
        runtime_metrics = _run_single_card_runtime_step(runtime_context, config, step, log)
        if runtime_metrics is not None:
            metrics = runtime_metrics
        elif hasattr(module, "train_step"):
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
                "single_card_runtime": runtime_context["status"],
                "dmp_wrapped": (runtime_context.get("dmp_report") or {}).get("wrapped"),
                "runtime_checkpoint": _runtime_checkpoint_payload(runtime_context),
                "checkpoint_payload": checkpoint_payload,
            },
            optimizer={"learning_rate": config.training.learning_rate}
            if config.checkpoint.save_optimizer
            else None,
            keep_last=config.checkpoint.keep_last,
            materialize_extra_files=_materialize_single_card_runtime_checkpoint(runtime_context),
        )
        log.write(f"Checkpoint saved: {checkpoint_dir}\n")
    if runtime_context["status"] == "single_card_runtime":
        return "SUCCEEDED_SINGLE_CARD_RUNTIME"
    return "SUCCEEDED_MINIMAL_LOOP"


def _run_multi_rank_runtime_loop(
    context: dict,
    config: PrototypeConfig,
    log,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    report = {
        "schema": "torchrec-v1-runtime-step",
        "rank": rank,
        "world_size": world_size,
        "runtime_step_executed": False,
        "runtime_loop_completed": False,
        "requested_steps": config.training.max_steps or max(1, config.training.epochs),
        "completed_steps": 0,
        "dmp_wrapped": (context.get("dmp_report") or {}).get("wrapped"),
        "runtime_kjt_batch": None,
        "train_loss": None,
        "accuracy": None,
        "steps": [],
        "error": None,
    }
    if context["status"] != "single_card_runtime":
        report["error"] = context.get("reason") or "runtime context is not available"
        return report
    requested_steps = int(report["requested_steps"])
    for step in range(1, requested_steps + 1):
        step_started = time.monotonic()
        try:
            metrics = _run_single_card_runtime_step(context, config, step=step, log=log)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            return report
        if metrics is None:
            report["error"] = context.get("reason") or "runtime step returned no metrics"
            return report
        step_report = {
            "step": step,
            "train_loss": metrics.get("train_loss"),
            "accuracy": metrics.get("accuracy"),
            "runtime_kjt_batch": metrics.get("runtime_kjt_batch"),
            "step_time_seconds": max(time.monotonic() - step_started, 1e-9),
        }
        report["steps"].append(step_report)
        report["completed_steps"] = step
        report["runtime_step_executed"] = True
        report["runtime_kjt_batch"] = metrics.get("runtime_kjt_batch")
        report["train_loss"] = metrics.get("train_loss")
        report["accuracy"] = metrics.get("accuracy")
        log.write(
            "multi_rank_runtime_loop rank={rank} step={step} metrics={metrics}\n".format(
                rank=rank,
                step=step,
                metrics=json.dumps(metrics, ensure_ascii=False),
            )
        )
        log.flush()
    report.update(
        {
            "runtime_step_executed": True,
            "runtime_loop_completed": report["completed_steps"] == requested_steps,
        }
    )
    return report


def _run_evaluation(
    module,
    config: PrototypeConfig,
    run_dir: Path,
    log,
    distributed_report: dict[str, Any],
) -> str:
    checkpoint_payload = read_checkpoint(config.checkpoint.load_path) if config.checkpoint.load_path else {}
    runtime_context = _prepare_single_card_runtime(
        module,
        config,
        log,
        run_dir=run_dir,
        distributed_report=distributed_report,
        checkpoint_path=config.checkpoint.load_path,
    )
    runtime_metrics = _run_single_card_runtime_evaluation(runtime_context, config, log)
    if runtime_metrics is not None:
        metrics = runtime_metrics
    elif hasattr(module, "evaluate"):
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
        "minimal_loop": runtime_context["status"] != "single_card_runtime",
        "single_card_runtime": runtime_context["status"],
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
    if runtime_context["status"] == "single_card_runtime":
        return "SUCCEEDED_SINGLE_CARD_RUNTIME_EVALUATION"
    return "SUCCEEDED_MINIMAL_LOOP"


def _write_status(
    run_dir: Path,
    report: dict,
    availability: dict,
    status: str,
    data_plan: dict | None = None,
    batch_materialization: dict | None = None,
    runtime_batch_report: dict | None = None,
    embedding_report: dict | None = None,
    sharding_report: dict | None = None,
    runtime_smoke: dict | None = None,
    training_plan: dict | None = None,
) -> None:
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    if runtime_batch_report is None:
        runtime_batch_report = _read_optional_json(artifacts_dir / "torchrec-runtime-batch.json")
    dmp_report = _read_optional_json(artifacts_dir / "torchrec-dmp-wrap.json")
    dmp_summary = _read_optional_json(artifacts_dir / "torchrec-dmp-wrap-summary.json")
    runtime_batch_summary = _read_optional_json(artifacts_dir / "torchrec-runtime-batch-summary.json")
    runtime_step_report = _read_optional_json(artifacts_dir / "torchrec-runtime-step.json")
    runtime_step_summary = _read_optional_json(artifacts_dir / "torchrec-runtime-step-summary.json")
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
                "runtime_batch_summary": {
                    "runtime_batch_created": runtime_batch_report.get("runtime_batch_created")
                    if runtime_batch_report
                    else None,
                    "keyed_jagged_tensor_created": runtime_batch_report.get(
                        "keyed_jagged_tensor_created"
                    )
                    if runtime_batch_report
                    else None,
                    "device": runtime_batch_report.get("device") if runtime_batch_report else None,
                    "error": runtime_batch_report.get("error") if runtime_batch_report else None,
                    "all_ranks_reported": runtime_batch_summary.get("all_ranks_reported")
                    if runtime_batch_summary
                    else None,
                    "all_ranks_created_runtime_batch": runtime_batch_summary.get(
                        "all_ranks_created_runtime_batch"
                    )
                    if runtime_batch_summary
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
                "dmp_wrap_summary": {
                    "can_attempt_wrap": dmp_report.get("can_attempt_wrap") if dmp_report else None,
                    "wrapped": dmp_report.get("wrapped") if dmp_report else None,
                    "wrapper_type": dmp_report.get("wrapper_type") if dmp_report else None,
                    "error": dmp_report.get("error") if dmp_report else None,
                    "all_ranks_reported": dmp_summary.get("all_ranks_reported")
                    if dmp_summary
                    else None,
                    "all_ranks_wrapped": dmp_summary.get("all_ranks_wrapped")
                    if dmp_summary
                    else None,
                },
                "runtime_step_summary": {
                    "runtime_step_executed": runtime_step_report.get("runtime_step_executed")
                    if runtime_step_report
                    else None,
                    "train_loss": runtime_step_report.get("train_loss")
                    if runtime_step_report
                    else None,
                    "accuracy": runtime_step_report.get("accuracy")
                    if runtime_step_report
                    else None,
                    "all_ranks_reported": runtime_step_summary.get("all_ranks_reported")
                    if runtime_step_summary
                    else None,
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
                    "error": runtime_step_report.get("error") if runtime_step_report else None,
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
                "minimal_training_loop": status == "SUCCEEDED_MINIMAL_LOOP",
                "single_card_random_training": status.startswith("SUCCEEDED_SINGLE_CARD_RUNTIME"),
                "multi_rank_dmp_smoke": status == "SUCCEEDED_MULTI_RANK_DMP_SMOKE",
                "multi_rank_runtime_step_smoke": status
                == "SUCCEEDED_MULTI_RANK_RUNTIME_STEP_SMOKE",
                "multi_rank_runtime_loop_smoke": status
                == "SUCCEEDED_MULTI_RANK_RUNTIME_LOOP_SMOKE",
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


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def _prepare_single_card_runtime(
    module,
    config: PrototypeConfig,
    log,
    run_dir: Path,
    distributed_report: dict[str, Any],
    checkpoint_path: str | None = None,
) -> dict:
    context = {
        "status": "fallback_minimal_loop",
        "reason": None,
        "torch": None,
        "model": None,
        "optimizer": None,
        "loss_fn": None,
        "loaded_runtime_checkpoint": False,
        "device": "cpu",
        "rank": int(distributed_report.get("rank", 0)),
        "world_size": int(distributed_report.get("world_size", 1)),
        "dmp_report": None,
    }
    if config.nproc_per_node != 1 and not distributed_report.get("process_group_initialized"):
        context["reason"] = "multi-process runtime requires an initialized torch.distributed process group."
        return context
    if config.data.format not in {"random", "criteo_binary"}:
        context["reason"] = f"single-card runtime does not support data.format={config.data.format} yet."
        return context
    try:
        import torch
    except Exception as exc:
        context["reason"] = f"torch unavailable: {type(exc).__name__}: {exc}"
        return context
    model = module.build_model(config.model_dump(mode="json"))
    try:
        from torch import nn
    except Exception as exc:
        context["reason"] = f"torch.nn unavailable: {type(exc).__name__}: {exc}"
        return context
    if not isinstance(model, nn.Module):
        context["reason"] = "model.build_model(config) did not return torch.nn.Module."
        return context
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        context["reason"] = "model has no trainable parameters."
        return context
    model, dmp_report = try_wrap_distributed_model_parallel(model, config, distributed_report, run_dir)
    optimizer = None
    if hasattr(module, "build_optimizer"):
        optimizer = module.build_optimizer(model, config.model_dump(mode="json"))
    if optimizer is None:
        optimizer = torch.optim.SGD(parameters, lr=config.training.learning_rate)
    context.update(
        {
            "status": "single_card_runtime",
            "torch": torch,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": torch.nn.BCEWithLogitsLoss(),
            "device": distributed_report.get("device") or "cpu",
            "dmp_report": dmp_report,
        }
    )
    if checkpoint_path:
        _load_single_card_runtime_checkpoint(context, checkpoint_path, log)
    log.write("Single-card PyTorch runtime path enabled for torchrec_v1.\n")
    return context


def _distributed_barrier(distributed_report: dict[str, Any], log) -> None:
    if not distributed_report.get("process_group_initialized"):
        return
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception as exc:
        log.write(f"distributed barrier failed: {type(exc).__name__}: {exc}\n")


def _run_single_card_runtime_step(
    context: dict,
    config: PrototypeConfig,
    step: int,
    log,
) -> dict[str, float] | None:
    if context["status"] != "single_card_runtime":
        if context.get("reason") and step == 1:
            log.write(f"Single-card runtime fallback reason: {context['reason']}\n")
        return None
    torch = context["torch"]
    model = context["model"]
    optimizer = context["optimizer"]
    loss_fn = context["loss_fn"]
    batch = build_runtime_batch(
        config,
        "train",
        device=context["device"],
        rank=int(context.get("rank", 0)),
        world_size=int(context.get("world_size", 1)),
    )
    model.train()
    optimizer.zero_grad()
    logits = _call_model_with_runtime_batch(model, batch)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.float().view_as(batch.labels)
    loss = loss_fn(logits, batch.labels)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == batch.labels).float().mean().item()
    return {
        "train_loss": float(loss.detach().item()),
        "accuracy": float(accuracy),
        "single_card_runtime_step": 1.0,
        "runtime_kjt_batch": 1.0 if batch.sparse_features is not None else 0.0,
    }


def _run_single_card_runtime_evaluation(
    context: dict,
    config: PrototypeConfig,
    log,
) -> dict[str, float] | None:
    if context["status"] != "single_card_runtime":
        if context.get("reason"):
            log.write(f"Single-card evaluation fallback reason: {context['reason']}\n")
        return None
    torch = context["torch"]
    model = context["model"]
    loss_fn = context["loss_fn"]
    batch = build_runtime_batch(
        config,
        "validation",
        device=context["device"],
        rank=int(context.get("rank", 0)),
        world_size=int(context.get("world_size", 1)),
    )
    model.eval()
    with torch.no_grad():
        logits = _call_model_with_runtime_batch(model, batch)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float().view_as(batch.labels)
        loss = loss_fn(logits, batch.labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == batch.labels).float().mean().item()
    return {
        "eval_log_loss": float(loss.detach().item()),
        "eval_accuracy": float(accuracy),
        "single_card_runtime_evaluation": 1.0,
        "runtime_kjt_batch": 1.0 if batch.sparse_features is not None else 0.0,
    }


def _call_model_with_runtime_batch(model, batch):
    if batch.sparse_features is not None:
        try:
            return model(batch.dense_features, batch.sparse_features)
        except (TypeError, AttributeError, RuntimeError, ValueError):
            pass
    try:
        return model(batch.dense_features, batch.sparse_tensor)
    except TypeError:
        return model(batch.dense_features)


def _materialize_single_card_runtime_checkpoint(context: dict):
    if context["status"] != "single_card_runtime":
        return None

    def materialize(checkpoint_dir: Path) -> dict[str, Any]:
        torch = context["torch"]
        model_path = checkpoint_dir / "model.pt"
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(context["model"].state_dict(), model_path)
        torch.save(context["optimizer"].state_dict(), optimizer_path)
        return {
            "model_state_dict": str(model_path),
            "optimizer_state_dict": str(optimizer_path),
        }

    return materialize


def _runtime_checkpoint_payload(context: dict) -> dict[str, Any] | None:
    if context["status"] != "single_card_runtime":
        return None
    return {
        "format": "torch_state_dict",
        "model_file": "model.pt",
        "optimizer_file": "optimizer.pt",
        "loaded_runtime_checkpoint": context.get("loaded_runtime_checkpoint", False),
        "dmp_wrapped": (context.get("dmp_report") or {}).get("wrapped", False),
    }


def _load_single_card_runtime_checkpoint(context: dict, checkpoint_path: str, log) -> None:
    checkpoint_dir = Path(checkpoint_path)
    model_path = checkpoint_dir / "model.pt"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if not model_path.exists():
        log.write(f"Single-card runtime checkpoint weights not found: {model_path}\n")
        return
    torch = context["torch"]
    model_state = torch.load(model_path, map_location="cpu")
    context["model"].load_state_dict(model_state)
    if optimizer_path.exists():
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        context["optimizer"].load_state_dict(optimizer_state)
    context["loaded_runtime_checkpoint"] = True
    log.write(f"Loaded single-card runtime weights from: {model_path}\n")


if __name__ == "__main__":
    raise SystemExit(main())
