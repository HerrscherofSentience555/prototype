from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from prototype.config import PrototypeConfig, RunMode
from prototype.runner.backends.base import RunnerBackend
from prototype.runner.checkpoints import read_checkpoint, write_checkpoint
from prototype.runner.metrics import append_metric
from prototype.runner.v1_metrics import append_v1_step_metrics


class CustomModelBackend(RunnerBackend):
    name = "custom"

    def run(self, config: PrototypeConfig, run_dir: Path) -> int:
        module = self._load_model_module(config.model.file)
        self._write_contract_report(module, run_dir)
        if config.mode == RunMode.EVALUATE:
            return self._run_evaluation(module, config, run_dir)
        return self._run_training(module, config, run_dir)

    def _run_training(self, module: ModuleType, config: PrototypeConfig, run_dir: Path) -> int:
        if not hasattr(module, "train_step"):
            raise ValueError(
                "custom backend requires model.py to define train_step(step: int, config: dict) -> dict"
            )
        checkpoint_payload = None
        start_step = 0
        if config.mode == RunMode.RESUME and config.checkpoint.load_path:
            checkpoint_payload = read_checkpoint(config.checkpoint.load_path)
            start_step = int(checkpoint_payload.get("step", 0))

        rank0_log = run_dir / "train-rank0.log"
        metrics_path = run_dir / "metrics.jsonl"
        steps = config.training.max_steps or max(1, config.training.epochs)
        latest_metrics: dict[str, Any] = {}
        with rank0_log.open("a", encoding="utf-8") as log:
            log.write("Starting custom model backend run.\n")
            log.write(f"Model file: {config.model.file}\n")
            if checkpoint_payload:
                log.write(f"Loaded checkpoint from: {config.checkpoint.load_path}\n")
            for step in range(start_step + 1, start_step + steps + 1):
                step_started = time.monotonic()
                metrics = module.train_step(step, config.model_dump(mode="json"))
                if not isinstance(metrics, dict):
                    raise ValueError("custom model train_step must return a dict of metric values")
                latest_metrics = metrics
                time.sleep(0.05)
                step_time_seconds = max(time.monotonic() - step_started, 1e-9)
                global_batch_size = config.data.batch_size * config.nproc_per_node
                samples_per_second = global_batch_size / step_time_seconds
                batches_per_second = 1.0 / step_time_seconds
                log.write(
                    "step={step} metrics={metrics} "
                    "step_time_seconds={step_time_seconds:.4f} "
                    "samples_per_second={samples_per_second:.2f}\n".format(
                        step=step,
                        metrics=json.dumps(metrics, ensure_ascii=False),
                        step_time_seconds=step_time_seconds,
                        samples_per_second=samples_per_second,
                    )
                )
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
                    backend=self.name,
                    payload={
                        "backend": self.name,
                        "mode": config.mode.value,
                        "step": start_step + steps,
                        "metrics": latest_metrics,
                        "resumed_from": config.checkpoint.load_path,
                    },
                    optimizer={"learning_rate": config.training.learning_rate}
                    if config.checkpoint.save_optimizer
                    else None,
                    supported=True,
                    keep_last=config.checkpoint.keep_last,
                )
                log.write(f"Checkpoint saved: {checkpoint_dir}\n")
            log.write("Custom model run completed.\n")
        return 0

    def _run_evaluation(self, module: ModuleType, config: PrototypeConfig, run_dir: Path) -> int:
        checkpoint_payload = read_checkpoint(config.checkpoint.load_path) if config.checkpoint.load_path else {}
        if hasattr(module, "evaluate"):
            metrics = module.evaluate(config.model_dump(mode="json"), checkpoint_payload)
        else:
            metrics = checkpoint_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("custom model evaluate must return a dict of metric values")
        evaluation = {
            "backend": self.name,
            "mode": config.mode.value,
            "source_checkpoint": config.checkpoint.load_path,
            "checkpoint_load_supported": True,
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
        with (run_dir / "train-rank0.log").open("a", encoding="utf-8") as log:
            log.write("Custom model evaluation completed.\n")
        return 0

    def _load_model_module(self, model_file: str) -> ModuleType:
        path = self._resolve_model_path(model_file)
        spec = importlib.util.spec_from_file_location("custom_model", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load custom model module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_model_path(self, model_file: str) -> Path:
        path = Path(model_file)
        if path.is_absolute() and path.exists():
            return path
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parents[2] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Custom model file does not exist: {model_file}")

    def _write_contract_report(self, module: ModuleType, run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        report = {
            "backend": self.name,
            "train_step_available": hasattr(module, "train_step"),
            "evaluate_available": hasattr(module, "evaluate"),
            "contract": {
                "train_step": "train_step(step: int, config: dict) -> dict[str, float]",
                "evaluate": "evaluate(config: dict, checkpoint: dict) -> dict[str, float]",
            },
        }
        (artifacts_dir / "custom-model-contract.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
