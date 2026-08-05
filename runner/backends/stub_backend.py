from __future__ import annotations

import json
import time
from pathlib import Path

from prototype.config import PrototypeConfig, RunMode
from prototype.runner.backends.base import RunnerBackend
from prototype.runner.checkpoints import read_checkpoint, write_checkpoint
from prototype.runner.metrics import append_metric
from prototype.runner.v1_metrics import append_v1_step_metrics


class StubBackend(RunnerBackend):
    name = "stub"

    def run(self, config: PrototypeConfig, run_dir: Path) -> int:
        if config.mode == RunMode.EVALUATE:
            return self.run_evaluation(config, run_dir)
        return self.run_training(config, run_dir)

    def run_training(self, config: PrototypeConfig, run_dir: Path) -> int:
        rank0_log = run_dir / "train-rank0.log"
        metrics_path = run_dir / "metrics.jsonl"

        with rank0_log.open("a", encoding="utf-8") as log:
            log.write("Starting prototype training run.\n")
            log.write(f"Backend: {self.name}\n")
            log.write(f"Mode: {config.mode}\n")
            log.write(f"Model file: {config.model.file}\n")
            log.write(f"Data format: {config.data.format}\n")
            log.write(f"nproc_per_node: {config.nproc_per_node}\n")

            loaded_checkpoint = None
            start_step = 0
            if config.mode == RunMode.RESUME and config.checkpoint.load_path:
                loaded_checkpoint = read_checkpoint(config.checkpoint.load_path)
                start_step = int(loaded_checkpoint.get("step", 0))
                log.write(f"Loaded checkpoint from: {config.checkpoint.load_path}\n")
                log.write(f"Resuming from step: {start_step}\n")

            steps = config.training.max_steps or max(5, config.training.epochs * 5)
            for step in range(start_step + 1, start_step + steps + 1):
                step_started = time.monotonic()
                loss = max(0.05, 1.0 / step)
                auc = min(0.5 + step * 0.03, 0.9)
                time.sleep(0.4)
                step_time_seconds = max(time.monotonic() - step_started, 1e-9)
                global_batch_size = config.data.batch_size * config.nproc_per_node
                samples_per_second = global_batch_size / step_time_seconds
                batches_per_second = 1.0 / step_time_seconds
                log.write(
                    "step={step} loss={loss:.4f} auc={auc:.4f} "
                    "step_time_seconds={step_time_seconds:.4f} "
                    "samples_per_second={samples_per_second:.2f}\n".format(
                        step=step,
                        loss=loss,
                        auc=auc,
                        step_time_seconds=step_time_seconds,
                        samples_per_second=samples_per_second,
                    )
                )
                log.flush()
                append_metric(metrics_path, step=step, metric="train_loss", value=loss)
                append_metric(metrics_path, step=step, metric="auc", value=auc)
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
                        "resumed_from": config.checkpoint.load_path,
                        "loaded_checkpoint": loaded_checkpoint,
                    },
                    optimizer={"learning_rate": config.training.learning_rate},
                    supported=True,
                    keep_last=config.checkpoint.keep_last,
                )
                log.write(f"Checkpoint saved: {checkpoint_dir}\n")
            log.write("Training completed.\n")
            log.flush()
        return 0

    def run_evaluation(self, config: PrototypeConfig, run_dir: Path) -> int:
        checkpoint_payload = read_checkpoint(config.checkpoint.load_path) if config.checkpoint.load_path else None
        evaluation = {
            "backend": self.name,
            "mode": config.mode.value,
            "auc": 0.5,
            "log_loss": 0.6931,
            "source_checkpoint": config.checkpoint.load_path,
            "checkpoint_load_supported": True,
            "loaded_checkpoint_step": checkpoint_payload.get("step") if checkpoint_payload else None,
        }
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics_path = run_dir / "metrics.jsonl"
        append_metric(metrics_path, step=None, metric="eval_auc", value=evaluation["auc"])
        append_metric(metrics_path, step=None, metric="eval_log_loss", value=evaluation["log_loss"])
        with (run_dir / "train-rank0.log").open("a", encoding="utf-8") as log:
            log.write("Evaluation completed.\n")
        return 0
