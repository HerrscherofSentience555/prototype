from __future__ import annotations

import json
import re
import shlex
import subprocess
import unicodedata
from collections import deque
from pathlib import Path

from prototype.config import PrototypeConfig, RunMode
from prototype.runner.backends.base import RunnerBackend
from prototype.runner.log_parser import parse_metric_line
from prototype.runner.metrics import append_metric


class DLRMBackend(RunnerBackend):
    name = "dlrm"

    def run(self, config: PrototypeConfig, run_dir: Path) -> int:
        self._validate_data_config(config)

        command = self.build_command(config, run_dir)
        self._record_backend_command(run_dir, command)
        self._write_launcher_log(run_dir, command)

        rank0_log = run_dir / "train-rank0.log"
        metrics_path = run_dir / "metrics.jsonl"
        with rank0_log.open("a", encoding="utf-8") as log:
            log.write("Starting real DLRM backend run.\n")
            log.write(f"DLRM root: {config.backend.dlrm_root}\n")
            log.write(f"WSL distribution: {config.backend.wsl_distribution}\n")
            log.write(f"Python env: {config.backend.python_env}\n")
            log.flush()

            recent_lines: deque[str] = deque(maxlen=20)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = self._decode_output_line(raw_line)
                recent_lines.append(line.rstrip())
                log.write(line)
                log.flush()
                for metric in parse_metric_line(line):
                    append_metric(metrics_path, **metric)
            exit_code = process.wait()
            if exit_code != 0:
                tail = "\n".join(line for line in recent_lines if line)
                guidance = (
                    "Check that the configured WSL distribution exists, the Python environment "
                    "can be activated, and the DLRM root is accessible from WSL."
                )
                raise RuntimeError(
                    f"DLRM backend command exited with code {exit_code}. {guidance}"
                    + (f"\nRecent output:\n{tail}" if tail else "")
                )
            if config.mode == RunMode.EVALUATE:
                self._write_evaluation_summary(config, run_dir)
            elif config.checkpoint.enabled:
                self._finalize_dlrm_checkpoint(config, run_dir)
                self._write_checkpoint_status(config, run_dir)
            self._write_capability_report(config, run_dir)
            return exit_code

    def build_command(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        dlrm_root = self._to_wsl_path(config.backend.dlrm_root)
        python_env = self._to_wsl_shell_path(config.backend.python_env.rstrip("/"))
        run_dir_wsl = self._to_wsl_path(str(run_dir))
        dlrm_args = self._build_dlrm_args(config, run_dir)
        cuda_visible_devices = ",".join(str(gpu_id) for gpu_id in config.device.gpu_ids)

        shell_script = "; ".join(
            [
                "set -e",
                "export PYTHONIOENCODING=utf-8",
                "export LANG=C.UTF-8",
                "export LC_ALL=C.UTF-8",
                f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)}",
                f"source {python_env}/bin/activate",
                f"cd {shlex.quote(dlrm_root)}",
                f"export TORCHREC_PROTOTYPE_RUN_DIR={shlex.quote(run_dir_wsl)}",
                "exec " + " ".join(shlex.quote(part) for part in dlrm_args),
            ]
        )
        return [
            "wsl",
            "-d",
            config.backend.wsl_distribution,
            "bash",
            "-lc",
            shell_script,
        ]

    def _build_dlrm_args(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        if config.mode == RunMode.EVALUATE:
            return self._build_evaluate_args(config, run_dir)
        return self._build_train_args(config, run_dir)

    def _base_dlrm_args(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        args = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={config.nproc_per_node}",
            "--log_dir",
            self._to_wsl_path(str(run_dir / "logs")),
            "--redirects",
            "3",
            "--tee",
            "3",
            "-m",
            "torchrec_dlrm.dlrm_main",
            "--epochs",
            str(config.training.epochs),
            "--batch_size",
            str(config.data.batch_size),
            "--learning_rate",
            str(config.training.learning_rate),
        ]
        if config.data.pin_memory:
            args.append("--pin_memory")
        if config.data.mmap_mode:
            args.append("--mmap_mode")
        if config.data.test_batch_size is not None:
            args.extend(["--test_batch_size", str(config.data.test_batch_size)])
        if config.model.num_embeddings is not None:
            args.extend(["--num_embeddings", str(config.model.num_embeddings)])
        if config.model.embedding_dim is not None:
            args.extend(["--embedding_dim", str(config.model.embedding_dim)])
        if config.model.dense_arch_layer_sizes:
            args.extend(["--dense_arch_layer_sizes", config.model.dense_arch_layer_sizes])
        if config.model.over_arch_layer_sizes:
            args.extend(["--over_arch_layer_sizes", config.model.over_arch_layer_sizes])
        if config.profile.enabled:
            args.extend(
                [
                    "--profile_dir",
                    self._to_wsl_path(str(run_dir / "profiles" / "dlrm")),
                    "--profile_record_shapes",
                    str(config.profile.record_shapes).lower(),
                    "--profile_memory",
                    str(config.profile.profile_memory).lower(),
                ]
            )
        if config.data.format in {"criteo_binary", "synthetic_multihot"}:
            args.extend(["--dataset_name", config.data.dataset_name])
        if config.data.format == "criteo_binary":
            args.extend(
                [
                    "--in_memory_binary_criteo_path",
                    self._to_wsl_path(config.data.criteo_binary_path or ""),
                ]
            )
        elif config.data.format == "synthetic_multihot":
            args.extend(
                [
                    "--synthetic_multi_hot_criteo_path",
                    self._to_wsl_path(config.data.synthetic_multi_hot_path or ""),
                ]
            )
        if config.checkpoint.load_path and config.mode in {RunMode.RESUME, RunMode.EVALUATE}:
            args.extend(["--checkpoint_load_path", self._to_wsl_path(config.checkpoint.load_path)])
        if config.checkpoint.enabled and config.mode != RunMode.EVALUATE:
            checkpoint_root = Path(config.checkpoint.save_dir) if config.checkpoint.save_dir else run_dir / "checkpoints"
            save_dir = str(checkpoint_root / "step-final")
            args.extend(["--checkpoint_save_dir", self._to_wsl_path(save_dir)])
            if config.checkpoint.save_optimizer:
                args.append("--checkpoint_save_optimizer")
        return args

    def _validate_data_config(self, config: PrototypeConfig) -> None:
        if config.data.format == "random":
            return
        if config.data.format == "parquet":
            raise ValueError(
                "The dlrm backend does not train parquet directly yet. Run the Phase 12 "
                "parquet validation/conversion path first or use data.format=criteo_binary."
            )
        if config.data.format == "criteo_binary":
            self._validate_optional_local_path(config.data.criteo_binary_path, "data.criteo_binary_path")
        if config.data.format == "synthetic_multihot":
            self._validate_optional_local_path(
                config.data.synthetic_multi_hot_path,
                "data.synthetic_multi_hot_path",
            )

    def _validate_optional_local_path(self, path_value: str | None, label: str) -> None:
        if not path_value:
            raise ValueError(f"{label} is required")
        path = Path(path_value)
        if path.is_absolute() and path.drive and not path.exists():
            raise ValueError(f"{label} does not exist on Windows: {path_value}")

    def _build_train_args(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        dlrm_args = self._base_dlrm_args(config, run_dir)
        if config.training.max_steps is not None:
            dlrm_args.extend(
                [
                    "--limit_train_batches",
                    str(config.training.max_steps),
                    "--limit_val_batches",
                    "1",
                    "--limit_test_batches",
                    "1",
                ]
            )
        if config.training.eval_every_n_steps:
            dlrm_args.extend(
                [
                    "--validation_freq_within_epoch",
                    str(config.training.eval_every_n_steps),
                ]
            )
        return dlrm_args

    def _build_evaluate_args(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        dlrm_args = self._base_dlrm_args(config, run_dir)
        self._set_arg_value(dlrm_args, "--epochs", "1")
        dlrm_args.extend(
            [
                "--limit_train_batches",
                "0",
                "--limit_val_batches",
                "1",
                "--limit_test_batches",
                "1",
            ]
        )
        return dlrm_args

    def _set_arg_value(self, args: list[str], flag: str, value: str) -> None:
        index = args.index(flag)
        args[index + 1] = value

    def _record_backend_command(self, run_dir: Path, command: list[str]) -> None:
        command_path = run_dir / "command.json"
        record = {}
        if command_path.exists():
            record = json.loads(command_path.read_text(encoding="utf-8"))
        record["backend_command"] = command
        record["backend_command_display"] = subprocess.list2cmdline(command)
        command_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_launcher_log(self, run_dir: Path, command: list[str]) -> None:
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            log.write("\nResolved DLRM backend command:\n")
            log.write(subprocess.list2cmdline(command))
            log.write("\n")

    def _write_evaluation_summary(self, config: PrototypeConfig, run_dir: Path) -> None:
        metrics = self._read_metrics(run_dir / "metrics.jsonl")
        latest_by_metric = {}
        for record in metrics:
            latest_by_metric[record.get("metric")] = record

        evaluation = {
            "backend": self.name,
            "mode": config.mode.value,
            "source_checkpoint": config.checkpoint.load_path,
            "checkpoint_load_supported": True,
            "checkpoint_load_note": (
                "The local torchrec_dlrm.dlrm_main.py command accepts --checkpoint_load_path "
                "and loads model.pt before validation/test."
            ),
            "val_auc": self._metric_value(latest_by_metric, "val_auc"),
            "test_auc": self._metric_value(latest_by_metric, "test_auc"),
            "log_loss": self._metric_value(latest_by_metric, "log_loss"),
            "val_samples": self._metric_value(latest_by_metric, "val_samples"),
            "test_samples": self._metric_value(latest_by_metric, "test_samples"),
            "metrics_record_count": len(metrics),
        }
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_checkpoint_status(self, config: PrototypeConfig, run_dir: Path) -> None:
        checkpoint_root = Path(config.checkpoint.save_dir) if config.checkpoint.save_dir else run_dir / "checkpoints"
        checkpoint_dir = checkpoint_root / "step-final"
        status = {
            "backend": self.name,
            "checkpoint_save_supported": True,
            "checkpoint_load_supported": True,
            "checkpoint_note": (
                "The local torchrec_dlrm.dlrm_main.py command path supports single-process "
                "smoke checkpoint save/load through model.pt and optional optimizer.pt. "
                "Multi-process sharded production checkpointing still requires Torch Distributed Checkpoint."
            ),
            "requested_save_dir": config.checkpoint.save_dir,
            "checkpoint_dir": str(checkpoint_dir),
            "success_marker": str(checkpoint_dir / "_SUCCESS"),
            "success_marker_exists": (checkpoint_dir / "_SUCCESS").exists(),
            "latest_path": str(checkpoint_root / "latest.json"),
        }
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "checkpoint-status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _finalize_dlrm_checkpoint(self, config: PrototypeConfig, run_dir: Path) -> None:
        checkpoint_root = Path(config.checkpoint.save_dir) if config.checkpoint.save_dir else run_dir / "checkpoints"
        checkpoint_dir = checkpoint_root / "step-final"
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            (checkpoint_dir / "_SUCCESS").write_text("completed", encoding="utf-8")

    def _write_capability_report(self, config: PrototypeConfig, run_dir: Path) -> None:
        report = {
            "backend": self.name,
            "mapped": {
                "cuda_visible_devices": ",".join(str(gpu_id) for gpu_id in config.device.gpu_ids),
                "nproc_per_node": config.nproc_per_node,
                "data_format": config.data.format,
                "checkpoint_save_load": True,
            },
            "recorded_not_fully_mapped": {
                "embedding_placement": config.device.embedding_placement.value,
                "cache_load_factor": config.device.cache_load_factor,
                "precision": config.precision.model_dump(mode="json"),
            },
            "notes": [
                "CUDA_VISIBLE_DEVICES is applied before torchrun.",
                "Embedding placement, GPU cache, and precision are retained in config/artifacts, "
                "but the current DLRM example backend does not expose complete CLI mapping for them.",
            ],
        }
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "capability-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_metrics(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _metric_value(self, latest_by_metric: dict, metric: str):
        record = latest_by_metric.get(metric)
        if not record:
            return None
        return record.get("value")

    def _to_wsl_path(self, path: str) -> str:
        normalized = path.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1:3] == ":/":
            drive = normalized[0].lower()
            return f"/mnt/{drive}{normalized[2:]}"
        return normalized

    def _to_wsl_shell_path(self, path: str) -> str:
        normalized = self._to_wsl_path(path)
        if normalized.startswith("~/"):
            return "$HOME/" + normalized[2:]
        return shlex.quote(normalized)

    def _decode_output_line(self, raw_line: bytes) -> str:
        if b"\x00" in raw_line:
            try:
                return self._sanitize_output_line(raw_line.decode("utf-16le", errors="replace"))
            except UnicodeDecodeError:
                pass
        return self._sanitize_output_line(raw_line.decode("utf-8", errors="replace"))

    def _sanitize_output_line(self, line: str) -> str:
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        return "".join(char for char in line if unicodedata.category(char) != "Cf")
