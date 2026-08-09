from __future__ import annotations

import json
import shlex
import subprocess
from collections import deque
from pathlib import Path

from prototype.config import PrototypeConfig
from prototype.runner.backends.base import RunnerBackend
from prototype.runner.backends.runtime_env import (
    DESKTOP_ROOT,
    build_shell_command,
    to_runtime_path,
    to_shell_path_literal,
    to_wsl_path,
)


class TorchRecV1Backend(RunnerBackend):
    name = "torchrec_v1"

    def run(self, config: PrototypeConfig, run_dir: Path) -> int:
        command = self.build_command(config, run_dir)
        self._record_backend_command(run_dir, command)
        self._write_launcher_log(run_dir, command)

        rank0_log = run_dir / "train-rank0.log"
        with rank0_log.open("a", encoding="utf-8") as log:
            log.write("Starting internal TorchRec V1 backend scaffold.\n")
            log.write(f"Model file: {config.model.file}\n")
            log.flush()

            recent_lines: deque[str] = deque(maxlen=20)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                recent_lines.append(line.rstrip())
                log.write(line)
                log.flush()
            exit_code = process.wait()
            if exit_code != 0:
                tail = "\n".join(line for line in recent_lines if line)
                raise RuntimeError(
                    "TorchRec V1 backend command exited with code "
                    f"{exit_code}."
                    + (f"\nRecent output:\n{tail}" if tail else "")
                )
        self._write_capability_report(config, run_dir)
        return exit_code

    def build_command(self, config: PrototypeConfig, run_dir: Path) -> list[str]:
        desktop_root = to_runtime_path(config, str(DESKTOP_ROOT))
        python_env = to_shell_path_literal(config, config.backend.python_env.rstrip("/"))
        run_dir_runtime = to_runtime_path(config, str(run_dir))
        config_path_runtime = to_runtime_path(config, str(run_dir / "resolved-config.yaml"))
        cuda_visible_devices = ",".join(str(gpu_id) for gpu_id in config.device.gpu_ids)
        torchrun_args = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={config.nproc_per_node}",
            "--log_dir",
            to_runtime_path(config, str(run_dir / "logs")),
            "--redirects",
            "3",
            "--tee",
            "3",
            "-m",
            "prototype.runner.torchrec_runner.entry",
            "--config",
            config_path_runtime,
            "--run-dir",
            run_dir_runtime,
        ]
        shell_script = "; ".join(
            [
                "set -e",
                "export PYTHONIOENCODING=utf-8",
                "export LANG=C.UTF-8",
                "export LC_ALL=C.UTF-8",
                f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)}",
                f"source {python_env}/bin/activate",
                f"cd {shlex.quote(desktop_root)}",
                "exec " + " ".join(shlex.quote(part) for part in torchrun_args),
            ]
        )
        return build_shell_command(config, shell_script)

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
            log.write("\nResolved TorchRec V1 backend command:\n")
            log.write(subprocess.list2cmdline(command))
            log.write("\n")

    def _write_capability_report(self, config: PrototypeConfig, run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        report = {
            "backend": self.name,
            "mapped": {
                "cuda_visible_devices": ",".join(str(gpu_id) for gpu_id in config.device.gpu_ids),
                "nproc_per_node": config.nproc_per_node,
                "torchrun_entrypoint": "prototype.runner.torchrec_runner.entry",
                "model_contract_validation": True,
            },
            "not_yet_complete": {
                "distributed_model_parallel_training_loop": True,
                "train_pipeline_sparse_dist": True,
                "torch_distributed_checkpoint": True,
                "gpu_cache_mapping": True,
                "precision_mapping": True,
            },
        }
        (artifacts_dir / "torchrec-v1-capability-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _to_wsl_path(self, path: str) -> str:
        return to_wsl_path(path)

    def _to_wsl_shell_path(self, path: str) -> str:
        return to_shell_path_literal(PrototypeConfig(backend={"name": "torchrec_v1"}), path)
