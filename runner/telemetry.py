from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from prototype.config import PrototypeConfig


class ResourceTelemetry:
    def __init__(
        self,
        run_dir: Path,
        config: PrototypeConfig | None = None,
        interval_seconds: float = 1.0,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.interval_seconds = interval_seconds
        self.path = run_dir / "resource-metrics.jsonl"
        self.summary_path = run_dir / "artifacts" / "resource-summary.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="resource-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._write_summary()

    def _loop(self) -> None:
        process = _get_process()
        while not self._stop.is_set():
            try:
                record = self._sample(process)
                self._records.append(record)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                with (self.run_dir / "launcher.log").open("a", encoding="utf-8") as log:
                    log.write(f"\nTelemetry sample failed: {type(exc).__name__}: {exc}\n")
            self._stop.wait(self.interval_seconds)

    def _sample(self, process) -> dict:
        disk_bytes = sum(path.stat().st_size for path in self.run_dir.rglob("*") if path.is_file())
        record = {
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "run_dir_disk_bytes": disk_bytes,
        }
        if process is not None:
            children = process.children(recursive=True)
            child_memory = sum(child.memory_info().rss for child in children if child.is_running())
            record.update(
                {
                    "cpu_percent": process.cpu_percent(interval=None),
                    "memory_rss_bytes": process.memory_info().rss,
                    "memory_percent": process.memory_percent(),
                    "child_process_count": len(children),
                    "child_memory_rss_bytes": child_memory,
                }
            )
        record.update(_sample_nvidia_smi())
        if self.config is not None:
            record.update(_sample_wsl_processes(self.config.backend.wsl_distribution))
        return record

    def _write_summary(self) -> None:
        self.summary_path.parent.mkdir(exist_ok=True)
        summary = {
            "record_count": len(self._records),
            "max_cpu_percent": _max_value(self._records, "cpu_percent"),
            "max_memory_rss_bytes": _max_value(self._records, "memory_rss_bytes"),
            "max_child_memory_rss_bytes": _max_value(self._records, "child_memory_rss_bytes"),
            "max_memory_percent": _max_value(self._records, "memory_percent"),
            "max_run_dir_disk_bytes": _max_value(self._records, "run_dir_disk_bytes"),
            "max_gpu_utilization_percent": _max_value(self._records, "gpu_utilization_percent"),
            "max_gpu_memory_used_mb": _max_value(self._records, "gpu_memory_used_mb"),
            "max_wsl_torch_cpu_percent": _max_value(self._records, "wsl_torch_cpu_percent"),
            "max_wsl_torch_memory_rss_kb": _max_value(self._records, "wsl_torch_memory_rss_kb"),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@contextmanager
def telemetry_span(run_dir: Path, config: PrototypeConfig | None = None) -> Iterator[None]:
    telemetry = ResourceTelemetry(run_dir, config=config)
    try:
        telemetry.start()
        yield
    finally:
        try:
            telemetry.stop()
        except Exception as exc:
            with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
                log.write(f"\nTelemetry stop failed: {type(exc).__name__}: {exc}\n")


def _get_process():
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)
        return process
    except Exception:
        return None


def _sample_nvidia_smi() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "gpu_telemetry_available": False,
            "gpu_telemetry_error": "nvidia-smi unavailable",
        }
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "gpu_telemetry_available": False,
            "gpu_telemetry_error": "nvidia-smi returned no data",
        }
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append(
            {
                "utilization": float(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
            }
        )
    if not rows:
        return {"gpu_telemetry_available": False, "gpu_telemetry_error": "nvidia-smi parse failed"}
    return {
        "gpu_telemetry_available": True,
        "gpu_count": len(rows),
        "gpu_utilization_percent": max(row["utilization"] for row in rows),
        "gpu_memory_used_mb": max(row["memory_used"] for row in rows),
        "gpu_memory_total_mb": max(row["memory_total"] for row in rows),
    }


def _sample_wsl_processes(distribution: str) -> dict:
    script = (
        "ps -eo comm=,pcpu=,rss= | "
        "awk 'BEGIN{cpu=0;rss=0;count=0} "
        "/python|torchrun/{cpu+=$2;rss+=$3;count+=1} "
        "END{printf \"%d %.2f %.0f\", count, cpu, rss}'"
    )
    try:
        result = subprocess.run(
            ["wsl", "-d", distribution, "bash", "-lc", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "wsl_telemetry_available": False,
            "wsl_telemetry_error": "wsl process sampling unavailable",
        }
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "wsl_telemetry_available": False,
            "wsl_telemetry_error": "wsl process sampling returned no data",
        }
    parts = result.stdout.strip().split()
    if len(parts) != 3:
        return {
            "wsl_telemetry_available": False,
            "wsl_telemetry_error": "wsl process sampling parse failed",
        }
    return {
        "wsl_telemetry_available": True,
        "wsl_torch_process_count": int(parts[0]),
        "wsl_torch_cpu_percent": float(parts[1]),
        "wsl_torch_memory_rss_kb": float(parts[2]),
    }


def _max_value(records: list[dict], key: str):
    values = [record.get(key) for record in records if record.get(key) is not None]
    return max(values) if values else None
