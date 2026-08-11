from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from prototype.config import PrototypeConfig


RUNS_DIR = Path(__file__).resolve().parent / "runs"


TERMINAL_STATES = {"STOPPED", "SUCCEEDED", "FAILED"}
ACTIVE_STATES = {"LAUNCHING", "RUNNING", "STOPPING"}
STOP_GRACE_SECONDS = 5.0
BUNDLE_EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".zip"}
BUNDLE_MAX_FILE_BYTES = 50 * 1024 * 1024


def _now_iso() -> str:
    return datetime.now().isoformat()


def _duration_seconds(started_at: Optional[str], ended_at: str) -> Optional[float]:
    if not started_at:
        return None
    return (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds()


@dataclass
class JobInfo:
    job_id: str
    status: str
    pid: Optional[int]
    run_dir: Path


class LocalTaskManager:
    def __init__(self, runs_dir: Path = RUNS_DIR, max_concurrent_jobs: int = 1) -> None:
        self.runs_dir = runs_dir
        self.max_concurrent_jobs = max_concurrent_jobs
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.recover_stale_jobs()

    def create_job(self, config: PrototypeConfig) -> JobInfo:
        job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        run_dir = self.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        profiles_dir = run_dir / "profiles"
        artifacts_dir = run_dir / "artifacts"
        checkpoints_dir.mkdir(exist_ok=True)
        profiles_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)

        config = config.model_copy(
            update={
                "checkpoint": config.checkpoint.model_copy(update={"save_dir": str(checkpoints_dir)})
            }
        )

        config_path = run_dir / "resolved-config.yaml"
        config_path.write_text(config.to_yaml(), encoding="utf-8")

        created_at = _now_iso()
        state = {
            "job_id": job_id,
            "job_name": config.job_name,
            "status": "CREATED",
            "backend": config.backend.name.value,
            "command": None,
            "cwd": None,
            "pid": None,
            "wsl_pid": None,
            "error_message": None,
            "created_at": created_at,
            "started_at": None,
            "updated_at": created_at,
            "ended_at": None,
            "duration_seconds": None,
            "exit_code": None,
            "stopped_at": None,
            "stop_reason": None,
            "stop_signal": None,
            "force_killed": False,
            "stop_error": None,
        }
        self._write_state(run_dir, state)
        return JobInfo(job_id=job_id, status="CREATED", pid=None, run_dir=run_dir)

    def launch_job(self, job_id: str) -> JobInfo:
        run_dir = self.runs_dir / job_id
        state = self._read_state(run_dir)
        active_jobs = self._active_jobs(exclude_job_id=job_id)
        if len(active_jobs) >= self.max_concurrent_jobs:
            active_ids = ", ".join(job.job_id for job in active_jobs)
            raise RuntimeError(
                f"Cannot launch job {job_id}: max_concurrent_jobs={self.max_concurrent_jobs} "
                f"and active job(s) already exist: {active_ids}"
            )
        config_path = run_dir / "resolved-config.yaml"
        config = PrototypeConfig.from_yaml_file(config_path)
        launcher_log = run_dir / "launcher.log"
        cwd = Path(__file__).resolve().parent.parent

        cmd = [
            sys.executable,
            "-m",
            "prototype.runner.cli",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]
        command_record = {
            "backend": config.backend.name.value,
            "backend_config": config.backend.model_dump(mode="json"),
            "command": cmd,
            "cwd": str(cwd),
            "launched_at": _now_iso(),
        }
        (run_dir / "command.json").write_text(
            json.dumps(command_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state["status"] = "LAUNCHING"
        state["backend"] = config.backend.name.value
        state["command"] = cmd
        state["cwd"] = str(cwd)
        state["updated_at"] = _now_iso()
        self._write_state(run_dir, state)

        try:
            with launcher_log.open("a", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
        except Exception as exc:
            stale_status = state.get("status")
            ended_at = _now_iso()
            state["status"] = "FAILED"
            state["error_message"] = f"Launch failed: {type(exc).__name__}: {exc}"
            state["ended_at"] = ended_at
            state["updated_at"] = ended_at
            state["duration_seconds"] = _duration_seconds(state.get("started_at"), ended_at)
            self._write_state(run_dir, state)
            raise

        state["status"] = "RUNNING"
        state["pid"] = process.pid
        state["started_at"] = _now_iso()
        state["updated_at"] = state["started_at"]
        self._write_state(run_dir, state)
        return JobInfo(job_id=job_id, status="RUNNING", pid=process.pid, run_dir=run_dir)

    def stop_job(self, job_id: str) -> JobInfo:
        run_dir = self.runs_dir / job_id
        state = self._read_state(run_dir)
        if state.get("status") in TERMINAL_STATES:
            return JobInfo(
                job_id=state["job_id"],
                status=state["status"],
                pid=state.get("pid"),
                run_dir=run_dir,
            )

        pid = state.get("pid")
        stop_error = None
        force_killed = False
        stop_signal = None
        stopped_at = _now_iso()

        state["status"] = "STOPPING"
        state["updated_at"] = stopped_at
        state["stop_reason"] = "user_requested"
        self._write_state(run_dir, state)

        if pid:
            try:
                if os.name == "nt":
                    stop_signal = "CTRL_BREAK_EVENT"
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                else:
                    stop_signal = "SIGTERM"
                    os.kill(pid, signal.SIGTERM)
            except OSError:
                stop_error = f"Process {pid} was not available for graceful termination."

            if self._pid_exists(pid) and not self._wait_for_pid_exit(pid, STOP_GRACE_SECONDS):
                force_killed = self._force_kill_process_tree(pid)
                if not force_killed:
                    stop_error = stop_error or f"Process {pid} did not exit after force kill attempt."
        else:
            stop_error = "No process id was recorded for this job."

        ended_at = _now_iso()
        state["status"] = "STOPPED"
        state["ended_at"] = ended_at
        state["updated_at"] = ended_at
        state["stopped_at"] = ended_at
        state["stop_reason"] = "user_requested"
        state["stop_signal"] = stop_signal
        state["force_killed"] = force_killed
        state["stop_error"] = stop_error
        state["exit_code"] = state.get("exit_code")
        state["duration_seconds"] = _duration_seconds(state.get("started_at"), ended_at)
        self._write_state(run_dir, state)
        return JobInfo(job_id=job_id, status="STOPPED", pid=pid, run_dir=run_dir)

    def list_jobs(self) -> list[JobInfo]:
        self.recover_stale_jobs()
        jobs: list[JobInfo] = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            state = self._read_state(run_dir)
            jobs.append(
                JobInfo(
                    job_id=state["job_id"],
                    status=state["status"],
                    pid=state.get("pid"),
                    run_dir=run_dir,
                )
            )
        return jobs

    def read_text_file(self, job_id: str, filename: str) -> str:
        path = self.runs_dir / job_id / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def get_run_dir(self, job_id: str) -> Path:
        return self.runs_dir / job_id

    def list_files(self, job_id: str, relative_dir: str) -> list[str]:
        directory = self.runs_dir / job_id / relative_dir
        if not directory.exists() or not directory.is_dir():
            return []
        files: list[str] = []
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(directory)))
        return files

    def update_terminal_state(self, run_dir: Path, exit_code: int, error_message: Optional[str] = None) -> None:
        state = self._read_state(run_dir)
        if state["status"] in TERMINAL_STATES:
            return
        ended_at = _now_iso()
        state["status"] = "SUCCEEDED" if exit_code == 0 else "FAILED"
        state["exit_code"] = exit_code
        state["error_message"] = error_message
        state["ended_at"] = ended_at
        state["updated_at"] = ended_at
        state["duration_seconds"] = _duration_seconds(state.get("started_at"), ended_at)
        self._write_state(run_dir, state)
        self._write_run_bundle(run_dir)

    def recover_stale_jobs(self) -> None:
        if not self.runs_dir.exists():
            return
        for run_dir in self.runs_dir.iterdir():
            state_path = run_dir / "state.json"
            if not run_dir.is_dir() or not state_path.exists():
                continue
            try:
                state = self._read_state(run_dir)
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") not in ACTIVE_STATES:
                continue
            pid = state.get("pid")
            if pid and self._pid_exists(int(pid)):
                continue
            stale_status = state.get("status")
            ended_at = _now_iso()
            state["status"] = "FAILED"
            state["ended_at"] = state.get("ended_at") or ended_at
            state["updated_at"] = ended_at
            state["exit_code"] = state.get("exit_code")
            state["error_message"] = (
                state.get("error_message")
                or f"Recovered stale {stale_status} job after app restart; recorded process is not running."
            )
            state["recovered_at"] = ended_at
            state["recovery_reason"] = "missing_process"
            state["duration_seconds"] = _duration_seconds(state.get("started_at"), state["ended_at"])
            self._write_state(run_dir, state)

    def _active_jobs(self, exclude_job_id: str | None = None) -> list[JobInfo]:
        active: list[JobInfo] = []
        for run_dir in self.runs_dir.iterdir():
            state_path = run_dir / "state.json"
            if not run_dir.is_dir() or not state_path.exists():
                continue
            state = self._read_state(run_dir)
            if state.get("job_id") == exclude_job_id:
                continue
            if state.get("status") in ACTIVE_STATES:
                active.append(
                    JobInfo(
                        job_id=state["job_id"],
                        status=state["status"],
                        pid=state.get("pid"),
                        run_dir=run_dir,
                    )
                )
        return active

    def _write_run_bundle(self, run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        bundle_path = artifacts_dir / "run-artifacts.zip"
        manifest = {
            "note": (
                "This bundle contains logs, configs, metrics, and reports. Large binary artifacts "
                "such as model checkpoints are excluded to keep the browser download reliable."
            ),
            "excluded_files": [],
        }
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if not path.is_file() or path == bundle_path:
                    continue
                if self._exclude_from_run_bundle(path):
                    manifest["excluded_files"].append(
                        {
                            "path": path.relative_to(run_dir).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "reason": "large or binary artifact",
                        }
                    )
                    continue
                archive.write(path, path.relative_to(run_dir))
            archive.writestr(
                "artifacts/bundle-manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )

    def _exclude_from_run_bundle(self, path: Path) -> bool:
        if path.suffix.lower() in BUNDLE_EXCLUDED_SUFFIXES:
            return True
        return path.stat().st_size > BUNDLE_MAX_FILE_BYTES

    def _read_state(self, run_dir: Path) -> dict:
        return json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

    def _write_state(self, run_dir: Path, state: dict) -> None:
        (run_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _wait_for_pid_exit(self, pid: int, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                return True
            time.sleep(0.2)
        return not self._pid_exists(pid)

    def _force_kill_process_tree(self, pid: int) -> bool:
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                return result.returncode == 0 or not self._pid_exists(pid)
            os.kill(pid, signal.SIGKILL)
            return self._wait_for_pid_exit(pid, 2.0)
        except (OSError, subprocess.SubprocessError):
            return not self._pid_exists(pid)
