from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from prototype.config import PrototypeConfig, RuntimePlatform
from prototype.local_settings import (
    LOCAL_SETTINGS_PATH,
    load_local_settings,
    local_settings_status,
    write_local_settings_template,
)
from prototype.runner.backends.dlrm_backend import clean_dlrm_log_text
from prototype.runner.backends.runtime_env import to_runtime_path, to_shell_path_literal


CHECK_COLUMNS = ["item", "status", "detail"]


def build_environment_tab() -> None:
    with gr.Tab("Environment"):
        settings_view = gr.Code(label="local settings", language="json")
        checks_table = gr.Dataframe(label="Environment Checks")
        action_status = gr.Textbox(label="Action Status")

        def refresh_settings() -> str:
            return json.dumps(local_settings_status(), ensure_ascii=False, indent=2)

        def create_template() -> tuple[str, str]:
            path = write_local_settings_template()
            return f"Local settings template is ready: {path}", refresh_settings()

        gr.Button("Refresh Settings").click(refresh_settings, outputs=settings_view)
        gr.Button("Create local_settings.yaml").click(
            create_template,
            outputs=[action_status, settings_view],
        )
        gr.Button("Run Environment Checks").click(run_environment_checks, outputs=checks_table)


def run_environment_checks() -> pd.DataFrame:
    settings = load_local_settings()
    rows: list[dict[str, str]] = []
    rows.append(_row("local_settings.yaml", "OK" if LOCAL_SETTINGS_PATH.exists() else "MISSING", str(LOCAL_SETTINGS_PATH)))
    rows.append(_row("settings source", "OK", settings.source))
    rows.append(_check_runtime_platform(settings.runtime.platform))
    rows.append(_check_python_env(settings.runtime.platform, settings.runtime.wsl_distribution, settings.runtime.python_env))
    rows.append(_check_python_import(settings.runtime.platform, settings.runtime.wsl_distribution, settings.runtime.python_env, "torch"))
    rows.append(_check_python_import(settings.runtime.platform, settings.runtime.wsl_distribution, settings.runtime.python_env, "torchrec"))
    rows.append(_check_path("DLRM root", settings.paths.dlrm_root, platform=settings.runtime.platform))
    rows.extend(_check_criteo_binary_dataset(settings.paths.criteo_binary_path))
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def _row(item: str, status: str, detail: str) -> dict[str, str]:
    return {"item": item, "status": status, "detail": detail}


def _check_runtime_platform(platform: str) -> dict[str, str]:
    if platform not in {item.value for item in RuntimePlatform}:
        return _row("runtime platform", "ERROR", f"unsupported platform: {platform}")
    if platform == RuntimePlatform.WINDOWS_WSL:
        completed = _run(["wsl", "-l", "-q"], timeout=10)
        return _row(
            "WSL available",
            "OK" if completed.returncode == 0 else "ERROR",
            _clean_check_detail(completed.output) or f"exit code {completed.returncode}",
        )
    completed = _run(["bash", "-lc", "echo linux_native"], timeout=10)
    return _row(
        "Linux shell",
        "OK" if completed.returncode == 0 else "ERROR",
        _clean_check_detail(completed.output) or f"exit code {completed.returncode}",
    )


def _check_python_env(platform: str, distro: str, python_env: str) -> dict[str, str]:
    config = _runtime_config(platform, distro, python_env)
    env_path = to_shell_path_literal(config, python_env.rstrip("/"))
    command = f"test -f {env_path}/bin/activate"
    completed = _run_runtime_shell(platform, distro, command)
    return _row(
        "Python env",
        "OK" if completed.returncode == 0 else "MISSING",
        python_env if completed.returncode == 0 else _clean_check_detail(completed.output),
    )


def _check_python_import(platform: str, distro: str, python_env: str, module: str) -> dict[str, str]:
    config = _runtime_config(platform, distro, python_env)
    env_path = to_shell_path_literal(config, python_env.rstrip("/"))
    command = (
        f"source {env_path}/bin/activate; "
        f"python -c 'import {module}; print({module}.__version__ if hasattr({module}, \"__version__\") else \"ok\")'"
    )
    completed = _run_runtime_shell(platform, distro, command, timeout=30)
    return _row(
        f"import {module}",
        "OK" if completed.returncode == 0 else "ERROR",
        _success_tail(completed.output) if completed.returncode == 0 else _clean_check_detail(completed.output),
    )


def _check_path(item: str, path_value: str, platform: str) -> dict[str, str]:
    if not path_value:
        return _row(item, "MISSING", "empty path")
    if platform == RuntimePlatform.WINDOWS_WSL:
        config = _runtime_config(platform, "", "")
        runtime_path = to_runtime_path(config, path_value)
        completed = _run_runtime_shell(platform, "", f"test -d {shlex.quote(runtime_path)}", timeout=10)
        return _row(item, "OK" if completed.returncode == 0 else "MISSING", runtime_path)
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return _row(item, "OK" if path.exists() else "MISSING", str(path))


def _check_criteo_binary_dataset(path_value: str) -> list[dict[str, str]]:
    root = Path(path_value).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    rows = [_row("Criteo binary path", "OK" if root.exists() else "MISSING", str(root))]
    expected = ["train_dense.npy", "train_sparse.npy", "train_labels.npy"]
    for filename in expected:
        path = root / filename
        rows.append(_row(filename, "OK" if path.exists() else "MISSING", str(path)))
    if all((root / filename).exists() for filename in expected):
        try:
            import numpy as np

            dense = np.load(root / "train_dense.npy", mmap_mode="r")
            sparse = np.load(root / "train_sparse.npy", mmap_mode="r")
            labels = np.load(root / "train_labels.npy", mmap_mode="r")
            rows.append(
                _row(
                    "Criteo binary shapes",
                    "OK",
                    f"dense={dense.shape}, sparse={sparse.shape}, labels={labels.shape}",
                )
            )
        except Exception as exc:
            rows.append(_row("Criteo binary shapes", "ERROR", f"{type(exc).__name__}: {exc}"))
    return rows


def _run_runtime_shell(platform: str, distro: str, command: str, timeout: int = 10):
    if platform == RuntimePlatform.WINDOWS_WSL:
        args = ["wsl", "-d", distro or "Ubuntu-22.04", "bash", "-lc", command]
    else:
        args = ["bash", "-lc", command]
    return _run(args, timeout=timeout)


class _Completed:
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output


def _run(args: list[str], timeout: int) -> _Completed:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return _Completed(completed.returncode, completed.stdout)
    except Exception as exc:
        return _Completed(1, f"{type(exc).__name__}: {exc}")


def _clean_check_detail(output: str) -> str:
    cleaned = clean_dlrm_log_text(output.replace("\x00", ""))
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("wsl: detected localhost proxy configuration"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _success_tail(output: str) -> str:
    cleaned = _clean_check_detail(output)
    if not cleaned:
        return "ok"
    return cleaned.split()[-1]


def _runtime_config(platform: str, distro: str, python_env: str) -> PrototypeConfig:
    backend = {
        "name": "dlrm",
        "runtime_platform": platform,
        "wsl_distribution": distro or "Ubuntu-22.04",
        "python_env": python_env or "~/venvs/torchrec17",
    }
    return PrototypeConfig(backend=backend)
