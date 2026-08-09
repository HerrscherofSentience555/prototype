from __future__ import annotations

import shlex
from pathlib import Path

from prototype.config import PrototypeConfig, RuntimePlatform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ROOT = PROJECT_ROOT.parent


def resolve_project_path(path_value: str | None) -> str:
    if not path_value:
        return ""
    normalized = path_value.replace("\\", "/")
    if normalized.startswith("/"):
        return normalized
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def to_wsl_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive = normalized[0].lower()
        return f"/mnt/{drive}{normalized[2:]}"
    return normalized


def to_runtime_path(config: PrototypeConfig, path_value: str | None) -> str:
    resolved = resolve_project_path(path_value)
    if config.backend.runtime_platform == RuntimePlatform.WINDOWS_WSL:
        return to_wsl_path(resolved)
    return resolved


def to_runtime_shell_path(config: PrototypeConfig, path_value: str) -> str:
    normalized = to_runtime_path(config, path_value)
    if normalized.startswith("~/") and config.backend.runtime_platform == RuntimePlatform.WINDOWS_WSL:
        return "$HOME/" + normalized[2:]
    return shlex.quote(normalized)


def to_shell_path_literal(config: PrototypeConfig, path_value: str) -> str:
    if config.backend.runtime_platform == RuntimePlatform.WINDOWS_WSL:
        normalized = to_wsl_path(path_value)
        if normalized.startswith("~/"):
            return "$HOME/" + normalized[2:]
        return shlex.quote(normalized)
    return shlex.quote(path_value)


def build_shell_command(config: PrototypeConfig, shell_script: str) -> list[str]:
    if config.backend.runtime_platform == RuntimePlatform.WINDOWS_WSL:
        return [
            "wsl",
            "-d",
            config.backend.wsl_distribution,
            "bash",
            "-lc",
            shell_script,
        ]
    return ["bash", "-lc", shell_script]
