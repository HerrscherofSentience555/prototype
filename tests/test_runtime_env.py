from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.backends.runtime_env import (  # noqa: E402
    PROJECT_ROOT,
    build_shell_command,
    resolve_project_path,
    to_runtime_path,
    to_shell_path_literal,
    to_wsl_path,
)


class RuntimeEnvTests(unittest.TestCase):
    def test_resolve_project_path_uses_current_clone_root(self) -> None:
        self.assertEqual(resolve_project_path("data/example"), str(PROJECT_ROOT / "data" / "example"))

    def test_resolve_project_path_keeps_unix_absolute_paths(self) -> None:
        self.assertEqual(resolve_project_path("/mnt/c/Users/example/dlrm"), "/mnt/c/Users/example/dlrm")

    def test_to_wsl_path_converts_windows_drive_path(self) -> None:
        self.assertEqual(to_wsl_path(r"C:\Users\han\Desktop\prototype"), "/mnt/c/Users/han/Desktop/prototype")

    def test_windows_wsl_runtime_converts_relative_paths(self) -> None:
        config = PrototypeConfig(backend={"name": "dlrm", "runtime_platform": "windows_wsl"})

        self.assertEqual(to_runtime_path(config, "data/example"), to_wsl_path(str(PROJECT_ROOT / "data" / "example")))
        self.assertEqual(to_shell_path_literal(config, "~/venvs/torchrec17"), "$HOME/venvs/torchrec17")
        self.assertEqual(build_shell_command(config, "echo ok")[:4], ["wsl", "-d", "Ubuntu-22.04", "bash"])

    def test_linux_native_runtime_keeps_native_absolute_paths(self) -> None:
        config = PrototypeConfig(backend={"name": "dlrm", "runtime_platform": "linux_native"})

        self.assertEqual(to_runtime_path(config, "data/example"), str(PROJECT_ROOT / "data" / "example"))
        self.assertEqual(build_shell_command(config, "echo ok"), ["bash", "-lc", "echo ok"])


if __name__ == "__main__":
    unittest.main()
