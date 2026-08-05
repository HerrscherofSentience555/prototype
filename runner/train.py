from __future__ import annotations

from pathlib import Path

from prototype.config import PrototypeConfig
from prototype.runner.backends.stub_backend import StubBackend


def run_training(config: PrototypeConfig, run_dir: Path) -> int:
    return StubBackend().run_training(config, run_dir)
