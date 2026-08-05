from __future__ import annotations

from pathlib import Path

from prototype.config import PrototypeConfig
from prototype.runner.backends.stub_backend import StubBackend


def run_evaluation(config: PrototypeConfig, run_dir: Path) -> int:
    return StubBackend().run_evaluation(config, run_dir)
