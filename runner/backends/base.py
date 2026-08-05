from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from prototype.config import PrototypeConfig


class RunnerBackend(ABC):
    name: str

    @abstractmethod
    def run(self, config: PrototypeConfig, run_dir: Path) -> int:
        raise NotImplementedError
