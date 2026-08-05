from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.backends import get_backend  # noqa: E402
from prototype.runner.backends.torchrec_v1_backend import TorchRecV1Backend  # noqa: E402


class TorchRecV1BackendTests(unittest.TestCase):
    def test_backend_registry_resolves_torchrec_v1(self) -> None:
        backend = get_backend(PrototypeConfig(backend={"name": "torchrec_v1"}))

        self.assertIsInstance(backend, TorchRecV1Backend)

    def test_build_command_invokes_internal_runner_entrypoint(self) -> None:
        config = PrototypeConfig(
            backend={"name": "torchrec_v1"},
            model={"file": "examples/models/torchrec_v1_model.py"},
            nproc_per_node=1,
            device={"gpu_ids": [0]},
        )
        command = TorchRecV1Backend().build_command(
            config,
            Path(r"C:\Users\han\Desktop\prototype\runs\torchrec-v1-dry-run"),
        )
        script = command[-1]

        self.assertEqual(command[:4], ["wsl", "-d", "Ubuntu-22.04", "bash"])
        self.assertIn("source $HOME/venvs/torchrec17/bin/activate", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", script)
        self.assertIn("torchrun --standalone", script)
        self.assertIn("--nproc_per_node=1", script)
        self.assertIn("-m prototype.runner.torchrec_runner.entry", script)
        self.assertIn("--config /mnt/c/Users/han/Desktop/prototype/runs/torchrec-v1-dry-run/resolved-config.yaml", script)
        self.assertIn("--run-dir /mnt/c/Users/han/Desktop/prototype/runs/torchrec-v1-dry-run", script)


if __name__ == "__main__":
    unittest.main()
