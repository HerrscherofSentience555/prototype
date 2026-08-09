from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.distributed import (  # noqa: E402
    build_distributed_environment_report,
    initialize_distributed_environment,
)


class TorchRecDistributedEnvironmentTests(unittest.TestCase):
    def test_report_defaults_to_single_process_without_torchrun_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            report = build_distributed_environment_report(PrototypeConfig(backend={"name": "torchrec_v1"}))

        self.assertEqual(report["rank"], 0)
        self.assertEqual(report["local_rank"], 0)
        self.assertEqual(report["world_size"], 1)
        self.assertFalse(report["torchrun_environment"])

    def test_initialize_writes_distributed_environment_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with patch.dict(os.environ, {}, clear=True):
                report = initialize_distributed_environment(
                    PrototypeConfig(backend={"name": "torchrec_v1"}),
                    run_dir,
                )
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-distributed-environment.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved["schema"], "torchrec-v1-distributed-environment")
        self.assertEqual(saved["world_size"], report["world_size"])

    def test_report_reads_torchrun_rank_environment(self) -> None:
        env = {
            "RANK": "2",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with patch.dict(os.environ, env, clear=True):
            report = build_distributed_environment_report(PrototypeConfig(backend={"name": "torchrec_v1"}))

        self.assertEqual(report["rank"], 2)
        self.assertEqual(report["local_rank"], 1)
        self.assertEqual(report["world_size"], 4)
        self.assertTrue(report["torchrun_environment"])


if __name__ == "__main__":
    unittest.main()
