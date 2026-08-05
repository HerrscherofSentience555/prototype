from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode  # noqa: E402
from prototype.runner.backends.custom_backend import CustomModelBackend  # noqa: E402


class CustomBackendTests(unittest.TestCase):
    def test_custom_backend_train_and_evaluate(self) -> None:
        model_file = Path(__file__).resolve().parents[1] / "examples" / "models" / "custom_simple_model.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_run = root / "train"
            eval_run = root / "eval"
            train_run.mkdir()
            eval_run.mkdir()
            train_config = PrototypeConfig(
                backend={"name": "custom"},
                model={"file": str(model_file)},
                training={"max_steps": 2},
            )
            CustomModelBackend().run(train_config, train_run)
            latest = json.loads((train_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))

            eval_config = PrototypeConfig(
                mode=RunMode.EVALUATE,
                backend={"name": "custom"},
                model={"file": str(model_file)},
                checkpoint={"load_path": latest["latest_checkpoint_dir"]},
            )
            CustomModelBackend().run(eval_config, eval_run)
            evaluation = json.loads((eval_run / "evaluation.json").read_text(encoding="utf-8"))

            self.assertTrue((train_run / "artifacts" / "custom-model-contract.json").exists())
            self.assertEqual(evaluation["backend"], "custom")
            self.assertTrue(evaluation["checkpoint_load_supported"])
            self.assertIn("auc", evaluation["metrics"])


if __name__ == "__main__":
    unittest.main()
