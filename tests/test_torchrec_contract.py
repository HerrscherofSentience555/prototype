from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode  # noqa: E402
from prototype.runner.torchrec_runner.entry import run as run_torchrec_v1_entry  # noqa: E402
from prototype.runner.torchrec_runner.contract import (  # noqa: E402
    TorchRecModelContractError,
    load_model_module,
    validate_model_contract,
    write_contract_report,
)


class TorchRecContractTests(unittest.TestCase):
    def test_valid_v1_contract_writes_report(self) -> None:
        model_file = Path(__file__).resolve().parents[1] / "examples" / "models" / "torchrec_v1_model.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            module = load_model_module(str(model_file))
            report = write_contract_report(module, run_dir)
            report_exists = (run_dir / "artifacts" / "torchrec-model-contract.json").exists()

        self.assertTrue(report["valid"])
        self.assertTrue(report["functions"]["build_model"]["signature_compatible"])
        self.assertTrue(report_exists)

    def test_missing_required_function_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "bad_model.py"
            model_file.write_text("def build_model(config):\n    return None\n", encoding="utf-8")
            module = load_model_module(str(model_file))

            with self.assertRaises(TorchRecModelContractError):
                validate_model_contract(module)

    def test_incompatible_required_signature_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "bad_signature_model.py"
            model_file.write_text(
                "\n".join(
                    [
                        "def build_model(settings):",
                        "    return None",
                        "def build_embedding_configs(config):",
                        "    return []",
                    ]
                ),
                encoding="utf-8",
            )
            module = load_model_module(str(model_file))

            with self.assertRaises(TorchRecModelContractError):
                validate_model_contract(module)

    def test_entry_minimal_training_and_evaluation_loop(self) -> None:
        model_file = Path(__file__).resolve().parents[1] / "examples" / "models" / "torchrec_v1_model.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_run = root / "train"
            eval_run = root / "eval"
            train_run.mkdir()
            eval_run.mkdir()
            train_config = PrototypeConfig(
                backend={"name": "torchrec_v1"},
                model={"file": str(model_file)},
                training={"max_steps": 1},
                data={"batch_size": 4},
            )
            run_torchrec_v1_entry(train_config, train_run)
            latest = json.loads((train_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
            success_exists = (Path(latest["latest_checkpoint_dir"]) / "_SUCCESS").exists()
            metrics = [
                json.loads(line)
                for line in (train_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            eval_config = PrototypeConfig(
                mode=RunMode.EVALUATE,
                backend={"name": "torchrec_v1"},
                model={"file": str(model_file)},
                checkpoint={"load_path": latest["latest_checkpoint_dir"]},
            )
            run_torchrec_v1_entry(eval_config, eval_run)
            evaluation = json.loads((eval_run / "evaluation.json").read_text(encoding="utf-8"))

        self.assertTrue(success_exists)
        self.assertIn("samples_per_second", [record["metric"] for record in metrics])
        self.assertEqual(evaluation["backend"], "torchrec_v1")
        self.assertTrue(evaluation["checkpoint_load_supported"])


if __name__ == "__main__":
    unittest.main()
