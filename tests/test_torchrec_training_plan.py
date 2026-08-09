from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.contract import load_model_module, validate_model_contract  # noqa: E402
from prototype.runner.torchrec_runner.data import build_data_plan  # noqa: E402
from prototype.runner.torchrec_runner.plan import build_training_plan, write_training_plan  # noqa: E402


class TorchRecTrainingPlanTests(unittest.TestCase):
    def test_training_plan_records_planned_dmp_steps(self) -> None:
        model_file = Path(__file__).resolve().parents[1] / "examples" / "models" / "torchrec_v1_model.py"
        config = PrototypeConfig(
            backend={"name": "torchrec_v1"},
            model={"file": str(model_file)},
            profile={"enabled": True, "start_step": 1, "end_step": 1},
        )
        module = load_model_module(str(model_file))
        contract = validate_model_contract(module)
        data_plan = build_data_plan(config, Path("unused"))
        plan = build_training_plan(config, contract, data_plan)

        status_by_name = {step["name"]: step["status"] for step in plan["steps"]}
        self.assertEqual(status_by_name["wrap_dmp"], "implemented_fallback_or_runtime")
        self.assertEqual(status_by_name["train_pipeline_sparse_dist"], "planned")
        self.assertEqual(status_by_name["profile"], "window_metric_implemented_trace_pending")

    def test_write_training_plan_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = PrototypeConfig(backend={"name": "torchrec_v1"})
            contract = {
                "required_functions": ["build_model", "build_embedding_configs"],
                "functions": {
                    "build_model": {"available": True},
                    "build_embedding_configs": {"available": True},
                    "build_dataloader": {"available": False},
                    "build_optimizer": {"available": False},
                },
            }
            data_plan = build_data_plan(config, run_dir)
            write_training_plan(config, run_dir, contract, data_plan)
            saved = json.loads((run_dir / "artifacts" / "torchrec-training-plan.json").read_text(encoding="utf-8"))

        self.assertEqual(saved["schema"], "torchrec-v1-training-plan")


if __name__ == "__main__":
    unittest.main()
