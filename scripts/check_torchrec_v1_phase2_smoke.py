from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode
from prototype.runner.torchrec_runner.entry import run


def main() -> int:
    parser = argparse.ArgumentParser(description="TorchRec V1 Phase 2 checkpoint smoke")
    parser.add_argument("--run-root", default="prototype/runs")
    parser.add_argument("--model-file", default="prototype/examples/models/torchrec_v1_model.py")
    args = parser.parse_args()

    root = Path(args.run_root) / f"phase2-wsl-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_run = root / "train"
    resume_run = root / "resume"
    eval_run = root / "evaluate"
    train_run.mkdir(parents=True)
    resume_run.mkdir()
    eval_run.mkdir()

    model_file = str(Path(args.model_file))
    train_config = PrototypeConfig(
        backend={"name": "torchrec_v1"},
        model={"file": model_file},
        training={"max_steps": 1},
        data={"batch_size": 4},
    )
    run(train_config, train_run)
    latest = json.loads((train_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    checkpoint_dir = Path(latest["latest_checkpoint_dir"])

    resume_config = PrototypeConfig(
        mode=RunMode.RESUME,
        backend={"name": "torchrec_v1"},
        model={"file": model_file},
        checkpoint={"load_path": str(checkpoint_dir)},
        training={"max_steps": 1},
        data={"batch_size": 4},
    )
    run(resume_config, resume_run)

    eval_config = PrototypeConfig(
        mode=RunMode.EVALUATE,
        backend={"name": "torchrec_v1"},
        model={"file": model_file},
        checkpoint={"load_path": str(checkpoint_dir)},
        data={"batch_size": 4},
    )
    run(eval_config, eval_run)

    resume_latest = json.loads((resume_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    resume_checkpoint_dir = Path(resume_latest["latest_checkpoint_dir"])
    resume_payload = json.loads((resume_checkpoint_dir / "model.json").read_text(encoding="utf-8"))
    evaluation = json.loads((eval_run / "evaluation.json").read_text(encoding="utf-8"))
    result = {
        "root": str(root),
        "train_checkpoint": str(checkpoint_dir),
        "model_pt_exists": (checkpoint_dir / "model.pt").exists(),
        "optimizer_pt_exists": (checkpoint_dir / "optimizer.pt").exists(),
        "resume_step": resume_payload.get("step"),
        "resume_loaded_runtime_checkpoint": (resume_payload.get("runtime_checkpoint") or {}).get(
            "loaded_runtime_checkpoint"
        ),
        "evaluation_single_card_runtime": evaluation.get("single_card_runtime"),
        "evaluation_minimal_loop": evaluation.get("minimal_loop"),
        "evaluation_metrics": evaluation.get("metrics"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["model_pt_exists"]:
        return 2
    if not result["optimizer_pt_exists"]:
        return 3
    if not result["resume_loaded_runtime_checkpoint"]:
        return 4
    if result["evaluation_single_card_runtime"] != "single_card_runtime":
        return 5
    if result["evaluation_minimal_loop"]:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
