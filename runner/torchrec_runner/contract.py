from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any


REQUIRED_FUNCTIONS = ["build_model", "build_embedding_configs"]
OPTIONAL_FUNCTIONS = ["build_optimizer", "build_dataloader", "train_step", "evaluate"]
RECOMMENDED_SIGNATURES = {
    "build_model": "build_model(config: dict)",
    "build_embedding_configs": "build_embedding_configs(config: dict) -> list",
    "build_optimizer": "build_optimizer(model, config: dict)",
    "build_dataloader": "build_dataloader(config: dict, split: str)",
    "train_step": "train_step(step: int, config: dict) -> dict[str, float]",
    "evaluate": "evaluate(config: dict, checkpoint: dict) -> dict[str, float]",
}


class TorchRecModelContractError(ValueError):
    pass


def load_model_module(model_file: str, project_root: Path | None = None) -> ModuleType:
    path = resolve_model_path(model_file, project_root=project_root)
    spec = importlib.util.spec_from_file_location("torchrec_v1_model", path)
    if spec is None or spec.loader is None:
        raise TorchRecModelContractError(f"Could not load model module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_model_path(model_file: str, project_root: Path | None = None) -> Path:
    path = Path(model_file)
    if path.is_absolute() and path.exists():
        return path
    roots = [Path.cwd()]
    if project_root is not None:
        roots.append(project_root)
    roots.append(Path(__file__).resolve().parents[2])
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"TorchRec V1 model file does not exist: {model_file}")


def inspect_model_contract(module: ModuleType) -> dict[str, Any]:
    functions = {}
    for name in [*REQUIRED_FUNCTIONS, *OPTIONAL_FUNCTIONS]:
        value = getattr(module, name, None)
        signature = inspect.signature(value) if callable(value) else None
        functions[name] = {
            "available": callable(value),
            "signature": str(signature) if signature else None,
            "recommended_signature": RECOMMENDED_SIGNATURES[name],
            "signature_compatible": _signature_compatible(name, signature),
        }
    missing_required = [name for name in REQUIRED_FUNCTIONS if not functions[name]["available"]]
    incompatible_required = [
        name
        for name in REQUIRED_FUNCTIONS
        if functions[name]["available"] and not functions[name]["signature_compatible"]
    ]
    return {
        "contract_version": "torchrec-v1",
        "required_functions": REQUIRED_FUNCTIONS,
        "optional_functions": OPTIONAL_FUNCTIONS,
        "recommended_signatures": RECOMMENDED_SIGNATURES,
        "functions": functions,
        "valid": not missing_required and not incompatible_required,
        "missing_required": missing_required,
        "incompatible_required": incompatible_required,
    }


def validate_model_contract(module: ModuleType) -> dict[str, Any]:
    report = inspect_model_contract(module)
    if not report["valid"]:
        problems = []
        if report["missing_required"]:
            problems.append("missing required function(s): " + ", ".join(report["missing_required"]))
        if report["incompatible_required"]:
            problems.append(
                "incompatible required signature(s): " + ", ".join(report["incompatible_required"])
            )
        raise TorchRecModelContractError(
            "TorchRec V1 model.py contract is incomplete. " + "; ".join(problems)
        )
    return report


def write_contract_report(module: ModuleType, run_dir: Path) -> dict[str, Any]:
    report = inspect_model_contract(module)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-model-contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _signature_compatible(function_name: str, signature: inspect.Signature | None) -> bool | None:
    if signature is None:
        return None
    parameters = list(signature.parameters)
    if function_name in {"build_model", "build_embedding_configs"}:
        return len(parameters) >= 1 and parameters[0] == "config"
    if function_name == "build_optimizer":
        return len(parameters) >= 2 and parameters[0] == "model" and parameters[1] == "config"
    if function_name == "build_dataloader":
        return len(parameters) >= 2 and parameters[0] == "config" and parameters[1] == "split"
    if function_name == "train_step":
        return len(parameters) >= 2 and parameters[0] == "step" and parameters[1] == "config"
    if function_name == "evaluate":
        return len(parameters) >= 2 and parameters[0] == "config" and parameters[1] == "checkpoint"
    return True
