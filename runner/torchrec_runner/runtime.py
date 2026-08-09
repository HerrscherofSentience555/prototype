from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_runtime_smoke(
    run_dir: Path,
    *,
    availability: dict[str, Any],
    batch_materialization: dict[str, Any] | None,
    embedding_report: dict[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "schema": "torchrec-v1-runtime-smoke",
        "torch_available": availability.get("torch_available", False),
        "torchrec_available": availability.get("torchrec_available", False),
        "dense_tensor_created": _get(batch_materialization, "torch", "dense_tensor_created"),
        "labels_tensor_created": _get(batch_materialization, "torch", "labels_tensor_created"),
        "keyed_jagged_tensor_created": _get(batch_materialization, "torchrec", "keyed_jagged_tensor_created"),
        "embedding_config_available": _get(
            embedding_report,
            "runtime",
            "torchrec_embedding_config_available",
        ),
        "ready_for_dmp_smoke": bool(
            availability.get("torch_available")
            and availability.get("torchrec_available")
            and _get(batch_materialization, "torchrec", "keyed_jagged_tensor_created")
            and _get(embedding_report, "runtime", "torchrec_embedding_config_available")
        ),
        "fallback_reasons": _fallback_reasons(availability, batch_materialization, embedding_report),
        "next_required_for_dmp_smoke": [
            "torch and torchrec import successfully in the WSL runtime",
            "KeyedJaggedTensor materialization succeeds",
            "EmbeddingBagConfig import/materialization succeeds",
            "torch.distributed process group initializes under torchrun",
            "a real torch.nn.Module is returned by model.build_model(config)",
            "EmbeddingShardingPlanner.collective_plan can run against that model",
        ],
    }
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-runtime-smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _get(payload: dict[str, Any] | None, *path: str):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fallback_reasons(
    availability: dict[str, Any],
    batch_materialization: dict[str, Any] | None,
    embedding_report: dict[str, Any] | None,
) -> list[str]:
    reasons = []
    if not availability.get("torch_available"):
        reasons.append(availability.get("torch_error") or "torch unavailable")
    if not availability.get("torchrec_available"):
        reasons.append(availability.get("torchrec_error") or "torchrec unavailable")
    torch_error = _get(batch_materialization, "torch", "error")
    if torch_error:
        reasons.append(torch_error)
    torchrec_error = _get(batch_materialization, "torchrec", "error")
    if torchrec_error:
        reasons.append(torchrec_error)
    embedding_error = _get(embedding_report, "runtime", "error")
    if embedding_error:
        reasons.append(embedding_error)
    return reasons
