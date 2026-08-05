from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prototype.config import PrototypeConfig
from prototype.runner.torchrec_runner.data import CRITEO_SPARSE_FEATURES


def materialize_embedding_configs(module, config: PrototypeConfig, run_dir: Path) -> dict[str, Any]:
    raw_configs = module.build_embedding_configs(config.model_dump(mode="json"))
    materialized = [_describe_embedding_config(item) for item in raw_configs or []]
    if not materialized:
        materialized = _default_embedding_descriptions(config)
    report = {
        "schema": "torchrec-v1-embedding-configs",
        "source": "model.py" if raw_configs else "default_criteo_like",
        "count": len(materialized),
        "embedding_dim": config.model.embedding_dim or 8,
        "num_embeddings": config.model.num_embeddings or 1024,
        "configs": materialized,
        "runtime": _runtime_status(),
    }
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "torchrec-embedding-configs.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _describe_embedding_config(item) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return {
        "name": getattr(item, "name", str(item)),
        "embedding_dim": getattr(item, "embedding_dim", None),
        "num_embeddings": getattr(item, "num_embeddings", None),
        "feature_names": list(getattr(item, "feature_names", []) or []),
        "runtime_type": type(item).__name__,
    }


def _default_embedding_descriptions(config: PrototypeConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": f"t_{feature_name}",
            "embedding_dim": config.model.embedding_dim or 8,
            "num_embeddings": config.model.num_embeddings or 1024,
            "feature_names": [feature_name],
            "runtime_type": "planned_EmbeddingBagConfig",
        }
        for feature_name in CRITEO_SPARSE_FEATURES
    ]


def _runtime_status() -> dict[str, Any]:
    try:
        from torchrec.modules.embedding_configs import EmbeddingBagConfig  # noqa: F401

        return {"torchrec_embedding_config_available": True}
    except Exception as exc:
        return {
            "torchrec_embedding_config_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
