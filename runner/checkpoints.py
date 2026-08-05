from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_checkpoint(
    run_dir: Path,
    step: int,
    backend: str,
    payload: dict[str, Any],
    optimizer: dict[str, Any] | None = None,
    supported: bool = True,
    keep_last: int | None = None,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    optimizer_path = None
    if optimizer is not None:
        optimizer_path = checkpoint_dir / "optimizer.json"
        optimizer_path.write_text(json.dumps(optimizer, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "backend": backend,
        "step": step,
        "created_at": datetime.now().isoformat(),
        "checkpoint_load_supported": supported,
        "model_path": str(model_path),
        "optimizer_path": str(optimizer_path) if optimizer_path else None,
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest = {
        "latest_checkpoint_dir": str(checkpoint_dir),
        "metadata_path": str(checkpoint_dir / "metadata.json"),
        "step": step,
        "backend": backend,
        "checkpoint_load_supported": supported,
    }
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (checkpoint_dir / "_SUCCESS").write_text(datetime.now().isoformat(), encoding="utf-8")
    if keep_last is not None:
        prune_old_checkpoints(run_dir, keep_last=keep_last)
    return checkpoint_dir


def read_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if path.is_dir():
        model_path = path / "model.json"
    else:
        model_path = path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model file does not exist: {model_path}")
    return json.loads(model_path.read_text(encoding="utf-8"))


def prune_old_checkpoints(run_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return
    step_dirs = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.startswith("step-")
    ]
    step_dirs.sort(key=_checkpoint_sort_key, reverse=True)
    for stale_dir in step_dirs[keep_last:]:
        _remove_tree(stale_dir)


def _checkpoint_sort_key(path: Path) -> int:
    suffix = path.name.removeprefix("step-")
    if suffix.isdigit():
        return int(suffix)
    return -1


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
