from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


MAXIMIZE_METRICS = {"auc", "val_auc", "test_auc", "eval_auc", "throughput", "samples_per_second"}


def write_run_summaries(run_dir: Path) -> None:
    config = _read_yaml(run_dir / "resolved-config.yaml")
    state = _read_json(run_dir / "state.json")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    metrics_summary = summarize_metrics(metrics)
    lineage = {
        "job_id": state.get("job_id"),
        "mode": config.get("mode"),
        "parent_checkpoint": (config.get("checkpoint") or {}).get("load_path"),
        "parent_run_id": _infer_parent_run_id((config.get("checkpoint") or {}).get("load_path")),
    }
    summary = {
        "job_id": state.get("job_id"),
        "job_name": state.get("job_name"),
        "status": state.get("status"),
        "backend": state.get("backend"),
        "mode": config.get("mode"),
        "data_format": (config.get("data") or {}).get("format"),
        "dataset_name": (config.get("data") or {}).get("dataset_name"),
        "created_at": state.get("created_at"),
        "started_at": state.get("started_at"),
        "ended_at": state.get("ended_at"),
        "duration_seconds": state.get("duration_seconds"),
        "metrics": metrics_summary,
        "lineage": lineage,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics-summary.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compare_runs(runs_dir: Path, job_ids: list[str], sort_metric: str = "auc") -> dict[str, Any]:
    rows = []
    for job_id in job_ids:
        summary_path = runs_dir / job_id / "summary.json"
        if not summary_path.exists():
            write_run_summaries(runs_dir / job_id)
        summary = _read_json(summary_path)
        metric_record = (summary.get("metrics") or {}).get(sort_metric)
        rows.append(
            {
                "job_id": job_id,
                "status": summary.get("status"),
                "backend": summary.get("backend"),
                "mode": summary.get("mode"),
                "data_format": summary.get("data_format"),
                "duration_seconds": summary.get("duration_seconds"),
                sort_metric: metric_record.get("best") if metric_record else None,
            }
        )
    best = _select_best(rows, sort_metric)
    return {"sort_metric": sort_metric, "best_run": best, "runs": rows}


def export_comparison_csv(comparison: dict[str, Any], output_path: Path) -> None:
    rows = comparison.get("runs") or []
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in metrics:
        metric = record.get("metric")
        value = record.get("value")
        if metric is None or value is None:
            continue
        current = summary.setdefault(
            metric,
            {"count": 0, "latest": None, "best": None, "best_step": None},
        )
        current["count"] += 1
        current["latest"] = value
        if current["best"] is None or _is_better(metric, value, current["best"]):
            current["best"] = value
            current["best_step"] = record.get("step")
    return summary


def _is_better(metric: str, candidate: float, current: float) -> bool:
    if metric in MAXIMIZE_METRICS or "auc" in metric.lower():
        return candidate > current
    return candidate < current


def _select_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    populated = [row for row in rows if row.get(metric) is not None]
    if not populated:
        return None
    reverse = metric in MAXIMIZE_METRICS or "auc" in metric.lower()
    return sorted(populated, key=lambda row: row[metric], reverse=reverse)[0]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _infer_parent_run_id(checkpoint_path: str | None) -> str | None:
    if not checkpoint_path:
        return None
    parts = Path(checkpoint_path).parts
    if "runs" not in parts:
        return None
    index = parts.index("runs")
    if index + 1 < len(parts):
        return parts[index + 1]
    return None
