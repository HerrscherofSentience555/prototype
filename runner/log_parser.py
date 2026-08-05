from __future__ import annotations

import re
from typing import Any


_TOTAL_ITERATIONS_RE = re.compile(r"Total number of iterations:\s*(?P<value>\d+)")
_AUROC_RE = re.compile(r"AUROC over (?P<stage>\w+) set:\s*(?P<value>[+-]?\d+(?:\.\d+)?)")
_SAMPLES_RE = re.compile(r"Number of (?P<stage>\w+) samples:\s*(?P<value>\d+)")
_LR_RE = re.compile(r"lr:\s*(?P<step>\d+)\s+(?P<group>\d+)\s+(?P<value>[+-]?\d+(?:\.\d+)?)")
_TQDM_RATE_RE = re.compile(r"(?P<value>[+-]?\d+(?:\.\d+)?)it/s")
_STEP_RE = re.compile(r"\bstep[=:]\s*(?P<value>\d+)")
_LOSS_RE = re.compile(r"\bloss[=:]\s*(?P<value>[+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_LOG_LOSS_RE = re.compile(r"\blog[_ -]?loss[=:]\s*(?P<value>[+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_STEP_TIME_RE = re.compile(r"\bstep_time_seconds[=:]\s*(?P<value>[+-]?\d+(?:\.\d+)?)")
_SAMPLES_PER_SECOND_RE = re.compile(r"\bsamples_per_second[=:]\s*(?P<value>[+-]?\d+(?:\.\d+)?)")
_BATCHES_PER_SECOND_RE = re.compile(r"\bbatches_per_second[=:]\s*(?P<value>[+-]?\d+(?:\.\d+)?)")


def parse_metric_line(line: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    clean_line = _strip_terminal_controls(line)

    total_iterations = _TOTAL_ITERATIONS_RE.search(clean_line)
    if total_iterations:
        step = int(total_iterations.group("value"))
        records.append({"step": step, "metric": "total_iterations", "value": step})

    auroc = _AUROC_RE.search(clean_line)
    if auroc:
        stage = auroc.group("stage")
        records.append(
            {
                "step": None,
                "metric": f"{stage}_auc",
                "value": float(auroc.group("value")),
                "stage": stage,
            }
        )

    samples = _SAMPLES_RE.search(clean_line)
    if samples:
        stage = samples.group("stage")
        records.append(
            {
                "step": None,
                "metric": f"{stage}_samples",
                "value": int(samples.group("value")),
                "stage": stage,
            }
        )

    lr = _LR_RE.search(clean_line)
    if lr:
        records.append(
            {
                "step": int(lr.group("step")),
                "metric": "learning_rate",
                "value": float(lr.group("value")),
                "param_group": int(lr.group("group")),
            }
        )

    loss = _LOSS_RE.search(clean_line)
    if loss:
        step_match = _STEP_RE.search(clean_line)
        records.append(
            {
                "step": int(step_match.group("value")) if step_match else None,
                "metric": "loss",
                "value": float(loss.group("value")),
            }
        )

    log_loss = _LOG_LOSS_RE.search(clean_line)
    if log_loss:
        records.append(
            {
                "step": None,
                "metric": "log_loss",
                "value": float(log_loss.group("value")),
            }
        )

    rate = _TQDM_RATE_RE.search(clean_line)
    if rate:
        records.append(
            {
                "step": None,
                "metric": "throughput_iter_per_sec",
                "value": float(rate.group("value")),
            }
        )

    for regex, metric_name in [
        (_STEP_TIME_RE, "step_time_seconds"),
        (_SAMPLES_PER_SECOND_RE, "samples_per_second"),
        (_BATCHES_PER_SECOND_RE, "batches_per_second"),
    ]:
        match = regex.search(clean_line)
        if match:
            step_match = _STEP_RE.search(clean_line)
            records.append(
                {
                    "step": int(step_match.group("value")) if step_match else None,
                    "metric": metric_name,
                    "value": float(match.group("value")),
                }
            )

    return records


def _strip_terminal_controls(line: str) -> str:
    return line.replace("\r", "").replace("\x1b", "")
