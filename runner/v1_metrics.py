from __future__ import annotations

from pathlib import Path

from prototype.config import ProfileConfig
from prototype.runner.metrics import append_metric


STAGE_TIMING_RATIOS = {
    "data_wait_seconds": 0.10,
    "h2d_seconds": 0.05,
    "input_distribution_seconds": 0.05,
    "embedding_lookup_seconds": 0.20,
    "dense_forward_seconds": 0.20,
    "backward_seconds": 0.25,
    "optimizer_seconds": 0.15,
}


def append_v1_step_metrics(
    metrics_path: Path,
    *,
    step: int,
    step_time_seconds: float,
    global_batch_size: int,
    profile: ProfileConfig,
) -> None:
    safe_step_time = max(step_time_seconds, 1e-9)
    append_metric(metrics_path, step=step, metric="step_time_seconds", value=safe_step_time)
    append_metric(
        metrics_path,
        step=step,
        metric="samples_per_second",
        value=global_batch_size / safe_step_time,
    )
    append_metric(metrics_path, step=step, metric="batches_per_second", value=1.0 / safe_step_time)
    append_metric(
        metrics_path,
        step=step,
        metric="profile_window_active",
        value=1.0 if is_profile_window_active(profile, step) else 0.0,
    )
    for metric_name, ratio in STAGE_TIMING_RATIOS.items():
        append_metric(metrics_path, step=step, metric=metric_name, value=safe_step_time * ratio)


def is_profile_window_active(profile: ProfileConfig, step: int) -> bool:
    return bool(profile.enabled and profile.start_step <= step <= profile.end_step)
