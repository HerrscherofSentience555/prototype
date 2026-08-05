from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from prototype.config import PrototypeConfig


@contextmanager
def profile_span(config: PrototypeConfig, run_dir: Path) -> Iterator[None]:
    if not config.profile.enabled:
        yield
        return

    profiles_dir = run_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    profiler, trace_path, profiler_error = _start_torch_profiler(config, profiles_dir)
    request = {
        "enabled": True,
        "start_step": config.profile.start_step,
        "end_step": config.profile.end_step,
        "record_shapes": config.profile.record_shapes,
        "profile_memory": config.profile.profile_memory,
        "status": "torch_profiler_active" if profiler is not None else "runner_wall_time_only",
        "note": (
            "When torch.profiler is available this span exports a Chrome trace for work executed "
            "inside the runner process. Child torchrun processes still require instrumentation "
            "inside the active DLRM training loop."
        ),
        "trace_path": str(trace_path) if trace_path else None,
        "profiler_error": profiler_error,
    }
    (profiles_dir / "profile-request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    started = time.perf_counter()
    started_at = datetime.now().isoformat()
    try:
        if profiler is None:
            yield
        else:
            with profiler:
                yield
    finally:
        ended_at = datetime.now().isoformat()
        trace_supported = False
        trace_error = profiler_error
        if profiler is not None and trace_path is not None:
            try:
                profiler.export_chrome_trace(str(trace_path))
                trace_supported = trace_path.exists()
            except Exception as exc:
                trace_error = f"{type(exc).__name__}: {exc}"
        summary = {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": time.perf_counter() - started,
            "profile_trace_supported": trace_supported,
            "trace_path": str(trace_path) if trace_path else None,
            "profile_trace_error": trace_error,
            "profile_trace_note": request["note"],
        }
        (profiles_dir / "runner-profile.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _start_torch_profiler(config: PrototypeConfig, profiles_dir: Path):
    trace_path = profiles_dir / "trace.json"
    try:
        import torch
    except Exception as exc:
        return None, trace_path, f"torch.profiler unavailable: {type(exc).__name__}: {exc}"
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=config.profile.record_shapes,
            profile_memory=config.profile.profile_memory,
            with_stack=False,
        )
        return profiler, trace_path, None
    except Exception as exc:
        return None, trace_path, f"torch.profiler setup failed: {type(exc).__name__}: {exc}"
