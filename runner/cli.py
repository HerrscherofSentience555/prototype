from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from prototype.config import PrototypeConfig
from prototype.runner.backends import get_backend
from prototype.runner.capability import write_v1_capability_report
from prototype.runner.data_validation import validate_dataset_if_needed
from prototype.runner.profile import profile_span
from prototype.runner.run_tracking import write_run_summaries
from prototype.runner.telemetry import telemetry_span
from prototype.task_manager import LocalTaskManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = PrototypeConfig.from_yaml_file(Path(args.config))
    run_dir = Path(args.run_dir)

    error_message = None
    try:
        with telemetry_span(run_dir, config=config):
            with profile_span(config, run_dir):
                validate_dataset_if_needed(config, run_dir)
                write_v1_capability_report(config, run_dir)
                backend = get_backend(config)
                exit_code = backend.run(config, run_dir)
    except Exception as exc:
        exit_code = 1
        error_message = f"{type(exc).__name__}: {exc}"
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            log.write("\nRunner failed with an unhandled exception.\n")
            log.write(traceback.format_exc())

    LocalTaskManager().update_terminal_state(run_dir, exit_code, error_message=error_message)
    try:
        write_run_summaries(run_dir)
    except Exception as exc:
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            log.write(f"\nRun summary generation failed: {type(exc).__name__}: {exc}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
