from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.runner.log_parser import parse_metric_line  # noqa: E402


class LogParserTests(unittest.TestCase):
    def test_parse_dlrm_evaluation_metrics(self) -> None:
        self.assertEqual(
            parse_metric_line("AUROC over val set: 0.12."),
            [{"step": None, "metric": "val_auc", "value": 0.12, "stage": "val"}],
        )
        self.assertEqual(
            parse_metric_line("Number of test samples: 4"),
            [{"step": None, "metric": "test_samples", "value": 4, "stage": "test"}],
        )

    def test_parse_iteration_loss_lr_and_rate(self) -> None:
        self.assertEqual(
            parse_metric_line("Total number of iterations: 7"),
            [{"step": 7, "metric": "total_iterations", "value": 7}],
        )
        self.assertEqual(
            parse_metric_line("step=3 loss=0.25"),
            [{"step": 3, "metric": "loss", "value": 0.25}],
        )
        self.assertEqual(
            parse_metric_line("lr: 2 0 0.010000"),
            [{"step": 2, "metric": "learning_rate", "value": 0.01, "param_group": 0}],
        )
        self.assertEqual(
            parse_metric_line("Epoch 0: 100%|x| 1/1 [00:00<00:00, 15.97it/s]"),
            [{"step": None, "metric": "throughput_iter_per_sec", "value": 15.97}],
        )

    def test_parse_log_loss(self) -> None:
        self.assertEqual(
            parse_metric_line("log_loss: 0.456"),
            [{"step": None, "metric": "log_loss", "value": 0.456}],
        )

    def test_parse_v1_timing_and_throughput_metrics(self) -> None:
        self.assertEqual(
            parse_metric_line("step=3 step_time_seconds=0.25 samples_per_second=128.0 batches_per_second=4.0"),
            [
                {"step": 3, "metric": "step_time_seconds", "value": 0.25},
                {"step": 3, "metric": "samples_per_second", "value": 128.0},
                {"step": 3, "metric": "batches_per_second", "value": 4.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
