from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode  # noqa: E402
from prototype.runner.backends.dlrm_backend import DLRMBackend, clean_dlrm_log_text  # noqa: E402
from prototype.runner.metrics import append_metric  # noqa: E402


class DLRMBackendTests(unittest.TestCase):
    def test_train_command_contains_expected_parts(self) -> None:
        config = PrototypeConfig(
            backend={"name": "dlrm", "dlrm_root": r"C:\Users\han\Desktop\dlrm"},
            nproc_per_node=2,
            device={"gpu_ids": [0, 1]},
            training={"epochs": 3, "max_steps": 5, "learning_rate": 0.02},
            data={"batch_size": 8},
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\dry-run"))
        script = command[-1]

        self.assertEqual(command[:4], ["wsl", "-d", "Ubuntu-22.04", "bash"])
        self.assertIn("source $HOME/venvs/torchrec17/bin/activate", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1", script)
        self.assertIn("cd /mnt/c/Users/han/Desktop/dlrm", script)
        self.assertIn("torchrun --standalone", script)
        self.assertIn("--log_dir /mnt/c/Users/han/Desktop/prototype/runs/dry-run/logs", script)
        self.assertIn("--redirects 3", script)
        self.assertIn("--tee 3", script)
        self.assertIn("--nproc_per_node=2", script)
        self.assertIn("-m torchrec_dlrm.dlrm_main", script)
        self.assertIn("--epochs 3", script)
        self.assertIn("--batch_size 8", script)
        self.assertIn("--learning_rate 0.02", script)
        self.assertIn("--limit_train_batches 5", script)

    def test_criteo_binary_command_contains_dataset_args(self) -> None:
        config = PrototypeConfig(
            backend={"name": "dlrm"},
            data={
                "format": "criteo_binary",
                "criteo_binary_path": r"C:\Users\han\Desktop\prototype\data\criteo_npy",
                "dataset_name": "criteo_kaggle",
                "batch_size": 16,
                "test_batch_size": 32,
                "mmap_mode": True,
            },
            model={
                "num_embeddings": 1024,
                "embedding_dim": 8,
                "dense_arch_layer_sizes": "8,8",
                "over_arch_layer_sizes": "8,1",
            },
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\real-run"))
        script = command[-1]

        self.assertIn("--dataset_name criteo_kaggle", script)
        self.assertIn("--in_memory_binary_criteo_path /mnt/c/Users/han/Desktop/prototype/data/criteo_npy", script)
        self.assertIn("--test_batch_size 32", script)
        self.assertIn("--mmap_mode", script)
        self.assertIn("--num_embeddings 1024", script)
        self.assertIn("--embedding_dim 8", script)
        self.assertIn("--dense_arch_layer_sizes 8,8", script)
        self.assertIn("--over_arch_layer_sizes 8,1", script)

    def test_evaluate_command_uses_zero_training_batches(self) -> None:
        config = PrototypeConfig(
            mode=RunMode.EVALUATE,
            backend={"name": "dlrm"},
            checkpoint={"load_path": "checkpoint-path"},
            training={"epochs": 3, "max_steps": 5},
            data={"batch_size": 4},
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\eval-run"))
        script = command[-1]

        self.assertIn("--epochs 1", script)
        self.assertIn("--limit_train_batches 0", script)
        self.assertIn("--limit_val_batches 1", script)
        self.assertIn("--limit_test_batches 1", script)
        self.assertIn("--checkpoint_load_path /mnt/c/Users/han/Desktop/prototype/checkpoint-path", script)

    def test_train_command_contains_checkpoint_save_args(self) -> None:
        config = PrototypeConfig(
            backend={"name": "dlrm"},
            checkpoint={"enabled": True, "save_optimizer": True},
            training={"max_steps": 1},
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\ckpt-run"))
        script = command[-1]

        self.assertIn(
            "--checkpoint_save_dir /mnt/c/Users/han/Desktop/prototype/runs/ckpt-run/checkpoints/step-final",
            script,
        )
        self.assertIn("--checkpoint_save_optimizer", script)

    def test_profile_command_contains_dlrm_profile_args(self) -> None:
        config = PrototypeConfig(
            backend={"name": "dlrm"},
            training={"max_steps": 1},
            profile={"enabled": True, "record_shapes": False, "profile_memory": True},
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\profile-run"))
        script = command[-1]

        self.assertIn("--profile_dir /mnt/c/Users/han/Desktop/prototype/runs/profile-run/profiles/dlrm", script)
        self.assertIn("--profile_record_shapes false", script)
        self.assertIn("--profile_memory true", script)

    def test_resume_command_contains_checkpoint_load_and_save_args(self) -> None:
        config = PrototypeConfig(
            mode=RunMode.RESUME,
            backend={"name": "dlrm"},
            checkpoint={"load_path": r"C:\Users\han\Desktop\prototype\runs\a\checkpoints\step-final"},
            training={"max_steps": 1},
        )
        command = DLRMBackend().build_command(config, Path(r"C:\Users\han\Desktop\prototype\runs\resume-run"))
        script = command[-1]

        self.assertIn(
            "--checkpoint_load_path /mnt/c/Users/han/Desktop/prototype/runs/a/checkpoints/step-final",
            script,
        )
        self.assertIn(
            "--checkpoint_save_dir /mnt/c/Users/han/Desktop/prototype/runs/resume-run/checkpoints/step-final",
            script,
        )

    def test_wsl_path_conversion(self) -> None:
        backend = DLRMBackend()
        self.assertEqual(
            backend._to_wsl_path(r"C:\Users\han\Desktop\prototype"),
            "/mnt/c/Users/han/Desktop/prototype",
        )

    def test_evaluation_summary_reads_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            metrics_path = run_dir / "metrics.jsonl"
            append_metric(metrics_path, step=None, metric="val_auc", value=0.1, stage="val")
            append_metric(metrics_path, step=None, metric="test_auc", value=0.2, stage="test")
            append_metric(metrics_path, step=None, metric="log_loss", value=0.3)
            config = PrototypeConfig(
                mode=RunMode.EVALUATE,
                backend={"name": "dlrm"},
                checkpoint={"load_path": "checkpoint-path"},
            )

            DLRMBackend()._write_evaluation_summary(config, run_dir)
            evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))

        self.assertEqual(evaluation["source_checkpoint"], "checkpoint-path")
        self.assertTrue(evaluation["checkpoint_load_supported"])
        self.assertEqual(evaluation["val_auc"], 0.1)
        self.assertEqual(evaluation["test_auc"], 0.2)
        self.assertEqual(evaluation["log_loss"], 0.3)

    def test_finalize_dlrm_checkpoint_writes_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint_dir = run_dir / "checkpoints" / "step-final"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.pt").write_text("placeholder", encoding="utf-8")
            config = PrototypeConfig(
                backend={"name": "dlrm"},
                checkpoint={"enabled": True, "save_dir": str(run_dir / "checkpoints")},
            )

            backend = DLRMBackend()
            backend._finalize_dlrm_checkpoint(config, run_dir)
            backend._write_checkpoint_status(config, run_dir)
            status = json.loads(
                (run_dir / "artifacts" / "checkpoint-status.json").read_text(encoding="utf-8")
            )
            success_exists = (checkpoint_dir / "_SUCCESS").exists()

        self.assertTrue(success_exists)
        self.assertTrue(status["success_marker_exists"])

    def test_clean_dlrm_log_text_normalizes_wsl_proxy_warning(self) -> None:
        raw = "wsl: xxx localhost yyy WSL zzz[default0]:PARAMS: ok\n"

        cleaned = clean_dlrm_log_text(raw)

        self.assertIn("wsl: detected localhost proxy configuration", cleaned)
        self.assertIn("[default0]:PARAMS: ok", cleaned)

    def test_clean_dlrm_log_text_simplifies_mojibake_progress_bar(self) -> None:
        cleaned = clean_dlrm_log_text("Epoch 0:  12%|鈻堚枏| 15/125 [00:00, 73.30it/s]\n")

        self.assertIn("Epoch 0:", cleaned)
        self.assertIn("|...|", cleaned)
        self.assertIn("73.30it/s", cleaned)

    def test_clean_dlrm_log_text_drops_unreadable_noise_line(self) -> None:
        raw = "\ufffd\u5b00\u654420\u617620\u6c75\u3074\u3a5d\n[default0]:PARAMS: ok\n"

        cleaned = clean_dlrm_log_text(raw)

        self.assertNotIn("\ufffd\u5b00\u6544", cleaned)
        self.assertIn("[default0]:PARAMS: ok", cleaned)


if __name__ == "__main__":
    unittest.main()
