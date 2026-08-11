from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig, RunMode  # noqa: E402
from prototype.runner.backends.stub_backend import StubBackend  # noqa: E402
from prototype.runner.checkpoints import read_checkpoint, write_checkpoint  # noqa: E402
from prototype.task_manager import LocalTaskManager  # noqa: E402


class StubAndTaskManagerTests(unittest.TestCase):
    def test_stub_training_outputs_logs_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = PrototypeConfig(training={"max_steps": 1})
            exit_code = StubBackend().run(config, run_dir)
            metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            log_exists = (run_dir / "train-rank0.log").exists()
            latest_exists = (run_dir / "checkpoints" / "latest.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(log_exists)
        self.assertTrue(latest_exists)
        metric_names = [record["metric"] for record in metrics]
        for expected_metric in [
            "train_loss",
            "auc",
            "step_time_seconds",
            "samples_per_second",
            "batches_per_second",
            "profile_window_active",
            "embedding_lookup_seconds",
            "backward_seconds",
            "optimizer_seconds",
        ]:
            self.assertIn(expected_metric, metric_names)

    def test_stub_resume_loads_checkpoint_and_continues_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_run = Path(tmpdir) / "first"
            second_run = Path(tmpdir) / "second"
            first_run.mkdir()
            second_run.mkdir()
            StubBackend().run(PrototypeConfig(training={"max_steps": 1}), first_run)
            latest = json.loads((first_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
            config = PrototypeConfig(
                mode=RunMode.RESUME,
                checkpoint={"load_path": latest["latest_checkpoint_dir"]},
                training={"max_steps": 1},
            )

            StubBackend().run(config, second_run)
            metrics = [
                json.loads(line)
                for line in (second_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(metrics[0]["step"], 2)

    def test_stub_checkpoint_writes_success_marker_and_prunes_old_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            StubBackend().run(
                PrototypeConfig(
                    training={"max_steps": 3},
                    checkpoint={"keep_last": 2},
                ),
                run_dir,
            )
            checkpoint_dirs = sorted(
                path.name
                for path in (run_dir / "checkpoints").iterdir()
                if path.is_dir() and path.name.startswith("step-")
            )
            success_exists = (run_dir / "checkpoints" / "step-000003" / "_SUCCESS").exists()

        self.assertEqual(checkpoint_dirs, ["step-000003"])
        self.assertTrue(success_exists)

    def test_write_checkpoint_keep_last_prunes_old_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for step in [1, 2, 3]:
                write_checkpoint(
                    run_dir,
                    step=step,
                    backend="stub",
                    payload={"step": step},
                    keep_last=2,
                )
            checkpoint_dirs = sorted(
                path.name
                for path in (run_dir / "checkpoints").iterdir()
                if path.is_dir() and path.name.startswith("step-")
            )

        self.assertEqual(checkpoint_dirs, ["step-000002", "step-000003"])

    def test_read_checkpoint_requires_success_marker_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "step-000001"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.json").write_text('{"step": 1}', encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                read_checkpoint(checkpoint_dir)

    def test_stub_evaluation_outputs_evaluation_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_run = Path(tmpdir) / "train"
            eval_run = Path(tmpdir) / "eval"
            train_run.mkdir()
            eval_run.mkdir()
            StubBackend().run(PrototypeConfig(training={"max_steps": 1}), train_run)
            latest = json.loads((train_run / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
            config = PrototypeConfig(
                mode=RunMode.EVALUATE,
                checkpoint={"load_path": latest["latest_checkpoint_dir"]},
            )
            exit_code = StubBackend().run(config, eval_run)
            evaluation = json.loads((eval_run / "evaluation.json").read_text(encoding="utf-8"))
            metrics = [json.loads(line) for line in (eval_run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertEqual(evaluation["source_checkpoint"], latest["latest_checkpoint_dir"])
        self.assertTrue(evaluation["checkpoint_load_supported"])
        self.assertEqual(evaluation["loaded_checkpoint_step"], 1)
        self.assertEqual([record["metric"] for record in metrics], ["eval_auc", "eval_log_loss"])

    def test_create_job_run_directory_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LocalTaskManager(runs_dir=Path(tmpdir))
            job = manager.create_job(PrototypeConfig(job_name="contract-test"))
            state = json.loads((job.run_dir / "state.json").read_text(encoding="utf-8"))

            self.assertTrue((job.run_dir / "resolved-config.yaml").exists())
            self.assertTrue((job.run_dir / "checkpoints").is_dir())
            self.assertTrue((job.run_dir / "profiles").is_dir())
            self.assertTrue((job.run_dir / "artifacts").is_dir())
            self.assertEqual(state["status"], "CREATED")
            self.assertEqual(state["backend"], "stub")
            self.assertIn("stopped_at", state)
            self.assertIn("force_killed", state)

    def test_stop_completed_job_does_not_overwrite_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LocalTaskManager(runs_dir=Path(tmpdir))
            job = manager.create_job(PrototypeConfig())
            state_path = job.run_dir / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "SUCCEEDED"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            stopped = manager.stop_job(job.job_id)
            after = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(stopped.status, "SUCCEEDED")
        self.assertEqual(after["status"], "SUCCEEDED")
        self.assertIsNone(after["stopped_at"])

    def test_manager_recovers_stale_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LocalTaskManager(runs_dir=Path(tmpdir))
            job = manager.create_job(PrototypeConfig())
            state_path = job.run_dir / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "RUNNING"
            state["pid"] = 99999999
            state["started_at"] = state["created_at"]
            state_path.write_text(json.dumps(state), encoding="utf-8")

            LocalTaskManager(runs_dir=Path(tmpdir))
            recovered = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(recovered["status"], "FAILED")
        self.assertEqual(recovered["recovery_reason"], "missing_process")
        self.assertIn("Recovered stale RUNNING job", recovered["error_message"])

    def test_launch_rejects_when_another_job_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LocalTaskManager(runs_dir=Path(tmpdir), max_concurrent_jobs=1)
            active = manager.create_job(PrototypeConfig(job_name="active"))
            active_state_path = active.run_dir / "state.json"
            active_state = json.loads(active_state_path.read_text(encoding="utf-8"))
            active_state["status"] = "RUNNING"
            active_state["pid"] = 99999999
            active_state_path.write_text(json.dumps(active_state), encoding="utf-8")
            pending = manager.create_job(PrototypeConfig(job_name="pending"))

            with self.assertRaises(RuntimeError):
                manager.launch_job(pending.job_id)

    def test_terminal_state_writes_run_artifact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LocalTaskManager(runs_dir=Path(tmpdir))
            job = manager.create_job(PrototypeConfig())
            (job.run_dir / "train-rank0.log").write_text("hello", encoding="utf-8")
            checkpoint_dir = job.run_dir / "checkpoints" / "step-final"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.pt").write_bytes(b"checkpoint")

            manager.update_terminal_state(job.run_dir, 0)

            bundle_path = job.run_dir / "artifacts" / "run-artifacts.zip"
            with zipfile.ZipFile(bundle_path) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("artifacts/bundle-manifest.json"))

            self.assertTrue(bundle_path.exists())
            self.assertIn("train-rank0.log", names)
            self.assertIn("resolved-config.yaml", names)
            self.assertNotIn("checkpoints/step-final/model.pt", names)
            self.assertEqual(manifest["excluded_files"][0]["path"], "checkpoints/step-final/model.pt")


if __name__ == "__main__":
    unittest.main()
