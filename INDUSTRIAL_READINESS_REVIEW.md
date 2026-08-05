# Industrial Readiness Review

Review date: 2026-07-28

## Current Status

The project has moved beyond a pure UI prototype. It now has a local job lifecycle,
real DLRM command bridging, Criteo tiny real-data smoke training, single-process DLRM checkpoint
save/load/resume/evaluate validation, parquet preflight validation, run comparison, telemetry,
stale job recovery, and reproducible setup scripts.
It also now maps configured GPU IDs into `CUDA_VISIBLE_DEVICES`, limits local concurrent launches,
exports run bundles, records profile requests, exports runner-level profiler traces when available,
samples GPU/WSL telemetry, supports a custom user model contract, converts parquet CTR data into
DLRM-compatible numpy arrays from CLI or UI, records V1 throughput/step-time metrics, and provides
DLRM checkpoint/profiler patch verification scripts.

It is not yet fully industrial-grade for TorchRec/DLRM production training because the remaining
gaps are now about scale, production sharded checkpointing, broader data adapters, and operational
hardening rather than basic real-data execution.

## Completed Industrial Foundations

- Gradio UI for creating, launching, monitoring, stopping, and inspecting jobs.
- Local run directory contract with `state.json`, `resolved-config.yaml`, logs, metrics, artifacts,
  checkpoints, and profiles.
- DLRM backend can launch local WSL TorchRec DLRM random-data training.
- DLRM backend supports real Criteo data command arguments.
- Criteo Kaggle tiny sample has been preprocessed to TorchRec numpy format.
- DLRM Criteo Kaggle tiny real-data smoke training has succeeded.
- DLRM single-process smoke checkpoints save `model.pt`, optional `optimizer.pt`, and metadata.
- DLRM EVALUATE and RESUME have loaded a saved checkpoint successfully.
- Parquet schema validation and data profiling are available before backend launch.
- Parquet CTR data can be converted into DLRM-compatible dense/sparse/label numpy arrays.
- Create Job can validate and convert parquet data, then populate the DLRM numpy path.
- Stub backend supports checkpoint save, resume, and evaluate-from-checkpoint.
- Custom backend supports user-provided `model.py` smoke experiments with checkpoint/evaluate flow.
- A stricter TorchRec V1 `model.py` contract validator and runner scaffold exist.
- DLRM multi-process sharded checkpoint limitation is explicit and recorded in artifacts.
- Run summaries, metric summaries, lineage files, and Compare Runs UI are available.
- Runner resource telemetry writes `resource-metrics.jsonl` and `resource-summary.json`.
- Telemetry samples `nvidia-smi` GPU utilization/memory when available.
- Telemetry samples visible WSL Python/torchrun process aggregates when available.
- App startup can recover stale active jobs.
- Local launch is guarded by `max_concurrent_jobs=1` by default.
- Logs can export `artifacts/run-artifacts.zip`.
- Profile requests produce `profiles/profile-request.json` and `profiles/runner-profile.json`.
- `torch.profiler` Chrome traces are exported from the runner process when torch is installed in the
  Windows runner environment.
- Patched DLRM runs can export child-process `torch.profiler` Chrome traces per rank.
- `torchrun` rank logs are redirected under each run's `logs/` directory.
- Project-owned training loops emit `step_time_seconds`, `samples_per_second`, and
  `batches_per_second`.
- Project-owned checkpoints write `_SUCCESS` after save completion and support `keep_last` pruning.
- Config includes precision fields and records unmapped backend capability gaps.
- `scripts/check_dlrm_checkpoint_patch.ps1` verifies local DLRM checkpoint patch presence.
- `scripts/check_dlrm_profiler_patch.ps1` verifies local DLRM profiler patch presence.
- Environment check scripts and example configs exist.
- Automated tests cover core platform behavior.

## Not Yet Industrial-Grade

These items block calling the project fully industrial-grade:

- DLRM checkpoint save/load is only validated for single-process smoke runs.
- Multi-process sharded checkpointing is not production-ready yet.
- Full Criteo Kaggle and Criteo 1TB scale runs have not been validated.
- Direct parquet-to-DLRM launch is partly closed in the UI through conversion and auto-filled
  `criteo_binary` path, but it is not yet a background job with progress tracking.
- The internal TorchRec runner scaffold validates the V1 model contract, but the full
  DistributedModelParallel / TrainPipelineSparseDist training loop is not complete yet.
- Embedding placement, GPU cache, and precision settings are recorded but not fully mapped into the
  current DLRM example backend.
- No model export or serving artifact exists.
- No multi-run database or remote experiment tracker exists.
- No CI pipeline is configured.

## Industrial Acceptance Milestone

The project can be called industrial-ready when this full UI-only workflow succeeds:

1. Configure Criteo or business parquet data from the UI.
2. Validate data and inspect a data profile.
3. Launch DLRM real-data training.
4. Save a real model checkpoint.
5. Resume DLRM training from that checkpoint.
6. Evaluate the checkpoint with restored weights.
7. Generate profiler traces and resource telemetry for the run.
8. Compare the training, resume, and evaluation runs.
9. Inspect metrics, logs, telemetry, artifacts, and lineage without editing source code.

## Recommended Next Engineering Order

1. Replace the single-process smoke checkpoint path with Torch Distributed Checkpoint for sharded
   multi-process DLRM.
2. Run a larger Criteo Kaggle subset and document runtime/memory expectations.
3. Add a UI action that converts parquet to DLRM numpy and wires the converted directory into a
   launch without manual command-line use.
4. Map precision, embedding placement, and cache fields into whichever TorchRec backend supports
   them in the selected run mode.
5. Add CI for unit tests and example config validation.
6. Add model export/evaluation report artifacts.
7. Add optional remote experiment tracking or a local SQLite-backed run index.

## Review Verdict

Current verdict: advanced local prototype with several industrial foundations complete.

Industrial-grade verdict: not yet. The decisive remaining blocker is production-grade sharded DLRM
checkpointing and validation on larger real datasets, not the basic real-data/checkpoint smoke path.
