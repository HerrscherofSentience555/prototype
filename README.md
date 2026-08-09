# TorchRec Prototype

Local-first Gradio prototype for configuring, launching, monitoring, and inspecting TorchRec /
DLRM recommendation jobs on a single machine.

This project is intentionally small and local:

- single user
- single machine
- local filesystem state
- Windows-hosted Gradio UI
- WSL2-hosted TorchRec / DLRM execution
- no Kubernetes
- no external scheduler
- no multi-user production concerns

## Current Status

Implemented:

- Gradio UI with Create Job, Logs, Monitor, and Artifacts tabs
- local run directory contract
- structured YAML config
- task lifecycle state in `state.json`
- `stub` backend for fast UI and lifecycle validation
- `dlrm` backend that launches local TorchRec DLRM through WSL2
- `custom` backend for user-provided `model.py` smoke experiments
- `torchrec_v1` backend that launches the internal TorchRec V1 runner scaffold through WSL2
- DLRM random-data training bridge
- DLRM Criteo Kaggle tiny real-data smoke path
- DLRM single-process checkpoint save/load/resume/evaluate smoke path
- DLRM log parsing into `metrics.jsonl`
- evaluation summary output in `evaluation.json`
- `CUDA_VISIBLE_DEVICES` mapping from configured GPU IDs
- WSL/GPU telemetry sampling into `resource-metrics.jsonl`
- single active job guard to avoid local GPU contention
- run artifact bundle export as `artifacts/run-artifacts.zip`
- profile request, runner profile artifacts, and DLRM child-process profiler trace support
- parquet schema validation and parquet-to-DLRM-numpy conversion CLI
- UI-driven parquet validation/conversion from the Create Job tab
- V1 model.py contract scaffolding for a future internal TorchRec runner
- step time and throughput metrics for project-owned runners
- V1 stage timing metrics and profile window activity metrics for project-owned runners
- explicit V1 capability reports for GPU placement/cache/precision mapping
- V1 TorchRec batch, embedding, runtime, and sharding-readiness artifacts
- checkpoint `_SUCCESS` markers and `keep_last` pruning for project-owned checkpoints
- local DLRM checkpoint patch verification script
- local DLRM profiler patch verification script
- local TorchRec V1 DMP readiness verification script for the WSL runtime
- improved Stop Job behavior and stop metadata

Important limitation:

- DLRM checkpointing is currently validated for single-process smoke runs through local
  `model.pt` / `optimizer.pt` files. Multi-process sharded production checkpointing still needs
  Torch Distributed Checkpoint or an equivalent production checkpoint path.
- Full Criteo Kaggle and Criteo 1TB scale runs have not been validated yet. The current real-data
  path is a tiny Criteo smoke dataset intended to prove the end-to-end contract quickly.
- Direct parquet training still uses a conversion step before DLRM launch.
- The internal `torchrec_v1` backend has readiness/materialization layers, but it does not yet
  perform a real `DistributedModelParallel` / `TrainPipelineSparseDist` training loop. The current
  Codex environment cannot execute WSL checks because `wsl.exe` returns `WSL/Service/E_ACCESSDENIED`;
  run the DMP readiness script from your own PowerShell session to verify the local WSL runtime.

## Project Layout

```text
prototype/
  app.py                         Gradio app entrypoint
  config.py                      Pydantic configuration model
  task_manager.py                local task lifecycle and subprocess management
  requirements.txt               Windows UI/runtime dependencies
  ui/
    create_tab.py                job creation and config preview
    logs_tab.py                  launcher/train logs and command viewer
    monitor_tab.py               metrics table and plots
    artifacts_tab.py             state/config/evaluation/artifact viewer and Stop Job
  runner/
    cli.py                       subprocess runner entrypoint
    metrics.py                   JSONL metric writer
    log_parser.py                DLRM log-to-metrics parser
    backends/
      stub_backend.py            simulated backend
      dlrm_backend.py            WSL2 TorchRec DLRM backend
      custom_backend.py          user-provided Python model contract backend
      torchrec_v1_backend.py     WSL2 launcher for the internal TorchRec V1 runner scaffold
    parquet_converter.py         parquet to Criteo-style numpy conversion
    convert_parquet.py           converter CLI entrypoint
    torchrec_runner/
      contract.py                V1 model.py contract validator
      entry.py                   internal TorchRec runner scaffold
      sharding.py                TorchRec sharding planner readiness artifact
  scripts/
    check_windows_env.ps1        Windows environment check
    check_wsl_dlrm.sh            WSL TorchRec/DLRM environment check
    check_dlrm_checkpoint_patch.ps1
                                  local DLRM checkpoint patch check
    check_dlrm_profiler_patch.ps1
                                  local DLRM profiler patch check
    check_torchrec_v1_dmp_readiness.ps1
                                  WSL TorchRec V1 DMP readiness check
  patches/
    README.md                    local DLRM patch notes
  runs/                          generated local job outputs
```

## Environments

The project is designed to be cloned to different machines. Keep shared defaults in git, and keep
machine-specific paths in `local_settings.yaml`.

Typical setup:

1. Windows Python virtual environment for the Gradio prototype UI.
2. WSL2 Ubuntu environment for real TorchRec / DLRM execution.
3. Optional Linux-native mode when the UI and TorchRec runtime are both started from Linux.

Create local settings:

```powershell
Copy-Item local_settings.example.yaml local_settings.yaml
notepad local_settings.yaml
```

Edit only the values that are local to your machine:

```yaml
runtime:
  platform: windows_wsl
  wsl_distribution: Ubuntu-22.04
  python_env: ~/venvs/torchrec17

paths:
  dlrm_root: /mnt/c/Users/<your-name>/Desktop/dlrm
  criteo_binary_path: data/criteo_kaggle_sample_npy
```

`local_settings.yaml` is ignored by git. Other users should create their own copy instead of
editing committed source files.

## Install And Start The UI

From PowerShell:

```powershell
cd <parent-folder-of-your-clone>
python -m venv prototype\.venv
prototype\.venv\Scripts\Activate.ps1
pip install -r prototype\requirements.txt
python -m prototype.app
```

If the virtual environment already exists:

```powershell
cd <parent-folder-of-your-clone>
prototype\.venv\Scripts\Activate.ps1
python -m prototype.app
```

Open the local Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Verify WSL / DLRM Environment

From PowerShell:

```powershell
wsl -l -v
```

Expected:

```text
Ubuntu-22.04    Running or Stopped    2
```

Check TorchRec / DLRM runtime:

```powershell
wsl -d Ubuntu-22.04 bash -lc "source ~/venvs/torchrec17/bin/activate; cd /mnt/c/Users/<your-name>/Desktop/dlrm; which torchrun; python -c 'import torchrec; print(\"torchrec ok\")'"
```

Expected:

```text
/home/han/venvs/torchrec17/bin/torchrun
torchrec ok
```

## Create Job Fields

Key fields in the Create Job tab:

- `Mode`: `COLD_START`, `RESUME`, or `EVALUATE`
- `Backend`: `stub`, `dlrm`, `custom`, or `torchrec_v1`
- `Runtime Platform`: `windows_wsl` for Windows UI + WSL training, or `linux_native` for Linux shell training
- `DLRM Root`: local DLRM repo path as seen by the selected runtime
- `Python Env`: Python virtual environment as seen by the selected runtime
- `WSL Distribution`: WSL distro name, used only by `windows_wsl`
- `Data Format`: `random`, `criteo_binary`, `synthetic_multihot`, or `parquet`
- `Batch Size`: DLRM/stub batch size
- `Epochs`: training epochs
- `Max Steps`: maps to DLRM `--limit_train_batches`
- `Learning Rate`: maps to DLRM `--learning_rate`
- `Processes per Node`: maps to `torchrun --nproc_per_node`
- `GPU IDs`: maps to `CUDA_VISIBLE_DEVICES`
- `Embedding Placement` and `Cache Load Factor`: recorded in config and capability report
- precision fields: recorded in config and capability report
- `Checkpoint Load Path`: required for `RESUME` and `EVALUATE`
- `Save Checkpoints`: saves DLRM single-process smoke checkpoints when using the patched local
  DLRM path
- `Profile Enabled`: writes profile request/runner artifacts and passes DLRM profiler arguments
  when using the patched local DLRM entrypoint

The Create Job tab also provides:

- `Validate Parquet`: validates parquet split paths and schema, then displays a JSON profile
- `Convert Parquet`: converts parquet splits to DLRM numpy arrays, switches `Data Format` to
  `criteo_binary`, and fills `Criteo Binary Path` with the output directory

## Environment Tab

Use this tab before launching jobs on a newly cloned machine.

- `Refresh Settings`: shows whether the app is reading `local_settings.yaml` or the example template.
- `Create local_settings.yaml`: copies `local_settings.example.yaml` if the local file does not exist.
- `Run Environment Checks`: checks WSL or Linux shell access, the configured Python environment,
  `torch` / `torchrec` imports, DLRM root, and the bundled Criteo sample numpy files.

When a check fails, fix `local_settings.yaml` first, then refresh the page or rerun the checks.

## Minimal Stub Validation

Use this first to confirm the UI and local task lifecycle.

Create Job:

```text
Backend: stub
Mode: COLD_START
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

Click:

```text
Validate Config
Launch Job
```

Expected:

- Launch Result shows job id, status, backend, PID, and run directory
- Logs tab shows `train-rank0.log`
- Monitor tab shows `train_loss` and `auc`
- Artifacts tab shows `state.json` with `status: SUCCEEDED`
- Logs tab can download `artifacts/run-artifacts.zip` after completion

## Minimal DLRM Training Validation

Create Job:

```text
Backend: dlrm
Mode: COLD_START
DLRM Root: /mnt/c/Users/<your-name>/Desktop/dlrm
Python Env: ~/venvs/torchrec17
WSL Distribution: Ubuntu-22.04
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
Learning Rate: 0.01
```

Expected:

- `command.json` contains `backend_command`
- `backend_command` invokes `wsl`, activates the venv, changes into DLRM root, and runs `torchrun`
- `train-rank0.log` contains real DLRM output
- `metrics.jsonl` contains parsed metrics such as `val_auc` and `test_auc`
- `state.json.status` becomes `SUCCEEDED`

## Custom Model Validation

Use the custom backend when you want to test a local Python model contract before integrating a full
TorchRec training loop.

Example config:

```text
examples/custom-model-smoke.yaml
```

Expected model file contract:

```python
def train_step(step: int, config: dict) -> dict[str, float]:
    ...

def evaluate(config: dict, checkpoint: dict) -> dict[str, float]:
    ...
```

Expected:

- `train-rank0.log` records custom backend execution
- `metrics.jsonl` contains metrics returned by `train_step` or `evaluate`
- `metrics.jsonl` also contains `step_time_seconds`, `samples_per_second`, and
  `batches_per_second` for training steps
- `artifacts/custom-model-contract.json` records the loaded model path and supported functions
- checkpoint files are created for COLD_START/RESUME when checkpoint saving is enabled

## TorchRec V1 Model Contract Validation

The internal TorchRec runner scaffold validates a stricter V1 `model.py` contract:

```python
def build_model(config: dict):
    ...

def build_embedding_configs(config: dict) -> list:
    ...
```

Optional functions:

```text
build_optimizer
build_dataloader
train_step
evaluate
```

Example:

```text
examples/models/torchrec_v1_model.py
```

Validation artifacts:

```text
artifacts/torchrec-model-contract.json
artifacts/torchrec-data-plan.json
artifacts/torchrec-batch-materialization.json
artifacts/torchrec-embedding-configs.json
artifacts/torchrec-runtime-smoke.json
artifacts/torchrec-sharding-plan-readiness.json
artifacts/torchrec-training-plan.json
artifacts/torchrec-runner-status.json
artifacts/torchrec-v1-capability-report.json
```

## Minimal TorchRec V1 Backend Validation

Create Job:

```text
Backend: torchrec_v1
Mode: COLD_START
Model File: examples/models/torchrec_v1_model.py
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

Expected:

- `command.json` invokes WSL, activates the WSL TorchRec env, and runs
  `torchrun -m prototype.runner.torchrec_runner.entry`
- `artifacts/torchrec-model-contract.json` exists
- `artifacts/torchrec-data-plan.json` exists and records dense/sparse/label batch schema
- `artifacts/torchrec-batch-materialization.json` exists and records whether real torch tensors
  and TorchRec `KeyedJaggedTensor` were created
- `artifacts/torchrec-embedding-configs.json` exists and records embedding config descriptions
- `artifacts/torchrec-runtime-smoke.json` exists and reports whether the run is ready for a DMP
  smoke test
- `artifacts/torchrec-sharding-plan-readiness.json` exists and reports planner import/topology
  readiness without falsely claiming a collective sharding plan
- `artifacts/torchrec-training-plan.json` exists and marks DMP/TrainPipeline steps as planned
- `artifacts/torchrec-runner-status.json` exists
- `metrics.jsonl` contains minimal-loop metrics
- checkpoint `_SUCCESS` is written when checkpoint saving is enabled

## Parquet Conversion Validation

Use this path for business-style CTR parquet data before launching DLRM.

```powershell
cd <path-to-your-prototype-clone>
python -m prototype.runner.convert_parquet --config examples\parquet-conversion-smoke.yaml --output-dir data\converted_criteo_npy
```

The config must use `data.format=parquet` and point at a schema YAML with:

```text
label
dense_features
sparse_features
```

Expected:

- `<split>_dense.npy`, `<split>_sparse.npy`, and `<split>_labels.npy` are created
- `conversion-manifest.json` records source files, schema, split names, row counts, and output files
- the output directory can be used as `Criteo Binary Path` with `Data Format=criteo_binary`

## Local DLRM Patch Check

Run:

```powershell
cd <path-to-your-prototype-clone>
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_profiler_patch.ps1
```

Expected:

```text
DLRM checkpoint patch is present.
DLRM profiler patch is present.
```

## Profile Validation

Create a short stub or DLRM job with `Profile Enabled=true`.

Expected:

- `profiles/profile-request.json` exists
- `profiles/runner-profile.json` exists
- `profiles/trace.json` exists when `torch.profiler` is available in the Windows runner environment
- `runner-profile.json.profile_trace_error` explains the fallback when torch is not installed
- DLRM runs pass `--profile_dir`, `--profile_record_shapes`, and `--profile_memory`
- patched DLRM runs can create `profiles/dlrm/rank<N>-trace.json`
- torchrun rank logs are redirected under `logs/`

## Throughput And Step-Time Validation

Run a stub or custom job with at least one training step.

Expected:

- `metrics.jsonl` contains `step_time_seconds`
- `metrics.jsonl` contains `samples_per_second`
- `metrics.jsonl` contains `batches_per_second`
- `metrics.jsonl` contains stage timing metrics such as `embedding_lookup_seconds`,
  `backward_seconds`, and `optimizer_seconds`
- `metrics.jsonl` contains `profile_window_active`
- Monitor tab shows `Throughput` and `Step Time` plots after refreshing metrics
- Monitor tab shows `Stage Timing` after refreshing metrics

## V1 Capability Report Validation

Every launched job writes:

```text
artifacts/v1-capability-report.json
```

Expected:

- requested GPU IDs, embedding placement, cache load factor, and precision fields are recorded
- mapped fields show what the selected backend actually applies
- `not_yet_mapped` explains GPU Cache, precision, DMP, or internal-runner gaps when applicable

## V1 Runner Artifact Views

The Artifacts tab displays these JSON artifacts directly:

- `artifacts/v1-capability-report.json`
- `artifacts/torchrec-model-contract.json`
- `artifacts/torchrec-data-plan.json`
- `artifacts/torchrec-batch-materialization.json`
- `artifacts/torchrec-embedding-configs.json`
- `artifacts/torchrec-runtime-smoke.json`
- `artifacts/torchrec-sharding-plan-readiness.json`
- `artifacts/torchrec-training-plan.json`

Use these views to verify whether a run is still on the minimal loop or has moved to real
TorchRec DMP / TrainPipeline execution.

## TorchRec V1 DMP Readiness Check

Before claiming real `DistributedModelParallel` execution is ready, run this from PowerShell:

```powershell
cd <path-to-your-prototype-clone>
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

Expected:

- the script exits with code `0`
- JSON output reports `true` for torch, torchrec, DMP, sharding planner, train pipeline, and
  distributed checkpoint checks
- if it exits non-zero, the JSON `errors` array is the next blocker to solve before a real DMP
  smoke run

Optional environment overrides:

```powershell
$env:TORCHREC_WSL_DISTRO = "Ubuntu-22.04"
$env:TORCHREC_PYTHON_ENV = "~/venvs/torchrec17"
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

## GPU And WSL Telemetry Validation

Run a short job and inspect:

```text
runs/<job_id>/resource-metrics.jsonl
runs/<job_id>/artifacts/resource-summary.json
```

Expected:

- records include `gpu_telemetry_available`
- records include GPU utilization and memory fields when `nvidia-smi` is available
- records include `wsl_telemetry_available`
- WSL process fields are populated while Python/torchrun processes are visible inside WSL

## Minimal DLRM Real-Data Validation

Preprocessed tiny Criteo sample:

```text
data/criteo_kaggle_sample_npy
```

Create Job:

```text
Backend: dlrm
Mode: COLD_START
Data Format: criteo_binary
Criteo Binary Path: data/criteo_kaggle_sample_npy
Dataset Name: criteo_kaggle
Batch Size: 4
Test Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

Expected:

- `state.json.status` becomes `SUCCEEDED`
- `train-rank0.log` contains `AUROC over val set` and `AUROC over test set`
- `metrics.jsonl` contains `val_auc`, `test_auc`, `val_samples`, and `test_samples`
- `checkpoints/step-final/model.pt` is created when checkpoint saving is enabled
- `checkpoints/step-final/_SUCCESS` is created after the DLRM smoke checkpoint is finalized

Validated smoke run:

```text
runs/20260728-150552-7e92c28f
```

## Minimal DLRM Checkpoint Validation

Create Job:

```text
Backend: dlrm
Mode: EVALUATE
Data Format: criteo_binary
Criteo Binary Path: data/criteo_kaggle_sample_npy
Dataset Name: criteo_kaggle
Batch Size: 4
Max Steps: 1
Processes per Node: 1
Checkpoint Load Path: runs/<job_id>/checkpoints/step-final
```

Expected:

- empty Checkpoint Load Path is rejected
- non-empty Checkpoint Load Path allows launch
- `command.json` contains `--limit_train_batches 0`
- `command.json` contains `--checkpoint_load_path`
- `train-rank0.log` contains `Loaded checkpoint from`
- `evaluation.json` is created
- `evaluation.json.source_checkpoint` matches the UI checkpoint path
- `evaluation.json.checkpoint_load_supported` is `true`
- `metrics.jsonl` contains parsed evaluation metrics when DLRM prints them

Validated smoke runs:

```text
EVALUATE: runs/20260728-150905-887685af
RESUME:   runs/20260728-150919-da311650
```

## Tabs

### Create Job

Builds a `PrototypeConfig`, previews it as YAML, creates a run directory, and launches the selected
backend.

### Environment

Shows the active local settings file and runs portability checks for the selected machine.

### Logs

Shows:

- job dropdown as `<job_id> [<status>]`
- `launcher.log`
- `train-rank0.log`
- `command.json`
- configurable tail line count

### Monitor

Shows:

- recent metric records from `metrics.jsonl`
- Train Loss line plot
- AUC line plot
- clear empty state when no metrics are available

### Artifacts

Shows:

- run directory
- `state.json`
- `resolved-config.yaml`
- `evaluation.json`
- checkpoint/profile/artifact file lists
- Stop Job action

## Run Directory Contract

Each job writes to:

```text
runs/<job_id>/
  resolved-config.yaml
  state.json
  launcher.log
  train-rank0.log
  logs/
  metrics.jsonl
  evaluation.json
  command.json
  checkpoints/
    step-000001/
      _SUCCESS
  profiles/
  artifacts/
```

Some files are mode-dependent. For example, `evaluation.json` may be absent for a training-only job.

## Task States

Common states:

```text
CREATED
LAUNCHING
RUNNING
STOPPING
STOPPED
SUCCEEDED
FAILED
```

`state.json` also records fields such as:

- backend
- command
- cwd
- pid
- error_message
- timestamps
- duration_seconds
- exit_code
- stop metadata

## Stop Job Behavior

Stop Job:

- does not overwrite already completed `SUCCEEDED` or `FAILED` jobs
- moves active jobs to `STOPPING`
- sends a graceful termination signal
- force kills the process tree if needed
- records `stopped_at`, `stop_reason`, `stop_signal`, `force_killed`, and `stop_error`

## Quality Checks

Run:

```powershell
cd <path-to-your-prototype-clone>
python -m compileall config.py task_manager.py runner ui
python -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_profiler_patch.ps1
```

If using the project venv:

```powershell
cd <path-to-your-prototype-clone>
.\.venv\Scripts\python.exe -m compileall config.py task_manager.py runner ui
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'gradio'`

Activate the project venv and install dependencies:

```powershell
cd <parent-folder-of-your-clone>
prototype\.venv\Scripts\Activate.ps1
pip install -r prototype\requirements.txt
```

### `WSL_E_DISTRO_NOT_FOUND`

Check the distro name:

```powershell
wsl -l -v
```

Then set `WSL Distribution` in the UI to the exact distro name.

### DLRM job fails before training

Check:

- `command.json`
- `launcher.log`
- `train-rank0.log`
- WSL distro name
- `Python Env`
- `DLRM Root`

### Monitor has no DLRM metrics

Check whether `train-rank0.log` contains parseable lines such as:

```text
AUROC over val set: ...
AUROC over test set: ...
Number of val samples: ...
```

If the log has no metric-like lines, `metrics.jsonl` may be empty even though the job ran.

### DLRM checkpoint load fails

Check:

- `Checkpoint Load Path` points to a checkpoint directory containing `model.pt`
- the new run uses the same DLRM model shape as the checkpoint run
- `Processes per Node` is `1` for the current smoke checkpoint path
- `train-rank0.log` contains the underlying `torch.load` or `load_state_dict` error

Current checkpoint support is intended for single-process smoke validation. Multi-process sharded
checkpointing still needs a production Torch Distributed Checkpoint implementation.
