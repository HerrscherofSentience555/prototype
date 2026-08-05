# TorchRec Prototype Development Plan

Last updated: 2026-07-23

## Overall Goal

Build a local-first, single-machine TorchRec / DLRM prototype that lets a user configure,
launch, monitor, and inspect recommendation model jobs from a browser UI.

The immediate goal is to move the project from:

- a runnable Gradio prototype with a simulated runner

to:

- a runnable Gradio prototype that can trigger the already validated local TorchRec / DLRM
  execution path through WSL2.

Out of scope for the current prototype:

- Kubernetes
- cluster schedulers
- multi-user production platform concerns
- production-grade distributed orchestration
- real dataset experimentation beyond the first local validation path

## Current Baseline

The project already has:

- Gradio app shell
- configuration model based on Pydantic
- local task manager
- run directory creation
- task state persistence
- subprocess launch path
- stub training runner
- stub evaluation runner
- basic logs, metrics, and artifacts UI tabs

The biggest missing piece is backend realism. The current runner simulates progress; it does not
yet invoke the user's local `torchrec_dlrm/dlrm_main.py`.

## Phase 1: Stabilize The Existing Prototype Foundation

Goal: make the current skeleton more reliable before integrating a real backend.

### 1.1 Standardize The Run Directory Contract

Use a stable structure:

```text
runs/<job_id>/
  resolved-config.yaml
  state.json
  launcher.log
  train-rank0.log
  metrics.jsonl
  evaluation.json
  command.json
  checkpoints/
  profiles/
  artifacts/
```

Add `command.json` to record:

- actual command
- working directory
- backend name
- WSL distribution
- Python environment
- DLRM root
- launch timestamp

### 1.2 Strengthen Task State

Current states:

```text
CREATED
RUNNING
SUCCEEDED
FAILED
STOPPED
```

Recommended states:

```text
CREATED
LAUNCHING
RUNNING
STOPPING
STOPPED
SUCCEEDED
FAILED
```

Extend `state.json` with:

```json
{
  "backend": "stub",
  "command": null,
  "cwd": null,
  "pid": null,
  "wsl_pid": null,
  "error_message": null,
  "created_at": null,
  "started_at": null,
  "updated_at": null,
  "ended_at": null,
  "duration_seconds": null,
  "exit_code": null
}
```

### 1.3 Add Basic Validation

Add validation for:

- `nproc_per_node >= 1`
- `batch_size > 0`
- `epochs > 0`
- `EVALUATE` mode requires `checkpoint.load_path`
- `RESUME` mode requires `checkpoint.load_path`
- `profile.end_step >= profile.start_step`
- `data.format=random` can run without real dataset paths
- `data.format=parquet` should require valid data paths or a clear warning

### 1.4 Phase 1 Completion Review

Status: completed on 2026-07-23.

Phase 1 development goal was to stabilize the existing prototype foundation before integrating a
real DLRM backend. The completed work focused on the local run-directory contract, task state
metadata, basic configuration validation, and safer UI behavior during validation and job refresh.

Completed implementation:

- `config.py`
  - added validation for `nproc_per_node`
  - added validation for data loader settings including `batch_size`
  - added validation for training settings including `epochs`, `max_steps`, and `learning_rate`
  - added validation for checkpoint settings including `save_every_n_steps` and `keep_last`
  - added validation for `profile.start_step`
  - added cross-field validation requiring `checkpoint.load_path` for `RESUME` and `EVALUATE`
- `task_manager.py`
  - added `artifacts/` creation for every run
  - added `command.json` creation during launch
  - added `LAUNCHING` and `STOPPING` state transitions
  - expanded `state.json` with `backend`, `command`, `cwd`, `wsl_pid`, `error_message`,
    `updated_at`, and `duration_seconds`
  - added terminal-state protection so completed jobs are not overwritten accidentally
- `runner/cli.py`
  - added exception handling around runner execution
  - records unhandled runner exceptions in `launcher.log`
  - marks failed runner execution as `FAILED` with an `error_message`
- `ui/create_tab.py`
  - displays Pydantic validation errors in the UI instead of allowing Gradio events to fail
- `ui/logs_tab.py`, `ui/monitor_tab.py`, and `ui/artifacts_tab.py`
  - fixed job dropdown refresh behavior with `gr.update(choices=..., value=...)`
  - added empty-selection guards for reading logs, metrics, and artifacts
- `requirements.txt`
  - added explicit runtime dependencies: `gradio`, `pandas`, `pydantic`, `pyyaml`
- `README.md`
  - added Windows PowerShell startup instructions using the project `.venv`

Automated checks performed:

```text
python -m compileall .
python -m compileall ui
```

Smoke validation performed:

```text
Created and launched a one-step stub job through LocalTaskManager.
Confirmed final status: SUCCEEDED.
Confirmed command.json exists.
Confirmed artifacts/ exists.
Confirmed duration_seconds is populated.
```

Example smoke run directory:

```text
runs/20260723-153417-fb43ddef
```

Manual acceptance steps:

```text
1. Open PowerShell.
2. cd C:\Users\han\Desktop
3. Activate the prototype virtual environment.
4. Run python -m prototype.app.
5. Open the Gradio local URL.
6. In Create Job, launch a default stub job.
7. In Logs, click Refresh Jobs and select the new job.
8. Confirm launcher.log and train-rank0.log can be displayed.
9. In Monitor, click Refresh Jobs and confirm metrics can be displayed.
10. In Artifacts, click Refresh Jobs and confirm state.json and resolved-config.yaml can be displayed.
11. Try EVALUATE mode without a checkpoint and confirm the UI shows a validation error.
```

Manual acceptance criteria:

- the Gradio app starts without missing dependency errors
- default stub job launch succeeds
- the job reaches `SUCCEEDED`
- `runs/<job_id>/` contains `resolved-config.yaml`, `state.json`, `launcher.log`,
  `train-rank0.log`, `metrics.jsonl`, `command.json`, `checkpoints/`, `profiles/`, and
  `artifacts/`
- `state.json` includes `backend`, `command`, `cwd`, `updated_at`, `duration_seconds`, and
  `exit_code`
- Logs, Monitor, and Artifacts job refresh buttons do not raise Dropdown value errors
- invalid `EVALUATE` / `RESUME` checkpoint relationships are rejected before launch

Known carry-over item:

- `data.format=parquet` path existence checks are intentionally left as a future refinement because
  the first real backend milestone focuses on the already validated random-data DLRM path.

## Phase 2: Add A Backend Abstraction

Goal: keep stub execution and real DLRM execution behind a shared interface.

Suggested new layout:

```text
runner/backends/
  __init__.py
  base.py
  stub_backend.py
  dlrm_backend.py
```

Responsibilities:

- `base.py`: define the backend interface
- `stub_backend.py`: hold the current simulated training and evaluation behavior
- `dlrm_backend.py`: build and execute real DLRM commands
- `runner/cli.py`: load config and dispatch to the selected backend

Add backend configuration:

```python
class BackendConfig(BaseModel):
    name: str = "stub"
    dlrm_root: str = "/mnt/c/Users/han/Desktop/dlrm"
    python_env: str = "~/venvs/torchrec17"
    wsl_distribution: str = "Ubuntu-22.04"
```

The first implementation should preserve the current stub path while introducing the real backend
incrementally.

### 2.1 Phase 2 Completion Review

Status: completed on 2026-07-23.

Phase 2 development goal was to introduce a backend abstraction so the current simulated runner and
the future real DLRM runner can share one dispatch path.

Completed implementation:

- `config.py`
  - added `BackendName`
  - added `BackendConfig`
  - added `backend` to `PrototypeConfig`
  - default backend is `stub`
  - default DLRM-related environment fields are present for the next phase:
    `dlrm_root`, `python_env`, and `wsl_distribution`
- `runner/backends/base.py`
  - added the abstract `RunnerBackend` interface
- `runner/backends/stub_backend.py`
  - moved the existing simulated training behavior into `StubBackend`
  - moved the existing simulated evaluation behavior into `StubBackend`
  - preserved the existing log and metric output contract
- `runner/backends/dlrm_backend.py`
  - registered a placeholder `DLRMBackend`
  - returns a clear `NotImplementedError` until Phase 3 connects the real WSL2 command
- `runner/backends/__init__.py`
  - added `get_backend(config)` registry dispatch
- `runner/cli.py`
  - now selects a backend through `get_backend(config)`
  - no longer directly branches between `run_training` and `run_evaluation`
- `runner/train.py` and `runner/evaluate.py`
  - kept as compatibility wrappers around `StubBackend`
- `task_manager.py`
  - now records `config.backend.name` in `state.json`
  - now records full backend config in `command.json`

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Backend dispatch checks performed:

```text
PrototypeConfig().to_yaml()
get_backend(PrototypeConfig()).name
get_backend(PrototypeConfig(backend={"name": "dlrm"})).name
```

Expected results:

```text
default backend appears in YAML as stub
stub backend resolves to StubBackend
dlrm backend resolves to DLRMBackend
```

Smoke validation performed:

```text
Created and launched a one-step stub job through LocalTaskManager.
Confirmed final status: SUCCEEDED.
Confirmed state.json backend: stub.
Confirmed command.json backend: stub.
Confirmed command.json backend_config.name: stub.
```

Example stub smoke run directory:

```text
runs/20260723-160508-d95641f9
```

DLRM placeholder validation performed:

```text
Created and launched a job with backend=dlrm.
Confirmed final status: FAILED.
Confirmed state.json backend: dlrm.
Confirmed error_message clearly says the dlrm backend is registered but not implemented yet.
```

Example DLRM placeholder run directory:

```text
runs/20260723-160508-893a3df1
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. In Create Job, keep the default settings and click Validate Config.
3. Confirm the YAML includes backend.name: stub.
4. Click Launch Job.
5. Open Logs, click Refresh Jobs, and select the new job.
6. Confirm train-rank0.log includes Backend: stub and step/loss/AUC lines.
7. Open Artifacts, refresh jobs, and inspect state.json.
8. Confirm state.json has backend: stub and final status SUCCEEDED.
9. Inspect command.json in the run directory.
10. Confirm command.json includes backend and backend_config.
```

Manual acceptance criteria:

- default job still works exactly as a stub job
- `resolved-config.yaml` includes a `backend` section
- `state.json` records the selected backend
- `command.json` records both `backend` and `backend_config`
- `runner/cli.py` can dispatch through the backend registry
- choosing `backend=dlrm` through a manually edited config fails clearly with the Phase 3
  placeholder message, not with an import error or unknown backend error

## Phase 3: Implement The Real DLRM Execution Bridge

Goal: launch the validated local DLRM command from the Windows-hosted Gradio prototype.

Known local paths:

- Windows DLRM path: `C:\Users\han\Desktop\dlrm`
- WSL DLRM path: `/mnt/c/Users/han/Desktop/dlrm`
- recommended WSL Python environment: `~/venvs/torchrec17`
- expected entrypoint: `torchrec_dlrm/dlrm_main.py`

First command shape:

```bash
wsl -d Ubuntu-22.04 bash -lc '
  source ~/venvs/torchrec17/bin/activate
  cd /mnt/c/Users/han/Desktop/dlrm
  python -m torchrec_dlrm.dlrm_main ...
'
```

If the previous successful local validation used `python torchrec_dlrm/dlrm_main.py ...`, preserve
that exact style for the first working bridge.

Implementation tasks:

- generate the DLRM command in `dlrm_backend.py`
- write the command to `command.json`
- append stdout and stderr to `train-rank0.log`
- record launcher lifecycle events in `launcher.log`
- update `state.json` when execution succeeds or fails
- write a helpful `error_message` when the process exits non-zero
- keep random-data mode as the first supported real backend mode

### 3.1 Phase 3 Completion Review

Status: completed on 2026-07-24.

Phase 3 development goal was to replace the placeholder `DLRMBackend` with a real local execution
bridge that can launch the TorchRec DLRM entrypoint through WSL2 and preserve the prototype's local
run-directory contract.

Completed implementation:

- `runner/backends/dlrm_backend.py`
  - implemented real command generation for the `dlrm` backend
  - uses `wsl -d <distribution> bash -lc ...`
  - activates the configured WSL Python environment
  - changes directory to the configured DLRM root
  - launches DLRM through `torchrun --standalone --nnodes=1 --nproc_per_node=<n>`
  - uses `-m torchrec_dlrm.dlrm_main`
  - maps prototype config fields into DLRM arguments:
    `epochs`, `batch_size`, `learning_rate`, `limit_train_batches`, `limit_val_batches`,
    `limit_test_batches`, and `validation_freq_within_epoch`
  - converts Windows paths such as `C:\Users\han\Desktop\prototype\runs\<job_id>` to WSL paths
    such as `/mnt/c/Users/han/Desktop/prototype/runs/<job_id>`
  - exports `TORCHREC_PROTOTYPE_RUN_DIR` for future metric/artifact integration
  - streams stdout and stderr into `train-rank0.log`
  - records the real backend command into `command.json`
  - writes the resolved command to `launcher.log`
  - raises a readable error when the DLRM process exits non-zero
- unsupported-but-registered modes:
  - `COLD_START` with `data.format=random` is the first supported real DLRM path
  - `RESUME` is intentionally rejected until checkpoint loading is implemented
  - `EVALUATE` is intentionally rejected until checkpoint loading / eval-only support is implemented
  - non-random data formats are intentionally rejected until dataset path mapping is implemented

Actual command shape now generated:

```text
wsl -d Ubuntu-22.04 bash -lc "set -e; source $HOME/venvs/torchrec17/bin/activate; cd /mnt/c/Users/han/Desktop/dlrm; export TORCHREC_PROTOTYPE_RUN_DIR=/mnt/c/Users/han/Desktop/prototype/runs/<job_id>; exec torchrun --standalone --nnodes=1 --nproc_per_node=1 -m torchrec_dlrm.dlrm_main --epochs 1 --batch_size 4 --learning_rate 0.01 --limit_train_batches 1 --limit_val_batches 1 --limit_test_batches 1 --validation_freq_within_epoch 50"
```

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Command generation validation performed:

```text
DLRMBackend().build_command(
    PrototypeConfig(
        backend={"name": "dlrm"},
        training={"epochs": 1, "max_steps": 1},
        data={"batch_size": 4},
    ),
    Path("C:\\Users\\han\\Desktop\\prototype\\runs\\dry-run"),
)
```

Expected command contents were confirmed:

- `wsl`
- `-d Ubuntu-22.04`
- `source $HOME/venvs/torchrec17/bin/activate`
- `cd /mnt/c/Users/han/Desktop/dlrm`
- `torchrun --standalone`
- `--nproc_per_node=1`
- `-m torchrec_dlrm.dlrm_main`
- `--limit_train_batches 1`
- WSL-converted run directory path

Smoke validation performed:

```text
Created and launched a dlrm backend job through LocalTaskManager.
Confirmed command.json includes backend_command.
Confirmed train-rank0.log receives output from the attempted WSL command.
Confirmed state.json records backend: dlrm.
```

Initial Codex-side environment result:

```text
The bridge reached wsl.exe, but the configured WSL distribution Ubuntu-22.04 was not available
from the Codex execution environment.

That Codex-side smoke job therefore ended as FAILED with a readable backend error.
This was later determined to be an environment visibility issue in the Codex context, not a
problem with the user's local WSL setup or the DLRM bridge implementation.
```

Example DLRM smoke run directory:

```text
runs/20260724-163412-1eb45c50
```

User-side environment verification:

```text
PowerShell:
wsl -l -v

Confirmed distributions:
- Ubuntu-22.04, WSL version 2
- docker-desktop, WSL version 2

Inside Ubuntu-22.04:
source ~/venvs/torchrec17/bin/activate
cd /mnt/c/Users/han/Desktop/dlrm
which torchrun
python -c 'import torchrec; print("torchrec ok")'

Confirmed:
- torchrun path: /home/han/venvs/torchrec17/bin/torchrun
- torchrec import succeeded
```

User-side final DLRM bridge validation:

```text
Created a dlrm backend job from PowerShell through LocalTaskManager:

job_id: 20260724-164800-f5155c60
run_dir: C:\Users\han\Desktop\prototype\runs\20260724-164800-f5155c60
```

Final validation evidence:

```text
state.json:
- status: SUCCEEDED
- backend: dlrm
- exit_code: 0
- duration_seconds: 13.264799
- error_message: null

command.json:
- backend: dlrm
- backend_config.name: dlrm
- backend_config.dlrm_root: /mnt/c/Users/han/Desktop/dlrm
- backend_config.python_env: ~/venvs/torchrec17
- backend_config.wsl_distribution: Ubuntu-22.04
- backend_command includes wsl, bash, venv activation, cd into DLRM root, and torchrun

train-rank0.log:
- Total number of iterations: 1
- AUROC over val set: 0.0.
- AUROC over test set: 0.25.
```

Final Phase 3 conclusion:

```text
The real local DLRM random-data execution bridge is validated end to end on the user's machine.
The third-phase blocking question about WSL availability is resolved.
```

Regression validation performed:

```text
Created and launched a default stub job after the DLRM bridge changes.
Confirmed final status: SUCCEEDED.
Confirmed backend: stub.
```

Example stub regression run directory:

```text
runs/20260724-163548-bead2eca
```

Manual acceptance steps:

```text
1. Confirm WSL has a usable Ubuntu distribution:
   wsl -l -v
2. If the distribution name differs from Ubuntu-22.04, edit resolved-config.yaml or future UI
   backend settings to use the actual distribution name.
3. Confirm the WSL environment exists:
   wsl -d <distribution> bash -lc "source ~/venvs/torchrec17/bin/activate; which torchrun"
4. Launch or manually create a config with backend.name: dlrm.
5. Use data.format: random.
6. Use a tiny validation config first:
   training.max_steps: 1
   data.batch_size: 4
   nproc_per_node: 1
7. Launch the job.
8. Open the run directory.
9. Inspect command.json.
10. Inspect launcher.log.
11. Inspect train-rank0.log.
12. Inspect state.json.
```

Manual acceptance criteria:

- `command.json` includes `backend_command`
- `backend_command` invokes `wsl`, activates `~/venvs/torchrec17`, changes into the DLRM root, and
  invokes `torchrun`
- `train-rank0.log` starts with `Starting real DLRM backend run.`
- with a valid WSL environment, real DLRM output appears in `train-rank0.log`
- with a valid WSL environment and successful DLRM run, `state.json.status` becomes `SUCCEEDED`
- if WSL, venv, or DLRM paths are wrong, `state.json.status` becomes `FAILED` and
  `state.json.error_message` contains a useful diagnostic

Final acceptance status:

```text
Accepted.

The user-side DLRM smoke job 20260724-164800-f5155c60 satisfied the Phase 3 criteria:
- real WSL command was generated and recorded
- real TorchRec DLRM command executed
- train-rank0.log contains DLRM training/evaluation output
- state.json finished with SUCCEEDED
```

Known carry-over items:

- Phase 4 should expose backend selection and WSL/DLRM environment fields in the Gradio UI.
- Later phases should add real checkpoint save/load handling for `RESUME` and `EVALUATE`.
- Later phases should parse real DLRM metrics from `train-rank0.log` into `metrics.jsonl`.
- Log display still contains some mojibake for WSL's Chinese proxy warning and progress-bar glyphs;
  this is a display cleanup item, not a Phase 3 execution blocker.

## Phase 4: Expand The Create Job UI

Goal: expose enough controls to run both stub jobs and real DLRM jobs.

Add fields to `Create Job`:

- Backend: `stub` or `dlrm`
- DLRM Root
- Python Env
- WSL Distribution
- Max Steps
- Learning Rate
- Checkpoint Load Path
- Save Checkpoints toggle
- Profile Enabled toggle

Improve launch output from:

```text
Job <id> launched with pid=<pid>
```

to:

```text
Job <id> launched
Status: RUNNING
Backend: dlrm
Run directory: C:\Users\han\Desktop\prototype\runs\<id>
```

### 4.1 Phase 4 Completion Review

Status: completed on 2026-07-24.

Phase 4 development goal was to expose backend and local execution settings in the Gradio Create
Job UI so users can launch stub or real DLRM jobs without manually editing YAML.

Completed implementation:

- `config.py`
  - added `checkpoint.enabled` so the UI can express the Save Checkpoints toggle
- `ui/create_tab.py`
  - added Backend dropdown with `stub` and `dlrm`
  - added DLRM Root textbox
  - added Python Env textbox
  - added WSL Distribution textbox
  - added Max Steps numeric input
  - added Learning Rate numeric input
  - added Checkpoint Load Path textbox
  - added Save Checkpoints checkbox
  - added Profile Enabled checkbox
  - maps all new UI values into `PrototypeConfig`
  - converts empty checkpoint path to `null`
  - converts empty or non-positive Max Steps to `null`
  - preserves validation error display in the UI
  - improves launch output with job id, status, backend, PID, and run directory
- `README.md`
  - documented the Create Job backend settings and default DLRM environment values

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Configuration validation performed:

```text
Created a PrototypeConfig with:
- backend.name: dlrm
- dlrm_root: /mnt/c/Users/han/Desktop/dlrm
- python_env: ~/venvs/torchrec17
- wsl_distribution: Ubuntu-22.04
- training.max_steps: 1
- training.learning_rate: 0.01
- data.batch_size: 4
- checkpoint.enabled: false
- profile.enabled: true
```

Expected YAML fields were confirmed:

- `backend.name`
- `backend.dlrm_root`
- `backend.python_env`
- `backend.wsl_distribution`
- `training.max_steps`
- `training.learning_rate`
- `checkpoint.enabled`
- `profile.enabled`

Validation guard confirmed:

```text
EVALUATE mode without checkpoint.load_path still fails validation.
```

Regression validation performed:

```text
Created and launched a default stub job after the UI/config changes.
Confirmed final status: SUCCEEDED.
Confirmed backend: stub.
Confirmed resolved-config.yaml includes the new Phase 4 fields.
```

Example stub regression run directory:

```text
runs/20260724-170711-3660e5dd
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. Open Create Job.
3. Confirm these fields are visible:
   Backend, DLRM Root, Python Env, WSL Distribution, Max Steps, Learning Rate,
   Checkpoint Load Path, Save Checkpoints, Profile Enabled.
4. Keep Backend as stub.
5. Set Max Steps to 1 and Batch Size to 4.
6. Click Validate Config.
7. Confirm the YAML includes backend, training.max_steps, training.learning_rate,
   checkpoint.enabled, and profile.enabled.
8. Click Launch Job.
9. Confirm Launch Result includes job id, status, backend, PID, and run directory.
10. Open Logs and Artifacts tabs and inspect the new run.
11. Return to Create Job, select Backend as dlrm, keep data.format as random, and validate config.
12. If running real validation, launch the dlrm job and inspect command.json and train-rank0.log.
```

Manual acceptance criteria:

- Create Job exposes all planned Phase 4 fields
- Validate Config shows the selected backend and environment fields in YAML
- default stub launch still succeeds
- launch output is a multi-line summary, not only `pid`
- `resolved-config.yaml` preserves all new UI values
- `command.json` records the selected backend config
- selecting `EVALUATE` without a checkpoint path still shows a validation failure before launch
- selecting `dlrm` with random data can trigger the Phase 3 DLRM bridge on a correctly configured
  local WSL environment

User-side manual acceptance evidence:

```text
Date: 2026-07-25

Create Job launch result shown in Gradio:

Job 20260725-221712-eef2abb2 launched
Status: RUNNING
Backend: stub
PID: 21236
Run directory: C:\Users\han\Desktop\prototype\runs\20260725-221712-eef2abb2
```

Acceptance interpretation:

```text
This launch result is expected for Phase 4.

RUNNING is the immediate post-launch status returned by LocalTaskManager.launch_job().
For short stub jobs, the final status should be checked afterward in Artifacts -> state.json,
where the job is expected to become SUCCEEDED with backend=stub, exit_code=0, and error_message=null.
```

## Phase 5: Improve Logs And Status Refresh

Goal: make the UI feel like a useful local task panel.

### Logs Tab

Recommended improvements:

- show job choices as `<job_id> [<status>]`
- add a `Refresh Logs` button
- add a tail-lines input, defaulting to 200
- display `launcher.log`
- display `train-rank0.log`
- display `command.json`

### Artifacts Tab

Recommended improvements:

- show run directory
- show `state.json`
- show `resolved-config.yaml`
- show `evaluation.json`
- list checkpoint files
- list profile files
- list artifact files

### Monitor Tab

Recommended improvements:

- preserve the metrics table
- add a loss chart
- add an AUC chart
- show a clear empty state when `metrics.jsonl` does not exist or has no records

### 5.1 Phase 5 Completion Review

Status: completed on 2026-07-25.

Phase 5 development goal was to improve the Logs, Monitor, and Artifacts tabs so the prototype
feels like a usable local task panel rather than a minimal file viewer.

Completed implementation:

- `task_manager.py`
  - added `get_run_dir(job_id)`
  - added `list_files(job_id, relative_dir)` for checkpoints, profiles, and artifacts listings
- `ui/logs_tab.py`
  - job dropdown now displays `<job_id> [<status>]`
  - added Tail Lines input, defaulting to 200
  - added Refresh Logs button
  - reads and displays `launcher.log`
  - reads and displays `train-rank0.log`
  - reads and displays `command.json`
  - parses the displayed job label back to the real job id before reading files
- `ui/monitor_tab.py`
  - job dropdown now displays `<job_id> [<status>]`
  - added Refresh Metrics button
  - added metric status textbox
  - preserves the recent metrics table
  - added Train Loss line plot
  - added AUC line plot
  - added plot fallback axis for metrics without `step` values
  - returns a clear empty state when no metrics are available
- `ui/artifacts_tab.py`
  - job dropdown now displays `<job_id> [<status>]`
  - added Run Directory display
  - added Refresh Artifacts button
  - displays `state.json`
  - displays `resolved-config.yaml`
  - displays `evaluation.json`
  - lists files under `checkpoints/`, `profiles/`, and `artifacts/`
  - parses the displayed job label back to the real job id before reading files or stopping a job

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Gradio app construction check performed:

```text
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '..'); from prototype.app import build_app; app=build_app(); print(type(app).__name__)"
```

Expected result:

```text
Blocks
```

Smoke validation performed:

```text
Created and launched a one-step stub job after the Phase 5 UI changes.
Confirmed final status: SUCCEEDED.
Confirmed metrics.jsonl has records.
Confirmed command.json exists.
Confirmed checkpoint/profile/artifact directory listings return safely when empty.
```

Example smoke run directory:

```text
runs/20260725-222554-98bf7355
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. Launch a one-step stub job from Create Job.
3. Open Logs.
4. Click Refresh Jobs.
5. Confirm the job dropdown shows entries like <job_id> [SUCCEEDED].
6. Select the latest job.
7. Set Tail Lines to 20.
8. Click Refresh Logs.
9. Confirm launcher.log, train-rank0.log, and command.json are visible.
10. Open Monitor.
11. Click Refresh Jobs.
12. Select the same job.
13. Click Refresh Metrics.
14. Confirm the metrics table is populated and loss/AUC plots render.
15. Open Artifacts.
16. Click Refresh Jobs.
17. Select the same job.
18. Click Refresh Artifacts.
19. Confirm Run Directory, state.json, resolved-config.yaml, evaluation.json, and file lists are visible.
```

Manual acceptance criteria:

- Logs job dropdown displays status alongside job id
- Refresh Logs does not change job selection unexpectedly
- Tail Lines limits long log display
- `command.json` can be inspected from the Logs tab
- Monitor shows recent metrics table for stub jobs
- Monitor shows Train Loss and AUC plots when `metrics.jsonl` contains matching records
- Monitor can plot DLRM `val_auc` / `test_auc` records even when their original `step` value is null
- Monitor shows a clear no-metrics message for jobs without metrics
- Artifacts shows the run directory
- Artifacts shows `state.json` and `resolved-config.yaml`
- Artifacts lists checkpoints, profiles, and artifacts directories, including a clear empty state
- Stop Job still works when the dropdown displays `<job_id> [<status>]`

User-side Monitor validation evidence:

```text
Date: 2026-07-26

Selected job:
20260726-174406-4add4868 [SUCCEEDED]

Metric Status:
Loaded 8 metric records.

Recent Metrics table displayed:
- total_iterations: 1
- throughput_iter_per_sec: 15.18
- val_auc: 0.0
- val_samples: 4
- throughput_iter_per_sec: 54.86
- test_auc: 0.0
- test_samples: 4
- throughput_iter_per_sec: 134.58
```

Acceptance interpretation:

```text
The Monitor tab successfully reads real DLRM metrics generated by Phase 6.
Train Loss is empty for this DLRM run because the current random-data DLRM log does not print
training loss.
AUC values are present and equal to 0.0 for both val and test in this run, so the AUC chart is
expected to be flat at 0.0.
```

Follow-up adjustment after user-side Monitor inspection:

```text
DLRM evaluation metrics such as val_auc and test_auc may not have a step value.
Monitor plot data now adds a plot_step fallback based on record order, so AUC records can be plotted
even when their original step is null.
```

## Phase 6: Parse Real Metrics From DLRM Logs

Goal: turn real DLRM output into the prototype's common `metrics.jsonl` format.

Add:

```text
runner/log_parser.py
```

Suggested API:

```python
def parse_metric_line(line: str) -> list[dict]:
    ...
```

Metrics to extract first:

- step
- loss
- AUC
- throughput
- learning rate
- elapsed time

Execution behavior:

- stream stdout and stderr from DLRM
- write original lines to `train-rank0.log`
- parse metric-like lines
- append parsed metric records to `metrics.jsonl`

Start with tolerant parsing rules. Tighten them later when the exact DLRM log format is stable.

### 6.1 Phase 6 Completion Review

Status: completed on 2026-07-25.

Phase 6 development goal was to parse real DLRM log output into the prototype's shared
`metrics.jsonl` format so the Monitor tab can display metrics for real DLRM jobs, not only stub
jobs.

Completed implementation:

- `runner/log_parser.py`
  - added `parse_metric_line(line)`
  - parses `Total number of iterations: <n>` into `total_iterations`
  - parses `AUROC over <stage> set: <value>.` into `<stage>_auc`
  - parses `Number of <stage> samples: <n>` into `<stage>_samples`
  - parses `lr: <step> <group> <value>` into `learning_rate`
  - parses tqdm-style `<value>it/s` into `throughput_iter_per_sec`
  - parses generic `step=<n> loss=<value>` or `loss: <value>` into `loss`
- `runner/backends/dlrm_backend.py`
  - calls `parse_metric_line()` while streaming DLRM stdout/stderr
  - appends parsed records to `metrics.jsonl` using the existing `append_metric()` helper
  - preserves original log output in `train-rank0.log`
  - sets UTF-8 locale / Python IO environment variables for the WSL-launched process
  - strips ANSI escape sequences and Unicode bidi control characters from captured log lines

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Parser validation performed:

```text
parse_metric_line("step=7 loss=0.1234")
parse_metric_line("AUROC over test set: 0.25.")
```

Expected results confirmed:

```text
loss record with step=7 and value=0.1234
test_auc record with value=0.25
```

Real DLRM sample-log validation performed:

```text
Input log:
runs/20260724-164800-f5155c60/train-rank0.log

Parsed metric records: 8
```

Parsed metrics included:

- `total_iterations`
- `throughput_iter_per_sec`
- `val_auc`
- `val_samples`
- `test_auc`
- `test_samples`

JSONL write validation performed:

```text
Parsed the Phase 3 successful DLRM log sample and wrote records using append_metric().
Confirmed the resulting JSONL records contain timestamp, metric, value, and optional stage fields.
```

Regression validation performed:

```text
Created and launched a default one-step stub job after the Phase 6 parser changes.
Confirmed final status: SUCCEEDED.
Confirmed metrics.jsonl still contains the existing stub train_loss and auc records.
```

Example stub regression run directory:

```text
runs/20260725-223619-11b3a1ea
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. In Create Job, select Backend=dlrm.
3. Keep Data Format=random.
4. Set Batch Size=4.
5. Set Max Steps=1.
6. Set Processes per Node=1.
7. Launch the job.
8. Wait for the job to finish.
9. Open the run directory.
10. Confirm train-rank0.log contains real DLRM output.
11. Confirm metrics.jsonl exists.
12. Open Monitor.
13. Refresh Jobs and select the DLRM job.
14. Click Refresh Metrics.
15. Confirm the metrics table contains parsed DLRM records such as val_auc and test_auc.
```

Manual acceptance criteria:

- real DLRM logs are still written verbatim to `train-rank0.log`
- `metrics.jsonl` is created for DLRM jobs when parseable metric lines appear
- `metrics.jsonl` contains records for `val_auc` and `test_auc` after the current random-data DLRM
  validation path
- Monitor table displays parsed DLRM metric rows
- Monitor AUC plot can use `val_auc` / `test_auc` records
- stub jobs continue to generate their original `train_loss` and `auc` metrics

User-side DLRM acceptance evidence:

```text
Date: 2026-07-26

Run directory:
runs/20260726-174406-4add4868

state.json:
- status: SUCCEEDED
- backend: dlrm
- exit_code: 0
- duration_seconds: 17.909475
- error_message: null

metrics.jsonl:
- total records: 8
- total_iterations: 1
- throughput_iter_per_sec: 15.18
- val_auc: 0.0
- val_samples: 4
- throughput_iter_per_sec: 54.86
- test_auc: 0.0
- test_samples: 4
- throughput_iter_per_sec: 134.58

train-rank0.log:
- real DLRM backend started
- DLRM root, WSL distribution, and Python env were recorded
- Total number of iterations: 1
- AUROC over val set: 0.0.
- Number of val samples: 4
- AUROC over test set: 0.0.
- Number of test samples: 4
```

Acceptance interpretation:

```text
Phase 6 metric parsing is validated on a real user-side DLRM run.
The Monitor tab should be able to read metrics.jsonl for this DLRM job and display the parsed
metric table and AUC-related records.
```

Follow-up adjustment after user-side log inspection:

```text
Some WSL / tqdm output contained mojibake and Unicode formatting control characters in
train-rank0.log. The DLRM backend capture path was adjusted to set UTF-8 locale variables and strip
ANSI / bidi control characters before writing captured lines.
```

Known carry-over items:

- The current DLRM random-data log does not print training loss, so `loss` parsing is ready but not
  exercised by the validated DLRM path yet.
- More detailed throughput, step-time, and checkpoint timing parsing can be added after richer real
  DLRM logs are available.

## Phase 7: Connect Real Evaluation Mode

Goal: make `EVALUATE` mode run a real local evaluation command.

Tasks:

- require `checkpoint.load_path` for `EVALUATE`
- build the evaluation command through the selected backend
- capture stdout and stderr
- parse AUC / log loss where possible
- write `evaluation.json`
- append evaluation metrics to `metrics.jsonl`

The first version can remain limited to the local random-data validation path if that is the
fastest route to a working end-to-end flow.

### 7.1 Phase 7 Completion Review

Status: completed on 2026-07-26.

Phase 7 development goal was to make `EVALUATE` mode run through the real local backend path and
produce evaluation artifacts instead of only relying on the earlier stub evaluation flow.

Important implementation note:

```text
The current local upstream torchrec_dlrm.dlrm_main.py does not expose checkpoint load, resume, or
eval-only CLI arguments.

For Phase 7, the first real DLRM evaluation mode is therefore implemented as a random-data
eval-only smoke path:

- checkpoint.load_path is still required by prototype validation and recorded in evaluation.json
- the path is not loaded into DLRM yet
- DLRM is launched with --limit_train_batches 0
- DLRM still runs its real model construction, dataloaders, distributed setup, validation, and test
  evaluation path
```

Completed implementation:

- `runner/backends/dlrm_backend.py`
  - removed the previous `EVALUATE` placeholder rejection
  - keeps `RESUME` rejected until checkpoint loading is implemented
  - splits DLRM argument construction into train and evaluate paths
  - `EVALUATE` mode generates a real DLRM command with:
    - `--epochs 1`
    - `--limit_train_batches 0`
    - `--limit_val_batches 1`
    - `--limit_test_batches 1`
  - preserves stdout/stderr capture into `train-rank0.log`
  - continues metric parsing into `metrics.jsonl`
  - writes `evaluation.json` after successful DLRM evaluate execution
  - summarizes `val_auc`, `test_auc`, `log_loss`, `val_samples`, and `test_samples` where present
  - records `source_checkpoint`
  - records `checkpoint_load_supported: false` with an explanatory note
- `runner/backends/stub_backend.py`
  - stub `EVALUATE` now also writes `eval_auc` and `eval_log_loss` to `metrics.jsonl`
- `runner/log_parser.py`
  - added generic `log_loss` parsing for future DLRM logs that expose it

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

DLRM evaluate command validation performed:

```text
PrototypeConfig(
    mode=RunMode.EVALUATE,
    backend={"name": "dlrm"},
    checkpoint={"load_path": "sample-checkpoint"},
    training={"max_steps": 1},
    data={"batch_size": 4},
)
```

Expected command contents were confirmed:

- `-m torchrec_dlrm.dlrm_main`
- `--limit_train_batches 0`
- `--limit_val_batches 1`
- `--limit_test_batches 1`

Validation guard confirmed:

```text
EVALUATE mode without checkpoint.load_path still fails validation.
EVALUATE mode with checkpoint.load_path passes validation.
```

Parser validation performed:

```text
parse_metric_line("log_loss: 0.456")
parse_metric_line("AUROC over val set: 0.12.")
```

Expected results confirmed:

```text
log_loss record with value=0.456
val_auc record with value=0.12
```

Stub evaluation smoke validation performed:

```text
Created and launched a stub EVALUATE job.
Confirmed final status: SUCCEEDED.
Confirmed evaluation.json contains source_checkpoint.
Confirmed metrics.jsonl contains eval_auc and eval_log_loss.
```

Example stub evaluation run directory:

```text
runs/20260726-180304-81ee8b24
```

DLRM evaluation summary validation performed:

```text
Created a sample metrics.jsonl with val_auc, test_auc, and log_loss.
Ran the DLRM evaluation summary writer.
Confirmed evaluation.json includes val_auc, test_auc, log_loss, and source_checkpoint.
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. Open Create Job.
3. Select Mode=EVALUATE.
4. Select Backend=dlrm.
5. Keep Data Format=random.
6. Set Batch Size=4.
7. Set Max Steps=1.
8. Set Processes per Node=1.
9. Enter a non-empty Checkpoint Load Path.
10. Click Validate Config.
11. Confirm the YAML contains mode: EVALUATE and checkpoint.load_path.
12. Launch the job.
13. Wait for completion.
14. Open Logs and inspect train-rank0.log.
15. Open Artifacts and inspect evaluation.json.
16. Open Monitor and inspect parsed metrics.
```

Manual acceptance criteria:

- empty Checkpoint Load Path is rejected for `EVALUATE`
- non-empty Checkpoint Load Path allows launch
- `command.json` contains the DLRM backend command
- backend command contains `--limit_train_batches 0`
- real DLRM output appears in `train-rank0.log`
- `metrics.jsonl` contains parsed evaluation metrics such as `val_auc` and/or `test_auc`
- `evaluation.json` exists
- `evaluation.json.source_checkpoint` matches the UI checkpoint path
- `evaluation.json.checkpoint_load_supported` is `false` for this phase
- successful local random-data evaluation ends with `state.json.status: SUCCEEDED`

Known carry-over items:

- True checkpoint restore is not implemented because the current local `torchrec_dlrm.dlrm_main.py`
  has no checkpoint load CLI argument.
- A later checkpoint-focused phase should either extend/wrap DLRM to load checkpoints or introduce a
  custom runner path that owns checkpoint save/load semantics.

## Phase 8: Strengthen Stop Job Behavior

Goal: make stopping real WSL-launched jobs more reliable.

Current limitation:

- the prototype stops the Windows-side subprocess PID
- real DLRM execution may involve `wsl.exe`, shell, Python, and child processes

Recommended improvements:

- record the Windows process PID
- record the WSL-side PID if practical
- move state to `STOPPING` before sending the signal
- try graceful termination first
- force termination only if needed
- record stop metadata in `state.json`

Example stop metadata:

```json
{
  "status": "STOPPED",
  "stopped_at": "2026-07-23T00:00:00",
  "stop_reason": "user_requested"
}
```

### 8.1 Phase 8 Completion Review

Status: completed on 2026-07-26.

Phase 8 development goal was to make Stop Job behavior safer and more useful for both stub jobs and
WSL-launched DLRM jobs.

Completed implementation:

- `task_manager.py`
  - added terminal-state protection for Stop Job
  - completed jobs in `SUCCEEDED`, `FAILED`, or `STOPPED` are no longer overwritten
  - moves active jobs to `STOPPING` before sending a signal
  - tries graceful termination first:
    - Windows: `CTRL_BREAK_EVENT`
    - non-Windows: `SIGTERM`
  - waits briefly for the process to exit
  - force kills the process tree when graceful termination does not finish in time:
    - Windows: `taskkill /PID <pid> /T /F`
    - non-Windows: `SIGKILL`
  - records stop metadata in `state.json`
- `ui/artifacts_tab.py`
  - Stop Job result now distinguishes completed `SUCCEEDED` / `FAILED` jobs from active jobs

New stop metadata fields:

```json
{
  "stopped_at": null,
  "stop_reason": null,
  "stop_signal": null,
  "force_killed": false,
  "stop_error": null
}
```

Automated checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Completed-job stop validation:

```text
Created and launched a one-step stub job.
Waited for final status SUCCEEDED.
Called stop_job().
Confirmed status remained SUCCEEDED.
Confirmed the completed job was not overwritten as STOPPED.
```

Example run directory:

```text
runs/20260726-180812-ac831a04
```

Running-job stop validation:

```text
Created and launched a longer stub job.
Called stop_job() while it was still running.
Confirmed final status STOPPED.
Confirmed stop_reason: user_requested.
Confirmed stop_signal: CTRL_BREAK_EVENT.
Confirmed stopped_at was populated.
Confirmed duration_seconds was populated.
Confirmed force_killed was false for the graceful stop case.
```

Example run directory:

```text
runs/20260726-180826-8af9dad0
```

No-pid stop validation:

```text
Created a job but did not launch it.
Called stop_job().
Confirmed final status STOPPED.
Confirmed stop_error: No process id was recorded for this job.
Confirmed stopped_at was populated.
```

Example run directory:

```text
runs/20260726-180839-9bf261bb
```

Manual acceptance steps:

```text
1. Start the Gradio UI.
2. Launch a stub job with Max Steps high enough that it remains RUNNING for a few seconds.
3. Open Artifacts.
4. Click Refresh Jobs.
5. Select the running job.
6. Click Stop Job.
7. Click Refresh Artifacts.
8. Inspect state.json.
9. Launch a short one-step stub job and wait for it to become SUCCEEDED.
10. Select that completed job in Artifacts.
11. Click Stop Job.
12. Refresh Artifacts and inspect state.json again.
```

Manual acceptance criteria:

- active job transitions to `STOPPING` and then `STOPPED`
- stopped active job records `stopped_at`
- stopped active job records `stop_reason: user_requested`
- stopped active job records a stop signal such as `CTRL_BREAK_EVENT`
- `force_killed` is recorded as `true` or `false`
- completed `SUCCEEDED` jobs are not changed to `STOPPED`
- completed `FAILED` jobs are not changed to `STOPPED`
- Stop Job still works when the Artifacts dropdown displays `<job_id> [<status>]`

Known carry-over items:

- WSL-side child PID discovery is not implemented yet. The prototype stops the Windows-side process
  group/tree, which is sufficient for the current local bridge but can be improved later if deeper
  WSL process inspection is needed.

## Phase 9: Update README And Operating Documentation

Goal: make the project reproducible and easier to continue.

Update `README.md` with:

- project purpose
- local scope
- install instructions
- Windows Gradio startup command
- WSL2 requirements
- DLRM path requirements
- TorchRec virtual environment requirements
- stub backend example
- DLRM backend example
- run directory contract
- common troubleshooting notes

Add a minimal validation workflow:

```text
1. Start the Gradio UI.
2. Select backend=stub.
3. Launch a job.
4. Check Logs, Monitor, and Artifacts.
5. Select backend=dlrm.
6. Use random data.
7. Launch a job.
8. Confirm real DLRM output appears in the run directory.
```

### 9.1 Phase 9 Completion Review

Status: completed on 2026-07-26.

Phase 9 development goal was to update `README.md` from an early short project note into a practical
operating document for running, validating, and troubleshooting the local prototype.

Completed implementation:

- `README.md`
  - documented project purpose and local-first scope
  - documented current implementation status
  - documented known checkpoint/evaluate limitation
  - documented project layout
  - documented the two-environment model:
    - Windows `.venv` for Gradio UI
    - WSL2 `~/venvs/torchrec17` for TorchRec / DLRM execution
  - documented Windows install and startup commands
  - documented WSL / DLRM environment verification commands
  - documented Create Job fields
  - documented minimal stub validation
  - documented minimal DLRM training validation
  - documented minimal DLRM evaluate validation
  - documented Create Job, Logs, Monitor, and Artifacts tabs
  - documented run directory contract
  - documented task states
  - documented Stop Job behavior
  - documented quality check commands
  - documented common troubleshooting cases

Automated/source checks performed:

```text
python -m compileall config.py task_manager.py runner ui
```

Documentation consistency checks performed:

```text
Confirmed README mentions the current UI fields:
- Backend
- DLRM Root
- Python Env
- WSL Distribution
- Max Steps
- Learning Rate
- Checkpoint Load Path
- Save Checkpoints
- Profile Enabled

Confirmed default PrototypeConfig values match README:
- backend.name: stub
- backend.dlrm_root: /mnt/c/Users/han/Desktop/dlrm
- backend.python_env: ~/venvs/torchrec17
- backend.wsl_distribution: Ubuntu-22.04
```

Manual acceptance steps:

```text
1. Open README.md.
2. Follow the Install And Start The UI section.
3. Confirm the Gradio app starts.
4. Follow Minimal Stub Validation.
5. Confirm stub job reaches SUCCEEDED.
6. Follow Verify WSL / DLRM Environment.
7. Follow Minimal DLRM Training Validation.
8. Confirm command.json, train-rank0.log, metrics.jsonl, and state.json are produced.
9. Follow Minimal DLRM Evaluate Validation.
10. Confirm evaluation.json is produced.
11. Read Troubleshooting and confirm it covers missing Gradio, WSL distro mismatch, DLRM startup
    failure, missing metrics, and checkpoint-load limitation.
```

Manual acceptance criteria:

- README can be used to bootstrap the Windows UI environment from scratch
- README can be used to verify the WSL/DLRM environment
- README explains both `stub` and `dlrm` validation paths
- README accurately describes the run directory files
- README accurately explains the current checkpoint/evaluate limitation
- README troubleshooting covers the known issues encountered during development

## Phase 10: Add Basic Tests And Quality Checks

Goal: keep the prototype easy to change without breaking the core contract.

### 10.1 Syntax Check

```bash
python -m compileall .
```

### 10.2 Configuration Tests

Cover:

- default config can generate YAML
- generated YAML can be loaded back
- invalid cache load factor fails
- invalid profile step range fails
- `EVALUATE` without checkpoint fails
- `RESUME` without checkpoint fails

### 10.3 Backend Command Tests

For the DLRM backend command builder, verify the command includes:

- WSL distribution
- virtual environment activation
- DLRM root
- DLRM entrypoint
- batch size
- epochs
- max steps when provided

### 10.4 Stub Runner Regression Test

Verify that a stub job produces:

- `train-rank0.log`
- `metrics.jsonl`
- final `state.json` with `SUCCEEDED`

### 10.5 Phase 10 Completion Review

Status: completed on 2026-07-26.

Phase 10 development goal was to add basic tests and quality checks so the prototype can keep
evolving without breaking the core local-run contract.

Completed implementation:

- `tests/__init__.py`
  - added test package marker
- `tests/test_config.py`
  - covers default config YAML round-trip
  - covers invalid cache load factor
  - covers invalid profile step range
  - covers `EVALUATE` without checkpoint rejection
  - covers `RESUME` without checkpoint rejection
  - covers `EVALUATE` with checkpoint success
  - covers positive numeric guards for `nproc_per_node`, `batch_size`, and `epochs`
- `tests/test_log_parser.py`
  - covers DLRM AUROC parsing
  - covers sample count parsing
  - covers total iteration parsing
  - covers generic loss parsing
  - covers learning-rate parsing
  - covers tqdm throughput parsing
  - covers log-loss parsing
- `tests/test_dlrm_backend.py`
  - covers DLRM training command generation
  - confirms WSL distribution, venv activation, DLRM root, `torchrun`, DLRM entrypoint, batch size,
    epochs, learning rate, and max-step flags
  - covers DLRM evaluate command generation with `--limit_train_batches 0`
  - covers Windows-to-WSL path conversion
  - covers DLRM `evaluation.json` summary generation from `metrics.jsonl`
- `tests/test_stub_and_task_manager.py`
  - covers stub training output contract
  - covers stub evaluation output contract
  - covers task manager run directory creation contract
  - covers terminal-state protection for Stop Job
- `README.md`
  - added unittest command to Quality Checks

Automated checks performed:

```text
python -m unittest discover -s tests
```

Result:

```text
Ran 18 tests in 0.561s
OK
```

Compile checks performed:

```text
python -m compileall config.py task_manager.py runner ui tests
```

Result:

```text
completed successfully
```

Manual acceptance steps:

```text
1. Open PowerShell.
2. cd C:\Users\han\Desktop\prototype
3. Run python -m unittest discover -s tests
4. Run python -m compileall config.py task_manager.py runner ui tests
```

Manual acceptance criteria:

- unittest discovery runs without importing external test dependencies
- all tests pass
- compileall succeeds
- tests do not require WSL or a real DLRM run
- tests use temporary directories instead of polluting the main runs/ directory
- README Quality Checks include both compile and unittest commands

## Recommended Implementation Order

1. Standardize run directory and `state.json` fields.
2. Add backend configuration to `config.py`.
3. Move current stub behavior into `runner/backends/stub_backend.py`.
4. Add backend selection in `runner/cli.py`.
5. Implement DLRM command generation.
6. Implement DLRM process execution and logging.
7. Add backend and environment fields to the Create Job UI.
8. Improve Logs, Monitor, and Artifacts tabs.
9. Add metric parsing from real DLRM logs.
10. Connect real evaluation mode.
11. Improve stop behavior.
12. Update README.
13. Add basic tests and compile checks.

## First Milestone

The first meaningful milestone is:

```text
From the Gradio UI, select backend=dlrm, click Launch Job, and start a real local DLRM random-data
training run inside WSL2 using ~/venvs/torchrec17.

The run should write real stdout/stderr to:

runs/<job_id>/train-rank0.log

and finish with state.json set to SUCCEEDED or FAILED.
```

Once this milestone is complete, the project will move from a simulated prototype to a real local
TorchRec / DLRM execution prototype.

## Industrialization Roadmap

Added on 2026-07-28.

The first 10 phases completed a local prototype milestone:

```text
Gradio UI -> local task manager -> WSL2 -> real TorchRec DLRM random-data execution.
```

This is a valid prototype foundation, but it is not yet an industrial-grade recommendation training
platform. To make the project genuinely useful beyond a small local experiment, the next roadmap
must focus on full TorchRec / DLRM capability, real data, reproducibility, checkpointing,
evaluation, observability, and operational reliability.

Updated product direction:

```text
Move from a local random-data TorchRec/DLRM prototype to a real-data, reproducible, extensible,
single-machine industrial training workbench for TorchRec recommendation workflows.
```

The project should still avoid prematurely adding Kubernetes or multi-user production complexity,
but it should stop treating real data, checkpointing, evaluation, and experiment tracking as
optional extras. These become core requirements for the next development cycle.

## Current Industrial Gap

Current capability:

- real `torchrec_dlrm.dlrm_main.py` can be launched from the UI
- random-data DLRM training runs through WSL2
- DLRM logs are captured
- DLRM metrics such as `val_auc` and `test_auc` are parsed
- local run artifacts are persisted
- stub backend remains available for UI/lifecycle validation

Current limitations:

- real dataset training is not implemented
- `data.format=parquet` is visible in the UI but not supported by the real DLRM backend
- custom business schemas are not mapped to DLRM dense/sparse/label inputs
- checkpoint save/load is not implemented
- `RESUME` is still rejected for the DLRM backend
- `EVALUATE` records checkpoint path but does not restore checkpoint weights
- model/export artifacts are not standardized
- experiment comparison and run lineage are minimal
- no resource telemetry such as CPU/GPU/memory utilization
- no robust data validation, schema validation, or feature statistics
- no industrial packaging or versioned release process

## Phase 11: Real Dataset Integration

Goal: enable training with real recommendation datasets instead of only random data.

This is the most important next phase.

### 11.1 Decide First Real Data Path

Recommended first target:

```text
Criteo-style local dataset path supported by torchrec_dlrm.dlrm_main.py
```

Reason:

- `torchrec_dlrm.dlrm_main.py` already has CLI arguments for Criteo-like inputs
- this keeps the first real-data milestone close to the official DLRM code
- it avoids inventing a custom dataset loader before proving real-data execution

Candidate upstream arguments to support:

- `--in_memory_binary_criteo_path`
- `--synthetic_multi_hot_criteo_path`
- `--dataset_name`
- `--limit_train_batches`
- `--limit_val_batches`
- `--limit_test_batches`
- `--mmap_mode`
- `--pin_memory`

### 11.2 Extend Config Model

Add richer data configuration:

```python
class DataConfig(BaseModel):
    format: str = "random"  # random | criteo_binary | synthetic_multihot | parquet
    train_path: str | None = None
    validation_path: str | None = None
    test_path: str | None = None
    criteo_binary_path: str | None = None
    synthetic_multi_hot_path: str | None = None
    dataset_name: str = "criteo_1t"
    batch_size: int = 32
    test_batch_size: int | None = None
    mmap_mode: bool = False
```

Validation requirements:

- `random` requires no data path
- `criteo_binary` requires `criteo_binary_path`
- `synthetic_multihot` requires `synthetic_multi_hot_path`
- `parquet` requires train/validation/test paths and schema config

### 11.3 Extend UI

Create Job should show format-specific fields:

- random: no dataset path required
- criteo_binary: Criteo Binary Path
- synthetic_multihot: Synthetic Multi-Hot Path
- parquet: train/validation/test paths plus schema file

### 11.4 Implement Backend Mapping

Map prototype config to DLRM CLI:

```text
data.format=random
  -> no dataset path args

data.format=criteo_binary
  -> --in_memory_binary_criteo_path <path>

data.format=synthetic_multihot
  -> --synthetic_multi_hot_criteo_path <path>
```

### 11.5 Acceptance Criteria

Manual acceptance:

```text
1. Select Backend=dlrm.
2. Select Data Format=criteo_binary or synthetic_multihot.
3. Enter a real WSL dataset path.
4. Set Max Steps=1.
5. Launch the job.
6. Confirm train-rank0.log shows real data loading.
7. Confirm state.json.status becomes SUCCEEDED.
8. Confirm metrics.jsonl has val/test metrics.
```

Pass criteria:

- real dataset path is validated before launch
- DLRM command includes the correct dataset argument
- random-data path still works
- invalid data paths fail with useful validation errors
- real dataset smoke training completes successfully

### Phase 11 Completion Review

Status: completed locally for configuration, UI wiring, backend command mapping, and automated
regression tests. Real-data smoke training is ready to run after the Criteo sample is preprocessed
to TorchRec numpy files.

Implemented:

- extended `DataConfig` with `criteo_binary_path`, `synthetic_multi_hot_path`, `schema_path`,
  `dataset_name`, `test_batch_size`, and `mmap_mode`
- added validation for `random`, `criteo_binary`, `synthetic_multihot`, and `parquet` formats
- exposed real-data fields in the Create Job tab
- mapped `criteo_binary` to `--in_memory_binary_criteo_path`
- mapped `synthetic_multihot` to `--synthetic_multi_hot_criteo_path`
- passed through `--dataset_name`, `--test_batch_size`, and `--mmap_mode`
- preserved the previous `random` DLRM path
- added tests for real data config validation and DLRM command generation

Validation run:

```text
python -m unittest discover -s tests
Ran 21 tests in 0.580s
OK
```

Manual acceptance method:

1. Preprocess the local Criteo tiny sample into TorchRec numpy format.
2. Start the Gradio app.
3. Open Create Job.
4. Select `Backend=dlrm`.
5. Select `Data Format=criteo_binary`.
6. Set `Dataset Name=criteo_kaggle`.
7. Enter the preprocessed numpy output directory in `Criteo Binary Path`.
8. Set `Max Steps=1`.
9. Launch the job.
10. Inspect `command.json`, `launcher.log`, `train-rank0.log`, `state.json`, and `metrics.jsonl`.

Manual acceptance standards:

- `command.json.backend_command_display` includes `--dataset_name criteo_kaggle`
- `command.json.backend_command_display` includes `--in_memory_binary_criteo_path <dataset path>`
- invalid missing Criteo paths fail before launch
- `random` data jobs still launch as before
- after preprocessing, a tiny real-data smoke job reaches `SUCCEEDED`

## Phase 12: Generic Parquet Dataset Adapter

Goal: support business-style recommendation data, not only Criteo-like DLRM inputs.

This phase is where the project starts becoming broadly industrial rather than only a wrapper
around one DLRM example.

### 12.1 Define Schema Contract

Add a schema file such as:

```yaml
label:
  name: clicked
  dtype: int
dense_features:
  - name: age
    dtype: float
  - name: price
    dtype: float
sparse_features:
  - name: user_id
    dtype: categorical
    num_embeddings: 1000000
    embedding_dim: 64
  - name: item_id
    dtype: categorical
    num_embeddings: 1000000
    embedding_dim: 64
```

### 12.2 Add Data Validation

Before training:

- confirm files exist
- confirm required columns exist
- confirm label values are valid
- confirm dense columns are numeric
- confirm sparse columns are categorical/int/string as expected
- compute row count sample
- compute null rates
- compute cardinality estimates for sparse features

Write output:

```text
runs/<job_id>/artifacts/data-profile.json
```

### 12.3 Choose Implementation Direction

Two possible paths:

1. Convert parquet to a DLRM-supported intermediate format.
2. Build a custom TorchRec dataset/dataloader and model runner.

Recommended industrial path:

```text
Start with conversion for speed, then graduate to a custom TorchRec runner when schema flexibility
becomes necessary.
```

### 12.4 Acceptance Criteria

- user can select `data.format=parquet`
- user can provide train/validation/test paths
- user can provide schema YAML
- data validation runs before training
- invalid schema/data fails before TorchRec launch
- valid parquet sample can run at least one real training/evaluation step

### Phase 12 Completion Review

Status: completed for the industrial preflight layer. Direct DLRM training from parquet remains a
future custom runner/conversion task, but parquet jobs now have schema-driven validation and produce
a reusable data profile before backend launch.

Implemented:

- added `runner/data_validation.py`
- added schema-driven parquet preflight validation
- validates schema path and train/validation/test paths
- validates required label, dense, and sparse columns
- validates binary label values
- validates dense features are numeric
- computes sample row counts, null rates, and sparse cardinality samples
- writes `artifacts/data-profile.json`
- connected validation into `runner/cli.py` before backend launch
- added `pyarrow` to `requirements.txt`
- added automated tests for parquet validation behavior

Validation run:

```text
python -m unittest discover -s tests
Ran 23 tests in 0.537s
OK (skipped=1)
```

The skipped test is the real parquet read/write test and will run automatically after installing
`pyarrow`.

Manual acceptance method:

1. Install requirements so `pyarrow` is available.
2. Prepare three small parquet files with the same columns.
3. Prepare a schema YAML with `label`, `dense_features`, and `sparse_features`.
4. Create a job with `data.format=parquet`.
5. Use the parquet file paths and schema path in the Create Job tab.
6. Launch with `Backend=stub` first to validate the preflight path.
7. Inspect `runs/<job_id>/artifacts/data-profile.json`.

Manual acceptance standards:

- missing schema/data paths fail before backend launch
- missing columns fail before backend launch
- non-binary labels fail before backend launch
- valid parquet data writes `data-profile.json`
- telemetry/profile failure is not mixed with data validation errors
- DLRM parquet launch still fails clearly, explaining direct parquet training is not implemented yet

## Phase 13: Checkpoint Save, Resume, And Evaluate From Checkpoint

Goal: make training reproducible and resumable.

Current limitation:

```text
The upstream torchrec_dlrm.dlrm_main.py used locally does not expose checkpoint load/resume CLI
arguments.
```

Therefore checkpoint support likely requires one of these:

1. patch/wrap local DLRM code to save/load model and optimizer state
2. introduce a custom runner that owns the training loop
3. contribute checkpoint CLI support into the local DLRM path

Required behavior:

- save model checkpoints under `checkpoints/`
- optionally save optimizer state
- record checkpoint metadata
- support `RESUME` mode
- support `EVALUATE` from checkpoint
- keep `checkpoint.latest.json`

Suggested files:

```text
checkpoints/
  step-000001/
    model.pt
    optimizer.pt
    metadata.json
  latest.json
```

Acceptance criteria:

- COLD_START creates checkpoint artifacts
- RESUME loads a previous checkpoint
- EVALUATE loads a checkpoint and runs validation/test
- `evaluation.json.checkpoint_load_supported` becomes `true`
- `state.json` records checkpoint source and final checkpoint path

### Phase 13 Completion Review

Status: completed for platform checkpoint structure and stub backend save/resume/evaluate.
DLRM checkpoint save/load is still explicitly unsupported because the local upstream
`torchrec_dlrm.dlrm_main.py` command path does not expose checkpoint CLI arguments.

Implemented:

- added `runner/checkpoints.py`
- standardized checkpoint directories as `checkpoints/step-000001/`
- writes `model.json`, optional `optimizer.json`, `metadata.json`, and `checkpoints/latest.json`
- stub COLD_START saves a loadable checkpoint
- stub RESUME loads a checkpoint and continues from the saved step
- stub EVALUATE loads the checkpoint and writes `checkpoint_load_supported=true`
- DLRM runs now write `artifacts/checkpoint-status.json` explaining checkpoint limitations
- added automated tests for save, resume, and evaluate-from-checkpoint behavior

Validation run:

```text
python -m unittest discover -s tests
Ran 24 tests in 1.920s
OK (skipped=1)
```

Manual acceptance method:

1. Launch a stub COLD_START job with `Max Steps=1`.
2. Inspect `runs/<job_id>/checkpoints/latest.json`.
3. Launch a stub RESUME job using `latest_checkpoint_dir` as `checkpoint.load_path`.
4. Confirm the resumed job starts at the next step.
5. Launch a stub EVALUATE job using the same checkpoint path.
6. Inspect `evaluation.json`.
7. Launch a short DLRM job and inspect `artifacts/checkpoint-status.json`.

Manual acceptance standards:

- COLD_START creates `checkpoints/step-000001/model.json`
- `checkpoints/latest.json` points to the latest checkpoint directory
- RESUME reads the checkpoint and continues step numbering
- EVALUATE writes `checkpoint_load_supported=true` for stub
- DLRM does not pretend to save/load weights and writes a clear unsupported status artifact
- missing/invalid checkpoint paths fail with a useful error

## Phase 14: Experiment Tracking And Run Comparison

Goal: make runs comparable and reproducible.

Add:

- run tags
- config diff between runs
- metric comparison table
- best metric summary
- run lineage
- parent run id for resume/evaluate
- exported run summary

Suggested files:

```text
summary.json
metrics-summary.json
lineage.json
```

UI additions:

- Compare Runs tab
- filter by backend/status/tag
- compare configs
- compare best AUC/log loss
- compare duration/throughput

Acceptance criteria:

- select two or more runs and compare configs/metrics
- identify best run by chosen metric
- export a summary JSON/CSV

### Phase 14 Completion Review

Status: completed for file-based experiment tracking and UI comparison.

Implemented:

- added `runner/run_tracking.py`
- writes `summary.json`
- writes `metrics-summary.json`
- writes `lineage.json`
- summarizes metric count/latest/best/best_step
- infers parent run id from checkpoint paths when possible
- exports run comparison CSV
- added `Compare Runs` tab
- Compare Runs can select multiple jobs and sort by a metric such as `auc`
- added tests for summary generation and best-run selection

Validation run:

```text
python -m unittest discover -s tests
Ran 25 tests in 2.095s
OK (skipped=1)
```

Manual acceptance method:

1. Launch at least two short stub jobs.
2. Open Compare Runs.
3. Click Refresh Jobs.
4. Select both jobs.
5. Set Sort Metric to `auc`.
6. Click Compare.
7. Inspect the comparison table and best-run JSON.
8. Inspect `runs/comparison-latest.csv`.

Manual acceptance standards:

- each completed run has `summary.json`, `metrics-summary.json`, and `lineage.json`
- Compare Runs table lists selected jobs
- best run matches the highest AUC or lowest loss-style metric
- `runs/comparison-latest.csv` is generated
- resume/evaluate jobs include checkpoint lineage when checkpoint path is under `runs/`

## Phase 15: Observability And Resource Telemetry

Goal: monitor not only model metrics but also system behavior.

Track:

- CPU utilization
- memory utilization
- GPU utilization if available
- GPU memory
- disk usage per run
- process runtime
- throughput
- step time

Implementation options:

- Windows-side process telemetry for UI runner
- WSL-side telemetry for DLRM process
- optional `nvidia-smi` polling when available

Output:

```text
resource-metrics.jsonl
```

Acceptance criteria:

- Monitor can show resource metrics over time
- high-level resource summary is written after run completion
- telemetry failure does not fail the training job

### Phase 15 Completion Review

Status: completed for Windows-side runner telemetry.

Implemented:

- added `runner/telemetry.py`
- records resource samples to `resource-metrics.jsonl`
- writes `artifacts/resource-summary.json`
- tracks runner pid, CPU percent, RSS memory, memory percent, and run directory disk bytes
- telemetry runs around backend execution in `runner/cli.py`
- telemetry failures are logged but do not fail training
- Monitor tab can refresh resource metrics and resource summary
- added telemetry tests

Validation run:

```text
python -m unittest discover -s tests
Ran 26 tests in 2.103s
OK (skipped=1)
```

Manual acceptance method:

1. Launch a stub job with several steps.
2. Open Monitor.
3. Select the job.
4. Click Refresh Resources while the job is running and after it completes.
5. Inspect `resource-metrics.jsonl`.
6. Inspect `artifacts/resource-summary.json`.

Manual acceptance standards:

- `resource-metrics.jsonl` contains timestamped samples
- `resource-summary.json` contains `record_count` and max resource values
- Monitor tab displays recent resource samples
- telemetry failure is logged in `launcher.log` and does not change job status to FAILED
- GPU-specific metrics are not required yet; this phase covers local runner telemetry

## Phase 16: Robust Job Lifecycle And Recovery

Goal: make local execution more reliable after crashes or app restarts.

Add:

- app startup scan of existing runs
- stale RUNNING detection
- process existence check
- state repair for orphaned jobs
- retry policy for launch failures
- structured error classes
- clearer failure summaries

Acceptance criteria:

- restarting the Gradio app does not lose prior runs
- stale RUNNING jobs are marked clearly
- failed launch has actionable error messages
- Stop Job remains safe after restart

### Phase 16 Completion Review

Status: completed for startup recovery and stale process repair.

Implemented:

- `LocalTaskManager` scans existing runs at startup
- active states `LAUNCHING`, `RUNNING`, and `STOPPING` are checked for missing processes
- stale active jobs are marked `FAILED`
- recovered jobs record `recovered_at` and `recovery_reason`
- stale job error messages preserve the original active status
- `list_jobs()` also performs recovery before returning jobs
- launch failures now update `state.json` with actionable error messages
- added automated stale RUNNING recovery test

Validation run:

```text
python -m unittest discover -s tests
Ran 27 tests in 2.241s
OK (skipped=1)
```

Manual acceptance method:

1. Launch a long-running job.
2. Close the Gradio app process while the job is running.
3. Stop or let the runner process exit.
4. Restart the Gradio app.
5. Open Logs/Artifacts and refresh jobs.
6. Inspect the affected `state.json`.

Manual acceptance standards:

- existing run directories remain visible after app restart
- orphaned RUNNING/LAUNCHING/STOPPING jobs become `FAILED`
- `state.json` includes `recovered_at` and `recovery_reason=missing_process`
- completed jobs are not modified by recovery
- Stop Job remains safe for terminal jobs

## Phase 17: Packaging And Reproducible Environment

Goal: make the project installable and reproducible.

Add:

- pinned requirements
- environment check script
- WSL environment check script
- optional setup script
- versioned release notes
- sample configs

Suggested files:

```text
scripts/check_windows_env.ps1
scripts/check_wsl_dlrm.sh
examples/stub-smoke.yaml
examples/dlrm-random-smoke.yaml
examples/dlrm-real-data-smoke.yaml
```

Acceptance criteria:

- fresh setup can follow README and scripts
- environment check reports missing dependencies clearly
- examples can be launched without hand-editing large configs

### Phase 17 Completion Review

Status: completed for reproducible local setup artifacts.

Implemented:

- added `scripts/check_windows_env.ps1`
- added `scripts/check_wsl_dlrm.sh`
- added `examples/stub-smoke.yaml`
- added `examples/dlrm-random-smoke.yaml`
- added `examples/dlrm-real-data-smoke.yaml`
- added `RELEASE_NOTES.md`
- added tests to ensure example YAML files parse as `PrototypeConfig`

Validation run:

```text
python -m unittest discover -s tests
Ran 28 tests in 2.231s
OK (skipped=1)
```

Manual acceptance method:

1. Run `powershell -ExecutionPolicy Bypass -File scripts/check_windows_env.ps1`.
2. In WSL, run `bash scripts/check_wsl_dlrm.sh`.
3. Inspect the `examples/` YAML files.
4. Use `examples/stub-smoke.yaml` as a known-good local smoke config.
5. Use `examples/dlrm-real-data-smoke.yaml` after Criteo numpy preprocessing is complete.

Manual acceptance standards:

- Windows check reports Python, WSL, and core package status clearly
- WSL check reports DLRM root, Python env, torch, and torchrec status clearly
- example YAML files parse successfully
- release notes describe the current capability set
- a fresh user has enough scripts/examples to reproduce a smoke run

## Phase 18: Industrial Readiness Review

Goal: evaluate whether the project has moved beyond prototype status.

Required before calling it industrial-grade:

- real dataset training works
- checkpoint save/load works
- evaluate from checkpoint works
- run comparison works
- failures are explainable
- core paths have automated tests
- documentation matches behavior
- run artifacts are reproducible
- no manual YAML edits are required for common workflows

Industrial acceptance milestone:

```text
From the UI, configure a real dataset, launch DLRM training, save checkpoints, resume from a
checkpoint, evaluate the checkpoint, compare runs, and inspect metrics/artifacts without touching
source code or hand-editing run files.
```

### Phase 18 Completion Review

Status: completed as an explicit readiness review.

Implemented:

- added `INDUSTRIAL_READINESS_REVIEW.md`
- documented completed industrial foundations
- documented remaining blockers
- defined the UI-only industrial acceptance milestone
- recommended the next engineering order
- recorded that the project is not yet fully industrial-grade

Validation run:

```text
python -m unittest discover -s tests
Ran 28 tests in 2.231s
OK (skipped=1)
```

Manual acceptance method:

1. Open `INDUSTRIAL_READINESS_REVIEW.md`.
2. Confirm the completed capabilities match the current UI and artifacts.
3. Confirm the blocker list is accurate.
4. Try the industrial acceptance milestone from the UI.

Manual acceptance standards:

- the review does not claim DLRM checkpoint save/load is complete
- the review clearly identifies real-data DLRM smoke and DLRM checkpointing as next blockers
- the review gives a concrete sequence for the next engineering cycle
- all current automated tests pass

## First Two Critical Gaps Resolution

Date: 2026-07-28

Goal:

1. Solve the real DLRM checkpoint save/load/resume/evaluate gap for the current smoke scope.
2. Solve the real Criteo data smoke training gap.

### DLRM Checkpoint Result

Status: completed for single-process smoke validation.

Implemented:

- patched local `C:\Users\han\Desktop\dlrm\torchrec_dlrm\dlrm_main.py`
- added DLRM CLI arguments:
  - `--checkpoint_save_dir`
  - `--checkpoint_load_path`
  - `--checkpoint_save_optimizer`
- saves `model.pt`, optional `optimizer.pt`, `metadata.json`, and `latest.json`
- loads `model.pt` before evaluation/resume
- loads `optimizer.pt` when available
- updated prototype DLRM backend to pass checkpoint save/load arguments
- DLRM `RESUME` is no longer rejected by the prototype backend
- added small-model DLRM config fields for smoke stability:
  - `model.num_embeddings`
  - `model.embedding_dim`
  - `model.dense_arch_layer_sizes`
  - `model.over_arch_layer_sizes`

Validated runs:

```text
COLD_START real-data checkpoint save:
runs/20260728-150552-7e92c28f

EVALUATE checkpoint load:
runs/20260728-150905-887685af

RESUME checkpoint load and save:
runs/20260728-150919-da311650
```

Validation evidence:

- `train-rank0.log` contains `Loaded checkpoint from`
- `evaluation.json.checkpoint_load_supported` is `true`
- `checkpoints/step-final/model.pt` exists
- `checkpoints/step-final/optimizer.pt` exists when optimizer saving is enabled
- `checkpoints/latest.json` points to the saved checkpoint

Remaining boundary:

- this is single-process smoke checkpointing
- multi-process sharded production checkpointing still requires Torch Distributed Checkpoint or an
  equivalent production checkpoint implementation

### Real Criteo Data Result

Status: completed for tiny real-data smoke validation.

Preprocessed dataset:

```text
C:\Users\han\Desktop\prototype\data\criteo_kaggle_sample_npy
```

Preprocessed file shapes:

```text
train_dense.npy  (1999, 13) float32
train_sparse.npy (1999, 26) int64
train_labels.npy (1999, 1) int32
```

Validated smoke run:

```text
runs/20260728-150552-7e92c28f
```

Validation evidence:

- `state.json.status` is `SUCCEEDED`
- `metrics.jsonl` contains `val_auc`, `test_auc`, `val_samples`, and `test_samples`
- `train-rank0.log` contains DLRM epoch/evaluation output
- checkpoint artifacts were saved after the real-data run

Manual acceptance method:

1. Start the Gradio app.
2. Create a DLRM job with `Data Format=criteo_binary`.
3. Set `Dataset Name=criteo_kaggle`.
4. Set `Criteo Binary Path` to
   `C:\Users\han\Desktop\prototype\data\criteo_kaggle_sample_npy`.
5. Use `Processes per Node=1`, `Batch Size=4`, `Test Batch Size=4`, `Max Steps=1`.
6. Enable checkpoint saving.
7. Launch the job.
8. Use the resulting `checkpoints/step-final` path for an EVALUATE job.
9. Use the same checkpoint path for a RESUME job.

Manual acceptance standards:

- COLD_START real-data job succeeds
- EVALUATE loads the saved checkpoint and succeeds
- RESUME loads the saved checkpoint, trains one step, saves a new checkpoint, and succeeds
- logs contain `Loaded checkpoint from`
- `evaluation.json.checkpoint_load_supported` is `true`
- no source-code editing is required for the UI workflow after this patch

## Design Document Alignment Update

Date: 2026-07-29

Goal: align the prototype more closely with
`C:\Users\han\Desktop\基于 TorchRec 的推荐模型训练平台原型工具设计文档.md`.

Implemented:

- added `PrecisionConfig`
- added `CheckpointLoadMode`
- validates `device.gpu_ids`
- validates DLRM `nproc_per_node <= len(device.gpu_ids)`
- expanded Create Job UI with:
  - model config file
  - DLRM small-model fields
  - GPU IDs
  - embedding placement
  - cache load factor
  - precision settings
  - checkpoint save frequency and keep-last fields
  - optimizer save toggle
  - profile start/end/shape/memory fields
- DLRM backend now exports `CUDA_VISIBLE_DEVICES` from configured GPU IDs
- DLRM backend writes `artifacts/capability-report.json`
- LocalTaskManager defaults to one active job at a time
- terminal jobs write `artifacts/run-artifacts.zip`
- Logs tab can download the run artifact bundle
- added `runner/profile.py`
- profile-enabled jobs write:
  - `profiles/profile-request.json`
  - `profiles/runner-profile.json`
- added `scripts/check_dlrm_checkpoint_patch.ps1`
- added `patches/README.md`
- documented local DLRM checkpoint patch verification

Validation run:

```text
python -m unittest discover -s tests
Ran 35 tests in 2.896s
OK (skipped=1)

python -m compileall config.py task_manager.py runner ui
OK

powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
DLRM checkpoint patch is present.
```

Manual acceptance method:

1. Start the Gradio app.
2. Open Create Job and confirm the new GPU, precision, checkpoint, model, and profile fields exist.
3. Launch a stub job with `Profile Enabled=true`.
4. Confirm `profiles/profile-request.json` and `profiles/runner-profile.json` exist.
5. Confirm `artifacts/run-artifacts.zip` exists after completion.
6. Open Logs and use Download Run Bundle.
7. Launch a DLRM job with `GPU IDs=0` and inspect `command.json`.
8. Confirm the backend command includes `export CUDA_VISIBLE_DEVICES=0`.
9. Run `scripts/check_dlrm_checkpoint_patch.ps1`.

Manual acceptance standards:

- invalid duplicate/negative/empty GPU IDs fail config validation
- DLRM `nproc_per_node` greater than configured GPU count fails config validation
- only one active job can be launched by default
- completed runs contain a downloadable zip bundle
- profile-enabled runs contain profile request and runner profile files
- DLRM capability report clearly distinguishes mapped settings from recorded-but-not-fully-mapped
  settings

Remaining design gaps:

- production multi-process sharded checkpointing
- full PyTorch operator-level profiler traces
- WSL/GPU telemetry
- direct parquet-to-DLRM training
- custom user-provided `model.py` runner
- complete embedding placement/GPU cache/precision backend mapping

## WSL/GPU Telemetry And Profiler Update

Date: 2026-07-29

Goal:

1. Add WSL/GPU telemetry.
2. Add a real PyTorch profiler path where the current environment supports it.

Implemented:

- `runner/telemetry.py` now samples multiple sources:
  - Windows runner process
  - Windows child process tree
  - run directory disk usage
  - `nvidia-smi` GPU utilization and memory
  - WSL Python/torchrun process aggregates when visible
- `resource-metrics.jsonl` now records telemetry availability flags instead of failing the job when
  GPU or WSL sampling is unavailable
- `artifacts/resource-summary.json` now includes max GPU and WSL aggregate fields
- `runner/profile.py` now attempts to use `torch.profiler`
- when `torch.profiler` is available, `profiles/trace.json` is exported as Chrome trace JSON
- when torch is not installed in the Windows runner environment, `runner-profile.json` records the
  fallback reason

Validation run:

```text
python -m unittest discover -s tests
Ran 35 tests in 2.566s
OK (skipped=1)

python -m compileall config.py task_manager.py runner ui
OK
```

Smoke validation:

```text
runs/20260729-003331-63e874cf
```

Observed:

- `resource-metrics.jsonl` included `gpu_telemetry_available: true`
- GPU utilization and memory were captured through `nvidia-smi`
- WSL sampling reported unavailable when no WSL torch process was visible
- Windows runner environment did not have torch, so profiler recorded:
  `torch.profiler unavailable: ModuleNotFoundError: No module named 'torch'`

Manual acceptance method:

1. Launch a short job with `Profile Enabled=true`.
2. Inspect `resource-metrics.jsonl`.
3. Inspect `artifacts/resource-summary.json`.
4. Inspect `profiles/profile-request.json`.
5. Inspect `profiles/runner-profile.json`.
6. If Windows runner environment has torch installed, confirm `profiles/trace.json` exists.
7. During a DLRM job, confirm WSL process aggregates appear when WSL `python` or `torchrun`
   processes are visible.

Manual acceptance standards:

- missing `nvidia-smi` does not fail the job
- missing WSL process samples do not fail the job
- GPU fields are present when `nvidia-smi` is available
- WSL telemetry fields are present with availability status
- profiler writes either `trace.json` or a clear fallback reason
- all telemetry/profile artifacts are local to the run directory

Remaining boundary:

- the current profiler wraps the Windows runner process
- DLRM itself runs in a child WSL `torchrun` process, so full DLRM operator-level traces still need
  instrumentation inside `torchrec_dlrm.dlrm_main.py` or a custom runner

## Design Document Gap Closure Update

Date: 2026-08-01

Goal: close the remaining practical gap between the current project and the design document's
prototype tool target.

This round focused on four areas:

1. User-defined `model.py` extensibility.
2. Business-style parquet data conversion into DLRM-compatible numpy arrays.
3. DLRM child-process profiler traces and torchrun rank logs.
4. Documentation and repeatable verification.

### Custom Model Backend

Status: completed for local smoke experiments.

Implemented:

- added `BackendName.CUSTOM`
- added `runner/backends/custom_backend.py`
- registered the `custom` backend in `runner/backends/__init__.py`
- added `examples/models/custom_simple_model.py`
- added `examples/custom-model-smoke.yaml`
- added automated tests in `tests/test_custom_backend.py`

Custom model contract:

```python
def train_step(step: int, config: dict) -> dict[str, float]:
    ...

def evaluate(config: dict, checkpoint: dict) -> dict[str, float]:
    ...
```

Behavior:

- COLD_START and RESUME call `train_step`
- EVALUATE calls `evaluate` when present
- returned numeric metrics are appended to `metrics.jsonl`
- checkpoints are saved through the prototype checkpoint contract
- `artifacts/custom-model-contract.json` records the model path and supported functions

Validation:

```text
python -m unittest tests.test_custom_backend tests.test_examples tests.test_config
Ran 13 tests
OK
```

Manual acceptance method:

1. Start the Gradio app.
2. Select `Backend=custom`.
3. Use `examples/models/custom_simple_model.py` as the model file path.
4. Launch a COLD_START job with `Max Steps=2`.
5. Inspect `train-rank0.log`, `metrics.jsonl`, `checkpoints/`, and
   `artifacts/custom-model-contract.json`.
6. Launch an EVALUATE job from the saved checkpoint.

Manual acceptance standards:

- invalid missing model files are rejected with a useful error
- COLD_START writes custom metrics to `metrics.jsonl`
- checkpoint save/load works through the prototype checkpoint format
- EVALUATE writes `evaluation.json`
- the custom contract artifact clearly identifies which functions were available

### Parquet To DLRM Numpy Conversion

Status: completed for small CTR parquet datasets.

Implemented:

- added `runner/parquet_converter.py`
- added CLI entrypoint `runner/convert_parquet.py`
- added `examples/parquet-conversion-smoke.yaml`
- added `examples/schemas/parquet-smoke-schema.yaml`
- added automated tests in `tests/test_parquet_converter.py`

Behavior:

- validates parquet schema and required columns before conversion
- reads train, validation, and test parquet splits
- writes Criteo-style numpy arrays:
  - `<split>_dense.npy`
  - `<split>_sparse.npy`
  - `<split>_labels.npy`
- writes `conversion-manifest.json`

Validation:

```text
python -m unittest tests.test_parquet_converter tests.test_data_validation
Ran 3 tests
OK (skipped=2 when pyarrow is unavailable in the Windows runner environment)
```

Manual acceptance method:

1. Prepare train, validation, and test parquet files.
2. Prepare a schema YAML with `label`, `dense_features`, and `sparse_features`.
3. Run:

```powershell
cd C:\Users\han\Desktop\prototype
python -m prototype.runner.convert_parquet --config examples\parquet-conversion-smoke.yaml --output-dir data\converted_criteo_npy
```

4. Launch DLRM with `Data Format=criteo_binary` and `Criteo Binary Path` set to the output
   directory.

Manual acceptance standards:

- missing parquet/schema paths fail clearly
- invalid labels or missing columns fail before conversion
- output arrays have aligned row counts across dense/sparse/labels
- `conversion-manifest.json` records source paths, schema, split names, row counts, and output files
- converted output can be used by the existing DLRM real-data path

### DLRM Profiler And Rank Logs

Status: completed for patched local DLRM smoke runs.

Implemented in the prototype:

- `runner/backends/dlrm_backend.py` passes torchrun log routing:
  - `--log_dir <run_dir>/logs`
  - `--redirects 3`
  - `--tee 3`
- profile-enabled DLRM jobs pass:
  - `--profile_dir <run_dir>/profiles/dlrm`
  - `--profile_record_shapes`
  - `--profile_memory`
- added `scripts/check_dlrm_profiler_patch.ps1`
- added DLRM backend tests for rank log and profiler arguments

Implemented in local DLRM:

- patched `C:\Users\han\Desktop\dlrm\torchrec_dlrm\dlrm_main.py`
- added DLRM CLI arguments:
  - `--profile_dir`
  - `--profile_record_shapes`
  - `--profile_memory`
- added `_maybe_profile`
- exports per-rank Chrome trace files such as `profiles/dlrm/rank0-trace.json`
- writes per-rank profile summaries such as `profiles/dlrm/rank0-summary.json`

Validation:

```text
python -m unittest tests.test_dlrm_backend
Ran 8 tests
OK

python -c "compile(open(r'C:\Users\han\Desktop\dlrm\torchrec_dlrm\dlrm_main.py', encoding='utf-8').read(), 'dlrm_main.py', 'exec')"
OK

powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
DLRM checkpoint patch is present.

powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_profiler_patch.ps1
DLRM profiler patch is present.
```

Manual acceptance method:

1. Start the Gradio app.
2. Launch a short DLRM job with `Profile Enabled=true`.
3. Use `Processes per Node=1` first.
4. Inspect `command.json`.
5. Inspect `logs/`.
6. Inspect `profiles/dlrm/`.

Manual acceptance standards:

- `command.json.backend_command` contains `--log_dir`, `--redirects 3`, and `--tee 3`
- `command.json.backend_command` contains `--profile_dir`, `--profile_record_shapes`, and
  `--profile_memory`
- `logs/` contains torchrun-routed output files when torchrun creates them
- `profiles/dlrm/rank0-summary.json` exists after a patched DLRM profile run
- `profiles/dlrm/rank0-trace.json` exists when torch profiler export succeeds
- profiler export failure writes a clear summary error instead of hiding the condition

### Documentation And Readiness Review

Status: completed.

Updated:

- `README.md`
- `INDUSTRIAL_READINESS_REVIEW.md`
- `DEVELOPMENT_PLAN.md`

Current remaining gaps after this update:

- production multi-process sharded checkpointing with Torch Distributed Checkpoint or equivalent
- larger Criteo Kaggle subset and Criteo 1TB validation
- one-click UI flow for parquet conversion plus DLRM launch
- deeper backend mapping for precision, embedding placement, and GPU cache controls
- model export and serving artifacts
- CI pipeline and optional remote/local database-backed experiment tracking

## V1 Prototype Gap Closure Pass

Date: 2026-08-02

Goal: move the project closer to the design document's V1 prototype target, not full industrial
production scope.

### Phase 1: UI Parquet Validation And Conversion

Status: completed.

Implemented:

- Create Job now includes `Parquet Conversion Output`
- added `Validate Parquet` button
- added `Convert Parquet` button
- validation displays a parquet profile JSON
- conversion displays `conversion-manifest.json`
- successful conversion updates `Data Format` to `criteo_binary`
- successful conversion fills `Criteo Binary Path` with the converted output directory

Validation:

```text
python -m unittest tests.test_examples tests.test_config tests.test_data_validation tests.test_parquet_converter
Ran 15 tests
OK (skipped=2 when pyarrow is unavailable)
```

Manual acceptance:

1. Start the Gradio app.
2. Set `Data Format=parquet`.
3. Fill train, validation, test, schema, and conversion output paths.
4. Click `Validate Parquet`.
5. Confirm a JSON profile appears.
6. Click `Convert Parquet`.
7. Confirm `Data Format` changes to `criteo_binary`.
8. Confirm `Criteo Binary Path` is set to the output directory.
9. Launch a DLRM job using that converted path.

Acceptance standards:

- invalid parquet paths fail with a clear message
- successful validation shows split profiles
- successful conversion writes DLRM numpy arrays and `conversion-manifest.json`
- no command-line conversion is required for the UI workflow

### Phase 2: V1 Throughput And Step-Time Metrics

Status: completed for project-owned runner loops.

Implemented:

- stub backend emits `step_time_seconds`
- stub backend emits `samples_per_second`
- stub backend emits `batches_per_second`
- custom backend emits the same metrics
- log parser can parse these fields from log lines
- Monitor tab shows `Throughput` and `Step Time` plots

Validation:

```text
python -m unittest tests.test_stub_and_task_manager tests.test_custom_backend tests.test_log_parser
Ran 13 tests
OK
```

Manual acceptance:

1. Launch a stub or custom training job.
2. Open Monitor.
3. Refresh metrics.
4. Inspect the metrics table and the new plots.

Acceptance standards:

- `metrics.jsonl` contains step time and throughput metrics
- Monitor displays throughput and step-time plots without errors
- existing loss and AUC plots still work

### Phase 3: TorchRec V1 Runner Scaffold And Model Contract

Status: completed as scaffold and validation layer.

Implemented:

- added `runner/torchrec_runner/contract.py`
- added `runner/torchrec_runner/entry.py`
- added `examples/models/torchrec_v1_model.py`
- added `tests/test_torchrec_contract.py`

V1 required model functions:

```python
def build_model(config: dict):
    ...

def build_embedding_configs(config: dict) -> list:
    ...
```

Validation:

```text
python -m unittest tests.test_torchrec_contract tests.test_examples
Ran 3 tests
OK
```

Manual acceptance:

1. Create or edit a model file with `build_model` and `build_embedding_configs`.
2. Run the TorchRec runner scaffold entry with a resolved config and run directory.
3. Inspect `artifacts/torchrec-model-contract.json`.
4. Inspect `artifacts/torchrec-runner-status.json`.

Acceptance standards:

- missing required functions fail clearly
- valid model files produce a contract report
- runtime availability for torch and torchrec is recorded

Remaining boundary:

- this is not yet the full DMP / TrainPipelineSparseDist training loop

### Phase 4: Checkpoint Success Markers And Keep-Last

Status: completed for project-owned checkpoints and DLRM smoke checkpoint finalization.

Implemented:

- project-owned checkpoints write `_SUCCESS` after save completion
- project-owned checkpoints support `keep_last` pruning
- stub/custom backends pass configured `checkpoint.keep_last`
- DLRM backend writes `_SUCCESS` when the smoke `model.pt` exists
- DLRM checkpoint status records marker path and existence

Validation:

```text
python -m unittest tests.test_stub_and_task_manager tests.test_dlrm_backend tests.test_custom_backend
Ran 20 tests
OK
```

Manual acceptance:

1. Launch a stub/custom job with checkpoint enabled.
2. Inspect `checkpoints/step-000001/_SUCCESS`.
3. Run enough project-owned checkpoint saves to exceed `keep_last`.
4. Confirm old step directories are pruned.
5. Launch a DLRM checkpoint smoke job.
6. Inspect `checkpoints/step-final/_SUCCESS` and `artifacts/checkpoint-status.json`.

Acceptance standards:

- `_SUCCESS` is written only after checkpoint files exist
- `latest.json` points to the latest checkpoint
- `keep_last` keeps only the newest configured number of step checkpoints
- DLRM smoke checkpoints expose success marker status in artifacts

### Current V1 Remaining Gaps

- full internal TorchRec Runner training loop using DMP and TrainPipelineSparseDist
- production-grade multi-process sharded checkpointing
- strict profiler start/end step scheduling inside the real training loop
- full GPU Cache / embedding placement / precision backend mapping
- deeper stage timing and embedding/cache table metrics

## V1 Prototype Gap Closure Continuation

Date: 2026-08-02

Goal: continue closing the design document V1 prototype gap by making the internal runner path
launchable and more observable from the UI/backend system.

### Phase 6: TorchRec V1 Backend Integration

Status: completed.

Implemented:

- added `BackendName.TORCHREC_V1`
- added `runner/backends/torchrec_v1_backend.py`
- registered `torchrec_v1` in the backend registry
- added `examples/torchrec-v1-contract-smoke.yaml`
- added backend command generation tests

Validation:

```text
python -m unittest tests.test_torchrec_v1_backend tests.test_torchrec_contract tests.test_examples tests.test_config
Ran 16 tests
OK
```

Manual acceptance:

1. Start Gradio.
2. Select `Backend=torchrec_v1`.
3. Set `Model File=examples/models/torchrec_v1_model.py`.
4. Launch a one-step job.
5. Inspect `command.json`.

Acceptance standards:

- backend command uses WSL and `torchrun`
- backend command invokes `prototype.runner.torchrec_runner.entry`
- command sets `CUDA_VISIBLE_DEVICES`
- command writes output under the selected run directory

### Phase 7: Minimal Internal Runner Loop

Status: completed.

Implemented:

- `runner/torchrec_runner/entry.py` now exposes `run(config, run_dir)`
- validates the V1 `model.py` contract
- writes `metrics.jsonl`
- writes project-owned checkpoint and `_SUCCESS`
- supports EVALUATE from the project-owned checkpoint format
- writes `evaluation.json`

Validation:

```text
python -m unittest tests.test_torchrec_contract tests.test_torchrec_v1_backend
Ran 5 tests
OK
```

Manual acceptance:

1. Launch `Backend=torchrec_v1`.
2. Inspect `artifacts/torchrec-model-contract.json`.
3. Inspect `artifacts/torchrec-runner-status.json`.
4. Inspect `metrics.jsonl`.
5. If checkpoint saving is enabled, inspect `checkpoints/latest.json` and `_SUCCESS`.

Acceptance standards:

- valid model contracts succeed
- metrics are emitted
- checkpoint save/evaluate works through the project-owned checkpoint format
- status artifact clearly says this is still a minimal loop

### Phase 8: V1 Stage Timing And Profile Window Metrics

Status: completed for project-owned loops.

Implemented:

- added `runner/v1_metrics.py`
- emits `profile_window_active`
- emits stage timing metrics:
  - `data_wait_seconds`
  - `h2d_seconds`
  - `input_distribution_seconds`
  - `embedding_lookup_seconds`
  - `dense_forward_seconds`
  - `backward_seconds`
  - `optimizer_seconds`
- stub/custom/torchrec_v1 loops use the shared V1 metric helper
- Monitor tab now includes `Stage Timing`

Validation:

```text
python -m unittest tests.test_v1_metrics tests.test_stub_and_task_manager tests.test_custom_backend tests.test_torchrec_contract
Ran 16 tests
OK
```

Manual acceptance:

1. Launch a stub, custom, or torchrec_v1 training job.
2. Refresh Monitor metrics.
3. Inspect metrics table and Stage Timing plot.

Acceptance standards:

- `metrics.jsonl` contains stage timing metrics
- profile window metrics are `1.0` only inside the configured profile step range
- Monitor renders Stage Timing without breaking loss/AUC/throughput plots

### Phase 9: V1 Capability Report

Status: completed.

Implemented:

- added `runner/capability.py`
- runner writes `artifacts/v1-capability-report.json`
- report records requested GPU IDs, placement, cache ratio, and precision
- report separates mapped behavior from not-yet-mapped behavior
- added capability report tests

Validation:

```text
python -m unittest tests.test_capability tests.test_torchrec_v1_backend tests.test_config
Ran 15 tests
OK
```

Manual acceptance:

1. Launch any backend job.
2. Inspect `artifacts/v1-capability-report.json`.
3. Launch a job with `Embedding Placement=MANAGED_CACHING` or BF16 precision.
4. Inspect `not_yet_mapped`.

Acceptance standards:

- report is written for launched jobs
- requested config values are preserved
- mapped fields do not overclaim unsupported runtime behavior
- pending GPU Cache/precision/DMP work is explicit

### Updated Current V1 Remaining Gaps

- full internal TorchRec Runner with actual DistributedModelParallel construction
- TrainPipelineSparseDist integration
- real TorchRec DataLoader that produces `Batch` / `KeyedJaggedTensor`
- production-grade sharded checkpointing with Torch Distributed Checkpoint
- real profiler schedule inside the actual TorchRec training loop
- true FBGEMM managed caching and precision mapping
- table-level embedding/cache metrics from real runtime APIs

## V1 Prototype Gap Closure Continuation 2

Date: 2026-08-02

Goal: continue moving the internal `torchrec_v1` backend toward the design document's Runner shape
by adding data, planning, contract, and artifact visibility layers.

### Phase 11: TorchRec V1 Data Plan

Status: completed.

Implemented:

- added `runner/torchrec_runner/data.py`
- writes `artifacts/torchrec-data-plan.json`
- records batch size per rank and global batch size
- records dense/sparse/label batch contract
- records Criteo-like feature schema for random/criteo_binary formats
- records expected DLRM numpy split files for `criteo_binary`

Validation:

```text
python -m unittest tests.test_torchrec_data_plan tests.test_torchrec_contract
Ran 6 tests
OK
```

Manual acceptance:

1. Launch `Backend=torchrec_v1`.
2. Open Artifacts.
3. Inspect `artifacts/torchrec-data-plan.json`.

Acceptance standards:

- data plan lists dense, sparse, and label contracts
- sparse features are identified as future `KeyedJaggedTensor`
- global batch size equals `batch_size * nproc_per_node`
- criteo_binary paths show expected numpy files

### Phase 12: TorchRec V1 Training Plan

Status: completed.

Implemented:

- added `runner/torchrec_runner/plan.py`
- writes `artifacts/torchrec-training-plan.json`
- records model loading, embedding config, dataloader, placement, precision, sharding, DMP,
  optimizer, checkpoint, TrainPipelineSparseDist, and profile steps
- distinguishes `implemented`, `contract_available`, `optional_missing`, and `planned`

Validation:

```text
python -m unittest tests.test_torchrec_training_plan tests.test_torchrec_contract
Ran 5 tests
OK
```

Manual acceptance:

1. Launch `Backend=torchrec_v1`.
2. Inspect `artifacts/torchrec-training-plan.json`.

Acceptance standards:

- plan includes DMP and TrainPipelineSparseDist steps
- unimplemented runner steps are marked `planned`
- required contract functions are marked available
- profile/checkpoint status reflects the selected config

### Phase 13: Model Contract Signature Checks

Status: completed.

Implemented:

- V1 contract now records recommended signatures
- required functions must accept `config` as the first parameter
- optional function signatures are recorded
- `examples/models/torchrec_v1_model.py` now includes `train_step` and `evaluate`

Validation:

```text
python -m unittest tests.test_torchrec_contract tests.test_torchrec_training_plan tests.test_examples
Ran 7 tests
OK
```

Manual acceptance:

1. Create a model file with an incompatible `build_model(settings)` signature.
2. Launch or validate with `torchrec_v1`.
3. Confirm the contract error mentions incompatible required signatures.

Acceptance standards:

- missing required functions fail
- incompatible required signatures fail
- valid example model passes

### Phase 14: Artifact UI Views

Status: completed.

Implemented:

- Artifacts tab displays:
  - `artifacts/v1-capability-report.json`
  - `artifacts/torchrec-model-contract.json`
  - `artifacts/torchrec-data-plan.json`
  - `artifacts/torchrec-training-plan.json`

Validation:

```text
python -m compileall ui/artifacts_tab.py
python -m unittest tests.test_stub_and_task_manager
Ran 10 tests
OK
```

Manual acceptance:

1. Launch a `torchrec_v1` job.
2. Open Artifacts.
3. Refresh the selected job.
4. Inspect the new JSON viewers.

Acceptance standards:

- new JSON viewers populate when files exist
- empty viewers do not break non-torchrec jobs
- artifact file list still works

### Updated V1 Remaining Gaps After Continuation 2

- actual TorchRec tensor and KeyedJaggedTensor materialization
- actual DMP model wrapping and sharding planner execution
- actual TrainPipelineSparseDist loop
- actual Torch Distributed Checkpoint for sharded multi-rank state
- actual FBGEMM managed caching runtime configuration
- actual mixed precision mapping
- runtime table-level embedding/cache metrics

## V1 Prototype Gap Closure Continuation 3

Date: 2026-08-05

Goal: start converting `torchrec_v1` planned runtime steps into materialization layers that can
use real torch/torchrec when available and report clear fallback status otherwise.

### Phase 16: Batch / KeyedJaggedTensor Materialization

Status: completed.

Implemented:

- added `runner/torchrec_runner/materialize.py`
- writes `artifacts/torchrec-batch-materialization.json`
- random format creates synthetic dense/sparse/label preview arrays
- criteo_binary reads converted numpy preview files when present
- creates torch dense and label tensors when torch is available
- creates TorchRec `KeyedJaggedTensor` when torchrec is available
- records fallback errors when torch/torchrec is unavailable

Validation:

```text
python -m unittest tests.test_torchrec_materialize tests.test_torchrec_contract
Ran 7 tests
OK
```

### Phase 17: Embedding Config Materialization

Status: completed.

Implemented:

- added `runner/torchrec_runner/embedding.py`
- calls `model.build_embedding_configs(config)`
- writes `artifacts/torchrec-embedding-configs.json`
- records actual config object fields when returned by model.py
- falls back to default Criteo-like embedding config descriptions when the example returns `[]`
- records TorchRec `EmbeddingBagConfig` import availability

Validation:

```text
python -m unittest tests.test_torchrec_embedding tests.test_torchrec_contract
Ran 5 tests
OK
```

### Phase 18: Runtime Smoke Readiness

Status: completed.

Implemented:

- added `runner/torchrec_runner/runtime.py`
- writes `artifacts/torchrec-runtime-smoke.json`
- summarizes torch availability, torchrec availability, tensor creation, KJT creation, and
  embedding config availability
- computes `ready_for_dmp_smoke`
- records fallback reasons

Validation:

```text
python -m unittest tests.test_torchrec_runtime tests.test_torchrec_contract
Ran 6 tests
OK
```

### Phase 19: Artifact Views And Plan Status Update

Status: completed.

Implemented:

- Artifacts tab displays:
  - `artifacts/torchrec-batch-materialization.json`
  - `artifacts/torchrec-embedding-configs.json`
  - `artifacts/torchrec-runtime-smoke.json`
- training plan marks batch and embedding materialization as `implemented_fallback_or_runtime`

Manual acceptance:

1. Launch `Backend=torchrec_v1`.
2. Open Artifacts.
3. Inspect batch materialization, embedding configs, runtime smoke, and training plan.

Acceptance standards:

- materialization artifacts exist
- when torch/torchrec are unavailable, fallback reasons are explicit
- when torch/torchrec are available, actual tensor/KJT creation is reflected
- training plan distinguishes implemented materialization from still-planned DMP/TrainPipeline

### Updated V1 Remaining Gaps After Continuation 3

- actual DMP model wrapping and sharding planner execution
- actual TrainPipelineSparseDist training/evaluation loop
- actual Torch Distributed Checkpoint for sharded multi-rank state
- actual FBGEMM managed caching runtime configuration
- actual mixed precision mapping
- runtime table-level embedding/cache metrics

## V1 Prototype Gap Closure Continuation 4

Date: 2026-08-05

Goal: begin closing the remaining actual TorchRec runtime gap without falsely claiming DMP is
complete when the current Codex environment cannot access WSL.

### Phase 21: Sharding Planner Readiness Artifact

Status: completed.

Implemented:

- added `runner/torchrec_runner/sharding.py`
- writes `artifacts/torchrec-sharding-plan-readiness.json`
- checks whether TorchRec planner components can be imported
- attempts `Topology` creation when available
- does not claim `collective_plan` succeeded unless a real model and distributed process group are
  available
- training plan now reports sharding status more precisely than plain `planned`

Validation:

```text
python -m unittest tests.test_torchrec_sharding tests.test_torchrec_training_plan tests.test_torchrec_contract
Ran 8 tests
OK
```

### Phase 22: DMP Readiness Environment Check

Status: blocked in Codex execution environment, script provided for user-side validation.

Observed blocker:

```text
wsl -d Ubuntu-22.04 ...
WSL/Service/E_ACCESSDENIED
```

This means Codex cannot currently execute WSL commands from this sandboxed context, so I cannot
honestly verify a real single-card DMP smoke here.

Implemented:

- added `scripts/check_torchrec_v1_dmp_readiness.ps1`
- checks WSL imports for:
  - torch
  - torchrec
  - DistributedModelParallel
  - EmbeddingShardingPlanner / Topology
  - TrainPipelineSparseDist
  - torch.distributed.checkpoint

Manual acceptance:

```powershell
cd C:\Users\han\Desktop\prototype
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

Acceptance standards:

- script exits 0
- JSON output shows all checked capabilities as `true`
- if it exits non-zero, the JSON `errors` list is the blocker to solve before real DMP smoke

### Phase 23: Runtime Smoke Next-Required Fields

Status: completed.

Implemented:

- `artifacts/torchrec-runtime-smoke.json` now includes `next_required_for_dmp_smoke`
- the list records the exact prerequisites before a real DMP smoke can be claimed:
  - torch and torchrec import successfully in WSL
  - KeyedJaggedTensor materialization succeeds
  - EmbeddingBagConfig materialization succeeds
  - torch.distributed process group can initialize under torchrun
  - `model.build_model(config)` returns a real `torch.nn.Module`
  - `EmbeddingShardingPlanner.collective_plan` can run against that model

Validation:

```text
python -m unittest tests.test_torchrec_runtime tests.test_torchrec_sharding tests.test_torchrec_contract
Ran 8 tests
OK
```

Manual acceptance:

1. Launch `Backend=torchrec_v1`.
2. Open Artifacts.
3. Inspect `artifacts/torchrec-runtime-smoke.json`.

Acceptance standards:

- `next_required_for_dmp_smoke` exists
- it does not claim real DMP execution
- missing runtime prerequisites are visible as fallback reasons

### Phase 24: Documentation, UI Visibility, And Regression Review

Status: completed.

Implemented:

- Artifacts tab now displays `artifacts/torchrec-sharding-plan-readiness.json`
- `README.md` documents:
  - the sharding readiness artifact
  - the TorchRec V1 DMP readiness check script
  - the exact PowerShell command for user-side WSL validation
  - the current boundary that `torchrec_v1` is not yet a real DMP / TrainPipelineSparseDist loop
- documented that Codex-side WSL execution is currently blocked by `WSL/Service/E_ACCESSDENIED`
  and must not be treated as a successful DMP validation

Validation:

```text
python -m compileall ui/artifacts_tab.py runner/torchrec_runner/sharding.py runner/torchrec_runner/entry.py runner/torchrec_runner/plan.py runner/torchrec_runner/runtime.py scripts
python -m unittest tests.test_torchrec_runtime tests.test_torchrec_sharding tests.test_torchrec_contract
```

Manual acceptance:

1. Start the Gradio app.
2. Launch a one-step `Backend=torchrec_v1` job.
3. Open Artifacts and click Refresh Jobs / Refresh Artifacts.
4. Confirm `artifacts/torchrec-sharding-plan-readiness.json` is visible.
5. Run the DMP readiness script from a normal PowerShell session:

```powershell
cd C:\Users\han\Desktop\prototype
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

Acceptance standards:

- Artifacts page can display the sharding readiness JSON without breaking older jobs
- readiness artifacts do not overclaim real `collective_plan` execution
- DMP readiness script exits 0 only when WSL imports all required TorchRec distributed components
- if the script exits non-zero, its JSON `errors` list becomes the next required fix

### Updated V1 Remaining Gaps After Continuation 4

- real `DistributedModelParallel` wrapping is not yet implemented
- real `EmbeddingShardingPlanner.collective_plan` is not yet validated in a distributed process
  group from this Codex environment
- real `TrainPipelineSparseDist` training/evaluation loop is not yet implemented
- production-grade sharded checkpointing with Torch Distributed Checkpoint is not yet implemented
- FBGEMM managed caching and mixed precision controls are still capability-reported but not mapped
  into real runtime behavior
