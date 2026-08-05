# TorchRec Prototype Project Context and Status

Last updated: 2026-07-23

## 1. Project Background

This prototype project exists to provide a single-machine, local-first visual interface for
configuring, launching, and monitoring TorchRec-based recommendation model jobs.

The project is based on the user's earlier local validation work:

- The local `TorchRec` source repository has already been deployed successfully.
- The local `DLRM` repository has already been deployed successfully.
- TorchRec's official installation verification path has already been run successfully.
- `torchrec_dlrm/dlrm_main.py` has already been run successfully in local random-data mode.

This means the current phase is no longer "prove TorchRec can run locally", but rather:

Build a usable prototype tool on top of that verified local foundation.

## 2. Confirmed Upstream Local Status

According to the prior project notes and discussion:

- Local TorchRec source path:
  - `C:\Users\han\Desktop\torchrec`
- Local DLRM source path:
  - `C:\Users\han\Desktop\dlrm`
- Main execution environment:
  - `WSL2 + Ubuntu 22.04`
- Recommended active Python environment:
  - `~/venvs/torchrec17`

Confirmed completed validation items:

- TorchRec editable install has been set up against the local source tree.
- TorchRec verification script has run successfully.
- DLRM `dlrm_main.py` has run successfully with random data.
- The training / validation / test main flow has been proven runnable locally.

Important scope note:

- The current verified DLRM run is a random-data baseline, not a real production dataset run.
- Real dataset integration, training experiments, and deeper model integration are future steps.

## 3. Current Task Goal

The current assignment is to build a prototype first.

The required scope is intentionally limited:

- Single machine
- Single user
- Local filesystem based
- Runnable locally
- No Kubernetes
- No distributed scheduler platform
- No multi-user production platform concerns for now

The prototype should let a user:

- Configure a TorchRec training or evaluation job in a browser UI
- Launch the job locally
- Observe logs and job state
- View basic metrics and artifacts
- Keep configuration and outputs on local disk for reproducibility

## 4. Product Direction Agreed in Conversation

The agreed implementation direction is:

- Start with a lightweight local prototype
- Prefer fast validation and low complexity
- Do not introduce K8S, cluster scheduling, or heavy platform infrastructure yet
- Use the already validated TorchRec / DLRM local chain as the future execution backend

The architectural preference captured so far is:

- `Python` as the implementation language
- `Gradio` for the UI
- Local subprocess execution for job launching
- A lightweight local task manager
- Local run directories to store config, state, logs, metrics, checkpoints, and evaluation outputs

## 5. Current Prototype Implementation Status

A first skeleton version of the prototype has already been created.

The project was first scaffolded in the Codex workspace and then moved to:

- `C:\Users\han\Desktop\prototype`

Current files and modules in that project include:

- `app.py`
- `config.py`
- `task_manager.py`
- `ui/`
- `runner/`
- `README.md`

### 5.1 What has been implemented

#### App shell

- A Gradio entrypoint has been created in `app.py`.
- The interface is divided into four tabs:
  - `Create Job`
  - `Logs`
  - `Monitor`
  - `Artifacts`

#### Configuration model

- `config.py` defines a structured configuration model using Pydantic.
- The current config includes:
  - run mode
  - model file
  - device settings
  - data settings
  - training settings
  - checkpoint settings
  - profile settings
- Config can be serialized to YAML as `resolved-config.yaml`.

#### Local task manager

- `task_manager.py` provides a lightweight local task manager.
- It currently supports:
  - creating a run directory
  - creating task metadata
  - writing `resolved-config.yaml`
  - writing `state.json`
  - launching a subprocess for the runner
  - basic stop behavior
  - listing jobs
  - reading log and artifact files
  - updating final task state

#### Runner skeleton

- `runner/cli.py` provides a unified runner entrypoint.
- `runner/train.py` currently contains a prototype stub training flow.
- `runner/evaluate.py` currently contains a prototype stub evaluation flow.
- `runner/metrics.py` writes JSONL metrics records.

#### UI tabs

- `ui/create_tab.py`
  - basic config entry
  - config validation preview
  - launch action
- `ui/logs_tab.py`
  - refresh job list
  - show launcher log
  - show rank-0 training log
- `ui/monitor_tab.py`
  - read and display recent metrics
- `ui/artifacts_tab.py`
  - show `state.json`
  - show `resolved-config.yaml`
  - stop job

#### Basic quality check already completed

- A syntax-level compile check was run successfully with `python -m compileall prototype`
  before the project was moved to the Desktop folder.

## 6. Current Functional Reality

The project currently provides a runnable prototype skeleton, not a full TorchRec-integrated trainer.

What it does today:

- Creates jobs from a UI
- Persists a run configuration
- Launches a local subprocess
- Writes logs
- Writes metrics
- Tracks simple task state
- Displays those artifacts in the UI

What it does not do yet:

- Run real TorchRec training
- Call the user's local `dlrm_main.py`
- Load real recommendation datasets
- Manage real checkpoints in a TorchRec-compatible way
- Parse rich training metrics from actual TorchRec execution
- Handle production-grade failure recovery or scheduling

## 7. Current Gaps

The biggest current gap is backend realism.

The UI and task lifecycle skeleton exist, but the runner is still a stub. That means the next
important engineering step is replacing the simulated runner behavior with a real local execution path.

Key missing capabilities:

- Integration with local TorchRec / DLRM execution commands
- Real training/evaluation mode mapping
- Real checkpoint save/load flow
- Real metric extraction and visualization
- Real data adapter path
- Better job refresh and task UX

## 8. Recommended Next Development Priorities

### Priority 1: Replace stub runner with real local execution

Suggested direction:

- Use the validated local environment and commands as the execution backend
- Decide whether the first real backend should:
  - wrap `torchrec_dlrm/dlrm_main.py` directly, or
  - introduce a custom runner that progressively absorbs DLRM logic

For speed, the first option is likely better:

- Start by wrapping the existing proven `dlrm_main.py` path
- Keep the UI and local run directory contract
- Capture stdout/stderr into job logs

### Priority 2: Strengthen run-directory contract

Standardize:

- `resolved-config.yaml`
- `state.json`
- `launcher.log`
- rank logs
- `metrics.jsonl`
- `evaluation.json`
- `checkpoints/`
- `profiles/`

This should remain stable as the backend becomes real.

### Priority 3: Improve UX for local validation

Add:

- better job list refresh behavior
- clearer job status transitions
- job summary panel
- explicit environment/backend information
- validation for mode/checkpoint relationships

### Priority 4: Connect metrics to real TorchRec outputs

Eventually the monitor tab should visualize real fields such as:

- training loss
- validation AUC
- throughput
- step time
- checkpoint timing
- resource utilization

## 9. Suggested Immediate Next Step

The most practical next milestone is:

Implement the first real execution bridge from the prototype UI to the already validated local
TorchRec / DLRM path.

That would convert the project from:

- "prototype shell with simulated runner"

to:

- "single-machine prototype that can actually trigger a real local TorchRec/DLRM job"

## 10. Overall Status Summary

Current overall status on 2026-07-23:

- Upstream TorchRec local deployment: done
- Upstream DLRM local deployment: done
- Local verification run for TorchRec: done
- Local verification run for DLRM random-data flow: done
- Prototype project scope alignment: done
- Initial project skeleton: done
- Real TorchRec backend integration: not done yet
- Production platform features: intentionally out of scope for now

## 11. One-Sentence Project State

This project is now in the "prototype skeleton completed, real TorchRec execution integration next"
phase.
