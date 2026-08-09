# Design Gap Closure Plan

This document tracks the gap between the V1 design document and the current local prototype.

## Acceptance Checklist

| Area | Design expectation | Current status | First closure target |
|---|---|---|---|
| UI | Gradio tabs for create, logs, monitor, artifacts | Implemented, plus Compare Runs | Add auto-refresh later |
| Config | Pydantic/YAML config for mode, model, data, GPU, checkpoint, profile | Implemented | Keep field names aligned with design terms |
| Modes | Cold start, resume, evaluate | Implemented across project-owned loops; DLRM smoke supports resume/evaluate | Enforce successful checkpoint markers on load |
| Local tasks | One active local job, state persisted in `state.json` | Implemented | Preserve existing lifecycle states |
| Logs | View and download complete logs | Implemented via refresh and run bundle | Add timer-based refresh later |
| Metrics | Loss, AUC, throughput, stage timing, resource metrics | Partially implemented | Emit from real V1 runtime path where available |
| TorchRec runner | Real model loading, DataLoader, DMP, TrainPipelineSparseDist | Partially implemented as V1 scaffold plus minimal loop | Phase 1 adds single-card PyTorch random training path |
| Data | Random plus at least one real format | Random, Criteo numpy, parquet validation/conversion | Wire V1 DataLoader beyond preview in later phase |
| Checkpoint | Save `_SUCCESS`, load only successful checkpoint, resume/evaluate | Project-owned checkpoints write `_SUCCESS` | Phase 1 enforces `_SUCCESS` for directory loads |
| GPU cache and precision | DEVICE/MANAGED_CACHING and precision affect runtime | Recorded in capability reports, not fully mapped | Later phase maps into real TorchRec runtime |

## Phase 1 Scope

The first closure phase is intentionally narrow:

- add this checklist so design acceptance is explicit
- enforce checkpoint `_SUCCESS` before loading a checkpoint directory
- make `torchrec_v1` run a real single-card PyTorch training step when:
  - `nproc_per_node == 1`
  - `data.format` is `random` or `criteo_binary`
  - `model.build_model(config)` returns a `torch.nn.Module`
  - trainable parameters are present
- keep the existing minimal loop as a fallback when those conditions are not met
- record the runtime path in logs, checkpoint payloads, metrics, and status artifacts

## Still Out Of Phase 1

- `DistributedModelParallel`
- `TrainPipelineSparseDist`
- multi-rank sharded checkpointing with Torch Distributed Checkpoint
- FBGEMM managed caching
- mixed precision communication/runtime mapping
- table-level embedding/cache metrics
- timer-based UI refresh

## Phase 2 Scope

The second closure phase extends the single-card runtime path into a checkpoint loop:

- write `model.pt` and `optimizer.pt` for `torchrec_v1` single-card PyTorch runs
- write those runtime files before `_SUCCESS`
- include runtime checkpoint metadata in `model.json` and `metadata.json`
- load `model.pt` and `optimizer.pt` during `RESUME` when they exist
- load `model.pt` during `EVALUATE` and compute runtime evaluation metrics when possible
- keep minimal-loop fallback for non-torch environments so UI and lightweight tests still run

## Still Out Of Phase 2

- Torch Distributed Checkpoint for multi-rank sharded state
- `DistributedModelParallel`
- `TrainPipelineSparseDist`
- strict failure mode for `torchrec_v1` when torch/torchrec are unavailable
- direct parquet DataLoader for the V1 runner

## Phase 3.1 Scope

The first part of Phase 3 introduces a real distributed runtime boundary without claiming DMP yet:

- read `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`
- initialize `torch.distributed.init_process_group(init_method="env://")` when running under `torchrun`
- choose `nccl` when CUDA is available, otherwise `gloo`
- set the current CUDA device from `LOCAL_RANK` when CUDA is available
- write `artifacts/torchrec-distributed-environment.json`
- expose the distributed environment artifact in the Artifacts tab
- destroy the process group on exit when this runner initialized it

## Still Out Of Phase 3.1

- wrapping the model with `DistributedModelParallel`
- distributed dataloader sharding
- DMP-compatible optimizer construction
- `TrainPipelineSparseDist`
- multi-rank checkpointing

## Phase 3.2 Scope

Phase 3.2 adds a real runtime batch boundary:

- build dense feature tensors and label tensors from the existing random/Criteo numpy source
- build a TorchRec `KeyedJaggedTensor` when torchrec is available
- pass the `KeyedJaggedTensor` into the single-card runtime model path first
- fall back to a dense sparse-id tensor only for model compatibility
- write `artifacts/torchrec-runtime-batch.json`
- expose the runtime batch artifact in the Artifacts tab
- include runtime batch status in `artifacts/torchrec-runner-status.json`

## Still Out Of Phase 3.2

- a real `torch.utils.data.DataLoader`
- per-rank distributed data sharding
- TorchRec `Batch` object integration if required by later DMP/TrainPipeline code
- table-wise sparse feature preprocessing beyond the current one-id-per-feature smoke shape

## Phase 3.3 And 3.4 Scope

Phase 3.3 and 3.4 move the V1 smoke path closer to a real TorchRec recommendation model:

- make the example `build_model(config)` return a dense+sparse model
- use TorchRec `EmbeddingBagCollection` when torchrec is available
- make `build_embedding_configs(config)` return real `EmbeddingBagConfig` objects when possible
- pass runtime `KeyedJaggedTensor` into the model path
- attempt `DistributedModelParallel` wrapping when a torchrun process group is initialized
- write `artifacts/torchrec-dmp-wrap.json`
- expose the DMP wrap artifact in the Artifacts tab
- include DMP wrap status in `artifacts/torchrec-runner-status.json`

## Still Out Of Phase 3.4

- constructing an explicit `EmbeddingShardingPlanner.collective_plan`
- requiring DMP success as a hard failure condition
- multi-rank training participation beyond rank 0
- DMP-aware optimizer separation for sparse and dense parameters
- replacing the single-card runtime loop with `TrainPipelineSparseDist`

## Phase 3.5 And 3.6 Scope

Phase 3.5 and 3.6 make the DMP smoke path truly multi-rank:

- allow the runtime model path when `nproc_per_node > 1` if a torchrun process group is initialized
- have every rank build the model and attempt `DistributedModelParallel` wrapping
- write rank-specific distributed environment reports
- write rank-specific DMP wrap reports
- aggregate rank DMP reports into `artifacts/torchrec-dmp-wrap-summary.json`
- expose the DMP wrap summary in the Artifacts tab
- add a torchrun smoke path that can validate `--nproc-per-node 2`

## Still Out Of Phase 3.6

- performing full multi-rank forward/backward training steps
- replacing the smoke path with `TrainPipelineSparseDist`
- constructing an explicit distributed DataLoader and sampler
- writing rank-sharded Torch Distributed Checkpoint state
- DMP-aware optimizer grouping for dense and sparse parameters

## Phase 3.7 Scope

Phase 3.7 adds a distributed data boundary to the DMP smoke path:

- shard runtime batch rows by `rank` and `world_size`
- keep rank0/rank1 from consuming the same synthetic rows in torchrun smoke
- write `artifacts/torchrec-runtime-batch-rank<N>.json` for each rank
- aggregate rank batch reports into `artifacts/torchrec-runtime-batch-summary.json`
- expose the runtime batch summary in the Artifacts tab
- include rank-sharded batch readiness in runner status and torchrun smoke output

## Still Out Of Phase 3.7

- replacing the rank-sharded runtime batch with a full `torch.utils.data.DataLoader`
- using a formal distributed sampler for file-backed datasets
- running multi-rank forward/backward training steps through DMP
- replacing the smoke path with `TrainPipelineSparseDist`
- writing rank-sharded Torch Distributed Checkpoint state

## Phase 3.8 Scope

Phase 3.8 turns the multi-rank DMP path from a wrap-only smoke into a training-step smoke:

- run one `forward -> loss -> backward -> optimizer.step()` per torchrun rank
- use the rank-sharded runtime batch from Phase 3.7 for that step
- write `artifacts/torchrec-runtime-step-rank<N>.json` for each rank
- aggregate rank step reports into `artifacts/torchrec-runtime-step-summary.json`
- expose the runtime step artifacts in the Artifacts tab
- make the torchrun smoke script require all ranks to execute the runtime step

## Still Out Of Phase 3.8

- replacing the smoke step with a full multi-step training loop
- replacing the runtime batch path with a full `torch.utils.data.DataLoader`
- using `TrainPipelineSparseDist`
- writing rank-sharded Torch Distributed Checkpoint state
- DMP-aware optimizer grouping for dense and sparse parameters

## Phase 3.9 Scope

Phase 3.9 closes the Phase 3 smoke loop by running multiple rank-sharded training steps:

- use `training.max_steps` in the multi-rank torchrun smoke path
- run repeated `forward -> loss -> backward -> optimizer.step()` steps on every rank
- record per-step loss, accuracy, KJT usage, and step timing in each rank step artifact
- aggregate completed step counts across ranks in `artifacts/torchrec-runtime-step-summary.json`
- mark the run as `SUCCEEDED_MULTI_RANK_RUNTIME_LOOP_SMOKE` only when every rank completes the loop
- make the torchrun smoke script validate the completed step count

## Still Out Of Phase 3.9

- replacing the smoke loop with `TrainPipelineSparseDist`
- replacing synthetic/runtime batches with full DataLoader iteration
- using formal distributed samplers for file-backed datasets
- writing rank-sharded Torch Distributed Checkpoint state
- DMP-aware optimizer grouping for dense and sparse parameters
