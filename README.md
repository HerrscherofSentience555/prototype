# TorchRec 推荐模型训练平台原型

一个本地优先的 Gradio 原型工具，用于在单机环境中配置、启动、监控和检查 TorchRec / DLRM 推荐模型训练任务。

用户 clone 仓库后，只需要配置本机路径，就能通过 UI 跑通推荐模型训练 smoke 流程，并查看日志、指标、checkpoint 和运行产物。

## 目录

- [项目亮点](#项目亮点)
- [原型边界](#原型边界)
- [快速开始](#快速开始)
- [迁移到另一台机器](#迁移到另一台机器)
- [UI 页面说明](#ui-页面说明)
- [人工测试流程](#人工测试流程)
- [数据格式说明](#数据格式说明)
- [高级验证](#高级验证)
- [运行产物](#运行产物)
- [项目结构](#项目结构)
- [质量检查](#质量检查)
- [常见问题](#常见问题)

## 项目亮点

- 提供 Gradio Web UI，不需要手写完整训练命令。
- 支持 `stub`、`custom`、`dlrm`、`torchrec_v1` 多种后端。
- 支持 Windows UI + WSL2 TorchRec / DLRM 训练。
- 支持本机配置文件 `local_settings.yaml`，方便不同电脑迁移。
- 支持 Environment 页面检查本机 WSL、Python env、TorchRec、DLRM 路径和样例数据。
- 支持 Criteo Kaggle 小样本真实数据 smoke 训练。
- 支持日志、指标、checkpoint、profile、资源采样和运行产物查看。
- 支持 parquet schema 校验和 parquet 到 DLRM numpy 格式转换。
- 提供 TorchRec V1 runner scaffold，用于后续接入更完整的 TorchRec 训练链路。

## 原型边界

当前项目面向：

- 单用户
- 单机
- 本地文件系统
- Windows 上启动 Gradio UI
- WSL2 Ubuntu 中执行 TorchRec / DLRM
- 可选 Linux native shell 运行模式

当前项目不处理：

- Kubernetes 集群训练
- 外部任务调度系统
- 多用户权限与隔离
- 生产级资源池管理
- 大规模 Criteo 1TB 训练验收
- 多进程分片生产级 checkpoint

需要特别注意：

- 当前 DLRM checkpoint 主要验证单进程 smoke 路径，使用 `model.pt` / `optimizer.pt`。
- DLRM 暂不直接训练 parquet，需要先转换成 DLRM numpy 格式。
- `torchrec_v1` 后端已有 readiness/materialization 层，但还不是完整真实的 `DistributedModelParallel` / `TrainPipelineSparseDist` 训练循环。
- `embedding_placement`、`cache_load_factor`、precision 等字段会被记录到配置和能力报告中，但当前 DLRM 示例后端没有完全映射到 DLRM 命令行。

## 快速开始

### 1. 创建本机配置

在仓库根目录执行：

```powershell
Copy-Item local_settings.example.yaml local_settings.yaml
notepad local_settings.yaml
```

把模板中的本机路径改成自己的实际路径，例如：

```yaml
runtime:
  platform: windows_wsl
  wsl_distribution: Ubuntu-22.04
  python_env: ~/venvs/torchrec17

paths:
  dlrm_root: /mnt/c/Users/<your-name>/Desktop/dlrm
  criteo_binary_path: data/criteo_kaggle_sample_npy
```

`local_settings.yaml` 已被 `.gitignore` 忽略，不会提交到仓库。

### 2. 安装依赖并启动 UI

推荐从仓库父目录启动，这样 `python -m prototype.app` 能正确识别包名：

```powershell
cd <your-parent-folder>
python -m venv prototype\.venv
prototype\.venv\Scripts\Activate.ps1
pip install -r prototype\requirements.txt
python -m prototype.app
```

终端会输出 Gradio 地址，通常是：

```text
http://127.0.0.1:7860
```

### 3. 先检查 Environment

打开 UI 后先进入 `Environment` 页面，点击：

```text
Refresh Settings
Run Environment Checks
```

期望看到：

```text
local_settings.yaml     OK
settings source         OK
WSL available           OK
Python env              OK
import torch            OK
import torchrec         OK
DLRM root               OK
Criteo binary path      OK
Criteo binary shapes    OK
```

如果这里失败，优先修改 `local_settings.yaml`，不要改源码。

### 4. 跑第一个 stub 任务

进入 `Create Job`，设置：

```text
Backend: stub
Mode: COLD_START
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

点击：

```text
Validate Config
Launch Job
```

任务成功后，`Artifacts` 页面中的 `state.json.status` 应为：

```text
SUCCEEDED
```

## 迁移到另一台机器

这个项目的迁移方式是：**源码保持通用，本机差异写入 `local_settings.yaml`。**

新机器上推荐按这个顺序：

1. clone 仓库。
2. 安装 Windows UI 虚拟环境。
3. 准备 WSL2 Ubuntu。
4. 在 WSL 中准备 TorchRec / DLRM Python 环境。
5. 复制 `local_settings.example.yaml` 为 `local_settings.yaml`。
6. 修改 `dlrm_root`、`python_env`、`wsl_distribution`。
7. 启动 UI。
8. 先跑 `Environment` 检查。
9. 再跑 `stub`、`custom`、`dlrm` smoke。

建议遵守：

- 不要把本机绝对路径写进源码。
- 不要提交 `local_settings.yaml`。
- 项目内数据、schema、示例模型尽量使用相对路径。
- 外部 DLRM 仓库路径按每台机器实际情况填写。

常见 Windows + WSL 路径示例：

```text
Windows 项目路径: C:\Users\<your-name>\Desktop\prototype
Windows DLRM 路径: C:\Users\<your-name>\Desktop\dlrm
WSL DLRM 路径:     /mnt/c/Users/<your-name>/Desktop/dlrm
WSL Python env:    ~/venvs/torchrec17
WSL distro:        Ubuntu-22.04
```

## UI 页面说明

### Create Job

用于创建训练任务、预览 YAML 配置并启动后端。

常用字段：

- `Job Name`：任务名称。
- `Mode`：`COLD_START`、`RESUME`、`EVALUATE`。
- `Backend`：`stub`、`dlrm`、`custom`、`torchrec_v1`。
- `Runtime Platform`：`windows_wsl` 或 `linux_native`。
- `DLRM Root`：运行环境看到的 DLRM 仓库路径。
- `Python Env`：运行环境中的 Python 虚拟环境。
- `WSL Distribution`：WSL 发行版名称。
- `Model File`：自定义模型或 TorchRec V1 模型文件。
- `Data Format`：`random`、`criteo_binary`、`synthetic_multihot`、`parquet`。
- `Criteo Binary Path`：Criteo numpy 数据目录。
- `Dataset Name`：`criteo_kaggle` 或 `criteo_1t`。
- `Batch Size` / `Test Batch Size`：训练和评估 batch size。
- `Max Steps`：最大训练步数，对 DLRM 映射为 `--limit_train_batches`。
- `Learning Rate`：学习率。
- `Processes per Node`：单机 torchrun 进程数。
- `GPU IDs`：映射为 `CUDA_VISIBLE_DEVICES`。
- `Checkpoint Load Path`：`RESUME` 和 `EVALUATE` 模式需要填写。
- `Save Checkpoints`：保存 smoke checkpoint。
- `Profile Enabled`：开启 profile 请求和相关产物。

常用按钮：

- `Validate Config`：校验并展示最终 YAML。
- `Launch Job`：创建 run 目录并启动任务。
- `Validate Parquet`：校验 parquet 数据和 schema。
- `Convert Parquet`：将 parquet 转为 DLRM numpy 格式。

选项解释：

`Mode` 是本项目定义的任务运行方式：

- `COLD_START`：从头开始训练一个新任务。第一次训练一般选这个。
- `RESUME`：从已有 checkpoint 继续训练。需要填写 `Checkpoint Load Path`。
- `EVALUATE`：加载已有 checkpoint 做评估，不继续训练。需要填写 `Checkpoint Load Path`。

`Backend` 是本项目定义的后端类型：

- `stub`：模拟训练，不依赖 TorchRec。适合快速检查 UI、任务状态、日志、指标和产物链路。
- `dlrm`：调用本机 WSL/Linux 中的 TorchRec DLRM。
- `custom`：加载用户自己写的 `model.py`。适合验证自定义训练逻辑或做轻量模型实验。
- `torchrec_v1`：项目内部的 TorchRec V1 runner scaffold。主要用于验证后续接入 TorchRec DMP/TrainPipeline 的准备情况。

`Runtime Platform` 是本项目定义的运行环境选择：

- `windows_wsl`：Windows 上打开 UI，训练命令通过 WSL 执行。当前主要测试路径是这个。
- `linux_native`：UI 和训练命令都在 Linux shell 中运行。适合把项目整体放到 Linux 环境时使用。

`Data Format` 是本项目对训练数据入口的约定：

- `random`：随机数据，用来快速确认任务能启动。
- `criteo_binary`：DLRM 可直接读取的 Criteo numpy 数据，真实小数据 smoke 用这个。
- `synthetic_multihot`：合成 multi-hot Criteo 风格数据，适合做数据管线 smoke。
- `parquet`：业务表格数据格式。当前不能直接给 DLRM 训练，需要先通过 `Convert Parquet` 转成 `criteo_binary`。

其他字段：

- `Embedding Placement`：记录 embedding 放置策略。当前 DLRM 示例后端主要记录该配置，不完整映射到 DLRM 命令行。
- `Cache Load Factor`：记录 GPU cache 相关比例。当前主要进入配置和能力报告。
- precision 字段：记录 embedding、dense compute、通信前向/反向的精度意图。当前主要进入配置和能力报告。
- `Checkpoint Load Path`：已有 checkpoint 目录，通常形如 `runs/<job_id>/checkpoints/step-final`。
- `Save Checkpoints`：保存本项目 smoke checkpoint，不等同于生产级分布式 checkpoint。
- `Profile Enabled`：开启 profile 请求和相关产物，主要用于观察耗时和 trace，不会自动优化模型。

### Environment

用于检查当前机器是否已经准备好。

- `Refresh Settings`：显示当前读取的本机配置。
- `Create local_settings.yaml`：从模板生成本机配置。
- `Run Environment Checks`：检查 WSL/Linux shell、Python env、`torch`、`torchrec`、DLRM root 和 Criteo 样例数据。

### Logs

用于查看任务日志和实际命令。

- `launcher.log`：runner 启动日志和异常栈。
- `train-rank0.log`：训练主日志。
- `command.json`：UI runner 命令和后端实际命令。
- tail line count：控制日志显示尾部行数。

### Monitor

用于查看训练指标。

- 展示 `metrics.jsonl` 中的最近记录。
- 展示 loss、AUC、throughput、step time、stage timing 等图表。
- 没有指标时显示空状态。

### Artifacts

用于查看运行产物。

- `state.json`
- `resolved-config.yaml`
- `evaluation.json`
- checkpoint 文件
- profile 文件
- artifact 文件
- `run-artifacts.zip`
- Stop Job 操作

### Compare

用于对比多个 run 的指标和摘要，适合观察不同参数、不同数据设置下的 smoke 结果。

## 人工测试流程

建议按下面顺序测试。前两步证明 UI 和任务系统正常，后两步证明真实 TorchRec / DLRM 路径可用。

### 1. Environment 检查

期望所有核心项为 `OK`：

```text
local_settings.yaml
settings source
WSL available
Python env
import torch
import torchrec
DLRM root
Criteo binary path
Criteo binary shapes
```

### 2. stub smoke

```text
Backend: stub
Mode: COLD_START
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

期望：

- 任务最终 `SUCCEEDED`。
- Logs 页面能看到 `train-rank0.log`。
- Monitor 页面能看到基础指标。
- Artifacts 页面能看到 `state.json` 和 `run-artifacts.zip`。

### 3. custom smoke

```text
Backend: custom
Model File: examples/models/custom_simple_model.py
Data Format: random
Batch Size: 4
Max Steps: 2
Save Checkpoints: true
```

期望：

- 任务最终 `SUCCEEDED`。
- `metrics.jsonl` 包含自定义模型返回的指标。
- `artifacts/custom-model-contract.json` 存在。
- 开启 checkpoint 时会生成 checkpoint 文件。

### 4. DLRM 随机数据 smoke

```text
Backend: dlrm
Mode: COLD_START
Runtime Platform: windows_wsl
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
Learning Rate: 0.01
```

期望：

- `command.json` 包含 `backend_command`。
- 后端命令调用 `wsl`、激活 venv、进入 DLRM root，并执行 `torchrun`。
- `train-rank0.log` 有真实 DLRM 输出。
- `state.json.status` 最终为 `SUCCEEDED`。

### 5. DLRM 真实小数据 smoke

使用项目内小样本：

```text
data/criteo_kaggle_sample_npy
```

推荐配置：

```text
Backend: dlrm
Mode: COLD_START
Data Format: criteo_binary
Criteo Binary Path: data/criteo_kaggle_sample_npy
Dataset Name: criteo_kaggle
Batch Size: 16
Test Batch Size: 16
Max Steps: 100
Processes per Node: 1
Save Checkpoints: true
```

期望：

- `state.json.status` 为 `SUCCEEDED`。
- `train-rank0.log` 包含 `Total number of iterations`。
- `train-rank0.log` 包含 `AUROC over val set` 和 `AUROC over test set`。
- `metrics.jsonl` 包含 `total_iterations`、`val_auc`、`test_auc`、`val_samples`、`test_samples`。
- `checkpoints/step-final/model.pt` 存在。
- `checkpoints/step-final/_SUCCESS` 存在。

### 6. checkpoint / evaluate smoke

先运行一次 DLRM 训练，并确认：

```text
runs/<job_id>/checkpoints/step-final/model.pt
runs/<job_id>/checkpoints/step-final/_SUCCESS
```

再新建评估任务：

```text
Backend: dlrm
Mode: EVALUATE
Data Format: criteo_binary
Criteo Binary Path: data/criteo_kaggle_sample_npy
Dataset Name: criteo_kaggle
Batch Size: 16
Test Batch Size: 16
Processes per Node: 1
Checkpoint Load Path: runs/<job_id>/checkpoints/step-final
```

期望：

- 空 `Checkpoint Load Path` 会被配置校验拒绝。
- 非空 checkpoint path 可以启动任务。
- `command.json` 包含 `--limit_train_batches 0`。
- `command.json` 包含 `--checkpoint_load_path`。
- `evaluation.json` 被创建。
- `metrics.jsonl` 包含评估指标。

## 数据格式说明

### `random`

随机数据，主要用于验证任务流程和 DLRM 命令桥接。

适合：

- 初次检查后端能不能启动。
- 检查日志、状态、指标是否写入。

### `criteo_binary`

DLRM 可以直接读取的 Criteo numpy 格式。项目内小样本路径为：

```text
data/criteo_kaggle_sample_npy
```

通常包含：

```text
train_dense.npy
train_sparse.npy
train_labels.npy
```

适合：

- 真实数据 smoke。
- checkpoint/evaluate smoke。
- 手工观察小数据训练日志和指标。

### `parquet`

Parquet 是业务系统和数据仓库常用的列式表格文件格式。它适合存储 CTR 类表格数据，但当前 DLRM 后端不直接训练 parquet。

当前流程是：

```text
parquet 数据
  -> schema 校验
  -> 转换为 Criteo numpy
  -> 使用 criteo_binary 训练
```

转换命令：

```powershell
cd <path-to-your-prototype-clone>
python -m prototype.runner.convert_parquet --config examples\parquet-conversion-smoke.yaml --output-dir data\converted_criteo_npy
```

期望：

- 生成 `<split>_dense.npy`、`<split>_sparse.npy`、`<split>_labels.npy`。
- `conversion-manifest.json` 记录输入、schema、split、行数和输出文件。
- 输出目录可作为 `Criteo Binary Path`。

## 高级验证

### TorchRec V1 model.py contract

内部 TorchRec runner scaffold 要求模型文件至少提供：

```python
def build_model(config: dict):
    ...

def build_embedding_configs(config: dict) -> list:
    ...
```

可选函数：

```text
build_optimizer
build_dataloader
train_step
evaluate
```

示例文件：

```text
examples/models/torchrec_v1_model.py
```

常见产物：

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

### TorchRec V1 后端 smoke

```text
Backend: torchrec_v1
Mode: COLD_START
Model File: examples/models/torchrec_v1_model.py
Data Format: random
Batch Size: 4
Max Steps: 1
Processes per Node: 1
```

期望：

- `command.json` 调用 WSL。
- 后端执行 `torchrun -m prototype.runner.torchrec_runner.entry`。
- TorchRec V1 contract、data plan、batch materialization、embedding configs、runtime smoke 等 artifact 被写出。

### DLRM patch 检查

```powershell
cd <path-to-your-prototype-clone>
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_profiler_patch.ps1
```

期望：

```text
DLRM checkpoint patch is present.
DLRM profiler patch is present.
```

### TorchRec V1 DMP readiness 检查

```powershell
cd <path-to-your-prototype-clone>
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

期望：

- 脚本退出码为 `0`。
- JSON 输出中 torch、torchrec、DMP、sharding planner、train pipeline、distributed checkpoint 检查为 true。
- 如果退出非零，JSON 中的 `errors` 是下一步 blocker。

可选环境变量：

```powershell
$env:TORCHREC_WSL_DISTRO = "Ubuntu-22.04"
$env:TORCHREC_PYTHON_ENV = "~/venvs/torchrec17"
powershell -ExecutionPolicy Bypass -File scripts\check_torchrec_v1_dmp_readiness.ps1
```

### profile 验证

启动一个短任务，并设置：

```text
Profile Enabled: true
```

期望：

- `profiles/profile-request.json` 存在。
- `profiles/runner-profile.json` 存在。
- Windows runner 环境有 torch 时，可能生成 `profiles/trace.json`。
- Windows runner 环境没有 torch 时，`runner-profile.json.profile_trace_error` 会说明 fallback 原因。
- DLRM 任务会传递 `--profile_dir`、`--profile_record_shapes`、`--profile_memory`。
- patched DLRM 可以生成 `profiles/dlrm/rank<N>-trace.json`。

### GPU / WSL 资源采样

运行一个短任务后检查：

```text
runs/<job_id>/resource-metrics.jsonl
runs/<job_id>/artifacts/resource-summary.json
```

期望：

- 记录中包含 `gpu_telemetry_available`。
- 如果 `nvidia-smi` 可用，会包含 GPU 利用率和显存字段。
- 记录中包含 `wsl_telemetry_available`。
- WSL 中可见 Python/torchrun 进程时，会填充 WSL 进程资源字段。

## 运行产物

每个任务写入：

```text
runs/<job_id>/
  resolved-config.yaml
  state.json
  launcher.log
  train-rank0.log
  command.json
  metrics.jsonl
  evaluation.json
  logs/
  checkpoints/
    step-final/
      model.pt
      optimizer.pt
      metadata.json
      _SUCCESS
  profiles/
  artifacts/
    run-artifacts.zip
```

有些文件与运行模式有关。例如训练任务不一定有 `evaluation.json`，未开启 checkpoint 时不会有 `model.pt`。

常见任务状态：

```text
CREATED
LAUNCHING
RUNNING
STOPPING
STOPPED
SUCCEEDED
FAILED
```

`state.json` 还会记录 backend、command、cwd、pid、error_message、时间戳、duration、exit_code 和 stop metadata。

## 项目结构

```text
prototype/
  app.py                         Gradio 应用入口
  config.py                      Pydantic 配置模型
  local_settings.py              本机配置读取逻辑
  local_settings.example.yaml    本机配置模板
  task_manager.py                本地任务生命周期和子进程管理
  requirements.txt               Windows UI/runtime 依赖
  ui/
    create_tab.py                创建任务和预览配置
    environment_tab.py           本机环境检查
    logs_tab.py                  日志和命令查看
    monitor_tab.py               指标表格和图表
    artifacts_tab.py             状态、配置、评估、产物查看和停止任务
    compare_tab.py               运行对比
  runner/
    cli.py                       子进程 runner 入口
    metrics.py                   JSONL 指标写入
    log_parser.py                DLRM 日志指标解析
    backends/
      stub_backend.py            模拟后端
      dlrm_backend.py            WSL2 TorchRec DLRM 后端
      custom_backend.py          用户自定义 Python 模型后端
      torchrec_v1_backend.py     TorchRec V1 runner scaffold 启动器
      runtime_env.py             Windows/WSL/Linux 路径与命令适配
    parquet_converter.py         parquet 到 Criteo 风格 numpy 转换
    convert_parquet.py           转换 CLI 入口
    torchrec_runner/
      contract.py                V1 model.py contract 校验
      entry.py                   内部 TorchRec runner scaffold
      sharding.py                TorchRec sharding planner readiness 产物
  examples/                      smoke 配置和示例模型
  scripts/                       环境、patch、readiness 检查脚本
  patches/                       本地 DLRM patch 说明
  runs/                          本地任务运行产物
```

## 质量检查

运行：

```powershell
cd <path-to-your-prototype-clone>
python -m compileall app.py config.py local_settings.py task_manager.py runner ui tests
python -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_checkpoint_patch.ps1
powershell -ExecutionPolicy Bypass -File scripts\check_dlrm_profiler_patch.ps1
```

如果使用项目虚拟环境：

```powershell
cd <path-to-your-prototype-clone>
.\.venv\Scripts\python.exe -m compileall app.py config.py local_settings.py task_manager.py runner ui tests
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 常见问题

### `ModuleNotFoundError: No module named 'gradio'`

说明当前 Python 环境没有安装 UI 依赖。激活项目虚拟环境并安装依赖：

```powershell
cd <your-parent-folder>
prototype\.venv\Scripts\Activate.ps1
pip install -r prototype\requirements.txt
```

### `ModuleNotFoundError: No module named 'prototype'`

通常是启动目录不对。请从仓库父目录运行：

```powershell
cd <your-parent-folder>
prototype\.venv\Scripts\Activate.ps1
python -m prototype.app
```

### `WSL_E_DISTRO_NOT_FOUND`

检查 WSL 发行版名称：

```powershell
wsl -l -v
```

然后把 UI 或 `local_settings.yaml` 里的 `WSL Distribution` 改成完全一致的名称。

### Environment 页面里 `DLRM root` 为 MISSING

通常是 `local_settings.yaml` 还在使用模板占位：

```text
/mnt/c/Users/<your-name>/Desktop/dlrm
```

请改成本机真实路径，例如：

```text
/mnt/c/Users/han/Desktop/dlrm
```

### DLRM 任务启动前失败

优先检查：

- `command.json`
- `launcher.log`
- `train-rank0.log`
- WSL 发行版名称
- `Python Env`
- `DLRM Root`
- `Criteo Binary Path`

如果日志里出现：

```text
cd '/mnt/c/Users/<your-name>/Desktop/dlrm'
```

说明还没有配置本机 DLRM 路径。

### Monitor 没有 DLRM 指标

先看 `train-rank0.log` 是否包含可解析的指标行：

```text
AUROC over val set: ...
AUROC over test set: ...
Number of val samples: ...
```

如果日志没有这些行，`metrics.jsonl` 可能为空，即使任务已经启动过。

### DLRM checkpoint 加载失败

检查：

- `Checkpoint Load Path` 是否指向包含 `model.pt` 的目录。
- 新任务的模型结构是否和 checkpoint 对应任务一致。
- 当前 smoke checkpoint 路径建议 `Processes per Node=1`。
- `train-rank0.log` 中是否有 `torch.load` 或 `load_state_dict` 错误。

当前 checkpoint 支持主要用于单进程 smoke 验证。多进程分片 checkpoint 需要生产级 Torch Distributed Checkpoint 实现。
