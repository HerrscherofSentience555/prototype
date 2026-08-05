$ErrorActionPreference = "Continue"

$distro = $env:TORCHREC_WSL_DISTRO
if (-not $distro) {
    $distro = "Ubuntu-22.04"
}

$pythonEnv = $env:TORCHREC_PYTHON_ENV
if (-not $pythonEnv) {
    $pythonEnv = "~/venvs/torchrec17"
}

Write-Host "TorchRec V1 DMP readiness check"
Write-Host "==============================="
Write-Host "WSL Distribution: $distro"
Write-Host "Python Env: $pythonEnv"

$pythonCode = @'
import json
result = {
    "torch": False,
    "torchrec": False,
    "distributed_model_parallel": False,
    "embedding_sharding_planner": False,
    "train_pipeline_sparse_dist": False,
    "torch_distributed_checkpoint": False,
    "errors": [],
}
try:
    import torch
    result["torch"] = True
    result["torch_version"] = torch.__version__
except Exception as exc:
    result["errors"].append(f"torch: {type(exc).__name__}: {exc}")
try:
    import torchrec
    result["torchrec"] = True
    result["torchrec_version"] = getattr(torchrec, "__version__", "unknown")
except Exception as exc:
    result["errors"].append(f"torchrec: {type(exc).__name__}: {exc}")
try:
    from torchrec.distributed.model_parallel import DistributedModelParallel
    result["distributed_model_parallel"] = True
except Exception as exc:
    result["errors"].append(f"DistributedModelParallel: {type(exc).__name__}: {exc}")
try:
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    result["embedding_sharding_planner"] = True
except Exception as exc:
    result["errors"].append(f"EmbeddingShardingPlanner: {type(exc).__name__}: {exc}")
try:
    from torchrec.distributed import TrainPipelineSparseDist
    result["train_pipeline_sparse_dist"] = True
except Exception as exc:
    result["errors"].append(f"TrainPipelineSparseDist: {type(exc).__name__}: {exc}")
try:
    import torch.distributed.checkpoint as tdc
    result["torch_distributed_checkpoint"] = True
except Exception as exc:
    result["errors"].append(f"torch.distributed.checkpoint: {type(exc).__name__}: {exc}")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not all([
    result["torch"],
    result["torchrec"],
    result["distributed_model_parallel"],
    result["embedding_sharding_planner"],
    result["train_pipeline_sparse_dist"],
    result["torch_distributed_checkpoint"],
]):
    raise SystemExit(1)
'@

$bashCommand = "set -e; source $pythonEnv/bin/activate; python -"
$pythonCode | wsl -d $distro bash -lc $bashCommand
exit $LASTEXITCODE
