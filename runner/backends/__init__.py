from __future__ import annotations

from prototype.config import BackendName, PrototypeConfig
from prototype.runner.backends.base import RunnerBackend
from prototype.runner.backends.custom_backend import CustomModelBackend
from prototype.runner.backends.dlrm_backend import DLRMBackend
from prototype.runner.backends.stub_backend import StubBackend
from prototype.runner.backends.torchrec_v1_backend import TorchRecV1Backend


def get_backend(config: PrototypeConfig) -> RunnerBackend:
    if config.backend.name == BackendName.STUB:
        return StubBackend()
    if config.backend.name == BackendName.DLRM:
        return DLRMBackend()
    if config.backend.name == BackendName.CUSTOM:
        return CustomModelBackend()
    if config.backend.name == BackendName.TORCHREC_V1:
        return TorchRecV1Backend()
    raise ValueError(f"Unsupported backend: {config.backend.name}")
