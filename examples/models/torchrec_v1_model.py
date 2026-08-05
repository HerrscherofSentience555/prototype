from __future__ import annotations


def build_model(config: dict):
    """Return a torch.nn.Module in a full TorchRec runtime."""
    return None


def build_embedding_configs(config: dict) -> list:
    """Return TorchRec EmbeddingBagConfig objects in a full TorchRec runtime."""
    return []


def build_optimizer(model, config: dict):
    return None


def build_dataloader(config: dict, split: str):
    return None


def train_step(step: int, config: dict) -> dict[str, float]:
    return {
        "train_loss": max(0.05, 1.0 / step),
        "auc": min(0.5 + 0.02 * step, 0.95),
    }


def evaluate(config: dict, checkpoint: dict) -> dict[str, float]:
    return checkpoint.get("metrics", {"auc": 0.5, "log_loss": 0.6931})
