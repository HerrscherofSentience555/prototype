from __future__ import annotations


def train_step(step: int, config: dict) -> dict[str, float]:
    learning_rate = float(config.get("training", {}).get("learning_rate", 0.01))
    loss = max(0.01, 1.0 / (step + 1))
    auc = min(0.99, 0.5 + step * 0.05 + learning_rate)
    return {
        "train_loss": loss,
        "auc": auc,
    }


def evaluate(config: dict, checkpoint: dict) -> dict[str, float]:
    metrics = checkpoint.get("metrics") or {}
    return {
        "auc": float(metrics.get("auc", 0.5)),
        "log_loss": float(metrics.get("train_loss", 0.6931)),
    }
