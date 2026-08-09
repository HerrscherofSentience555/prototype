from __future__ import annotations


def build_model(config: dict):
    """Return a tiny dense+sparse recommendation model for V1 runtime smoke paths."""
    try:
        import torch
    except Exception:
        return None

    dense_features = 13
    sparse_features = 26
    embedding_dim = config.get("model", {}).get("embedding_dim") or 8
    num_embeddings = config.get("model", {}).get("num_embeddings") or 1024
    hidden_dim = 16
    embedding_configs = build_embedding_configs(config)

    if embedding_configs:
        try:
            from torchrec.modules.embedding_modules import EmbeddingBagCollection
        except Exception:
            embedding_configs = []

    if embedding_configs:
        class TinyTorchRecRecommendationModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ebc = EmbeddingBagCollection(tables=embedding_configs)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(dense_features + sparse_features * embedding_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )

            def forward(self, dense, sparse=None):
                if sparse is None:
                    sparse_values = torch.zeros(
                        (dense.shape[0], sparse_features * embedding_dim),
                        dtype=dense.dtype,
                        device=dense.device,
                    )
                elif hasattr(sparse, "values") and not isinstance(sparse, torch.Tensor):
                    sparse_values = self.ebc(sparse).values().to(dtype=dense.dtype)
                else:
                    sparse_values = sparse.to(dtype=dense.dtype)
                    if sparse_values.shape[1] != sparse_features * embedding_dim:
                        sparse_values = sparse_values.repeat_interleave(embedding_dim, dim=1)
                features = torch.cat([dense, sparse_values], dim=1)
                return self.net(features)

        return TinyTorchRecRecommendationModel()

    class TinyRecommendationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dense_features + sparse_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, dense, sparse=None):
            if sparse is None:
                sparse = torch.zeros(
                    (dense.shape[0], sparse_features),
                    dtype=dense.dtype,
                    device=dense.device,
                )
            if not isinstance(sparse, torch.Tensor) and hasattr(sparse, "values"):
                values = sparse.values()
                expected = dense.shape[0] * sparse_features
                if values.numel() == expected:
                    sparse = values.view(dense.shape[0], sparse_features)
                else:
                    sparse = torch.zeros(
                        (dense.shape[0], sparse_features),
                        dtype=dense.dtype,
                        device=dense.device,
                    )
            sparse_float = sparse.to(dtype=dense.dtype)
            features = torch.cat([dense, sparse_float], dim=1)
            return self.net(features)

    return TinyRecommendationModel()


def build_embedding_configs(config: dict) -> list:
    """Return TorchRec EmbeddingBagConfig objects when torchrec is available."""
    try:
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
    except Exception:
        return []
    embedding_dim = config.get("model", {}).get("embedding_dim") or 8
    num_embeddings = config.get("model", {}).get("num_embeddings") or 1024
    return [
        EmbeddingBagConfig(
            name=f"t_C{index}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"C{index}"],
        )
        for index in range(1, 27)
    ]


def build_optimizer(model, config: dict):
    try:
        import torch
    except Exception:
        return None
    if model is None:
        return None
    return torch.optim.SGD(model.parameters(), lr=config.get("training", {}).get("learning_rate", 0.01))


def build_dataloader(config: dict, split: str):
    return None


def train_step(step: int, config: dict) -> dict[str, float]:
    return {
        "train_loss": max(0.05, 1.0 / step),
        "auc": min(0.5 + 0.02 * step, 0.95),
    }


def evaluate(config: dict, checkpoint: dict) -> dict[str, float]:
    return checkpoint.get("metrics", {"auc": 0.5, "log_loss": 0.6931})
