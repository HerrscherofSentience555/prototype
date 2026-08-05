from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402
from prototype.runner.torchrec_runner.contract import load_model_module  # noqa: E402
from prototype.runner.torchrec_runner.embedding import materialize_embedding_configs  # noqa: E402


class TorchRecEmbeddingTests(unittest.TestCase):
    def test_default_embedding_configs_are_generated_when_model_returns_empty_list(self) -> None:
        model_file = Path(__file__).resolve().parents[1] / "examples" / "models" / "torchrec_v1_model.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            module = load_model_module(str(model_file))
            report = materialize_embedding_configs(
                module,
                PrototypeConfig(
                    backend={"name": "torchrec_v1"},
                    model={"embedding_dim": 4, "num_embeddings": 16},
                ),
                run_dir,
            )
            saved = json.loads(
                (run_dir / "artifacts" / "torchrec-embedding-configs.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["source"], "default_criteo_like")
        self.assertEqual(report["count"], 26)
        self.assertEqual(saved["configs"][0]["embedding_dim"], 4)
        self.assertEqual(saved["configs"][0]["num_embeddings"], 16)


if __name__ == "__main__":
    unittest.main()
