from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.config import PrototypeConfig  # noqa: E402


class ExampleConfigTests(unittest.TestCase):
    def test_example_configs_parse(self) -> None:
        examples_dir = Path(__file__).resolve().parents[1] / "examples"
        for path in examples_dir.glob("*.yaml"):
            with self.subTest(path=path.name):
                config = PrototypeConfig.from_yaml_file(path)
                self.assertTrue(config.job_name)


if __name__ == "__main__":
    unittest.main()
