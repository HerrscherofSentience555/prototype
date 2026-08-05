from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototype.config import PrototypeConfig
from prototype.runner.parquet_converter import convert_parquet_to_criteo_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert schema-backed parquet data to Criteo-like numpy files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = PrototypeConfig.from_yaml_file(Path(args.config))
    manifest = convert_parquet_to_criteo_numpy(config, Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
