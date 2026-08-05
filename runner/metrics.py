from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_metric(path: Path, **payload: Any) -> None:
    record = {"timestamp": datetime.now().isoformat(), **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
