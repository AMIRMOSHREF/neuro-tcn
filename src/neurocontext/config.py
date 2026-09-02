from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"bin_size_s", "delay_bins", "response_bins", "regions", "model", "training"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if len(config["regions"]) != 4:
        raise ValueError("Exactly four canonical regions are required")
    return config
