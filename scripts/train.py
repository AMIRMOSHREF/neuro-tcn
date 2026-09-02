#!/usr/bin/env python3
"""Train SPEC-TCNN on the discovered Data + Data2 catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rodent_tcnn.config import load_config
from rodent_tcnn.train.engine import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tf", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    result = train_model(cfg, compute_tf=args.tf)
    print(f"trained on {result['n_train']} trials, validated on {result['n_val']}")
    if result["history"]:
        print("best val acc", max(h["val_acc"] for h in result["history"]))


if __name__ == "__main__":
    main()
