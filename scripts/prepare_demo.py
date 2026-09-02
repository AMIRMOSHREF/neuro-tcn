#!/usr/bin/env python3
"""Build a demo Data + Data2 tree that matches the real folder/NPZ schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rodent_tcnn.data.synthetic import generate_demo_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--trials-per-class", type=int, default=8)
    parser.add_argument("--figure-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    info = generate_demo_tree(
        Path(args.out),
        n_per_region=args.units,
        trials_per_class=args.trials_per_class,
        seed=args.seed,
        figure_units=args.figure_units,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
