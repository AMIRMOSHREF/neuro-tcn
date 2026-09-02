#!/usr/bin/env python3
"""Render Figure 1 from a trial NPZ (uses the demo figure trial by default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rodent_tcnn.config import load_config
from rodent_tcnn.viz.selection_figure import render_selection_figure
from run_pipeline import figure_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--npz", default="data/demo/figure_trial/trial_figure_right.npz")
    parser.add_argument("--out", default="figures/fig1_neuron_selection.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scores, label, _ = figure_scores(cfg, Path(args.npz))
    path = render_selection_figure(args.npz, scores, args.out, label=label)
    print(path)


if __name__ == "__main__":
    main()
