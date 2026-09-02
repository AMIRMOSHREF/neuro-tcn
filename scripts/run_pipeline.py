#!/usr/bin/env python3
"""End-to-end SPEC-TCNN pipeline on Data + Data2 (or the demo tree)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rodent_tcnn.claims import CLAIMS
from rodent_tcnn.config import load_config
from rodent_tcnn.constants import CLASS_TO_ID, KEY_TO_REGION, REGION_KEYS
from rodent_tcnn.data.catalog import catalog_frame, discover_trials
from rodent_tcnn.data.dataset import DualDatasetRasterDataset, encode_trial
from rodent_tcnn.data.epochs import delay_window, lick_window
from rodent_tcnn.data.io_npz import load_trial_npz, pad_or_crop, split_by_region
from rodent_tcnn.data.synthetic import generate_matched_population
from rodent_tcnn.features.selection import collect_population, select_neurons, selection_table
from rodent_tcnn.train.engine import collect_attention, train_model
from rodent_tcnn.viz.selection_figure import render_schematic, render_selection_figure


def _encode_payload(payload: dict, label: str, cfg) -> dict | None:
    d0, d1 = delay_window(payload)
    l0, l1 = lick_window(payload, cfg.epochs.lick_window_s)
    delay = split_by_region(payload, d0, d1, cfg.epochs.delay_bins)
    lick = split_by_region(payload, l0, l1, cfg.epochs.lick_bins)
    n = cfg.model.units_per_region
    # for figure population keep native unit count
    n_use = delay["left_ALM"]["raster"].shape[0] or n
    delay_stack = np.stack([pad_or_crop(delay[k]["raster"], n_use) for k in REGION_KEYS], axis=0)
    lick_stack = np.stack([pad_or_crop(lick[k]["raster"], n_use) for k in REGION_KEYS], axis=0)
    return {
        "delay": delay_stack.astype(np.float32),
        "lick": lick_stack.astype(np.float32),
        "label": CLASS_TO_ID[label],
    }


def _figure_priors(pop: dict, bin_size: float) -> tuple[dict, dict]:
    """Stand-in attention / gain when the trained model has a different unit count."""
    attn, gain = {}, {}
    for key, pack in pop.items():
        delay, lick, labels = pack["delay"], pack["lick"], pack["labels"]
        if delay.size == 0:
            continue
        n_u = delay.shape[1]
        rate = delay.sum(axis=2) / max(delay.shape[2] * bin_size, 1e-6)
        dprime = np.zeros(n_u)
        coupling = np.zeros(n_u)
        for u in range(n_u):
            fr = rate[:, u]
            left, right = labels == 1, labels == 2
            if left.any() and right.any():
                pos, neg = fr[left], fr[right]
                v = 0.5 * (pos.var() + neg.var())
                dprime[u] = abs(pos.mean() - neg.mean()) / np.sqrt(v + 1e-8)
            a = delay[:, u].mean(0)
            b = lick[:, u].mean(0)
            b = np.interp(np.linspace(0, 1, a.size), np.linspace(0, 1, max(b.size, 1)), b)
            if a.std() > 1e-8 and b.std() > 1e-8:
                coupling[u] = max(float(np.corrcoef(a, b)[0, 1]), 0.0)
        z = np.exp(dprime - dprime.max())
        attn[key] = z / z.sum()
        gain[key] = coupling * (dprime / (dprime.max() + 1e-8))
    return attn, gain


def figure_scores(cfg, figure_npz: Path):
    data = load_trial_npz(figure_npz)
    label = str(data.get("label", "Right"))
    if isinstance(label, np.ndarray):
        label = str(label.item() if label.size else "Right")
    matched = generate_matched_population(data, n_per_class=6, seed=21)
    items = []
    for trial in matched:
        enc = _encode_payload(trial["payload"], trial["label"], cfg)
        if enc:
            items.append(enc)
    # include the figure trial itself
    enc0 = _encode_payload(data, label, cfg)
    if enc0:
        items.append(enc0)
    pop = collect_population(items, cfg.epochs.bin_size)
    attn, gain = _figure_priors(pop, cfg.epochs.bin_size)
    types = {}
    ids = {}
    if "neuron_type" in data:
        regions = np.asarray(data["brain_region"])
        ntypes = np.asarray(data["neuron_type"])
        uids = np.asarray(data["unit_ids"])
        from rodent_tcnn.data.io_npz import normalize_region

        for key in REGION_KEYS:
            mask = np.array([normalize_region(str(r)) == key for r in regions])
            types[key] = ntypes[mask]
            ids[key] = uids[mask]
    scores = select_neurons(
        pop,
        cfg,
        attention_by_region=attn,
        pred_gain_by_region=gain,
        unit_ids_by_region=ids,
        neuron_types_by_region=types,
    )
    return scores, label, data


def laterality_index(scores) -> dict:
    alm_l = [s for s in scores if s.region_key == "left_ALM" and s.selected]
    alm_r = [s for s in scores if s.region_key == "right_ALM" and s.selected]
    def pref_frac(block, cls):
        if not block:
            return 0.0
        return sum(1 for s in block if s.preferred_class == cls) / len(block)
    return {
        "left_ALM_right_pref": pref_frac(alm_l, "Right"),
        "right_ALM_left_pref": pref_frac(alm_r, "Left"),
        "n_selected_left_ALM": len(alm_l),
        "n_selected_right_ALM": len(alm_r),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tf", action="store_true", help="compute wavelet/STFT inside the train loop (slower)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    out = Path(cfg.paths.output_dir)
    fig_dir = Path(cfg.paths.figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = discover_trials(cfg)
    catalog = catalog_frame(records)
    catalog.to_csv(out / "trial_catalog.csv", index=False)
    print(f"discovered {len(records)} trials")
    if not catalog.empty:
        print(catalog.groupby(["dataset", "label"]).size().unstack(fill_value=0))

    attn = None
    history = []
    metrics = {}
    history_path = Path(cfg.paths.checkpoint_dir) / "history.json"
    if args.skip_train and history_path.exists():
        prior = json.loads(history_path.read_text())
        if prior:
            metrics = {
                "n_train": None,
                "n_val": prior[-1].get("n_eval"),
                "best_val_acc": max(h.get("val_acc", 0) for h in prior),
                "last": prior[-1],
            }
    if records and not args.skip_train:
        result = train_model(cfg, records, compute_tf=args.tf)
        history = result["history"]
        dataset = DualDatasetRasterDataset(records, cfg)
        attn = collect_attention(result["model"], dataset, cfg, result["device"])
        metrics = {
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "best_val_acc": max(h["val_acc"] for h in history) if history else None,
            "last": history[-1] if history else None,
        }
        # training-set selection
        items = [encode_trial(r, cfg) for r in records]
        items = [x for x in items if x is not None]
        pop = collect_population(items, cfg.epochs.bin_size)
        train_scores = select_neurons(pop, cfg, attention_by_region=attn)
        selection_table(train_scores).to_csv(out / "selection_training_set.csv", index=False)

    figure_npz = Path(cfg.paths.demo_root) / "figure_trial" / "trial_figure_right.npz"
    if not figure_npz.exists():
        # fall back to first Right trial
        right = [r for r in records if r.label == "Right"]
        figure_npz = right[0].path if right else None

    scores, label, _ = figure_scores(cfg, Path(figure_npz))
    table = selection_table(scores)
    table.to_csv(out / "selection_figure_trial.csv", index=False)
    (out / "laterality.json").write_text(json.dumps(laterality_index(scores), indent=2))

    fig_path = render_selection_figure(figure_npz, scores, fig_dir / "fig1_neuron_selection.png", label=label)
    sch_path = render_schematic(fig_dir / "fig0_spec_tcnn_schematic.png")

    # dashboard JSON
    selected = table[table["selected"]].sort_values("score", ascending=False)
    payload = {
        "n_trials": int(len(records)),
        "catalog_summary": catalog.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_dict() if not catalog.empty else {},
        "metrics": metrics,
        "laterality": laterality_index(scores),
        "n_selected": int(selected.shape[0]),
        "n_scored": int(table.shape[0]),
        "selected": selected.head(12).to_dict(orient="records"),
        "claims": CLAIMS,
        "figures": {
            "selection": str(fig_path),
            "schematic": str(sch_path),
        },
    }
    (out / "dashboard.json").write_text(json.dumps(payload, indent=2, default=str))
    dash_public = ROOT / "dashboard" / "public"
    dash_public.mkdir(parents=True, exist_ok=True)
    (dash_public / "dashboard.json").write_text(json.dumps(payload, indent=2, default=str))
    # copy figures into the Next.js public folder
    for src in (fig_path, sch_path):
        dest = dash_public / Path(src).name
        dest.write_bytes(Path(src).read_bytes())

    print(f"figure -> {fig_path}")
    print(f"schematic -> {sch_path}")
    print(f"selected {payload['n_selected']} / {payload['n_scored']} units")


if __name__ == "__main__":
    main()
