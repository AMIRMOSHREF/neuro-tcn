"""Figure 4: decoding and forecasting results (confusion matrix, context-length sweep, region
ablation, forecasting deviance explained, neuron-set ablation and linear baselines)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import CLASSES, REGION_COLORS, REGION_LABELS, REGIONS
from .style import apply_style, panel_label


def plot_results(results_by_mode: dict[str, dict], cfg, out_path: Path) -> Path:
    apply_style()
    main = results_by_mode.get("criteria") or next(iter(results_by_mode.values()))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    # A confusion matrix
    ax = axes[0, 0]
    cm = np.asarray(main["confusion"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    m = main["classification"]
    ax.set_title(f"Test confusion — bal. acc {m['balanced_accuracy']:.2f}, F1 {m['macro_f1']:.2f} (n={m['n']})", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    panel_label(ax, "A")

    # B context sweep
    ax = axes[0, 1]
    for mode, res in results_by_mode.items():
        sw = res.get("context_sweep", [])
        if sw:
            ax.plot([s["context_ms"] for s in sw], [s["balanced_accuracy"] for s in sw], marker="o", lw=1.5,
                    label={"criteria": "criteria-selected", "rate": "top-K by rate", "random": "random K"}.get(mode, mode))
    ch = main["chance_balanced_accuracy"]
    ax.axhspan(0, ch["p95"], color="#dddddd", zorder=0, label="shuffle 95th pct")
    ax.set_xlabel("delay context available before go (ms)")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("How much past context is needed?", loc="left")
    ax.legend(fontsize=6.5)
    panel_label(ax, "B")

    # C region ablation
    ax = axes[0, 2]
    abl = main.get("region_ablation", [])
    base = main["classification"]["balanced_accuracy"]
    ax.bar(range(len(abl)), [base - a["balanced_accuracy"] for a in abl], color=[REGION_COLORS[a["dropped_region"]] for a in abl])
    ax.set_xticks(range(len(abl)))
    ax.set_xticklabels([REGION_LABELS[a["dropped_region"]] for a in abl], rotation=20)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ balanced accuracy when region removed")
    ax.set_title("Region ablation (drop one region's input)", loc="left")
    panel_label(ax, "C")

    # D forecast deviance explained
    ax = axes[1, 0]
    fc = main.get("forecast", {})
    w = 0.38
    model_v = [fc.get(f"deviance_explained_{r}", np.nan) for r in REGIONS]
    pers_v = [fc.get(f"deviance_explained_persistence_{r}", np.nan) for r in REGIONS]
    ax.bar(np.arange(4) - w / 2, model_v, width=w, color=[REGION_COLORS[r] for r in REGIONS], label="DelayCAST forecast")
    ax.bar(np.arange(4) + w / 2, pers_v, width=w, color="#bbbbbb", label="persistence (late-delay rate)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS], rotation=20)
    ax.set_ylabel("Poisson deviance explained\n(vs training PSTH null)")
    ax.set_title("Response-epoch forecast from delay activity", loc="left")
    ax.legend(fontsize=6.5)
    panel_label(ax, "D")

    # E neuron-set ablation + baselines
    ax = axes[1, 1]
    names, vals = [], []
    for mode, res in results_by_mode.items():
        names.append({"criteria": "DelayCAST\ncriteria-selected", "rate": "DelayCAST\ntop-K by rate", "random": "DelayCAST\nrandom K"}.get(mode, mode))
        vals.append(res["classification"]["balanced_accuracy"])
    for b in main.get("baselines", []):
        names.append(b["model"].replace("logreg_", "log-reg\n").replace("_", " "))
        vals.append(b["balanced_accuracy"])
    ax.bar(range(len(vals)), vals, color=["#1f77b4"] * len(results_by_mode) + ["#999999"] * (len(vals) - len(results_by_mode)))
    ax.axhline(ch["mean"], color="k", ls=":", lw=0.8, label="chance")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0, 1); ax.set_ylabel("balanced accuracy")
    ax.set_title("Neuron-set ablation and linear baselines", loc="left")
    ax.legend(fontsize=6.5)
    panel_label(ax, "E")

    # F per-session accuracy
    ax = axes[1, 2]
    ps = main.get("per_session", [])
    ax.bar(range(len(ps)), [p["balanced_accuracy"] for p in ps], color="#4c72b0")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([p["session"].replace("_behavior+ecephys+image+ogen", "") for p in ps], rotation=35, ha="right", fontsize=6)
    ax.axhline(ch["mean"], color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 1); ax.set_ylabel("balanced accuracy")
    ax.set_title("Per-session test performance (both datasets)", loc="left")
    panel_label(ax, "F")

    fig.suptitle("DelayCAST: predicting the upcoming action and response-epoch activity from the delay epoch", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path


def load_results(out_dir: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(Path(out_dir).glob("run_*")):
        p = d / "results.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                out[d.name.replace("run_", "")] = json.load(f)
    return out
