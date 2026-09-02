"""Figure 3: what the model attends to — temporal attention over the delay per region and class,
cross-region attention, and learned neuron gates vs the statistical selection score."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .. import CLASSES, REGION_COLORS, REGION_LABELS, REGIONS
from .style import CLASS_COLORS, apply_style, panel_label


def plot_attention(attention_npz: Path, selections: dict[str, pd.DataFrame], unit_index: dict[str, dict[str, np.ndarray]],
                   bin_ms: float, cfg, out_path: Path) -> Path:
    apply_style()
    att = np.load(attention_npz)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), gridspec_kw={"hspace": 0.5, "wspace": 0.35})
    # Row 1: temporal attention per region.
    for j, r in enumerate(REGIONS):
        ax = axes[0, j]
        for c in CLASSES:
            key = f"temporal_{r}_{c}"
            if key in att.files:
                w = att[key]
                t = (np.arange(len(w)) + 0.5) * bin_ms / 1000.0
                ax.plot(t, w * len(w), color=CLASS_COLORS[c], lw=1.4, label=c)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{REGION_LABELS[r]} — temporal attention", color=REGION_COLORS[r], loc="left")
        ax.set_xlabel("time from delay onset (s)")
        ax.set_ylabel("attention (× uniform)")
        if j == 0:
            ax.legend()
    panel_label(axes[0, 0], "A")
    # Row 2 left: cross-region attention per class.
    ax = axes[1, 0]
    w = 0.25
    for c_i, c in enumerate(CLASSES):
        key = f"region_{c}"
        if key in att.files:
            ax.bar(np.arange(len(REGIONS)) + (c_i - 1) * w, att[key], width=w, color=CLASS_COLORS[c], label=c)
    ax.set_xticks(np.arange(len(REGIONS)))
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS], rotation=20)
    ax.set_ylabel("region attention weight")
    ax.set_title("Cross-region attention (test trials)", loc="left")
    ax.legend()
    panel_label(ax, "B")
    # Row 2 rest: gates vs selection score, pooled over regions and sessions.
    ax_g = axes[1, 1]
    xs, ys, cs = [], [], []
    for sess, tab in selections.items():
        for r in REGIONS:
            key = f"gates_{sess.replace('/', '__')}_{r}"
            if key not in att.files:
                continue
            g = att[key]
            ui = unit_index[sess][r]
            sub = tab[tab.region == r].set_index("unit_index")
            for k, u in enumerate(ui):
                if u >= 0 and u in sub.index:
                    xs.append(sub.loc[u, "score"]); ys.append(g[k]); cs.append(REGION_COLORS[r])
    if xs:
        ax_g.scatter(xs, ys, c=cs, s=12, alpha=0.7)
        if len(xs) > 3:
            rho, p = stats.spearmanr(xs, ys)
            ax_g.text(0.02, 0.02, f"Spearman ρ = {rho:.2f}, p = {p:.2g}", transform=ax_g.transAxes, fontsize=7.5)
    ax_g.set_xlabel("statistical selection score")
    ax_g.set_ylabel("learned neuron gate")
    ax_g.set_title("Learned gates vs criteria score", loc="left")
    panel_label(ax_g, "C")
    # Gate distributions per region.
    ax_h = axes[1, 2]
    for r in REGIONS:
        vals = np.concatenate([att[k] for k in att.files if k.startswith("gates_") and k.endswith(f"_{r}")]) if any(
            k.startswith("gates_") and k.endswith(f"_{r}") for k in att.files) else np.empty(0)
        if vals.size:
            ax_h.hist(vals, bins=20, range=(0, 1), histtype="step", color=REGION_COLORS[r], lw=1.4, label=REGION_LABELS[r])
    ax_h.set_xlabel("gate value")
    ax_h.set_ylabel("units")
    ax_h.set_title("Distribution of learned gates", loc="left")
    ax_h.legend(fontsize=6.5)
    panel_label(ax_h, "D")
    # Criteria overlap of gated-in units.
    ax_k = axes[1, 3]
    crit = ["c_selectivity", "c_coupling", "c_spectral", "c_ramp"]
    frac_sel = []
    for c in crit:
        vals = [tab.loc[tab.selected, c].mean() for tab in selections.values() if tab.selected.any()]
        frac_sel.append(np.mean(vals) if vals else 0)
    ax_k.bar(range(4), frac_sel, color="#444444")
    ax_k.set_xticks(range(4))
    ax_k.set_xticklabels(["selectivity", "coupling", "wavelet", "ramp"], rotation=20)
    ax_k.set_ylabel("fraction of selected units")
    ax_k.set_title("Criteria satisfied by selected units", loc="left")
    panel_label(ax_k, "E")
    fig.suptitle("DelayCAST attention: which past context, which regions and which neurons drive the prediction", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path
