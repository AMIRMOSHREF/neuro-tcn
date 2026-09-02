"""Figure 1: one trial's raster for all neurons of the four regions, the criterion-based neuron
selection overlaid, per-neuron criterion badges, class-conditional PSTHs of the top unit of each
region, and the bullet-point reasons for the selection."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

from .. import CLASSES, REGION_COLORS, REGION_LABELS, REGIONS
from ..data.cache import SessionCache
from ..data.rasters import normalize_region, read_epochs
from ..features.spectral import smooth_rates
from .style import CLASS_COLORS, CRITERIA, EPOCH_COLORS, apply_style, panel_label


def _trial_spikes(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    ep = read_epochs(data)
    regions = np.array([normalize_region(r) or "unknown" for r in np.asarray(data["brain_region"]).astype(str)])
    spikes = data["spike_times"]
    per_region = {}
    for r in REGIONS:
        idx = np.where(regions == r)[0]
        per_region[r] = [np.asarray(spikes[i], dtype=float).ravel() for i in idx]
    licks = {"Left": np.asarray(data["left_lick_times"], dtype=float).ravel() if "left_lick_times" in data.files else np.empty(0),
             "Right": np.asarray(data["right_lick_times"], dtype=float).ravel() if "right_lick_times" in data.files else np.empty(0)}
    return per_region, ep, licks


def _shade_epochs(ax, ep, t0):
    spans = [("sample", ep["sample_start_times"], ep["sample_stop_times"]),
             ("delay", ep["delay_start_times"], ep["go_start_times"]),
             ("response", ep["go_start_times"], ep["go_stop_times"] if not np.isnan(ep["go_stop_times"]) else ep["go_start_times"] + 1.5)]
    for name, a, b in spans:
        if np.isnan(a) or np.isnan(b):
            continue
        ax.axvspan(a - t0, b - t0, color=EPOCH_COLORS[name], zorder=0, lw=0)
    for k in ("sample_start_times", "delay_start_times", "go_start_times"):
        if not np.isnan(ep[k]):
            ax.axvline(ep[k] - t0, color="k", ls=":", lw=0.8, zorder=3)


def _badges(ax, row: pd.Series, x: float, y: float, w: float = 0.07, h: float = 0.8):
    """Four small squares (S C W R) right of a raster row; filled when the criterion is satisfied."""
    for j, (col, letter, _) in enumerate(CRITERIA):
        on = bool(row[col])
        ax.add_patch(Rectangle((x + j * w * 1.2, y - h / 2), w, h, transform=ax.transData,
                               facecolor="#333333" if on else "white", edgecolor="#333333", lw=0.5, clip_on=False, zorder=6))
        ax.text(x + j * w * 1.2 + w / 2, y, letter, ha="center", va="center", fontsize=5,
                color="white" if on else "#999999", zorder=7, clip_on=False)


def selection_bullets(table: pd.DataFrame, cfg, per_region: int = 2) -> list[str]:
    """Human-readable reasons: global criteria, then the top units of every region."""
    sel = cfg.selection
    n_tot, n_sel = len(table), int(table.selected.sum())
    bullets = [
        f"{n_sel}/{n_tot} units selected (top-{sel.top_k_per_region} per region, >= {sel.min_criteria} criteria, BH-FDR q < {sel.fdr_q}).",
        f"Activity floor: mean delay rate >= {sel.min_rate_hz} Hz and spiking on >= {int(sel.min_active_trial_frac * 100)}% of trials "
        f"({int((~table.pass_floor).sum())} units removed).",
        "S  choice selectivity: Kruskal-Wallis on delay spike counts across Ignore/Left/Right "
        f"({int(table.c_selectivity.sum())} units).",
        f"C  coupling: late-delay ({sel.late_delay_ms} ms) rate predicts the unit's own response-epoch rate (Spearman) "
        f"({int(table.c_coupling.sum())} units).",
        "W  wavelet: class-dependent Morlet band power (slow/theta/beta) in the delay "
        f"({int(table.c_spectral.sum())} units).",
        f"R  ramping: monotonic delay PSTH trend ({int(table.c_ramp.sum())} units).",
    ]
    for r in REGIONS:
        top = table[(table.region == r) & table.selected].sort_values("rank").head(per_region)
        for _, row in top.iterrows():
            bullets.append(f"{REGION_LABELS[r]} #{int(row['rank'])} (unit {row.unit_id}): {row.reasons}")
    return bullets


def plot_raster_selection(npz_path, cache: SessionCache, table: pd.DataFrame, cfg, out_path: Path, trial_label: str = "") -> Path:
    apply_style()
    per_region, ep, licks = _trial_spikes(npz_path)
    t0 = ep["delay_start_times"]
    x_min = (ep["sample_start_times"] if not np.isnan(ep["sample_start_times"]) else t0 - 0.65) - t0 - 0.3
    x_max = (ep["go_start_times"] - t0) + 1.6
    counts = {r: len(per_region[r]) for r in REGIONS}

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(6, 3, figure=fig, width_ratios=[1.35, 0.9, 1.0], height_ratios=[1, 1, 1, 1, 0.95, 0.05], hspace=0.45, wspace=0.3)

    # ---- A: all neurons, selected highlighted -----------------------------------------------
    axes_a = []
    for i, r in enumerate(REGIONS):
        ax = fig.add_subplot(gs[i, 0], sharex=axes_a[0] if axes_a else None)
        axes_a.append(ax)
        _shade_epochs(ax, ep, t0)
        sub = table[table.region == r].set_index("unit_index")
        for u, st in enumerate(per_region[r]):
            st = st[(st - t0 >= x_min) & (st - t0 <= x_max)]
            is_sel = bool(sub.loc[u, "selected"]) if u in sub.index else False
            ax.vlines(st - t0, u, u + 0.9, color=REGION_COLORS[r] if is_sel else "#b0b0b0", lw=0.6 if is_sel else 0.4, zorder=2)
            if is_sel:
                ax.add_patch(Rectangle((x_min, u), 0.08, 0.9, color=REGION_COLORS[r], lw=0, zorder=5, clip_on=False))
        n_sel = int(sub.selected.sum()) if len(sub) else 0
        ax.set_ylim(0, max(counts[r], 1))
        ax.set_ylabel(f"{REGION_LABELS[r]}\n{n_sel}/{counts[r]} selected", color=REGION_COLORS[r], fontweight="bold")
        ax.set_xlim(x_min, x_max)
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
    for name, lk in licks.items():
        if lk.size:
            axes_a[0].vlines(lk - t0, counts[REGIONS[0]] * 1.02, counts[REGIONS[0]] * 1.10, color=CLASS_COLORS[name], lw=1.0, clip_on=False)
            axes_a[0].text(x_max, counts[REGIONS[0]] * 1.06, f"{name.lower()} licks", color=CLASS_COLORS[name], fontsize=7, va="center", ha="left")
    for k, lab in (("sample_start_times", "sample"), ("delay_start_times", "delay"), ("go_start_times", "go")):
        if not np.isnan(ep[k]):
            axes_a[0].text(ep[k] - t0, counts[REGIONS[0]] * 1.14, lab, ha="center", fontsize=7.5, clip_on=False)
    axes_a[-1].set_xlabel("Time from delay onset (s)")
    axes_a[0].set_title(f"All recorded units, one trial ({trial_label}); selected units in colour", loc="left")
    panel_label(axes_a[0], "A", y=1.22)

    # ---- B: selected neurons only, sorted by score, with criterion badges ---------------------
    axes_b = []
    for i, r in enumerate(REGIONS):
        ax = fig.add_subplot(gs[i, 1], sharex=axes_a[0])
        axes_b.append(ax)
        _shade_epochs(ax, ep, t0)
        sub = table[(table.region == r) & table.selected].sort_values("rank", ascending=False)
        for row_i, (_, row) in enumerate(sub.iterrows()):
            st = per_region[r][int(row.unit_index)]
            st = st[(st - t0 >= x_min) & (st - t0 <= x_max)]
            ax.vlines(st - t0, row_i, row_i + 0.9, color=REGION_COLORS[r], lw=0.7)
            _badges(ax, row, x_max + 0.08, row_i + 0.45)
        ax.set_ylim(0, max(len(sub), 1))
        ax.set_yticks([i + 0.45 for i in range(len(sub))])
        ax.set_yticklabels([f"#{int(k)}" for k in sub["rank"]], fontsize=6)
        ax.set_ylabel("rank", fontsize=7)
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes_b[-1].set_xlabel("Time from delay onset (s)")
    axes_b[0].set_title("Selected units (rank-ordered) with criterion badges  S C W R", loc="left")
    panel_label(axes_b[0], "B", y=1.22)

    # ---- C: score bars for the selected units --------------------------------------------------
    ax_c = fig.add_subplot(gs[0:2, 2])
    sel_tab = table[table.selected].sort_values(["region", "score"], ascending=[True, False])
    ypos = np.arange(len(sel_tab))
    ax_c.barh(ypos, sel_tab.score, color=[REGION_COLORS[r] for r in sel_tab.region], height=0.8)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([f"{REGION_LABELS[r].split()[0][0]}{REGION_LABELS[r].split()[1][:3]} u{u}" for r, u in zip(sel_tab.region, sel_tab.unit_id)], fontsize=5.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("combined score  Σ w·(−log10 q)")
    ax_c.set_title("Selection score per selected unit", loc="left")
    panel_label(ax_c, "C", x=-0.25)

    # ---- D: class-conditional delay PSTH of the top unit of every region ----------------------
    y = cache.labels
    ax_d = [fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[3, 2])]
    t_ctx = cache.context[REGIONS[0]].shape[2]
    tvec = (np.arange(t_ctx) + 0.5) * cache.bin_ms / 1000.0
    t_tgt = cache.target[REGIONS[0]].shape[2]
    tvec_t = 1.2 + (np.arange(t_tgt) + 0.5) * cache.target_bin_ms / 1000.0
    for ax, pair in zip(ax_d, (REGIONS[:2], REGIONS[2:])):
        for r in pair:
            top = table[(table.region == r) & table.selected].sort_values("rank").head(1)
            if not len(top):
                continue
            u = int(top.unit_index.iloc[0])
            for c_i, c in enumerate(CLASSES):
                m = y == c_i
                if not m.any():
                    continue
                psth = smooth_rates(cache.context[r][m, u].mean(0), cache.bin_ms, cfg.data.smoothing_sigma_ms)
                psth_t = cache.target[r][m, u].mean(0) / (cache.target_bin_ms / 1000.0)
                ls = "-" if r.endswith("L") else "--"
                ax.plot(tvec, psth, color=CLASS_COLORS[c], ls=ls, lw=1.2, label=f"{REGION_LABELS[r]} u{top.unit_id.iloc[0]} · {c}")
                ax.plot(tvec_t, psth_t, color=CLASS_COLORS[c], ls=ls, lw=1.2, alpha=0.6)
        ax.axvspan(0, 1.2, color=EPOCH_COLORS["delay"], zorder=0, lw=0)
        ax.axvspan(1.2, 1.2 + t_tgt * cache.target_bin_ms / 1000.0, color=EPOCH_COLORS["response"], zorder=0, lw=0)
        ax.axvline(1.2, color="k", ls=":", lw=0.8)
        ax.set_ylabel("rate (Hz)")
        ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    ax_d[0].set_title("Top-ranked unit per region: class-conditional PSTH (delay → response)", loc="left")
    ax_d[1].set_xlabel("Time from delay onset (s)")
    panel_label(ax_d[0], "D", x=-0.25)

    # ---- E: bullet reasons ----------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[4, :])
    ax_e.axis("off")
    bullets = selection_bullets(table, cfg)
    text = "\n".join("• " + b for b in bullets)
    ax_e.text(0.0, 1.0, "Why these units were selected", fontsize=9.5, fontweight="bold", va="top", transform=ax_e.transAxes)
    ax_e.text(0.0, 0.86, text, fontsize=6.9, va="top", transform=ax_e.transAxes, family="DejaVu Sans", wrap=True)
    panel_label(ax_e, "E", x=-0.02, y=0.98)

    handles = [Patch(color=EPOCH_COLORS[k], label=k) for k in EPOCH_COLORS]
    fig.legend(handles=handles, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.005))
    fig.suptitle(f"Criterion-based neuron selection — session {cache.session}", fontsize=12, y=0.995)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path
