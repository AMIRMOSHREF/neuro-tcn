"""Publication figure: all-neuron rasters → selected neurons + reasons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from ..constants import CLASS_COLORS, KEY_TO_REGION, REGION_COLORS, REGION_KEYS, REGIONS
from ..data.epochs import delay_window, lick_window
from ..data.io_npz import load_trial_npz, normalize_region
from ..features.selection import NeuronScore
from ..features.spectral import wavelet_power


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _region_cmap(hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("r", ["#ffffff", hex_color])


def _bin_full_trial(data: dict, bin_size: float = 0.01) -> dict[str, np.ndarray]:
    t0 = float(data["trial_start"])
    t1 = float(data["trial_stop"])
    n_bins = int(np.ceil((t1 - t0) / bin_size))
    edges = np.linspace(t0, t1, n_bins + 1)
    regions = np.asarray(data["brain_region"])
    spikes = data["spike_times"]
    out = {}
    for display in REGIONS:
        key = None
        for r in regions:
            if normalize_region(str(r)) and KEY_TO_REGION.get(normalize_region(str(r)), "") == display:
                key = normalize_region(str(r))
                break
        key = key or next(k for k, v in KEY_TO_REGION.items() if v == display)
        mask = np.array([normalize_region(str(r)) == key for r in regions])
        idx = np.where(mask)[0]
        raster = np.zeros((len(idx), n_bins), dtype=np.float32)
        for row, ui in enumerate(idx):
            s = np.asarray(spikes[ui], dtype=np.float64)
            s = s[(s >= t0) & (s < t1)]
            raster[row] = np.histogram(s, bins=edges)[0]
        out[key] = {"raster": raster, "indices": idx, "unit_ids": np.asarray(data["unit_ids"])[idx]}
    return out


def _epoch_spans(data: dict) -> list[tuple[str, float, float, str]]:
    return [
        ("presample", float(data["presample_start_times"]), float(data["presample_stop_times"]), "#e5e7eb"),
        ("sample", float(data["sample_start_times"]), float(data["sample_stop_times"]), "#fde68a"),
        ("DELAY", float(data["delay_start_times"]), float(data["delay_stop_times"]), "#bfdbfe"),
        ("go / lick", float(data["go_start_times"]), float(data["go_stop_times"]), "#fecaca"),
    ]


def _draw_raster(ax, raster: np.ndarray, t0: float, t1: float, color: str, selected_mask=None, title: str = ""):
    n, tb = raster.shape
    show = np.clip(raster, 0, 3)
    if selected_mask is None:
        img = show
        cmap = _region_cmap(color)
        ax.imshow(img, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2.2, extent=[t0, t1, n, 0])
    else:
        faded = np.clip(show * 0.18, 0, 3)
        ax.imshow(faded, aspect="auto", interpolation="nearest", cmap=_region_cmap("#9ca3af"), vmin=0, vmax=2.2, extent=[t0, t1, n, 0])
        sel = show.copy()
        sel[~selected_mask] = np.nan
        cmap = _region_cmap(color)
        cmap.set_bad(alpha=0)
        ax.imshow(sel, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2.2, extent=[t0, t1, n, 0])
        for i, flag in enumerate(selected_mask):
            if flag:
                ax.plot([t0 - 0.04 * (t1 - t0)], [i + 0.5], marker=">", color="#ca8a04", markersize=3.2, clip_on=False)
    ax.set_xlim(t0, t1)
    ax.set_ylim(n, 0)
    ax.set_ylabel(f"{title}\n{n} units", fontsize=7.5, color=color, fontweight="bold")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_yticks([])


def _shade_epochs(ax, data: dict, y0: float, y1: float, labeled: bool = False):
    for name, a, b, col in _epoch_spans(data):
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        ax.axvspan(a, b, color=col, alpha=0.35, zorder=0)
        if labeled:
            ax.text((a + b) / 2, y0 - 0.02 * (y1 - y0), name, ha="center", va="bottom", fontsize=6.5, color="#374151")


def render_selection_figure(
    npz_path: str | Path,
    scores: list[NeuronScore],
    out_path: str | Path,
    label: str = "Right",
    bin_size: float = 0.01,
) -> Path:
    data = load_trial_npz(npz_path)
    packed = _bin_full_trial(data, bin_size)
    t0, t1 = float(data["trial_start"]), float(data["trial_stop"])
    d0, d1 = delay_window(data)
    l0, l1 = lick_window(data, 0.8)

    selected = {(s.region_key, s.local_index) for s in scores if s.selected}
    by_region = {k: [s for s in scores if s.region_key == k] for k in REGION_KEYS}

    fig = plt.figure(figsize=(13.4, 16.2), facecolor="white")
    gs = GridSpec(
        11,
        6,
        figure=fig,
        height_ratios=[0.42, 1, 1, 1, 1, 0.28, 1, 1, 1, 1, 2.15],
        width_ratios=[1.15, 1.15, 1.15, 1.15, 0.08, 1.35],
        hspace=0.38,
        wspace=0.28,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.035,
    )

    fig.text(0.07, 0.965, "Figure 1", fontsize=11, fontweight="bold", color="#111827")
    fig.text(
        0.16,
        0.965,
        "Causal predictive neuron selection across ALM and striatum",
        fontsize=12.5,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.07,
        0.942,
        f"Single {label.lower()}-lick trial  ·  four-region raster  ·  delay context used to forecast lick-period activity  ·  "
        f"selected units marked in gold",
        fontsize=8,
        color="#4b5563",
    )

    # --- A: full population ---
    fig.text(0.07, 0.922, "A   All recorded units", fontsize=10, fontweight="bold")
    axes_a = [fig.add_subplot(gs[i, 0:4]) for i in range(1, 5)]
    for ax, key in zip(axes_a, REGION_KEYS):
        display = KEY_TO_REGION[key]
        raster = packed[key]["raster"]
        _shade_epochs(ax, data, 0, raster.shape[0], labeled=(ax is axes_a[0]))
        _draw_raster(ax, raster, t0, t1, REGION_COLORS[display], title=display)
        ax.axvline(d0, color="#1d4ed8", ls="--", lw=0.9, alpha=0.85)
        ax.axvline(d1, color="#1d4ed8", ls="--", lw=0.9, alpha=0.85)
        ax.axvline(l0, color="#b91c1c", ls=":", lw=0.9, alpha=0.9)
        if ax is not axes_a[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time from trial start (s)", fontsize=8)
    axes_a[0].plot([], [], color="#1d4ed8", ls="--", label="delay")
    axes_a[0].plot([], [], color="#b91c1c", ls=":", label="first lick / go-aligned")
    axes_a[0].legend(loc="upper right", frameon=False, fontsize=6.5, ncol=2)

    # --- C: selection scores (right of A) ---
    ax_heat = fig.add_subplot(gs[1:5, 5])
    fig.text(0.78, 0.922, "C   Selection scores", fontsize=10, fontweight="bold")
    crit_names = ["d′", "delay→lick r", "TF sel.", "attn", "ΔMSE"]
    heat_rows = []
    heat_labels = []
    heat_sel = []
    for key in REGION_KEYS:
        block = sorted(by_region[key], key=lambda s: -s.score)
        # show top 10 per region for readability
        for s in block[:10]:
            heat_rows.append([s.dprime, s.delay_lick_coupling, s.tf_selectivity, s.attention, s.prediction_gain])
            heat_labels.append(f"{KEY_TO_REGION[key].split()[0][0]}{KEY_TO_REGION[key].split()[1][:3]} u{s.unit_id}")
            heat_sel.append(s.selected)
    heat = np.asarray(heat_rows, dtype=np.float64)
    if heat.size:
        heat_z = (heat - heat.min(axis=0, keepdims=True)) / (np.ptp(heat, axis=0, keepdims=True) + 1e-8)
        im = ax_heat.imshow(heat_z, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax_heat.set_yticks(range(len(heat_labels)))
        ax_heat.set_yticklabels(heat_labels, fontsize=5.6)
        ax_heat.set_xticks(range(len(crit_names)))
        ax_heat.set_xticklabels(crit_names, fontsize=6.5, rotation=35, ha="right")
        for i, flag in enumerate(heat_sel):
            if flag:
                ax_heat.add_patch(plt.Rectangle((-0.5, i - 0.5), 5, 1, fill=False, edgecolor="#ca8a04", lw=0.7))
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("within-column scaled", fontsize=6.5)

    # spacer label B
    fig.text(0.07, 0.545, "B   Units retained by SPEC selection (same trial; unselected faded)", fontsize=10, fontweight="bold")
    axes_b = [fig.add_subplot(gs[i, 0:4]) for i in range(6, 10)]
    for ax, key in zip(axes_b, REGION_KEYS):
        display = KEY_TO_REGION[key]
        raster = packed[key]["raster"]
        n = raster.shape[0]
        mask = np.zeros(n, dtype=bool)
        for s in by_region[key]:
            if s.selected and 0 <= s.local_index < n:
                mask[s.local_index] = True
        _shade_epochs(ax, data, 0, n, labeled=False)
        _draw_raster(ax, raster, t0, t1, REGION_COLORS[display], selected_mask=mask, title=display)
        ax.axvline(d0, color="#1d4ed8", ls="--", lw=0.9, alpha=0.85)
        ax.axvline(d1, color="#1d4ed8", ls="--", lw=0.9, alpha=0.85)
        n_sel = int(mask.sum())
        ax.text(t1, -1.5, f"{n_sel} selected", ha="right", va="bottom", fontsize=6.5, color="#ca8a04", fontweight="bold")
        if ax is not axes_b[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time from trial start (s)", fontsize=8)

    # D: TF of top selected vs rejected
    ax_tf = fig.add_subplot(gs[6:8, 5])
    fig.text(0.78, 0.545, "D   Delay CWT, selected vs rejected", fontsize=10, fontweight="bold")
    top_sel = next((s for s in scores if s.selected and s.region_key == "left_ALM"), None)
    top_rej = next((s for s in reversed(scores) if (not s.selected) and s.region_key == "left_ALM"), None)
    if top_sel is not None and packed["left_ALM"]["raster"].size:
        full = packed["left_ALM"]["raster"]
        # delay slice
        edges = np.linspace(t0, t1, full.shape[1] + 1)
        d_idx = np.where((edges[:-1] >= d0) & (edges[:-1] < d1))[0]
        if len(d_idx) >= 8:
            for arr, name, cmap in (
                (full[top_sel.local_index : top_sel.local_index + 1, d_idx], "selected", "magma"),
                (
                    full[min(top_rej.local_index, full.shape[0] - 1) : min(top_rej.local_index, full.shape[0] - 1) + 1, d_idx]
                    if top_rej
                    else full[-1:, d_idx],
                    "rejected",
                    "Greys",
                ),
            ):
                pass
            sel_tf = wavelet_power(full[top_sel.local_index : top_sel.local_index + 1, d_idx], bin_size, 16)[0]
            rej_i = top_rej.local_index if top_rej is not None else full.shape[0] - 1
            rej_tf = wavelet_power(full[rej_i : rej_i + 1, d_idx], bin_size, 16)[0]
            pair = np.concatenate([sel_tf, rej_tf], axis=0)
            ax_tf.imshow(pair, aspect="auto", origin="lower", cmap="magma", extent=[d0, d1, 0, 2])
            ax_tf.axhline(1.0, color="white", lw=0.8)
            ax_tf.set_yticks([0.5, 1.5])
            ax_tf.set_yticklabels(["selected\nleft ALM", "rejected\nleft ALM"], fontsize=6)
            ax_tf.set_xlabel("Delay time (s)", fontsize=7)
            ax_tf.set_title("β/γ structure is richer in selected units", fontsize=7, pad=4)

    # E: region composition
    ax_bar = fig.add_subplot(gs[8:10, 5])
    fig.text(0.78, 0.395, "E   Selected count by region", fontsize=10, fontweight="bold")
    counts = []
    totals = []
    cols = []
    labs = []
    for key in REGION_KEYS:
        display = KEY_TO_REGION[key]
        labs.append(display.replace(" ", "\n"))
        cols.append(REGION_COLORS[display])
        counts.append(sum(1 for s in by_region[key] if s.selected))
        totals.append(len(by_region[key]))
    ax_bar.barh(range(4), totals, color="#e5e7eb", height=0.55, label="all")
    ax_bar.barh(range(4), counts, color=cols, height=0.55, label="selected")
    ax_bar.set_yticks(range(4))
    ax_bar.set_yticklabels(labs, fontsize=6.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Units", fontsize=7)
    for i, (c, t) in enumerate(zip(counts, totals)):
        ax_bar.text(c + 0.3, i, f"{c}/{t}", va="center", fontsize=6.5, color="#374151")

    # F: bullets
    ax_txt = fig.add_subplot(gs[10, :])
    ax_txt.set_axis_off()
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.text(0.0, 0.96, "F   Why these neurons were selected", fontsize=10, fontweight="bold", transform=ax_txt.transAxes)
    ax_txt.text(
        0.0,
        0.88,
        "Composite score  =  0.28·z(attention) + 0.22·z(prediction gain) + 0.22·z(d′) + 0.16·z(delay→lick r) + 0.12·z(TF selectivity).  "
        "Top 18% per region kept. Silent units (delay rate < 0.4 Hz) are penalized.",
        fontsize=7,
        color="#4b5563",
        transform=ax_txt.transAxes,
        wrap=True,
    )
    top = [s for s in scores if s.selected][:6]
    col_x = [0.0, 0.34, 0.68]
    for i, s in enumerate(top):
        col = i % 3
        row = i // 3
        x = col_x[col]
        y = 0.74 - row * 0.36
        box = FancyBboxPatch(
            (x, y - 0.30),
            0.31,
            0.32,
            transform=ax_txt.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#fffbeb" if "ALM" in s.region else "#f0fdf4",
            edgecolor=REGION_COLORS[s.region],
            linewidth=1.0,
            clip_on=False,
        )
        ax_txt.add_patch(box)
        header = f"{s.region}  ·  unit {s.unit_id}  ·  prefers {s.preferred_class}  ·  score {s.score:.2f}"
        ax_txt.text(x + 0.008, y + 0.005, header, fontsize=6.4, fontweight="bold", color="#111827", transform=ax_txt.transAxes)
        body = "\n".join(f"• {r}" for r in s.reasons[:3])
        ax_txt.text(x + 0.008, y - 0.28, body, fontsize=5.9, color="#1f2937", va="bottom", transform=ax_txt.transAxes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def render_schematic(out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.6), facecolor="white")
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title("SPEC-TCNN  ·  delay context → lick-period forecast + 3-class action", fontsize=12, fontweight="bold", pad=8)

    boxes = [
        (0.25, 2.55, 2.2, 1.6, "#dbeafe", "Delay rasters\nALM-L · ALM-R\nSTR-L · STR-R"),
        (0.25, 0.45, 2.2, 1.6, "#f3e8ff", "Wavelet + STFT\n4–80 Hz on delay\nβ / low-γ gates"),
        (2.85, 1.15, 2.35, 2.3, "#ffedd5", "4× TCNN + DCC\ndilations 1,2,4,8\ncausal (no leak)"),
        (5.55, 2.45, 2.15, 1.55, "#fef3c7", "Neuron attention\nsparse unit gate"),
        (5.55, 0.45, 2.15, 1.55, "#e0f2fe", "Temporal attention\nlate-delay context"),
        (8.05, 2.45, 2.0, 1.55, "#fee2e2", "Predict lick\nrasters (4 regions)"),
        (8.05, 0.45, 2.0, 1.55, "#dcfce7", "Classify\nIgnore / Left / Right"),
        (10.35, 1.15, 1.8, 2.3, "#fce7f3", "Selected\npredictive\nneurons"),
    ]
    for x, y, w, h, c, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=c, edgecolor="#111827", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="#111827")

    arrows = [
        ((2.45, 3.35), (2.85, 2.6)),
        ((2.45, 1.25), (2.85, 2.0)),
        ((5.2, 2.3), (5.55, 3.2)),
        ((5.2, 2.1), (5.55, 1.2)),
        ((7.7, 3.2), (8.05, 3.2)),
        ((7.7, 1.2), (8.05, 1.2)),
        ((10.05, 3.2), (10.35, 2.6)),
        ((10.05, 1.2), (10.35, 2.0)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#111827", lw=1.1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path
