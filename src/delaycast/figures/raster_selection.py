"""Figure 1 - one real trial, every recorded unit of the four regions, the selected units and the
evidence behind every selection.

Layout (portrait 7.5 x 10.5 in, 300 dpi, PNG + PDF)::

    title / source note / qc note
    header strip: epoch bracket + Left / Right lick rows           (shares x with column A)
    A  all recorded units       B  selected units + evidence      C  exemplar (rank-1 unit)
       (status-sorted rows,        (rank-ordered raster and          class-conditional rate,
        equal pixel height)         heat-strip of the criteria)      coupling scatter inset
    D  criteria legend + reason table                              selection funnel per region

Design decisions and the scientific reason for each
---------------------------------------------------
* **Column A gives every unit the same pixel height** (``height_ratios = n_units per region``).  A reader
  must be able to judge *how many* units a region contributes and *how sparse* the selection is; equal row
  height makes the four regions directly comparable, unlike equal-height panels that compress a 674-unit
  region 2x more than a 380-unit one.
* **Rows are sorted by status** (selected by rank, eligible, floor-pass, below floor) so the selection
  reads as a contiguous block at the top of every region and the status strip on the left is a funnel in
  disguise.  ``figures.raster_row_order = recording`` restores the recording order for readers who want to
  see the depth structure of the probe.
* **Background spikes are rasterized, selected spikes are vector.**  Two thousand units x ~50 spikes is
  ~10^5 segments; rasterizing the grey background keeps the PDF small and fast to open while the coloured
  selected rows stay crisp and editable.
* **Column B shows the evidence, not just the verdict.**  Every selected unit gets one row of a heat-strip
  with the effect sizes (AUROC), the four scored criteria as -log10 q, the information onset, the stability
  and (when a trained model exists) the learned gate and the occlusion importance.  Non-significant cells
  are white with a grey dot, untested cells are hatched: absence of evidence is drawn differently from
  evidence of absence.
* **Column C shows what the criteria mean on real rates.**  The rank-1 unit of each region is drawn as
  class-conditional mean +- SEM rate over the delay *and* the response epoch, with the late-delay window
  used by the coupling criterion shaded and the per-trial coupling scatter inset, so that a reader can
  check the numbers of the strip against a picture.
* **Colours** follow ``style.py``: region colours only for selected spikes / region labels / exemplar
  titles, class colours only for trial-conditioned quantities, neutral colormaps for every heat-strip.
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from .. import CLASSES, REGION_COLORS, REGION_LABELS, REGIONS
from ..data.cache import SessionCache
from ..data.rasters import read_epochs
from ..features.spectral import smooth_rates
from .style import (CLASS_COLORS, EPOCH_COLORS, REGION_SHORT, STABLE_COLOR, STATUS_CODES, STATUS_COLORS,
                    apply_style, small_colorbar, status_cmap)

try:  # the loader may be renamed to a public name by the data-layer agent
    from ..data.rasters import spikes_by_region as _spikes_by_region  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the sibling module version
    from ..data.rasters import _spikes_by_region

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------- constants
FIG_SIZE = (7.5, 10.5)
B_LABEL_GAP = 0.042         # figure fraction reserved left of the column-B rasters for the '#rank uID' labels
C_TICK_GAP = 0.012          # figure fraction taken from the evidence strip for the y tick labels of column C
FUNNEL_LABEL_GAP = 0.062    # figure fraction reserved left of the funnel bars for the region names
BG_SPIKE = "#b8b8b8"
NS_DOT = "#9a9a9a"          # "not significant" marker inside a white heat-strip cell
HATCH_EDGE = "#b0b0b0"      # "not tested" hatching
Q_MAX = 6.0                 # -log10 q saturates at 1e-6: beyond that the exact value carries no information
ONSET_EDGES_MS = np.arange(0, 1201, 100)  # discrete onset bins (window starts are multiples of 50 ms)
STRIP_HEADERS = {"auroc_left_right": "AUROC L/R", "auroc_ignore": "AUROC I", "q_selectivity": "S",
                 "q_coupling": "C", "q_spectral": "W", "q_ramp": "R", "onset_ms": "onset",
                 "stability": "stab", "gate_rel": "gate", "delta_log_loss": "occl"}


# ----------------------------------------------------------------------------------------------- data access
def _lick_times(data, key: str) -> np.ndarray:
    if key not in data.files:
        return np.empty(0)
    vals = np.asarray(data[key], dtype=object).ravel()
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return np.asarray(out, dtype=float)


def _load_trial(npz_path: Path, cache: SessionCache):
    """Spike times of one trial per region, rows aligned to the cache's unit order.

    Alignment is by ``unit_id`` whenever the NPZ carries ids that match the cache (Dataset A); Dataset B has
    positional ids in both, so the same code path applies.  When the two disagree (a unit dropped mid-session)
    the figure falls back to positional order with a warning rather than failing - a misaligned row only
    affects this illustration, never the statistics, which come from the cache.
    """
    data = np.load(npz_path, allow_pickle=True)
    ep = read_epochs(data)
    by_region = _spikes_by_region(data)
    spikes: dict[str, list[np.ndarray]] = {}
    for r in REGIONS:
        raw, ids = by_region[r]
        st = [np.asarray(s, dtype=float).ravel() for s in raw]
        ids = [x.item() if hasattr(x, "item") else x for x in np.asarray(ids).ravel()]
        cache_ids = [x.item() if hasattr(x, "item") else x for x in np.asarray(cache.unit_ids[r]).ravel()]
        n_cache = int(cache.n_units[r])
        aligned = False
        if len(st) == n_cache and len(cache_ids) == n_cache and len(set(ids)) == n_cache and set(ids) == set(cache_ids):
            pos = {uid: i for i, uid in enumerate(ids)}
            st = [st[pos[uid]] for uid in cache_ids]
            aligned = True
        if not aligned:
            if len(st) != n_cache:
                warnings.warn(f"{npz_path.name}: {r} has {len(st)} units in the NPZ but {n_cache} in the cache; "
                              "rows are shown in positional order", stacklevel=2)
            elif n_cache:
                warnings.warn(f"{npz_path.name}: {r} unit ids differ from the cache; rows shown in positional order",
                              stacklevel=2)
            st = (st + [np.empty(0)] * n_cache)[:n_cache]
        spikes[r] = st
    licks = {"Left": _lick_times(data, "left_lick_times"), "Right": _lick_times(data, "right_lick_times")}
    return spikes, ep, licks


def _col(df: pd.DataFrame, name: str, default=np.nan, dtype=float) -> np.ndarray:
    """Column as an array with a default for tables written by older versions of the selection code."""
    if name in df.columns:
        v = df[name]
        if dtype is bool:
            return v.fillna(False).astype(bool).to_numpy()
        return pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    if dtype is bool:
        return np.zeros(len(df), dtype=bool)
    return np.full(len(df), default, dtype=float)


def _unit_status(tab_r: pd.DataFrame, n_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Status code (``STATUS_CODES``) and rank per unit index of one region."""
    code = np.full(n_units, STATUS_CODES["below_floor"], dtype=int)
    rank = np.full(n_units, np.nan)
    if not len(tab_r):
        return code, rank
    idx = tab_r.unit_index.to_numpy(dtype=int)
    ok = (idx >= 0) & (idx < n_units)
    sel, elig, floor = (_col(tab_r, "selected", dtype=bool), _col(tab_r, "eligible", dtype=bool),
                        _col(tab_r, "pass_floor", dtype=bool))
    code[idx[ok]] = np.select([sel, elig, floor], [STATUS_CODES["selected"], STATUS_CODES["eligible"],
                                                    STATUS_CODES["floor"]], STATUS_CODES["below_floor"])[ok]
    rank[idx[ok]] = _col(tab_r, "rank")[ok]
    return code, rank


def _row_order(code: np.ndarray, rank: np.ndarray, mode: str) -> np.ndarray:
    """Unit indices from top to bottom of the raster (status mode: selected by rank, then the greys)."""
    n = len(code)
    if mode == "recording":
        return np.arange(n)
    rank_key = np.where(np.isfinite(rank), rank, np.inf)
    return np.lexsort((np.arange(n), rank_key, code))


def _reason_lines(table: pd.DataFrame, cfg, per_region: int = 2) -> list[str]:
    """``reason_short`` of the top-ranked selected units per region (recomputed from the long form when
    the table predates the column)."""
    lines = []
    for r in REGIONS:
        top = table[(table.region == r) & _col(table, "selected", dtype=bool)].sort_values("rank").head(per_region)
        for _, row in top.iterrows():
            text = row.get("reason_short", np.nan)
            if not isinstance(text, str) or not text:
                try:
                    from ..features.selection import _reason_short
                    text = _reason_short(row, cfg)
                except Exception:  # noqa: BLE001 - any missing field: fall back to the long text
                    text = f"{REGION_SHORT[r]} u{row.unit_id} #{int(row['rank'])}: {str(row.get('reasons', ''))[:100]}"
            lines.append(text)
    return lines


# ----------------------------------------------------------------------------------------------- drawing helpers
def _segments(spikes: list[np.ndarray], rows: np.ndarray, x_lim: tuple[float, float], pad: float = 0.08) -> np.ndarray:
    """Vertical tick segments ((n, 2, 2) array) for ``spikes[k]`` drawn on raster row ``rows[k]`` (row j spans y in
    [j, j+1]; the y axis is inverted so row 0 is at the top)."""
    xs, ys = [], []
    for st, j in zip(spikes, rows):
        st = st[(st >= x_lim[0]) & (st <= x_lim[1])]
        xs.append(st)
        ys.append(np.full(st.size, float(j)))
    if not xs:
        return np.zeros((0, 2, 2))
    x, y = np.concatenate(xs), np.concatenate(ys)
    seg = np.empty((x.size, 2, 2))
    seg[:, 0, 0] = seg[:, 1, 0] = x
    seg[:, 0, 1] = y + pad
    seg[:, 1, 1] = y + 1 - pad
    return seg


def _epoch_background(ax, ep: dict, t0: float, first_lick: float, lick_class: str | None) -> None:
    """Light-grey epoch spans and dotted boundaries (delay start, go) plus the first lick of this trial."""
    sample_a, sample_b = ep["sample_start_times"], ep["sample_stop_times"]
    go = ep["go_start_times"]
    if np.isfinite(sample_a) and np.isfinite(sample_b):
        ax.axvspan(sample_a - t0, sample_b - t0, color=EPOCH_COLORS["sample"], lw=0, zorder=0)
        ax.axvline(sample_a - t0, color="#999999", ls=":", lw=0.4, zorder=1)
    ax.axvspan(0.0, go - t0, color=EPOCH_COLORS["delay"], lw=0, zorder=0)
    ax.axvline(0.0, color="k", ls=":", lw=0.5, zorder=3)
    ax.axvline(go - t0, color="k", ls=":", lw=0.5, zorder=3)
    if np.isfinite(first_lick):
        ax.axvline(first_lick - t0, color=CLASS_COLORS.get(lick_class, "k"), ls=":", lw=0.6, zorder=3)


def _tidy_raster(ax) -> None:
    ax.tick_params(left=False, labelleft=False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)


def _draw_header(ax, ep: dict, t0: float, licks: dict, x_lim: tuple[float, float]) -> None:
    """Epoch bracket with durations and two lick rows (Left above Right) that share x with column A."""
    go = ep["go_start_times"]
    spans = [("sample", ep["sample_start_times"], ep["sample_stop_times"]),
             ("delay", ep["delay_start_times"], go),
             ("response", go, ep["go_stop_times"] if np.isfinite(ep["go_stop_times"]) else go + 1.5)]
    for name, a, b in spans:
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ax.add_patch(Rectangle((a - t0, 2.05), b - a, 0.95, facecolor=EPOCH_COLORS[name], edgecolor="#777777",
                               lw=0.4, ls=(0, (1, 1.5)), zorder=1))
        ax.text((a + b) / 2 - t0, 2.52, f"{name} {b - a:.2f} s".replace(".00 s", " s").replace("0 s", " s"),
                ha="center", va="center", fontsize=5.5, zorder=2)
    for y, cls in ((1.0, "Left"), (0.0, "Right")):
        lk = licks[cls] - t0
        lk = lk[(lk >= x_lim[0]) & (lk <= x_lim[1])]
        if lk.size:
            ax.vlines(lk, y + 0.08, y + 0.92, color=CLASS_COLORS[cls], lw=0.7)
        ax.text(x_lim[0] - 0.04, y + 0.5, f"{cls[0]} lick", ha="right", va="center", fontsize=4.5,
                color=CLASS_COLORS[cls], clip_on=False)
    ax.set_ylim(0, 3.05)
    ax.set_xlim(*x_lim)
    ax.axis("off")


# ----------------------------------------------------------------------------------------------- column A
def _margin_label(fig, ax, n_sel: int, n_el: int, n_fl: int, n: int) -> None:
    """Rotated count label in the right margin of one region; the wording is shortened (and, as a last resort,
    the font reduced) when the region is too short for the full text, so that labels of neighbouring regions
    never overlap."""
    avail_pt = ax.get_position().height * fig.get_figheight() * 72 * 0.98
    candidates = [f"{n_sel} selected | {n_el} eligible | {n_fl} >= floor | {n} recorded",
                  f"{n_sel} sel | {n_el} elig | {n_fl} floor | {n} rec"]
    for text in candidates:
        fs = 6.0
        if len(text) * 0.56 * fs <= avail_pt:
            break
    else:
        fs = max(4.5, avail_pt / (0.56 * len(text)))
    ax.text(1.006, 0.5, text, transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=fs,
            color="#333333", clip_on=False)


def _column_a(fig, gs_cell, spikes, table, cache, ep, t0, x_lim, first_lick, lick_class, row_mode, k_per_region):
    """All recorded units, status-sorted, equal pixel height per unit, status strip on the left."""
    counts = [max(int(cache.n_units[r]), 1) for r in REGIONS]
    gs = gs_cell.subgridspec(4, 2, height_ratios=counts, width_ratios=[0.06, 1], wspace=0.02, hspace=0.035)
    axes, ax_ref = [], None
    for i, r in enumerate(REGIONS):
        n = int(cache.n_units[r])
        tab_r = table[table.region == r]
        code, rank = _unit_status(tab_r, n)
        order = _row_order(code, rank, row_mode)
        ax_s = fig.add_subplot(gs[i, 0])
        ax = fig.add_subplot(gs[i, 1], sharex=ax_ref)
        ax_ref = ax_ref or ax
        _epoch_background(ax, ep, t0, first_lick, lick_class)
        if n:
            rows = np.arange(n)
            rel = [spikes[r][u] - t0 for u in order]
            is_sel = code[order] == STATUS_CODES["selected"]
            bg = _segments([s for s, m in zip(rel, is_sel) if not m], rows[~is_sel], x_lim)
            fg = _segments([s for s, m in zip(rel, is_sel) if m], rows[is_sel], x_lim)
            if len(bg):
                ax.add_collection(LineCollection(bg, colors=BG_SPIKE, linewidths=0.3, rasterized=True, zorder=2))
            if len(fg):
                ax.add_collection(LineCollection(fg, colors=REGION_COLORS[r], linewidths=0.7, zorder=4))
            ax_s.imshow(code[order][:, None], cmap=status_cmap(r), vmin=-0.5, vmax=3.5, aspect="auto",
                        interpolation="nearest", extent=(0, 1, n, 0))
            n_sel = int(is_sel.sum())
            if row_mode != "recording" and 0 < n_sel < n:
                ax.axhline(n_sel, color="#333333", lw=0.5, zorder=5)
                ax_s.axhline(n_sel, color="#333333", lw=0.5, zorder=5)
        else:
            ax.text(0.5, 0.5, "no units recorded", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#888888")
        n_sel = int(_col(tab_r, "selected", dtype=bool).sum())
        n_el = int(_col(tab_r, "eligible", dtype=bool).sum())
        n_fl = int(_col(tab_r, "pass_floor", dtype=bool).sum())
        _margin_label(fig, ax, n_sel, n_el, n_fl, n)
        ax.set_ylim(n, 0)
        ax.set_xlim(*x_lim)
        _tidy_raster(ax)
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        for s in ax_s.spines.values():
            s.set_visible(False)
        ax_s.set_ylabel(REGION_LABELS[r], color=REGION_COLORS[r], fontsize=7, fontweight="bold", labelpad=2)
        if i < 3:
            ax.tick_params(labelbottom=False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
        axes.append(ax)
    axes[-1].set_xlabel("time from delay onset (s)", labelpad=1.5)
    return axes


# ----------------------------------------------------------------------------------------------- column B
def _strip_column(fig, gs_cell, ax_ref, values, cmap, norm, dots, hatch, k, header=None, sustained=None):
    """One column of the evidence strip: an imshow of ``values`` (K,) sharing y with the raster.

    Cells that are NaN or non-significant are masked (white) with a grey dot; ``hatch`` marks cells whose
    test was not run.  ``sustained`` marks onset cells whose information persists until the go cue."""
    ax = fig.add_subplot(gs_cell, sharey=ax_ref)
    vals = np.ma.masked_invalid(np.asarray(values, dtype=float))
    vals = np.ma.masked_where(dots | hatch, vals)
    cm = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).with_extremes(bad="white")
    ax.imshow(vals[:, None], cmap=cm, norm=norm, aspect="auto", interpolation="nearest", extent=(0, 1, k, 0))
    for j in np.flatnonzero(dots & ~hatch):
        ax.plot(0.5, j + 0.5, "o", ms=1.1, color=NS_DOT, mew=0, zorder=3)
    for j in np.flatnonzero(hatch):
        ax.add_patch(Rectangle((0, j), 1, 1, facecolor="white", edgecolor=HATCH_EDGE, hatch="////", lw=0, zorder=2))
    if sustained is not None:
        for j in np.flatnonzero(sustained & ~dots):
            ax.plot(0.5, j + 0.5, "o", ms=1.6, mfc="white", mec="#222222", mew=0.35, zorder=3)
    ax.set_xticks([])
    ax.tick_params(left=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.35)
        s.set_color("#c8c8c8")
    if header:
        ax.text(0.5, 1.02, header, rotation=90, ha="center", va="bottom", fontsize=5, transform=ax.transAxes, clip_on=False)
    return ax


def _onset_cmap():
    """Discrete viridis with *dark = late*: a late onset is the interesting case for the late-delay claim, and
    dark cells pop out of a mostly-bright strip."""
    base = plt.get_cmap("viridis_r")
    return ListedColormap(base(np.linspace(0, 1, len(ONSET_EDGES_MS) - 1)), name="onset"), BoundaryNorm(ONSET_EDGES_MS, len(ONSET_EDGES_MS) - 1)


def _evidence_columns(imp_max: float | None):
    """Ordered column specification of the evidence strip; ``None`` entries are spacer columns."""
    q_norm = Normalize(0, Q_MAX)
    auroc = Normalize(0, 1)
    onset_cmap, onset_norm = _onset_cmap()
    cols = [("auroc_left_right", "RdBu_r", auroc), ("auroc_ignore", "RdBu_r", auroc), None,
            ("q_selectivity", "Blues", q_norm), ("q_coupling", "Blues", q_norm), ("q_spectral", "Blues", q_norm),
            ("q_ramp", "Blues", q_norm), None, ("onset_ms", onset_cmap, onset_norm), None,
            ("stability", "Greys", Normalize(0, 1))]
    if imp_max is not None:
        cols += [None, ("gate_rel", "Greys", Normalize(0, 1)), ("delta_log_loss", "RdBu_r", Normalize(-imp_max, imp_max))]
    return cols


def _column_b(fig, gs_cell, spikes, table, cache, ep, t0, x_lim, first_lick, lick_class, k_per_region, importance, imp_max):
    """Rank-ordered raster of the selected units and the evidence strip (one imshow per column group)."""
    gs = gs_cell.subgridspec(4, 2, width_ratios=[1, 0.9], wspace=0.03, hspace=0.30)
    cols = _evidence_columns(imp_max)
    widths = [0.35 if c is None else 1.0 for c in cols]
    axes, ax_ref = [], None
    for i, r in enumerate(REGIONS):
        sel = table[(table.region == r) & _col(table, "selected", dtype=bool)].sort_values("rank").reset_index(drop=True)
        k = len(sel)
        ax = fig.add_subplot(gs[i, 0], sharex=ax_ref)
        ax_ref = ax_ref or ax
        # leave room on the left for the '#rank uID' labels (the outer gap is shared with column A's margin label)
        pos = ax.get_position()
        ax.set_position([pos.x0 + B_LABEL_GAP, pos.y0, pos.width - B_LABEL_GAP, pos.height])
        _epoch_background(ax, ep, t0, first_lick, lick_class)
        if k:
            rel = [spikes[r][int(u)] - t0 for u in sel.unit_index]
            ax.add_collection(LineCollection(_segments(rel, np.arange(k), x_lim), colors=REGION_COLORS[r], linewidths=0.8, zorder=4))
            ranks = sel["rank"].to_numpy(dtype=float)
            lab_rows = [j for j in range(k) if (int(ranks[j]) - 1) % 4 == 0]
            ax.set_yticks([j + 0.5 for j in lab_rows])
            ax.set_yticklabels([f"#{int(ranks[j])} u{sel.unit_id.iloc[j]}" for j in lab_rows], fontsize=5.5)
            ax.tick_params(left=False, labelleft=True, pad=1.2, labelsize=5.5)
            if k < k_per_region:
                ax.text(1.0, 1.005, f"{k} of K={k_per_region} selected", transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                        color="#555555")
        else:
            ax.text(0.5, 0.5, "no unit selected in this region", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#888888")
        ax.set_ylim(max(k, 1), 0)
        ax.set_xlim(*x_lim)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        if i < 3:
            ax.tick_params(labelbottom=False, bottom=False)
            ax.spines["bottom"].set_visible(False)
        if k == 0:
            ax.tick_params(left=False, labelleft=False)
            axes.append(ax)
            continue
        # ---- evidence strip ---------------------------------------------------------------------------
        gs_e = gs[i, 1].subgridspec(1, len(cols), width_ratios=widths, wspace=0.12)
        imp_r = None
        if importance is not None and len(importance):
            imp_r = sel[["unit_index"]].merge(importance[importance.region == r], on="unit_index", how="left")
        for c_i, spec in enumerate(cols):
            if spec is None:
                continue
            key, cmap, norm = spec
            header = STRIP_HEADERS[key] if i == 0 else None
            n_ = len(sel)
            dots = np.zeros(n_, bool)
            hatch = np.zeros(n_, bool)
            sustained = None
            if key.startswith("q_"):
                q = _col(sel, key)
                crit = _col(sel, "c_" + key[2:], dtype=bool)
                vals = np.clip(-np.log10(np.clip(q, 1e-300, None)), 0, Q_MAX)
                dots = ~crit | ~np.isfinite(q)
                if key == "q_spectral":
                    hatch = ~np.isfinite(_col(sel, "p_spectral"))
            elif key == "auroc_ignore":
                vals = _col(sel, key)
                dots = ~np.isfinite(_col(sel, "p_ignore")) | ~np.isfinite(vals)
            elif key == "onset_ms":
                vals = _col(sel, key)
                vals = np.where(np.isfinite(vals), np.clip(vals, 0, ONSET_EDGES_MS[-1] - 1), np.nan)
                dots = ~np.isfinite(vals)
                sustained = _col(sel, "sustained_to_go", dtype=bool)
            elif key in ("gate_rel", "delta_log_loss"):
                vals = _col(imp_r, key) if imp_r is not None else np.full(n_, np.nan)
                dots = ~np.isfinite(vals)
            else:
                vals = _col(sel, key)
                dots = ~np.isfinite(vals)
            ax_e = _strip_column(fig, gs_e[0, c_i], ax, vals, cmap, norm, dots, hatch, k, header=header, sustained=sustained)
            # leave a little room on the right for the tick labels of column C
            strip = gs[i, 1].get_position(fig)
            ep_ = ax_e.get_position()
            scale = (strip.width - C_TICK_GAP) / strip.width
            ax_e.set_position([strip.x0 + (ep_.x0 - strip.x0) * scale, ep_.y0, ep_.width * scale, ep_.height])
        ax.set_ylim(k, 0)
        axes.append(ax)
    return axes


# ----------------------------------------------------------------------------------------------- column C
def _column_c(fig, gs_cell, table, cache, cfg, k_per_region, fit_idx=None):
    """Exemplar per region: class-conditional mean +- SEM rate of the rank-1 unit over delay -> response, the
    late-delay coupling window, the information onset and the per-trial coupling scatter.

    ``fit_idx`` restricts every trial-conditioned quantity to the trials the selection statistics were computed
    on (the training split of a run), so the panel never shows test-trial labels or spikes."""
    gs = gs_cell.subgridspec(4, 1, hspace=0.30)
    fit = np.arange(cache.n_trials) if fit_idx is None else np.asarray(fit_idx, dtype=int)
    y = cache.labels[fit]
    bin_s, tbin_s = cache.bin_ms / 1000.0, cache.target_bin_ms / 1000.0
    delay_s = float(cfg.data.get_path("context.delay_ms", 1200)) / 1000.0
    late_s = float(cfg.selection.late_delay_ms) / 1000.0
    sigma = float(cfg.data.smoothing_sigma_ms)
    t_ctx = cache.context[REGIONS[0]].shape[2]
    t_tgt = cache.target[REGIONS[0]].shape[2]
    tvec = delay_s - (t_ctx - np.arange(t_ctx) - 0.5) * bin_s          # context ends at the go cue
    tvec_t = delay_s + (np.arange(t_tgt) + 0.5) * tbin_s
    late_bins = max(int(round(late_s / bin_s)), 1)
    axes = []
    for i, r in enumerate(REGIONS):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        top = table[(table.region == r) & _col(table, "selected", dtype=bool)].sort_values("rank").head(1)
        ax.axvspan(tvec[0] - bin_s / 2, delay_s, color=EPOCH_COLORS["delay"], lw=0, zorder=0)
        ax.axvspan(delay_s - late_s, delay_s, color="#d9d9d9", lw=0, zorder=0)
        ax.axvline(delay_s, color="k", ls=":", lw=0.5, zorder=3)
        ax.set_xlim(tvec[0] - bin_s / 2, tvec_t[-1] + tbin_s / 2)
        if not len(top):
            ax.text(0.5, 0.5, "no unit selected", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#888888")
            ax.set_yticks([])
            continue
        row = top.iloc[0]
        u = int(row.unit_index)
        ctx = cache.context[r][fit, u].astype(float)    # (n_fit_trials, T_ctx)
        tgt = cache.target[r][fit, u].astype(float)     # (n_fit_trials, T_tgt)
        rate_ctx = smooth_rates(ctx, cache.bin_ms, sigma)
        rate_tgt = tgt / tbin_s
        for c_i, c in enumerate(CLASSES):
            m = y == c_i
            if m.sum() < 2:
                continue
            for t, rr in ((tvec, rate_ctx[m]), (tvec_t, rate_tgt[m])):
                mu, se = rr.mean(0), rr.std(0, ddof=1) / np.sqrt(m.sum())
                ax.fill_between(t, mu - se, mu + se, color=CLASS_COLORS[c], alpha=0.18, lw=0, zorder=1)
                ax.plot(t, mu, color=CLASS_COLORS[c], lw=0.9, zorder=2)
        ax.text(delay_s - late_s / 2, 0.985, "C window", transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="top", fontsize=5, color="#555555")
        onset = float(row.get("onset_ms", np.nan))
        if np.isfinite(onset):
            ax.plot(onset / 1000.0, 0, marker="^", ms=3.2, color="#222222", mec="none", clip_on=False, zorder=6,
                    transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
        stab = float(row.get("stability", np.nan))
        ax.set_title(f"{REGION_LABELS[r]} u{row.unit_id} - #{int(row['rank'])}/{k_per_region} - stab {stab:.0%}",
                     color=REGION_COLORS[r], fontsize=6.5, loc="left", pad=2.5)
        ax.set_ylabel("rate (Hz)", labelpad=1.0, fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_ylim(bottom=0)
        # inset: per-trial late-delay rate vs own response rate (the C criterion)
        ins = ax.inset_axes([0.6, 0.55, 0.38, 0.42])
        late_rate = ctx[:, -late_bins:].mean(1) / bin_s
        resp_rate = tgt.mean(1) / tbin_s
        for c_i, c in enumerate(CLASSES):
            m = y == c_i
            if m.any():
                ins.scatter(late_rate[m], resp_rate[m], s=2.5, color=CLASS_COLORS[c], alpha=0.75, lw=0, zorder=2)
        rho = float(row.get("rho_coupling", np.nan))
        ins.text(0.03, 0.97, f"rho_within={rho:+.2f}" if np.isfinite(rho) else "rho_within=n/a", transform=ins.transAxes,
                 ha="left", va="top", fontsize=4.8)
        ins.set_xlabel("late-delay Hz", fontsize=4.8, labelpad=0.5)
        ins.set_ylabel("response Hz", fontsize=4.8, labelpad=0.5)
        ins.tick_params(labelsize=4.3, length=1.5, width=0.4, pad=1)
        ins.patch.set_alpha(0.85)
        for s in ins.spines.values():
            s.set_linewidth(0.4)
        if i < 3:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time from delay onset (s)", labelpad=1.5)
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    handles = [Line2D([], [], color=CLASS_COLORS[c], lw=1.2, label=f"{c} n={counts[c]}") for c in CLASSES]
    return axes, handles


# ----------------------------------------------------------------------------------------------- footer
def _criteria_legend(cfg, table: pd.DataFrame) -> list[str]:
    sel = cfg.selection
    n_sub = int(table.n_subsamples.iloc[0]) if "n_subsamples" in table and len(table) else int(sel.get_path("n_subsamples", 50))
    return [
        (f"floor: delay rate >= {sel.min_rate_hz} Hz and spikes on >= {sel.min_active_trial_frac:.0%} of trials   |   "
         f"S: Mann-Whitney U, delay rate Left vs Right (AUROC)"),
        (f"C: within-class Spearman, late-delay ({int(sel.late_delay_ms)} ms) rate vs own response rate   |   "
         "W: rate-normalised Morlet band power, Left vs Right"),
        ("R: Wilcoxon late vs early delay within class   |   T: sliding-window AUROC cluster test (onset; descriptive)   |   "
         "I: Ignore vs lick (descriptive)"),
        (f"all BH-FDR q < {sel.fdr_q} across units   |   eligible = floor and >= {int(sel.min_criteria)} of {{S, C, W, R}}   |   "
         f"stable = selected in >= {float(sel.min_stability):.0%} of {n_sub} half-subsamples"),
        ("ranked by stability, then score = sum w (-log10 q)   |   strip: white + dot = n.s., hatched = not tested, "
         "ring = information sustained to go"),
    ]


def _footer_text(fig, gs_cell, cfg, table):
    """Three-line criteria legend and the monospace reason table (top-2 selected units per region)."""
    ax = fig.add_subplot(gs_cell)
    ax.axis("off")
    bbox = ax.get_position()
    width_pt = bbox.width * fig.get_figwidth() * 72
    legend = _criteria_legend(cfg, table)
    fs_leg = min(5.6, width_pt / (0.55 * max(len(s) for s in legend)))
    lines = _reason_lines(table, cfg)
    y = 1.0
    for s in legend:
        ax.text(0.0, y, s, transform=ax.transAxes, fontsize=fs_leg, va="top", ha="left", color="#333333")
        y -= 1.18 * fs_leg / (bbox.height * fig.get_figheight() * 72)
    y -= 0.03
    if lines:
        # DejaVu Sans Mono advances 0.602 em per glyph: shrink below 5.8 pt only when a 110-char reason would overflow the cell
        fs_tab = min(5.8, width_pt / (0.605 * max(len(s) for s in lines)))
        step = 1.15 * fs_tab / (bbox.height * fig.get_figheight() * 72)
        for k, s in enumerate(lines):
            ax.text(0.0, y - k * step, s, transform=ax.transAxes, fontsize=fs_tab, family="monospace", va="top", ha="left",
                    color="#111111")
    return ax


def _footer_funnel(fig, gs_cell, table, cache):
    """Nested funnel bar per region (recorded -> floor -> eligible -> stable -> selected) that doubles as the
    legend of the status colours in column A."""
    ax = fig.add_subplot(gs_cell)
    pos = ax.get_position()  # the region tick labels hang to the left: keep them inside this cell
    ax.set_position([pos.x0 + FUNNEL_LABEL_GAP, pos.y0, pos.width - FUNNEL_LABEL_GAP, pos.height])
    stages = ["recorded", ">= floor", "eligible", "stable", "selected"]
    greys = [STATUS_COLORS["below_floor"], STATUS_COLORS["floor"], STATUS_COLORS["eligible"], STABLE_COLOR]
    x_max = 1
    for i, r in enumerate(REGIONS):
        tab_r = table[table.region == r]
        counts = [int(cache.n_units[r]), int(_col(tab_r, "pass_floor", dtype=bool).sum()), int(_col(tab_r, "eligible", dtype=bool).sum()),
                  int((_col(tab_r, "eligible", dtype=bool) & _col(tab_r, "stable", dtype=bool)).sum()), int(_col(tab_r, "selected", dtype=bool).sum())]
        x_max = max(x_max, counts[0])
        for c, col in zip(counts, greys + [REGION_COLORS[r]]):
            ax.barh(i + 0.62, c, height=0.42, left=0, color=col, lw=0)
        ax.text(0, i + 0.2, "  >  ".join(str(c) for c in counts), fontsize=5.2, va="center", ha="left", color="#333333")
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(len(REGIONS) + 0.05, -0.85)  # head-room for the stage legend
    ax.set_yticks([i + 0.5 for i in range(len(REGIONS))])
    ax.set_yticklabels([REGION_SHORT[r] for r in REGIONS], fontsize=6)
    for lab, r in zip(ax.get_yticklabels(), REGIONS):
        lab.set_color(REGION_COLORS[r])
        lab.set_fontweight("bold")
    ax.tick_params(left=False, bottom=False, labelbottom=False, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = ([Patch(facecolor=c, edgecolor="#999999", lw=0.3, label=s) for s, c in zip(stages, greys)]
               + [Patch(facecolor=REGION_COLORS["ALM_L"], edgecolor="#999999", lw=0.3, label="selected (region colour)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0), ncol=3, fontsize=5, handlelength=0.9,
              handleheight=0.9, columnspacing=0.8, handletextpad=0.4, borderaxespad=0, borderpad=0)
    return ax


def _colorbars(fig, x0, x1, y_top, imp_max):
    """Small horizontal colour bars for the evidence strip, drawn in the gap above the footer under column B
    (the only place where they sit next to the strip without stealing height from the rasters)."""
    onset_cmap, onset_norm = _onset_cmap()
    specs = [("RdBu_r", Normalize(0, 1), "AUROC\n0.5 = chance", [0, 0.5, 1], ["0", "0.5", "1"]),
             ("Blues", Normalize(0, Q_MAX), "-log10 q\ndot = n.s.", [0, 3, 6], ["0", "3", "6"]),
             (onset_cmap, onset_norm, "onset (ms)\ndark = late", [0, 600, 1200], ["0", "600", "1200"]),
             ("Greys", Normalize(0, 1), "stability, gate", [0, 0.5, 1], ["0", "0.5", "1"])]
    if imp_max is not None:
        specs.append(("RdBu_r", Normalize(-imp_max, imp_max), f"occlusion\ndelta log-loss (+-{imp_max:.1g})",
                      [-imp_max, 0, imp_max], ["-", "0", "+"]))
    n = len(specs)
    gap = 0.014
    w = (x1 - x0 - gap * (n - 1)) / n
    for k, (cmap, norm, label, ticks, ticklabels) in enumerate(specs):
        small_colorbar(fig, [x0 + k * (w + gap), y_top - 0.004, w, 0.004], cmap, norm, label, ticks, ticklabels, fontsize=4.5)


# ----------------------------------------------------------------------------------------------- main entry point
def plot_raster_selection(npz_path, cache: SessionCache, table: pd.DataFrame, cfg, out_path: Path, trial_label: str = "",
                          source_note: str = "", importance: pd.DataFrame | None = None, qc_note: str = "",
                          fit_idx: np.ndarray | None = None) -> Path:
    """Render Figure 1 for one trial of ``cache`` and write ``out_path`` (PNG) plus the same name with ``.pdf``.

    Parameters
    ----------
    npz_path : path of the trial NPZ (either Dataset schema).
    cache : the session cache the selection table refers to (unit order, labels, class-conditional rates).
    table : selection DataFrame (one row per unit) as written by ``delaycast select`` / the training run.
    cfg : configuration; uses ``selection.*``, ``data.smoothing_sigma_ms``, ``figures.dpi`` and the optional
        ``figures.raster_row_order`` (``status`` | ``recording``, default ``status``).
    trial_label : text after the session key in the title (e.g. ``trial 331 - Left - first lick +0.21 s after go``).
    source_note : sub-title stating which trials / settings the criteria were computed on.
    importance : ``neuron_importance.csv`` of a trained run (adds the gate / occlusion columns), or ``None``.
    qc_note : printed under the title when non-empty (e.g. ``trial excluded by QC: early lick``).
    """
    t_start = time.perf_counter()
    dpi = max(int(cfg.figures.get_path("dpi", 300)), 300)  # a publication raster is never below 300 dpi
    apply_style(dpi)
    npz_path = Path(npz_path)
    row_mode = str(cfg.figures.get_path("raster_row_order", "status"))
    if row_mode not in ("status", "recording"):
        warnings.warn(f"figures.raster_row_order={row_mode!r} unknown; using 'status'", stacklevel=2)
        row_mode = "status"
    k_per_region = int(cfg.selection.top_k_per_region)
    table = table.copy()
    if "rank" not in table.columns:
        table["rank"] = np.nan

    spikes, ep, licks = _load_trial(npz_path, cache)
    t0 = ep["delay_start_times"]
    go = ep["go_start_times"]
    if not (np.isfinite(t0) and np.isfinite(go)):
        raise ValueError(f"{npz_path}: delay_start_times / go_start_times missing")
    sample_start = ep["sample_start_times"] if np.isfinite(ep["sample_start_times"]) else t0 - 0.65
    x_lim = (sample_start - t0 - 0.1, go - t0 + 1.5)
    after_go = {c: lk[lk >= go] for c, lk in licks.items()}
    first_lick, lick_class = np.nan, None
    for c, lk in after_go.items():
        if lk.size and (not np.isfinite(first_lick) or lk.min() < first_lick):
            first_lick, lick_class = float(lk.min()), c

    imp = None
    imp_max = None
    if importance is not None and len(importance):
        imp = importance.copy()
        if "session" in imp.columns:
            imp = imp[imp.session == cache.session]
        if len(imp) and {"region", "unit_index", "gate_rel", "delta_log_loss"} <= set(imp.columns):
            imp["unit_index"] = imp.unit_index.astype(int)
            imp_max = float(np.nanmax(np.abs(imp.delta_log_loss.to_numpy(dtype=float)))) if imp.delta_log_loss.notna().any() else 0.0
            imp_max = max(imp_max, 1e-3)
        else:
            imp, imp_max = None, None

    fig = plt.figure(figsize=FIG_SIZE)
    outer = GridSpec(2, 3, figure=fig, height_ratios=[8.3, 1.5], width_ratios=[1.15, 1.25, 0.85], left=0.05, right=0.99,
                     top=0.94, bottom=0.04, hspace=0.12, wspace=0.10)

    axes_a = _column_a(fig, outer[0, 0], spikes, table, cache, ep, t0, x_lim, first_lick, lick_class, row_mode, k_per_region)
    axes_b = _column_b(fig, outer[0, 1], spikes, table, cache, ep, t0, x_lim, first_lick, lick_class, k_per_region, imp, imp_max)
    axes_c, class_handles = _column_c(fig, outer[0, 2], table, cache, cfg, k_per_region, fit_idx=fit_idx)
    gs_f = outer[1, :].subgridspec(1, 2, width_ratios=[2.2, 1], wspace=0.06)
    ax_text = _footer_text(fig, gs_f[0, 0], cfg, table)
    ax_funnel = _footer_funnel(fig, gs_f[0, 1], table, cache)

    # header strip above column A (shares x with the rasters)
    pos_a = axes_a[0].get_position()
    strip_h = 0.15 / FIG_SIZE[1]
    ax_h = fig.add_axes([pos_a.x0, pos_a.y1 + 0.003, pos_a.width, strip_h], sharex=axes_a[0])
    _draw_header(ax_h, ep, t0, licks, x_lim)

    # colour bars of the evidence strip in the gap under column B
    pos_b = outer[0, 1].get_position(fig)
    pos_f = gs_f[0, 0].get_position(fig)
    _colorbars(fig, pos_b.x0 + B_LABEL_GAP, pos_b.x1 - C_TICK_GAP, pos_f.y1 + 0.031, imp_max)

    # class legend of column C, above its top axis
    pos_c = axes_c[0].get_position()
    fig.legend(handles=class_handles, loc="lower right", bbox_to_anchor=(0.99, pos_c.y1 + 0.014), ncol=3, fontsize=5.2,
               handlelength=1.0, columnspacing=0.7, handletextpad=0.35, borderaxespad=0)

    # titles and panel letters
    title = f"Session {cache.session}" + (f" - {trial_label}" if trial_label else "")
    fig.text(0.05, 0.9905, title, fontsize=8, fontweight="bold", va="bottom", ha="left")
    if source_note:
        fig.text(0.05, 0.9795, source_note, fontsize=6, va="bottom", ha="left", color="#333333")
    if qc_note:
        fig.text(0.05, pos_a.y1 + strip_h + 0.008, qc_note, fontsize=6, va="bottom", ha="left", color="#b03030")
    letter_y = pos_a.y1 + 0.004
    fig.text(0.012, letter_y, "A", fontsize=9, fontweight="bold", va="bottom")
    fig.text(pos_b.x0 - 0.005, letter_y, "B", fontsize=9, fontweight="bold", va="bottom")
    fig.text(pos_c.x0 - 0.03, letter_y, "C", fontsize=9, fontweight="bold", va="bottom")
    fig.text(0.012, pos_f.y1 - 0.004, "D", fontsize=9, fontweight="bold", va="top")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    log.info("fig1 %s rendered in %.1f s", out_path.name, time.perf_counter() - t_start)
    return out_path
