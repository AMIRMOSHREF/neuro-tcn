"""Figure 3 - "what the model uses": when in the delay, which region and which neurons.

Layout (7.5 x 8 in, ``figures.dpi``)::

    A  temporal attention over the delay, one panel per region, one curve per class      (row 1, 4 panels)
    B  temporal occlusion map: delta balanced accuracy per occluded window (bars)          (row 2, 2 columns)
       + delta forecast deviance explained of the backbone (thin line, right axis)
    C  the same map for the log-loss and the Left-vs-Right accuracy                       (row 2, 1 column)
    D  cross-region attention per class                                                    (row 2, 1 column)
    E  learned gate vs criteria score   F  occlusion importance vs score                   (row 3, 5 panels)
    G  gate vs importance   H  criteria satisfied by the selected units   I  onset of choice information

Design decisions and the scientific reason for each
---------------------------------------------------
* **Attention is plotted in units of the uniform weight** (``w * T``, dotted line at 1) so that "the model
  looks at the late delay 2x more than at a uniform average" is readable directly; the raw softmax weights
  scale with the context length and would not compare across context sweeps.  The attention centre of mass
  (``results['attention_centre_of_mass_ms']``) is marked at the top of each panel because a single number
  per region x class is what the report compares across arms.
* **Attention is descriptive, occlusion is causal.**  Attention tells where the model *looks*; only the
  occlusion map (B, C) tells what it *needs*: each 200 ms window is replaced by the same window of another
  test trial of the same session, so marginal statistics are preserved and only trial information is
  destroyed.  The two rows share the x axis (ms from delay onset) so that a reader can check whether the
  windows the model attends to are the ones whose removal hurts.  Bars are filled with a diverging map
  centred at 0 (blue = accuracy drops when the window is removed = the window carries information).
* **Whiskers are the spread across sessions, not a standard error.**  The session is the unit of
  replication; with 4-8 sessions a min-max whisker is more honest than an SEM.  When several seeds are
  available the bar is the seed mean and a thin dark line shows the seed range (optimisation noise).
* **Model-based vs model-free evidence (E-G).**  The permutation-occlusion importance of every selected
  neuron (delta log-loss) and the learned gate are plotted against the model-free selection score; the
  agreement statistics printed in the corner come from ``results['importance_agreement']`` (Spearman rho
  within each session x region cell, sign test across cells), i.e. exactly the numbers the report uses for
  prediction P6, never a pooled correlation that would be confounded by between-session differences.
* **H and I describe the selected population** with the criteria fractions (S/C/W/R scored, T descriptive)
  and the onset of choice information: if the median onset is late, the "last 500 ms retain >= 95 % of the
  accuracy" prediction (P3) is expected from the single-unit statistics alone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from ._fig_common import (CLASS_COLORS, CLASSES, REGION_COLORS, REGION_LABELS, REGION_TICK, REGIONS, apply_style,
                          nested_get, not_run, panel_label, per_session_dict_seed_mean, rows_by_field, seed_stat,
                          session_spread)

log = logging.getLogger(__name__)

CRITERIA_COLS = [("c_selectivity", "S", "choice\nselectivity"), ("c_coupling", "C", "delay-response\ncoupling"),
                 ("c_spectral", "W", "wavelet\nband power"), ("c_ramp", "R", "ramping"), ("c_locus", "T", "temporal\nlocus")]
_SEED_RANGE_COLOR = "#222222"
_SESSION_WHISKER_COLOR = "#8a8a8a"


# ----------------------------------------------------------------------------- data access
def _load_optional_npz(path: Path):
    try:
        return np.load(path) if path.is_file() else None
    except Exception as e:  # corrupt file: the figure still renders, the panel says so
        log.warning("cannot read %s: %s", path, e)
        return None


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.is_file() else None
    except Exception as e:
        log.warning("cannot read %s: %s", path, e)
        return None


def _occlusion_table(seeds: list[dict]) -> pd.DataFrame | None:
    """One row per occluded window with seed-mean / seed-range of every delta and the session spread."""
    by_start = rows_by_field(seeds, "temporal_occlusion", "window_start_ms")
    if not by_start:
        return None
    rows = []
    for start in sorted(by_start):
        rs = by_start[start]
        rec = {"start": float(start), "end": float(rs[0].get("window_end_ms", start))}
        for k in ("delta_balanced_accuracy", "delta_log_loss", "delta_balanced_accuracy_lr",
                  "delta_forecast_deviance_explained", "delta_forecast_deviance_explained_backbone"):
            rec[k], rec[k + "_lo"], rec[k + "_hi"] = seed_stat([r.get(k) for r in rs])
        rec["sess_lo"], rec["sess_hi"] = session_spread(per_session_dict_seed_mean([r.get("per_session") for r in rs]))
        rows.append(rec)
    return pd.DataFrame(rows)


def _bar_width(tab: pd.DataFrame) -> float:
    """Windows overlap (200 ms every 100 ms), so a bar is one *step* wide, centred on its window."""
    starts = tab["start"].to_numpy()
    if len(starts) > 1:
        return float(np.min(np.diff(starts)))
    return float((tab["end"] - tab["start"]).iloc[0]) if len(tab) else 100.0


# ----------------------------------------------------------------------------- panels
def _panel_temporal_attention(fig, gs_row, att, main: dict, bin_ms: float) -> None:
    axes = []
    n_per_class = nested_get(main, "classification", "n_per_class", default={}) or {}
    com = main.get("attention_centre_of_mass_ms") or {}
    for j, r in enumerate(REGIONS):
        ax = fig.add_subplot(gs_row[0, j], sharey=axes[0] if axes else None)
        axes.append(ax)
        drawn = False
        if att is not None:
            for c in CLASSES:
                key = f"temporal_{r}_{c}"
                if key not in att.files:
                    continue
                w = np.asarray(att[key], dtype=float).ravel()
                if w.size == 0 or not np.isfinite(w).any():
                    continue
                w = w / max(float(np.nansum(w)), 1e-12) * w.size  # x uniform
                t = (np.arange(w.size) + 0.5) * bin_ms
                n = n_per_class.get(c)
                ax.plot(t, w, color=CLASS_COLORS[c], lw=1.1, label=f"{c} (n = {n})" if n is not None else c)
                cm = com.get(f"{r}_{c}")
                if cm is not None and np.isfinite(cm):
                    ax.axvline(float(cm), ymin=0.9, ymax=1.0, color=CLASS_COLORS[c], lw=1.2, clip_on=False)
                drawn = True
        if not drawn:
            not_run(ax, title=REGION_LABELS[r], text="attention.npz missing")
            continue
        ax.axhline(1.0, color="k", ls=":", lw=0.7)
        ax.set_title(REGION_LABELS[r], color=REGION_COLORS[r], loc="left")
        ax.set_xlabel("time from delay onset (ms)")
        ax.set_xlim(0, None)
        ax.margins(x=0)
        if j == 0:
            ax.set_ylabel("temporal attention\n(x uniform)")
            # the legend is shared by the four panels and lives under the figure title, never on the curves
            handles = ax.get_legend_handles_labels()[0]
            handles.append(Line2D([], [], color="#555555", marker="|", ms=6, ls="none", label="attention centre of mass"))
            handles.append(Line2D([], [], color="k", ls=":", lw=0.7, label="uniform attention"))
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6, bbox_to_anchor=(0.5, 0.965),
                       title="test trials per class", title_fontsize=6, handlelength=1.6, columnspacing=1.2)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    if axes:
        panel_label(axes[0], "A", x=-0.16)
        # y range hugs the data (attention is close to uniform); the top 10 % is reserved for the CoM marks
        lo = min([np.nanmin(l.get_ydata()) for ax in axes for l in ax.get_lines() if len(l.get_xdata()) > 2] + [1.0])
        hi = max([np.nanmax(l.get_ydata()) for ax in axes for l in ax.get_lines() if len(l.get_xdata()) > 2] + [1.0])
        span = max(hi - lo, 0.2)
        axes[0].set_ylim(max(0.0, lo - 0.1 * span), hi + 0.2 * span)


def _panel_occlusion_bacc(ax, tab: pd.DataFrame | None, n_seeds: int) -> None:
    title = "Temporal occlusion of the classifier"
    if tab is None or tab.empty:
        not_run(ax, title=title)
        return
    w = _bar_width(tab)
    centre = (tab["start"] + tab["end"]).to_numpy() / 2
    d = tab["delta_balanced_accuracy"].to_numpy()
    vmax = max(float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0, 0.02)
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(-vmax, vmax)
    ax.bar(centre, d, width=w * 0.92, color=cmap(norm(d)), edgecolor="#555555", lw=0.4, zorder=2)
    ok = np.isfinite(tab["sess_lo"].to_numpy()) & np.isfinite(tab["sess_hi"].to_numpy())
    if ok.any():
        ax.vlines(centre[ok], tab["sess_lo"].to_numpy()[ok], tab["sess_hi"].to_numpy()[ok], color=_SESSION_WHISKER_COLOR,
                  lw=0.8, zorder=3)
        for x0, lo, hi in zip(centre[ok], tab["sess_lo"].to_numpy()[ok], tab["sess_hi"].to_numpy()[ok]):
            ax.hlines([lo, hi], x0 - w * 0.18, x0 + w * 0.18, color=_SESSION_WHISKER_COLOR, lw=0.8, zorder=3)
    if n_seeds > 1:
        ax.vlines(centre + w * 0.3, tab["delta_balanced_accuracy_lo"], tab["delta_balanced_accuracy_hi"],
                  color=_SEED_RANGE_COLOR, lw=0.6, zorder=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("occluded window, time from delay onset (ms)")
    ax.set_ylabel("delta balanced accuracy\n(occluded - intact)")
    ax.set_xlim(0, float(tab["end"].max()))
    ax.set_title(title, loc="left")
    # forecaster on a twin axis: the backbone-only variant holds the persistence input fixed, so the
    # line measures what the *delay context* contributes to the response-epoch forecast.
    fc = tab["delta_forecast_deviance_explained_backbone"].to_numpy()
    handles = [Line2D([], [], color=_SESSION_WHISKER_COLOR, lw=0.8, label="session spread (min-max)")]
    if n_seeds > 1:
        handles.append(Line2D([], [], color=_SEED_RANGE_COLOR, lw=0.6, label=f"seed range (n = {n_seeds})"))
    if np.isfinite(fc).any():
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(centre, fc, color="#333333", lw=0.7, marker=".", ms=2.5, zorder=5)
        if n_seeds > 1:
            ax2.fill_between(centre, tab["delta_forecast_deviance_explained_backbone_lo"],
                             tab["delta_forecast_deviance_explained_backbone_hi"], color="#333333", alpha=0.12, lw=0)
        ax2.axhline(0, color="#333333", lw=0.4, ls=":")
        ax2.tick_params(labelsize=5.5)
        handles.append(Line2D([], [], color="#333333", lw=0.7, marker=".", ms=2.5,
                              label="forecaster: delta deviance explained,\nbackbone only (right axis)"))
    ax.legend(handles=handles, loc="best", fontsize=5.2)


def _panel_occlusion_secondary(ax, tab: pd.DataFrame | None) -> None:
    title = "Occlusion: log-loss, L-vs-R"
    if tab is None or tab.empty:
        not_run(ax, title=title)
        return
    centre = (tab["start"] + tab["end"]).to_numpy() / 2
    w = _bar_width(tab)
    ax.bar(centre, tab["delta_log_loss"], width=w * 0.92, color="#c9c9c9", edgecolor="#777777", lw=0.4, label="delta log-loss")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("occluded window (ms)")
    ax.set_ylabel("delta log-loss")
    ax.set_xlim(0, float(tab["end"].max()))
    ax.set_title(title, loc="left")
    lr = tab["delta_balanced_accuracy_lr"].to_numpy()
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#c9c9c9", ec="#777777", lw=0.4, label="delta log-loss")]
    if np.isfinite(lr).any():
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(centre, lr, color="#b2182b", lw=0.8, marker=".", ms=2.5)
        ax2.axhline(0, color="#b2182b", lw=0.4, ls=":")
        ax2.tick_params(labelsize=5.5, colors="#b2182b")
        handles.append(Line2D([], [], color="#b2182b", lw=0.8, marker=".", ms=2.5, label="delta L-vs-R accuracy\n(right axis)"))
    ax.legend(handles=handles, loc="best", fontsize=5.2)


def _panel_region_attention(ax, att) -> None:
    title = "Region attention"
    keys = [c for c in CLASSES if att is not None and f"region_{c}" in att.files]
    if not keys:
        not_run(ax, title=title, text="attention.npz missing")
        return
    w = 0.8 / len(keys)
    x = np.arange(len(REGIONS))
    for i, c in enumerate(keys):
        v = np.asarray(att[f"region_{c}"], dtype=float).ravel()[: len(REGIONS)]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, v, width=w, color=CLASS_COLORS[c], label=c)
    ax.axhline(1.0 / len(REGIONS), color="k", ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_TICK[r] for r in REGIONS], fontsize=5.5)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.0 / len(REGIONS)) * 1.45)  # head-room for the legend
    ax.set_ylabel("region attention weight")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=5.2, ncol=3, columnspacing=0.6, handlelength=1.0)


def _agreement_text(agree: dict | None) -> str:
    if not agree:
        return "agreement: not run"
    rho = agree.get("mean_rho")
    n = agree.get("n_cells")
    p = agree.get("sign_test_p")
    rho_s = f"{rho:.2f}" if rho is not None and np.isfinite(rho) else "n/a"
    n_s = f"{int(n)}" if n is not None else "n/a"
    p_s = f"{p:.2g}" if p is not None and np.isfinite(p) else "n/a"
    return f"mean rho = {rho_s}\n{n_s} session x region cells\nsign test p = {p_s}"


def _panel_scatter(ax, imp: pd.DataFrame | None, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str,
                   agree: dict | None, legend: bool = False) -> None:
    if imp is None or xcol not in imp.columns or ycol not in imp.columns:
        not_run(ax, title=title, text="neuron_importance.csv missing")
        return
    sub = imp[[xcol, ycol, "region"]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        not_run(ax, title=title, text="no finite values")
        return
    for r in REGIONS:
        s = sub[sub.region == r]
        if len(s):
            ax.scatter(s[xcol], s[ycol], s=7, color=REGION_COLORS[r], alpha=0.75, lw=0, label=REGION_LABELS[r])
    if ycol == "delta_log_loss" or xcol == "delta_log_loss":
        (ax.axhline if ycol == "delta_log_loss" else ax.axvline)(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.45 * (hi - lo), hi)  # bottom strip reserved for the agreement statistics
    ax.text(0.98, 0.02, _agreement_text(agree), transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color="#333333", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    if legend:
        ax.legend(loc="upper left", fontsize=5.0, handletextpad=0.2, markerscale=1.2)


def _panel_criteria_fraction(ax, selections: dict[str, pd.DataFrame]) -> None:
    title = "Criteria met"
    per_session = []  # rows = sessions, cols = criteria
    for tab in selections.values():
        if tab is None or "selected" not in tab.columns:
            continue
        sel = tab[tab["selected"].astype(bool)]
        if sel.empty:
            continue
        per_session.append([float(sel[c].astype(bool).mean()) if c in sel.columns else np.nan for c, _, _ in CRITERIA_COLS])
    if not per_session:
        not_run(ax, title=title, text="selection tables missing")
        return
    m = np.asarray(per_session, dtype=float)
    mean = np.nanmean(m, axis=0)
    x = np.arange(len(CRITERIA_COLS))
    colors = ["#444444"] * 4 + ["#9a9a9a"]  # T is descriptive (never scored) -> lighter
    ax.bar(x, mean, color=colors, width=0.7, zorder=2)
    rng = np.random.default_rng(0)
    for row in m:
        ax.scatter(x + rng.uniform(-0.15, 0.15, size=len(x)), row, s=6, color="white", edgecolor="#222222", lw=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s for _, s, _ in CRITERIA_COLS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"fraction of selected units\n(bar = mean, dot = session, n = {len(per_session)})")
    ax.set_xlabel("S selectivity, C coupling,\nW wavelet, R ramp (scored);\nT temporal locus (descriptive)", fontsize=5.5)
    ax.set_title(title, loc="left")


def _panel_onset_hist(ax, selections: dict[str, pd.DataFrame], delay_ms: float) -> None:
    title = "Selectivity onset"
    with_locus, all_sel = [], []
    for tab in selections.values():
        if tab is None or "selected" not in tab.columns or "onset_ms" not in tab.columns:
            continue
        sel = tab[tab["selected"].astype(bool)]
        on = pd.to_numeric(sel["onset_ms"], errors="coerce")
        all_sel.append(on.dropna().to_numpy(float))
        if "c_locus" in sel.columns:
            with_locus.append(on[sel["c_locus"].astype(bool)].dropna().to_numpy(float))
    onsets = np.concatenate(with_locus) if with_locus else np.empty(0)
    note = "selected units with a temporal locus (T)"
    if onsets.size == 0:  # T is descriptive: when no selected unit passes it, the onset of all selected units is shown
        onsets = np.concatenate(all_sel) if all_sel else np.empty(0)
        note = "all selected units (none passed T)"
    if onsets.size == 0:
        not_run(ax, title=title, text="no selected unit\nwith an onset")
        return
    step = 50.0
    edges = np.arange(0, delay_ms + step, step)
    ax.hist(onsets, bins=edges, color="#7a7a7a", edgecolor="white", lw=0.3)
    med = float(np.median(onsets))
    ax.axvline(med, color="#b2182b", lw=1.0, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(0.98, 0.98, f"median {med:.0f} ms\nn = {onsets.size}", color="#b2182b", fontsize=5.5, ha="right", va="top",
            transform=ax.transAxes)
    ax.set_xlabel("onset (ms)")
    ax.set_ylabel(f"units, {note}", fontsize=5.5)
    ax.set_xlim(0, delay_ms)
    ax.set_title(title, loc="left")


# ----------------------------------------------------------------------------- public API
def plot_attention(run_dir: Path, results: dict | list[dict], selections: dict[str, pd.DataFrame], cfg, out_path: Path) -> Path:
    """Render Figure 3 for one run.

    ``run_dir`` holds ``attention.npz`` and ``neuron_importance.csv`` (both optional: the affected panels
    print what is missing); ``results`` is that run's ``results.json`` dict - or a list of such dicts, one
    per seed, in which case the occlusion map shows the seed mean and range; ``selections`` maps session ->
    the selection table the run was trained with (``run_dir/selection_<session__>.csv``)."""
    apply_style()
    run_dir = Path(run_dir)
    seeds = [r for r in (results if isinstance(results, (list, tuple)) else [results]) if isinstance(r, dict)]
    main = seeds[0] if seeds else {}
    bin_ms = float(cfg.get_path("data.bin_ms", 10) if hasattr(cfg, "get_path") else 10)
    att = _load_optional_npz(run_dir / "attention.npz")
    imp = _load_optional_csv(run_dir / "neuron_importance.csv")
    occ = _occlusion_table(seeds)
    t_ctx = next((int(np.asarray(att[k]).size) for k in (att.files if att is not None else []) if k.startswith("temporal_")), 0)
    delay_ms = t_ctx * bin_ms if t_ctx else (float(occ["end"].max()) if occ is not None and len(occ) else 1200.0)
    agree = main.get("importance_agreement") or {}

    fig = plt.figure(figsize=(7.5, 8.0))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.62, left=0.08, right=0.975, top=0.905, bottom=0.08)
    row1 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.28)
    row2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.85, width_ratios=[1.0, 1.0, 1.0, 0.85])
    bot = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.8)

    _panel_temporal_attention(fig, row1, att, main, bin_ms)

    ax_b = fig.add_subplot(row2[0, 0:2])
    _panel_occlusion_bacc(ax_b, occ, len(seeds))
    panel_label(ax_b, "B", x=-0.1)
    ax_c = fig.add_subplot(row2[0, 2])
    _panel_occlusion_secondary(ax_c, occ)
    panel_label(ax_c, "C", x=-0.25)
    ax_d = fig.add_subplot(row2[0, 3])
    _panel_region_attention(ax_d, att)
    panel_label(ax_d, "D", x=-0.25)

    ax_e = fig.add_subplot(bot[0, 0])
    _panel_scatter(ax_e, imp, "score", "gate_rel", "selection score", "learned gate (relative)", "Gate vs score",
                   agree.get("gate_vs_score"), legend=True)
    panel_label(ax_e, "E", x=-0.42)
    ax_f = fig.add_subplot(bot[0, 1])
    _panel_scatter(ax_f, imp, "score", "delta_log_loss", "selection score", "importance\n(delta log-loss)",
                   "Importance vs score", agree.get("importance_vs_score"))
    panel_label(ax_f, "F", x=-0.42)
    ax_g = fig.add_subplot(bot[0, 2])
    _panel_scatter(ax_g, imp, "delta_log_loss", "gate_rel", "importance (delta log-loss)", "learned gate (relative)",
                   "Gate vs importance", agree.get("gate_vs_importance"))
    panel_label(ax_g, "G", x=-0.42)
    ax_h = fig.add_subplot(bot[0, 3])
    _panel_criteria_fraction(ax_h, selections or {})
    panel_label(ax_h, "H", x=-0.42)
    ax_i = fig.add_subplot(bot[0, 4])
    _panel_onset_hist(ax_i, selections or {}, delay_ms)
    panel_label(ax_i, "I", x=-0.42)

    mode = main.get("mode", "criteria")
    fig.suptitle(f"DelayCAST ({mode} K): where in the delay, which region and which neurons the prediction uses",
                 fontsize=9, y=0.985)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path
