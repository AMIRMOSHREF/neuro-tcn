"""Figure 4 - "does the claim hold": every arm of the protocol on one page.

Layout (7.5 x 8.6 in, ``figures.dpi``)::

    A  confusion matrix of the criteria run      B  context sweep (all arms, linear sweep, tau95)   C  region ablation
    D  forecast deviance explained per region    E  arms: DelayCAST variants and linear baselines (fixed order)
    F  per-session dot plot (within / cross-session / cross-dataset / negative control)             G  ablation per class

Design decisions and the scientific reason for each
---------------------------------------------------
* **Chance is estimated, never assumed.**  Every accuracy panel shades the 95th percentile of the
  within-session label-permutation null from ``results['chance']`` (pooled over seeds) rather than
  drawing 1/3: the sessions are class-imbalanced and pooled balanced accuracy has a null above 1/3.
* **Whiskers are the spread across sessions** (C, D) or **a 95 % trial bootstrap CI / the seed range**
  (A, E).  The session is the replicate of every claim, so for ablations the reader must see whether the
  effect is consistent across sessions; for the arm comparison the CI answers "is this arm's accuracy
  distinguishable at all", while the report's paired Wilcoxon test answers the arm-vs-arm question.
* **Seeds are averaged first.**  Any quantity of an arm trained with several seeds is the mean over seeds
  (bar / dot) with the seed range as a thin whisker (F) or the min-max band (B); the spread between seeds
  is optimisation noise and must not be confused with biological variability.
* **The context sweep carries the whole P3 story**: the DelayCAST arms, the tuned linear decoder on all
  units (dashed grey, model-free), the 95 % line of the full-context accuracy and the tau95 marker with its
  bootstrap CI (``results['csi']``), so that a reader can check the sufficiency claim without the report.
* **Region ablation shows both operations** (permutation occlusion of a region = out-of-distribution
  removal that keeps the marginals; in-distribution drop = the augmentation the model was trained with).
  A region matters only if *both* hurt.  The Left-vs-Right delta is annotated because prediction P5a is
  about the lick direction, not about the Ignore class.
* **Panel G isolates the class**: a region whose removal only hurts the Ignore recall supports the
  exploratory striatal prediction (P5b), not the ALM prediction (P5a); the pooled balanced accuracy of C
  cannot tell these apart.
* **Arms in a fixed order (E)** so that the same bar is at the same place in every re-render and across
  datasets; missing arms are skipped and named in the panel so the reader knows what was not run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ._fig_common import (ARM_COLORS, ARM_LABELS, ARM_ORDER, CLASS_COLORS, CLASSES, KIND_STYLE, REGION_COLORS,
                          REGION_TICK, REGIONS, apply_style, dataset_of, nested_get, not_run, panel_label,
                          per_session_dict_seed_mean, per_session_seed_mean, rows_by_field, run_color, run_label,
                          seed_stat, session_spread, short_session, split_run_name)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- shared numbers
def _chance(seeds: list[dict]) -> tuple[float, float]:
    """(mean, p95) of the permutation null, pooled over seeds; NaN if never computed."""
    mean = seed_stat([nested_get(r, "chance", "mean", default=nested_get(r, "chance_balanced_accuracy", "mean")) for r in seeds])[0]
    p95 = seed_stat([nested_get(r, "chance", "p95", default=nested_get(r, "chance_balanced_accuracy", "p95")) for r in seeds])[0]
    return mean, p95


def _chance_band(ax, p95: float, mean: float, horizontal: bool = True) -> None:
    if np.isfinite(p95):
        (ax.axhspan if horizontal else ax.axvspan)(0, p95, color="#e6e6e6", zorder=0, lw=0)
    if np.isfinite(mean):
        (ax.axhline if horizontal else ax.axvline)(mean, color="#9a9a9a", lw=0.6, ls=":", zorder=1)


def _bacc(seeds: list[dict]) -> tuple[float, float, float]:
    return seed_stat([nested_get(r, "classification", "balanced_accuracy") for r in seeds])


# ----------------------------------------------------------------------------- panels
def _panel_confusion(ax, crit: list[dict]) -> None:
    title = "Confusion, criteria K (test trials)"
    cms = [np.asarray(r["confusion"], dtype=float) for r in crit if isinstance(r.get("confusion"), list)]
    if not cms:
        not_run(ax, title=title)
        return
    cm = np.sum(cms, axis=0)  # pooled counts over seeds: the same trials, several fits
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(round(cm[i, j]))}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    m, lo, hi = _bacc(crit)
    if len(crit) == 1:
        ci = nested_get(crit[0], "classification_ci", "balanced_accuracy")
        if isinstance(ci, list) and len(ci) == 2:
            lo, hi = float(ci[0]), float(ci[1])
        ci_note = "95 % trial-bootstrap CI"
    else:
        ci_note = f"seed range, n = {len(crit)} seeds"
    f1 = seed_stat([nested_get(r, "classification", "macro_f1") for r in crit])[0]
    n = nested_get(crit[0], "classification", "n", default=int(cm.sum()))
    ax.set_title(title, loc="left")
    ax.set_anchor("N")  # square image at the top of its cell, statistics directly under the x label
    ax.text(0.0, -0.24, f"bal. acc {m:.2f} [{lo:.2f}, {hi:.2f}]\n({ci_note})\nmacro-F1 {f1:.2f}, n = {n} trials"
            + ("\ncounts pooled over seeds" if len(crit) > 1 else ""),
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color="#333333")


def _panel_context_sweep(ax, results_by_run: dict[str, list[dict]], crit: list[dict]) -> None:
    title = "How much delay context is needed?"
    drawn = False
    for name, seeds in results_by_run.items():
        kind, _ = split_run_name(name)
        by_ms = rows_by_field(seeds, "context_sweep", "context_ms")
        if not by_ms:
            continue
        xs = sorted(by_ms)
        stats = [seed_stat([r.get("balanced_accuracy") for r in by_ms[x]]) for x in xs]
        mean, lo, hi = (np.array([s[i] for s in stats]) for i in range(3))
        col = run_color(name)
        ax.plot(xs, mean, color=col, lw=1.2 if name == "criteria" else 0.9, marker="o", ms=2.2, label=run_label(name),
                zorder=3 if name == "criteria" else 2, **KIND_STYLE.get(kind, {}))
        if len(seeds) > 1:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        drawn = True
    lin = rows_by_field(crit, "linear_sweep", "context_ms")
    if lin:
        xs = sorted(lin)
        ax.plot(xs, [seed_stat([r.get("balanced_accuracy") for r in lin[x]])[0] for x in xs], color="#8a8a8a", ls="--",
                lw=1.0, label="log-reg, all units", zorder=2)
        drawn = True
    if not drawn:
        not_run(ax, title=title)
        return
    mean, p95 = _chance(crit)
    _chance_band(ax, p95, mean)
    full, _, _ = _bacc(crit)
    csi = crit[0].get("csi") if crit else None
    if csi and np.isfinite(full):
        frac = float(csi.get("fraction", 0.95))
        # both reference lines are named in the legend: any text placed at the line would sit on the
        # full-context end of the curves by construction (the line *is* 95 % of that end point)
        ax.axhline(frac * full, color="#222222", lw=0.6, ls="--", label=f"{frac:.0%} of full-context accuracy")
        tau = seed_stat([nested_get(r, "csi", "tau95_median_ms", default=nested_get(r, "csi", "tau95_ms")) for r in crit])[0]
        ci = csi.get("tau95_ci_ms")
        if tau is not None and np.isfinite(tau):
            ci_txt = f" [{ci[0]:.0f}, {ci[1]:.0f}]" if isinstance(ci, list) and len(ci) == 2 else ""
            ax.axvline(tau, color="#b2182b", lw=0.8, ls="-.", label=f"tau95 = {tau:.0f} ms{ci_txt}")
    ax.set_xlabel("delay context available before the go cue (ms)")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=5.2, ncol=2, columnspacing=0.8)


def _ablation_rows(crit: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for r in crit:
        for row in (r.get("region_ablation") or []):
            if isinstance(row, dict) and "dropped_region" in row:
                out.setdefault((row["dropped_region"], row.get("method", "drop")), []).append(row)
    return out


def _panel_region_ablation(ax, crit: list[dict]) -> None:
    title = "Region ablation"
    rows = _ablation_rows(crit)
    if not rows:
        not_run(ax, title=title)
        return
    methods = [m for m in ("permute", "drop", "zero") if any(k[1] == m for k in rows)]
    w = 0.8 / len(methods)
    x = np.arange(len(REGIONS))
    hatch = {"permute": "", "drop": "////", "zero": "...."}
    ymin = 0.0
    for i, m in enumerate(methods):
        for j, r in enumerate(REGIONS):
            rs = rows.get((r, m))
            if not rs:
                continue
            xpos = x[j] + (i - (len(methods) - 1) / 2) * w
            d, _, _ = seed_stat([q.get("delta_balanced_accuracy") for q in rs])
            lo, hi = session_spread(per_session_dict_seed_mean([q.get("per_session") for q in rs]))
            ax.bar(xpos, d, width=w * 0.95, color=REGION_COLORS[r], edgecolor="#333333", lw=0.4, hatch=hatch[m],
                   alpha=1.0 if m == "permute" else 0.55, zorder=2)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(xpos, d, yerr=[[max(d - lo, 0)], [max(hi - d, 0)]], color="#333333", lw=0.7, capsize=1.5, zorder=3)
                ymin = min(ymin, lo)
            d_lr, _, _ = seed_stat([q.get("delta_balanced_accuracy_lr") for q in rs])
            if np.isfinite(d_lr):
                ax.plot(xpos, d_lr, marker="D", ms=2.8, color="white", markeredgecolor="#b2182b", mew=0.8, ls="none", zorder=4)
            ymin = min(ymin, d, d_lr if np.isfinite(d_lr) else 0.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_TICK[r] for r in REGIONS])
    ax.set_ylabel("delta balanced accuracy\n(region removed - intact)")
    ax.set_title(title, loc="left")
    handles = [Patch(fc="#bbbbbb", ec="#333333", lw=0.4, hatch=hatch[m], alpha=1.0 if m == "permute" else 0.55,
                     label={"permute": "permutation occlusion", "drop": "region drop (in-distribution)", "zero": "zeroed"}[m])
               for m in methods]
    handles += [Line2D([], [], color="#333333", lw=0.7, label="session spread"),
                Line2D([], [], marker="D", ms=2.8, color="white", markeredgecolor="#b2182b", ls="none", label="Left-vs-Right delta")]
    ax.legend(handles=handles, loc="best", fontsize=5.0)


def _panel_forecast(ax, crit: list[dict]) -> None:
    title = "Response-epoch forecast"
    fc = [r.get("forecast") or {} for r in crit]
    if not any(f"deviance_explained_{r}" in f for f in fc for r in REGIONS):
        not_run(ax, title=title)
        return
    w = 0.38
    x = np.arange(len(REGIONS))
    for kind, off, label in (("", -w / 2, "DelayCAST forecast"), ("persistence_", w / 2, "persistence (late-delay rate)")):
        for j, r in enumerate(REGIONS):
            key = f"deviance_explained_{kind}{r}"
            v, _, _ = seed_stat([f.get(key) for f in fc])
            if not np.isfinite(v):
                continue
            ps = per_session_seed_mean(crit, "forecast", "per_session", metric=key)
            lo, hi = session_spread({s: t[0] for s, t in ps.items()})
            ax.bar(x[j] + off, v, width=w, color=REGION_COLORS[r] if kind == "" else "#c4c4c4", edgecolor="#333333", lw=0.4, zorder=2)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(x[j] + off, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], color="#333333", lw=0.7, capsize=1.5, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_TICK[r] for r in REGIONS])
    ax.set_ylabel("Poisson deviance explained\n(vs training-PSTH null)")
    ax.set_title(title, loc="left")
    ax.legend(handles=[Patch(fc="#555555", label="DelayCAST forecast (region colour)"),
                       Patch(fc="#c4c4c4", ec="#333333", lw=0.4, label="persistence (late-delay rate)"),
                       Line2D([], [], color="#333333", lw=0.7, label="session spread")], loc="best", fontsize=5.0)


def _arm_values(results_by_run: dict[str, list[dict]], crit: list[dict]) -> dict[str, tuple[float, float, float, str]]:
    """arm -> (value, lo, hi, whisker kind) for the fixed arm order: DelayCAST arms from their own runs,
    linear baselines from the criteria run's ``baselines`` list."""
    out: dict[str, tuple[float, float, float, str]] = {}
    for arm in ARM_ORDER:
        if arm.startswith("logreg_"):
            rows = [b for r in crit for b in (r.get("baselines") or []) if isinstance(b, dict) and b.get("model") == arm]
            if not rows:
                continue
            m, lo, hi = seed_stat([b.get("balanced_accuracy") for b in rows])
            out[arm] = (m, lo, hi, "seeds") if len(rows) > 1 else (m, float("nan"), float("nan"), "")
        else:
            seeds = results_by_run.get(arm) or []
            if not seeds:
                continue
            m, lo, hi = _bacc(seeds)
            if len(seeds) > 1:
                out[arm] = (m, lo, hi, "seeds")
            else:
                ci = nested_get(seeds[0], "classification_ci", "balanced_accuracy")
                if isinstance(ci, list) and len(ci) == 2:
                    out[arm] = (m, float(ci[0]), float(ci[1]), "ci")
                else:
                    out[arm] = (m, float("nan"), float("nan"), "")
    return out


def _panel_arms(ax, results_by_run: dict[str, list[dict]], crit: list[dict]) -> None:
    title = "Neuron sets, model ablations and linear baselines"
    arms = _arm_values(results_by_run, crit)
    if not arms:
        not_run(ax, title=title)
        return
    names = list(arms)
    x = np.arange(len(names))
    vals = np.array([arms[a][0] for a in names])
    ax.bar(x, vals, color=[ARM_COLORS.get(a, "#555555") for a in names], edgecolor="#333333", lw=0.4, width=0.7, zorder=2)
    kinds = set()
    for i, a in enumerate(names):
        _, lo, hi, kind = arms[a]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(x[i], vals[i], yerr=[[max(vals[i] - lo, 0)], [max(hi - vals[i], 0)]], color="#333333", lw=0.7,
                        capsize=1.5, zorder=3)
            kinds.add(kind)
        ax.text(x[i], 0.02, f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=5, color="white" if vals[i] > 0.12 else "#333333",
                transform=ax.get_xaxis_transform())
    mean, p95 = _chance(crit)
    _chance_band(ax, p95, mean)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in names], fontsize=6)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("balanced accuracy")
    ax.set_title(title, loc="left")
    missing = [ARM_LABELS.get(a, a).replace("\n", " ") for a in ARM_ORDER if a not in arms]
    note = {"ci": "whiskers: 95 % trial-bootstrap CI", "seeds": "whiskers: seed range"}
    lines = [note[k] for k in ("ci", "seeds") if k in kinds] + ["grey band: permutation null (95th pct)"]
    if missing:
        lines.append("not run: " + ", ".join(missing))
    ax.text(1.0, -0.2, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color="#333333")


def _panel_per_session(ax, results_by_run: dict[str, list[dict]], crit: list[dict]) -> None:
    title = "Per-session balanced accuracy (criteria K)"
    within = per_session_seed_mean(crit, "per_session")
    cross_s = per_session_seed_mean(results_by_run.get("cross_session/criteria") or [], "per_session")
    cross_d = per_session_seed_mean(results_by_run.get("cross_dataset/criteria") or [], "per_session")
    neg = per_session_seed_mean(results_by_run.get("negative_control/criteria") or [], "per_session")
    sessions = sorted(set(within) | set(cross_s) | set(cross_d) | set(neg), key=lambda s: (dataset_of(s), s))
    if not sessions:
        not_run(ax, title=title)
        return
    # rows grouped by dataset with a one-row gap and a separator line between groups
    ypos, y, prev = {}, 0.0, None
    for s in sessions:
        ds = dataset_of(s)
        if prev is not None and ds != prev:
            y += 1.0
            ax.axhline(y - 0.5, color="#bbbbbb", lw=0.5)
        ypos[s] = y
        y += 1.0
        prev = ds
    mean, p95 = _chance(crit)
    _chance_band(ax, p95, mean, horizontal=False)
    n_seeds = len(crit)
    for s, (m, lo, hi) in within.items():
        if n_seeds > 1 and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [ypos[s]] * 2, color="#222222", lw=0.8, zorder=2)
        ax.plot(m, ypos[s], marker="o", ms=4, color="#222222", ls="none", zorder=4)
    for s, (m, _, _) in cross_s.items():
        ax.plot(m, ypos[s], marker="o", ms=4, mfc="white", mec="#222222", mew=0.8, ls="none", zorder=3)
    for s, (m, _, _) in cross_d.items():
        ax.plot(m, ypos[s], marker="^", ms=4.5, mfc="white", mec="#d55e00", mew=0.8, ls="none", zorder=3)
    for s, (m, _, _) in neg.items():
        ax.plot(m, ypos[s], marker="x", ms=4, color="#8a8a8a", mew=0.9, ls="none", zorder=3)
    ax.set_yticks([ypos[s] for s in sessions])
    ax.set_yticklabels([short_session(s) for s in sessions], fontsize=5.5)
    ax.set_ylim(y - 0.5, -0.5)  # first session at the top
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("balanced accuracy")
    ax.set_title(title, loc="left")
    handles = [Line2D([], [], marker="o", ms=4, color="#222222", ls="none", label="within-session" + (f" (mean of {n_seeds} seeds, whisker = range)" if n_seeds > 1 else "")),
               Line2D([], [], marker="o", ms=4, mfc="white", mec="#222222", ls="none", label="cross-session (held out)"),
               Line2D([], [], marker="^", ms=4.5, mfc="white", mec="#d55e00", ls="none", label="cross-dataset (adapters only)"),
               Line2D([], [], marker="x", ms=4, color="#8a8a8a", ls="none", label="negative control (labels permuted)"),
               Line2D([], [], color="#9a9a9a", lw=0.6, ls=":", label="chance (permutation null, band = 95th pct)")]
    ax.legend(handles=handles, loc="lower right", fontsize=5.0, ncol=1)
    for ds in sorted(set(dataset_of(s) for s in sessions)):
        rows = [ypos[s] for s in sessions if dataset_of(s) == ds]
        ax.text(1.005, float(np.mean(rows)), f"dataset {ds}" if ds else "", transform=ax.get_yaxis_transform(), fontsize=5.5,
                rotation=90, va="center", ha="left", color="#555555")


def _panel_ablation_per_class(ax, crit: list[dict]) -> None:
    title = "Region ablation per class"
    rows = _ablation_rows(crit)
    method = next((m for m in ("permute", "drop", "zero") if any(k[1] == m for k in rows)), None)
    base = {c: seed_stat([nested_get(r, "classification", "recall", c) for r in crit])[0] for c in CLASSES}
    if method is None or not any(np.isfinite(v) for v in base.values()):
        not_run(ax, title=title)
        return
    w = 0.8 / len(CLASSES)
    x = np.arange(len(REGIONS))
    for i, c in enumerate(CLASSES):
        vals = []
        for r in REGIONS:
            rs = rows.get((r, method)) or []
            rec, _, _ = seed_stat([nested_get(q, "recall", c) for q in rs])
            vals.append(rec - base[c] if np.isfinite(rec) and np.isfinite(base[c]) else np.nan)
        ax.bar(x + (i - (len(CLASSES) - 1) / 2) * w, vals, width=w * 0.95, color=CLASS_COLORS[c], edgecolor="#333333", lw=0.3,
               label=f"{c} (intact recall {base[c]:.2f})" if np.isfinite(base[c]) else c, zorder=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_TICK[r] for r in REGIONS])
    ax.set_ylabel(f"delta recall ({method} region)")
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=5.0)


# ----------------------------------------------------------------------------- public API
def plot_results(results_by_run: dict[str, list[dict]], cfg, out_path: Path) -> Path:
    """Render Figure 4 from ``runs.load_results`` output (run name -> list of results dicts, one per seed).

    Every panel degrades to a "not run" note when its inputs are missing; the criteria run is the
    reference arm for chance, the confusion matrix, the ablations and the baselines."""
    apply_style()
    results_by_run = {k: [r for r in (v if isinstance(v, (list, tuple)) else [v]) if isinstance(r, dict)]
                      for k, v in (results_by_run or {}).items()}
    results_by_run = {k: v for k, v in results_by_run.items() if v}
    crit = results_by_run.get("criteria") or []
    if not crit and results_by_run:
        # no criteria arm: use the first within-session arm as reference so the figure still says something
        fallback = next((v for k, v in results_by_run.items() if "/" not in k), next(iter(results_by_run.values())))
        crit = fallback
        log.warning("no 'criteria' run: Figure 4 uses '%s' as the reference arm", fallback[0].get("_name", "?"))

    fig = plt.figure(figsize=(7.5, 8.6))
    gs = GridSpec(3, 3, figure=fig, left=0.1, right=0.975, top=0.94, bottom=0.06, hspace=0.62, wspace=0.42,
                  height_ratios=[1.0, 1.0, 1.15], width_ratios=[1.0, 1.15, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    _panel_confusion(ax, crit)
    panel_label(ax, "A", x=-0.25)
    ax = fig.add_subplot(gs[0, 1])
    _panel_context_sweep(ax, results_by_run, crit)
    panel_label(ax, "B", x=-0.14)
    ax = fig.add_subplot(gs[0, 2])
    _panel_region_ablation(ax, crit)
    panel_label(ax, "C", x=-0.2)
    ax = fig.add_subplot(gs[1, 0])
    _panel_forecast(ax, crit)
    panel_label(ax, "D", x=-0.25)
    ax = fig.add_subplot(gs[1, 1:])
    _panel_arms(ax, results_by_run, crit)
    panel_label(ax, "E", x=-0.06)
    ax = fig.add_subplot(gs[2, 0:2])
    _panel_per_session(ax, results_by_run, crit)
    panel_label(ax, "F", x=-0.12)
    ax = fig.add_subplot(gs[2, 2])
    _panel_ablation_per_class(ax, crit)
    panel_label(ax, "G", x=-0.2)

    ref = crit[0] if crit else {}
    tag = f"{ref.get('mode', 'criteria')} K, {len(crit)} seed{'s' if len(crit) != 1 else ''}"
    fig.suptitle(f"DelayCAST: decoding the upcoming action and forecasting the response epoch from the delay ({tag})",
                 fontsize=9, y=0.985)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path


def load_results(out_dir: Path) -> dict[str, list[dict]]:
    """Backward-compatible alias of :func:`delaycast.runs.load_results` (run name -> list of results dicts)."""
    from ..runs import load_results as _load
    return _load(Path(out_dir))
