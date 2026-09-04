"""Helpers shared by the two results-driven figures (Fig 3 ``attention_fig`` and Fig 4 ``results_fig``).

Both figures are drawn from ``results.json`` files (see ``delaycast.evaluate.evaluate_run`` and
``delaycast.runs.load_results``); nothing here touches the data cache or the model.  Three concerns live
here so that the two figure modules do not duplicate them:

* **Colour system with a fallback.**  ``style.py`` is the single source of truth for the palette, but this
  module tolerates a partially written ``style.py`` (the constants are being consolidated by another
  module owner) by falling back to the values of the implementation contract.  A figure must never fail
  to render because a colour constant is missing - the numbers are the deliverable, the colour is the
  presentation.
* **Seed aggregation.**  Every arm may have been trained with several seeds.  The scientific unit of
  replication is the *session*, so per-session quantities are averaged over seeds *first* and the spread
  across sessions is what the whiskers show; the spread across seeds is reported separately (a thin
  line) because it measures optimisation noise, not biological variability.
* **Graceful degradation.**  A panel whose inputs are missing prints "not run" instead of raising, so a
  partial protocol (``--quick``, a crashed cross-dataset run) still yields a complete figure that says
  what is missing.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import matplotlib as mpl
import numpy as np

from .. import CLASSES, REGION_LABELS, REGIONS
from .. import REGION_COLORS as _PKG_REGION_COLORS

# ----------------------------------------------------------------------------- colours (style.py with fallback)
_FALLBACK = {
    "REGION_COLORS": {"ALM_L": "#1f4e9c", "ALM_R": "#7fb2e5", "STR_L": "#6a2c91", "STR_R": "#c39bd3"},
    "CLASS_COLORS": {"Ignore": "#7f7f7f", "Left": "#009e73", "Right": "#e69f00"},
    "EPOCH_COLORS": {"sample": "#f4f4f4", "delay": "#e9e9e9", "response": "#ffffff"},
    "MODE_COLORS": {"criteria": "#222222", "criteria_nospec": "#d55e00", "criteria_popmean": "#cc79a7",
                    "rate": "#888888", "random": "#bbbbbb", "logreg_all_units": "#56b4e9",
                    "logreg_selected_units": "#0072b2"},
    "STATUS_COLORS": {"selected": None, "eligible": "#9a9a9a", "floor": "#cfcfcf", "below_floor": "#ededed"},
}

try:  # style.py may be mid-rewrite by its owner; every name is optional here
    from . import style as _style
except Exception:  # pragma: no cover - only hit while style.py is broken
    _style = None


def _from_style(name: str, default):
    value = getattr(_style, name, None) if _style is not None else None
    return value if value is not None else default


REGION_COLORS: dict[str, str] = _from_style("REGION_COLORS", dict(_PKG_REGION_COLORS) or _FALLBACK["REGION_COLORS"])
CLASS_COLORS: dict[str, str] = _from_style("CLASS_COLORS", _FALLBACK["CLASS_COLORS"])
EPOCH_COLORS: dict[str, str] = _from_style("EPOCH_COLORS", _FALLBACK["EPOCH_COLORS"])
MODE_COLORS: dict[str, str] = dict(_FALLBACK["MODE_COLORS"])
MODE_COLORS.update(_from_style("MODE_COLORS", {}))
STATUS_COLORS: dict[str, str | None] = _from_style("STATUS_COLORS", _FALLBACK["STATUS_COLORS"])

# Two-line region tick labels for narrow bar panels ("left Striatum" does not fit under a 0.2 in bar).
REGION_TICK = {"ALM_L": "left\nALM", "ALM_R": "right\nALM", "STR_L": "left\nSTR", "STR_R": "right\nSTR"}

# Arms that only the linear baselines contribute; kept in the blue family of ``logreg_all_units`` so that
# every linear decoder reads as "the same family, a different feature set".
ARM_COLORS: dict[str, str] = dict(MODE_COLORS)
ARM_COLORS.setdefault("logreg_pca50_all_units", "#a7d8f2")
ARM_COLORS.setdefault("logreg_l1_all_units", "#2f8fc7")
ARM_COLORS.setdefault("logreg_trial_index", "#dddddd")
ARM_COLORS.setdefault("logreg_selected_ALM", "#5a9fd4")
ARM_COLORS.setdefault("logreg_selected_STR", "#7a6fb0")

# Fixed arm order of Figure 4E (missing arms are skipped) and two-line tick labels.
ARM_ORDER: tuple[str, ...] = ("criteria", "criteria_nospec", "criteria_popmean", "rate", "random",
                              "logreg_all_units", "logreg_pca50_all_units", "logreg_l1_all_units",
                              "logreg_selected_units", "logreg_trial_index")
ARM_LABELS: dict[str, str] = {
    "criteria": "DelayCAST\ncriteria K",
    "criteria_nospec": "DelayCAST\nno spectral",
    "criteria_popmean": "DelayCAST\npop. mean",
    "rate": "DelayCAST\nrate K",
    "random": "DelayCAST\nrandom K",
    "logreg_all_units": "log-reg\nall units",
    "logreg_pca50_all_units": "log-reg\nPCA-50",
    "logreg_l1_all_units": "log-reg\nL1 all units",
    "logreg_selected_units": "log-reg\nselected K",
    "logreg_selected_ALM": "log-reg\nselected ALM",
    "logreg_selected_STR": "log-reg\nselected STR",
    "logreg_trial_index": "log-reg\ntrial index",
}
# One-line names for legends (context sweep, per-session panel).
RUN_LABELS: dict[str, str] = {
    "criteria": "criteria K", "criteria_nospec": "criteria K, no spectral", "criteria_popmean": "criteria K, pop. mean",
    "rate": "rate K", "random": "random K",
}
KIND_STYLE: dict[str, dict] = {  # line style per run kind so cross_* arms never masquerade as within-session arms
    "within": {"ls": "-"}, "cross_session": {"ls": "--"}, "cross_dataset": {"ls": ":"}, "negative_control": {"ls": (0, (1, 1))},
}


def split_run_name(name: str) -> tuple[str, str]:
    """'cross_dataset/criteria' -> ('cross_dataset', 'criteria'); 'rate' -> ('within', 'rate')."""
    if "/" in name:
        kind, mode = name.split("/", 1)
        return kind, mode
    return "within", name


def run_color(name: str) -> str:
    return ARM_COLORS.get(split_run_name(name)[1], "#555555")


def run_label(name: str) -> str:
    kind, mode = split_run_name(name)
    base = RUN_LABELS.get(mode, mode.replace("_", " "))
    return base if kind == "within" else f"{base} ({kind.replace('_', '-')})"


# ----------------------------------------------------------------------------- typography (style.py with fallback)
def apply_style(dpi: float | None = None) -> None:
    fn = getattr(_style, "apply_style", None) if _style is not None else None
    if fn is not None:
        fn(dpi)
        return
    mpl.rcParams.update({  # journal typography of the contract
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None, "figure.facecolor": "white",
    })


def panel_label(ax, text: str, x: float = -0.08, y: float = 1.04, **kw) -> None:
    fn = getattr(_style, "panel_label", None) if _style is not None else None
    if fn is not None:
        fn(ax, text, x, y, **kw)
        return
    style = dict(fontsize=9, fontweight="bold", va="bottom", ha="left", clip_on=False)
    style.update(kw)
    ax.text(x, y, text, transform=ax.transAxes, **style)


# ----------------------------------------------------------------------------- "not run" stubs
def not_run(ax, title: str | None = None, text: str = "not run") -> None:
    """Empty a panel and print why: a missing analysis is shown, never silently skipped."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#8a8a8a", fontsize=7,
            style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, loc="left")


# ----------------------------------------------------------------------------- numeric helpers
def _finite(values: Iterable) -> np.ndarray:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def seed_stat(values: Iterable) -> tuple[float, float, float]:
    """(mean, min, max) over the finite entries; NaNs when nothing is finite."""
    v = _finite(values)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(v.min()), float(v.max())


def nested_get(d: dict | None, *keys, default=None):
    node = d
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return default if node is None else node


def per_session_seed_mean(seeds: Sequence[dict], *keys, session_key: str = "session", metric: str = "balanced_accuracy"
                          ) -> dict[str, tuple[float, float, float]]:
    """Session -> (mean, min, max over seeds) of ``metric`` in the ``per_session`` list found under ``keys``.

    The per-seed values are averaged per session *before* anything else is computed because the session,
    not the seed, is the replicate of every claim in the report."""
    acc: dict[str, list[float]] = {}
    for r in seeds:
        rows = nested_get(r, *keys, default=[]) or []
        for row in rows:
            if not isinstance(row, dict) or session_key not in row:
                continue
            acc.setdefault(str(row[session_key]), []).append(row.get(metric))
    return {s: seed_stat(v) for s, v in acc.items()}


def per_session_dict_seed_mean(dicts: Sequence[dict | None]) -> dict[str, float]:
    """Session -> mean over seeds for ``per_session`` stored as {session: value} (occlusion / ablation rows)."""
    acc: dict[str, list[float]] = {}
    for d in dicts:
        for s, v in (d or {}).items():
            if isinstance(v, dict):   # region ablation stores both the 3-class and the Left/Right delta per session
                v = v.get("delta_balanced_accuracy", v.get("delta_balanced_accuracy_lr"))
            acc.setdefault(str(s), []).append(v)
    return {s: seed_stat(v)[0] for s, v in acc.items()}


def session_spread(values: dict[str, float]) -> tuple[float, float]:
    v = _finite(values.values())
    return (float(v.min()), float(v.max())) if v.size else (float("nan"), float("nan"))


def rows_by_field(seeds: Sequence[dict], key: str, field: str) -> dict:
    """field value -> list of row dicts (one per seed) for a list-of-rows result key such as ``context_sweep``."""
    out: dict = {}
    for r in seeds:
        for row in (r.get(key) or []):
            if isinstance(row, dict) and field in row:
                out.setdefault(row[field], []).append(row)
    return out


# ----------------------------------------------------------------------------- session labels
_SES_B = re.compile(r"sub-(\d+)_ses-(\d{4})(\d{2})(\d{2})")
_SES_A = re.compile(r"^Session\s*(\d+)$", re.IGNORECASE)


def short_session(session: str) -> str:
    """Dense session label: 'A/Session1' -> 'A-S1'; 'B/sub-440957_ses-20190211T143614' -> 'B-440957-0211'."""
    ds, _, rest = session.partition("/")
    if not rest:
        return session
    m = _SES_B.search(rest)
    if m:
        return f"{ds}-{m.group(1)}-{m.group(3)}{m.group(4)}"
    m = _SES_A.match(rest)
    if m:
        return f"{ds}-S{m.group(1)}"
    return f"{ds}-{rest[-12:]}"


def dataset_of(session: str) -> str:
    return session.partition("/")[0] if "/" in session else ""


__all__ = ["REGIONS", "REGION_LABELS", "CLASSES", "REGION_COLORS", "CLASS_COLORS", "EPOCH_COLORS", "MODE_COLORS",
           "STATUS_COLORS", "REGION_TICK", "ARM_COLORS", "ARM_ORDER", "ARM_LABELS", "RUN_LABELS", "KIND_STYLE", "apply_style",
           "panel_label", "not_run", "seed_stat", "nested_get", "per_session_seed_mean", "per_session_dict_seed_mean",
           "session_spread", "rows_by_field", "short_session", "dataset_of", "split_run_name", "run_color", "run_label"]
