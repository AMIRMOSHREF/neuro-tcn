"""Shared colour system and typography for every DelayCAST figure.

Design rules (from the figure review) that all figure modules follow:

* **Region colours** are used *only* for things that belong to a region: selected-unit spikes, region
  labels, exemplar titles, the "selected" segment of a funnel.  Left/right hemisphere pairs share a hue
  (blue for ALM, purple for striatum) at two lightness levels so that a colour-blind reader still
  separates the two areas, and a grey-scale print still separates the two hemispheres.
* **Class colours** (Okabe-Ito) are used *only* for trial-conditioned quantities (licks, class-conditional
  PSTHs, class scatters).  They never encode a region, so the two colour families never compete.
* **Epoch colours** are light greys with dotted boundaries: the task structure is background information
  and must not draw the eye away from the spikes.
* **Heatmaps** use neutral colormaps (Blues, Greys, RdBu_r, viridis) so that the colour of a cell is
  never confused with a region or class colour.
* **Typography** is fixed at journal sizes: 7 pt body, 6 pt ticks and legends, 7.5 pt axis titles and
  9 pt bold panel letters.  Fonts are embedded as TrueType (``pdf.fonttype 42``) so the PDF text remains
  editable, and ``savefig.bbox`` is left at ``None`` so the figure is exactly the requested size.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import ListedColormap

from .. import REGION_COLORS  # single source of truth for the region colours

CLASS_COLORS = {"Ignore": "#7f7f7f", "Left": "#009e73", "Right": "#e69f00"}
EPOCH_COLORS = {"sample": "#f4f4f4", "delay": "#e9e9e9", "response": "#ffffff"}
MODE_COLORS = {
    "criteria": "#222222",
    "criteria_nospec": "#d55e00",
    "criteria_popmean": "#cc79a7",
    "rate": "#888888",
    "random": "#bbbbbb",
    "logreg_all_units": "#56b4e9",
    "logreg_selected_units": "#0072b2",
}
# Status strip of Figure 1.  "selected" takes the region colour (see ``status_colors``); the other three
# stages are greys of decreasing darkness so the strip reads as a funnel even without a legend.
STATUS_COLORS = {"selected": None, "eligible": "#9a9a9a", "floor": "#cfcfcf", "below_floor": "#ededed"}
STATUS_ORDER = ("selected", "eligible", "floor", "below_floor")
STATUS_LABELS = {"selected": "selected", "eligible": "eligible, not selected",
                 "floor": ">= floor, < 2 criteria", "below_floor": "below floor"}

CRITERIA = [("c_selectivity", "S", "choice selectivity"), ("c_coupling", "C", "delay->response coupling"),
            ("c_spectral", "W", "wavelet band-power selectivity"), ("c_ramp", "R", "ramping")]

FONT_STACK = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def status_colors(region: str) -> list[str]:
    """Status colours in ``STATUS_ORDER`` with the region colour substituted for "selected"."""
    return [REGION_COLORS[region] if STATUS_COLORS[s] is None else STATUS_COLORS[s] for s in STATUS_ORDER]


def status_cmap(region: str) -> ListedColormap:
    """Categorical colormap (code 0..3 = ``STATUS_ORDER``) for the status strip of one region."""
    return ListedColormap(status_colors(region), name=f"status_{region}")


def apply_style(dpi: int | float | None = None) -> None:
    """Install the journal typography.  ``dpi`` (from ``cfg.figures.dpi``) only sets the screen dpi; the
    output resolution is passed to ``savefig`` by every figure module."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1.0,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "figure.dpi": float(dpi) if dpi else 110.0,
    })


def panel_label(ax, text: str, x: float = -0.08, y: float = 1.04, **kw) -> None:
    """9 pt bold panel letter in axes coordinates (defaults sit just outside the top-left corner)."""
    style = dict(fontsize=9, fontweight="bold", va="bottom", ha="left", clip_on=False)
    style.update(kw)
    ax.text(x, y, text, transform=ax.transAxes, **style)
