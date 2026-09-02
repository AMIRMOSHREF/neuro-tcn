import matplotlib as mpl

EPOCH_COLORS = {"sample": "#f2e8cf", "delay": "#dbe9f6", "response": "#e6f4e6"}
CLASS_COLORS = {"Ignore": "#7f7f7f", "Left": "#1b9e77", "Right": "#d95f02"}
CRITERIA = [("c_selectivity", "S", "choice selectivity"), ("c_coupling", "C", "delay->response coupling"),
            ("c_spectral", "W", "wavelet band-power selectivity"), ("c_ramp", "R", "ramping")]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def panel_label(ax, text: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
