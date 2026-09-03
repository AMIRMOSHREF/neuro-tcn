"""Figure 1 (raster + selection evidence) renders for the synthetic session, with and without model importance."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRATCH = Path("/tmp/claude-0/-home-user-neuro-tcn/45bbd087-4c58-57cb-936a-9f32df8042b3/scratchpad")
CACHE_NPZ = SCRATCH / "cache_synth" / "bin10_tbin50_ctx0_0_resp1500" / "A__Session1.npz"
SELECTION_CSV = SCRATCH / "out_synth" / "selection" / "A__Session1.csv"
IMPORTANCE_CSV = SCRATCH / "out_synth" / "run_criteria" / "neuron_importance.csv"
MIN_BYTES = 50_000

pytestmark = pytest.mark.skipif(not (CACHE_NPZ.is_file() and SELECTION_CSV.is_file()), reason="synthetic cache missing")


@pytest.fixture(scope="module")
def session():
    from delaycast.config import load_config
    from delaycast.data.cache import SessionCache

    cfg = load_config(None, [f"data.cache_dir={SCRATCH / 'cache_synth'}", f"output_dir={SCRATCH / 'out_synth'}"])
    cache = SessionCache.load(CACHE_NPZ)
    table = pd.read_csv(SELECTION_CSV)
    ti = int(np.flatnonzero(cache.labels == 1)[0])  # first Left trial
    npz = Path(cache.meta.npz_path.iloc[ti])
    if not npz.is_file():
        pytest.skip("synthetic trial NPZ missing")
    return cfg, cache, table, ti, npz


def _check_outputs(png: Path) -> None:
    pdf = png.with_suffix(".pdf")
    assert png.is_file() and png.stat().st_size > MIN_BYTES
    assert pdf.is_file() and pdf.stat().st_size > MIN_BYTES


def test_fig1_with_importance(session, tmp_path):
    from delaycast.figures.raster_selection import plot_raster_selection

    cfg, cache, table, ti, npz = session
    imp = pd.read_csv(IMPORTANCE_CSV) if IMPORTANCE_CSV.is_file() else None
    if imp is not None:
        imp = imp[imp.session == cache.session]
    t0 = time.perf_counter()
    out = plot_raster_selection(npz, cache, table, cfg, tmp_path / "fig1_imp.png",
                                trial_label=f"trial {int(cache.trials[ti])} - Left - first lick +0.25 s after go",
                                source_note="criteria on n=92 training trials (I 12 / L 40 / R 40); 50 stratified half-subsamples; BH-FDR q<0.05; K=32 per region",
                                importance=imp)
    print(f"fig1 with importance rendered in {time.perf_counter() - t0:.1f} s")
    assert out == tmp_path / "fig1_imp.png"
    _check_outputs(out)


def test_fig1_without_importance_recording_order_and_empty_region(session, tmp_path):
    """No importance frame, recording row order, a QC note, and one region with zero selected units."""
    from delaycast.figures.raster_selection import plot_raster_selection

    cfg, cache, table, ti, npz = session
    cfg = type(cfg)(cfg)
    cfg.set_path("figures.raster_row_order", "recording")
    tab = table.copy()
    tab.loc[tab.region == "STR_R", ["selected", "stable"]] = False
    tab.loc[tab.region == "STR_R", "rank"] = np.nan
    out = plot_raster_selection(npz, cache, tab, cfg, tmp_path / "fig1_noimp.png", trial_label="trial 4 - Left",
                                qc_note="trial 4 is not in the cache (excluded by QC: early lick)", importance=None)
    _check_outputs(out)


def test_status_sort_and_alignment_helpers():
    from delaycast.figures.raster_selection import _row_order, _unit_status

    tab = pd.DataFrame({"unit_index": [0, 1, 2, 3, 4], "selected": [False, True, False, True, False],
                        "eligible": [False, True, True, True, False], "pass_floor": [True, True, True, True, False],
                        "rank": [np.nan, 2, np.nan, 1, np.nan]})
    code, rank = _unit_status(tab, 5)
    assert code.tolist() == [2, 0, 1, 0, 3]
    assert _row_order(code, rank, "status").tolist() == [3, 1, 2, 0, 4]   # rank 1, rank 2, eligible, floor, below floor
    assert _row_order(code, rank, "recording").tolist() == [0, 1, 2, 3, 4]


def test_style_colour_system():
    from delaycast import REGION_COLORS
    from delaycast.figures.style import CLASS_COLORS, MODE_COLORS, STATUS_COLORS, status_colors

    assert REGION_COLORS == {"ALM_L": "#1f4e9c", "ALM_R": "#7fb2e5", "STR_L": "#6a2c91", "STR_R": "#c39bd3"}
    assert CLASS_COLORS["Left"] == "#009e73" and CLASS_COLORS["Right"] == "#e69f00"
    assert MODE_COLORS["criteria"] == "#222222"
    assert STATUS_COLORS["eligible"] == "#9a9a9a"
    assert status_colors("STR_L")[0] == REGION_COLORS["STR_L"]
