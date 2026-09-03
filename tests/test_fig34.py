"""Figures 3 and 4 rendered from a real ``results.json`` (the synthetic example run) and from degraded inputs.

The example run is used under several run names so that every code path of Figure 4 (within-session arms
with one and several seeds, a cross-dataset arm, baselines, chance band) is exercised; the degraded cases
check that missing files / keys turn into "not run" panels instead of exceptions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

EXAMPLE = Path("/tmp/claude-0/-home-user-neuro-tcn/45bbd087-4c58-57cb-936a-9f32df8042b3/scratchpad/out_synth/run_criteria")


def _cfg(dpi: int = 60):
    from delaycast.config import load_config
    cfg = load_config(None)
    cfg.set_path("figures.dpi", dpi)  # small previews keep the test fast
    return cfg


@pytest.fixture(scope="module")
def example():
    if not (EXAMPLE / "results.json").is_file():
        pytest.skip(f"example run {EXAMPLE} not available")
    with open(EXAMPLE / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _selections(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {p.stem[len("selection_"):].replace("__", "/"): pd.read_csv(p)
            for p in run_dir.glob("selection_*.csv") if p.stem != "selection_funnel"}


def test_fig3_from_example_run(example, tmp_path):
    from delaycast.figures.attention_fig import plot_attention
    sels = _selections(EXAMPLE)
    assert sels, "example run has no selection tables"
    out = plot_attention(EXAMPLE, example, sels, _cfg(), tmp_path / "fig3_attention.png")
    assert out.is_file() and out.stat().st_size > 0
    # several seeds: the occlusion map averages and shows the range
    out2 = plot_attention(EXAMPLE, [example, example], sels, _cfg(), tmp_path / "fig3_two_seeds.png")
    assert out2.is_file()


def test_fig4_from_example_run(example, tmp_path):
    from delaycast.figures.results_fig import plot_results
    results_by_run = {"criteria": [example, example], "rate": [example], "cross_dataset/criteria": [example]}
    out = plot_results(results_by_run, _cfg(), tmp_path / "fig4_results.png")
    assert out.is_file() and out.stat().st_size > 0


def test_fig3_fig4_degrade_gracefully(tmp_path):
    """No attention/importance files, no selection tables, results with almost no keys: still a figure."""
    from delaycast.figures.attention_fig import plot_attention
    from delaycast.figures.results_fig import plot_results
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    minimal = {"mode": "rate", "classification": {"balanced_accuracy": 0.5, "n": 10}}
    out3 = plot_attention(empty_run, minimal, {}, _cfg(), tmp_path / "fig3_empty.png")
    assert out3.is_file()
    out4 = plot_results({"rate": [minimal]}, _cfg(), tmp_path / "fig4_min.png")
    assert out4.is_file()
    out4b = plot_results({}, _cfg(), tmp_path / "fig4_nothing.png")
    assert out4b.is_file()


def test_load_results_alias(tmp_path):
    """``results_fig.load_results`` delegates to ``runs.load_results`` (new run layout)."""
    from delaycast.figures.results_fig import load_results
    d = tmp_path / "runs" / "within" / "criteria" / "seed0"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"mode": "criteria", "classification": {"balanced_accuracy": 0.6}}))
    res = load_results(tmp_path)
    assert list(res) == ["criteria"] and len(res["criteria"]) == 1
    assert res["criteria"][0]["seed"] == 0


def test_short_session_labels():
    from delaycast.figures._fig_common import short_session
    assert short_session("A/Session1") == "A-S1"
    assert short_session("B/sub-440957_ses-20190211T143614") == "B-440957-0211"
