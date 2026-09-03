"""Claims report on a fabricated run layout: verdicts follow the contract's rule and every prediction is written.

The fixtures build the ``runs/<kind>/<mode>/seed0/results.json`` layout that ``delaycast.runs.load_results``
reads, with six sessions of per-session numbers chosen so that each prediction lands on a known verdict.
The key structure mirrors the synthetic ``run_criteria/results.json`` (see CONTRACTS.md), reduced to the keys
the report consumes, because the report must tolerate exactly this kind of partial results file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from delaycast import REGIONS
from delaycast.config import load_config

SESSIONS = ["A/Session1", "A/Session2", "B/sub-100001_ses-20190101T100000", "B/sub-100001_ses-20190102T100000",
            "B/sub-100002_ses-20190103T100000", "B/sub-100003_ses-20190104T100000"]
CRIT = [0.85, 0.88, 0.82, 0.90, 0.86, 0.84]
RATE_LOW = [0.60, 0.65, 0.58, 0.62, 0.61, 0.63]
RATE_MIXED = [c + d for c, d in zip(CRIT, [0.02, -0.02, 0.03, -0.03, 0.01, -0.01])]
RANDOM = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53]
PREDICTIONS = ("P0", "P1a", "P1b", "P2", "P3", "P4", "P5a", "P5b", "P6", "P7", "P8")


def _cls(bacc: float, lr: float | None = None, n_ignore: int = 40, ignore_recall: float = 0.6) -> dict:
    return {"accuracy": bacc, "balanced_accuracy": bacc, "balanced_accuracy_lr": lr if lr is not None else bacc,
            "macro_f1": bacc, "log_loss": 0.8, "n": n_ignore + 120, "n_per_class": {"Ignore": n_ignore, "Left": 60, "Right": 60},
            "n_classes_present": 3, "recall": {"Ignore": ignore_recall, "Left": bacc, "Right": bacc}}


def _per_session(values: list[float], lr: list[float] | None = None) -> list[dict]:
    lr = lr or values
    return [{"session": s, **_cls(v, l, n_ignore=7, ignore_recall=0.6)} for s, v, l in zip(SESSIONS, values, lr)]


def make_results(bacc: list[float], mode: str, *, chance_p95: float = 0.42, negative_control: bool = False,
                 full: bool = False, seed: int = 0) -> dict:
    """A results.json dict; ``full=True`` adds the analyses only the criteria run needs (baselines, forecast,
    CSI, occlusion, ablation, importance agreement)."""
    pooled = float(np.mean(bacc))
    res = {"mode": mode, "seed": seed, "holdout": [], "eval_mode": "within_session", "negative_control": negative_control,
           "classification": _cls(pooled), "classification_ci": {"balanced_accuracy": [pooled - 0.05, pooled + 0.05]},
           "per_session": _per_session(bacc),
           "chance": {"mean": 0.335, "p95": chance_p95, "p99": chance_p95 + 0.03, "analytic": 1 / 3, "n_shuffles": 200,
                      "scheme": "within_session_label_permutation"},
           "chance_balanced_accuracy": {"mean": 0.335, "p95": chance_p95}}
    if not full:
        return res
    lin_all = [0.90] * 6
    lin_sel = [0.90, 0.91, 0.89, 0.90, 0.92, 0.90]
    res["baselines"] = [
        {"model": "logreg_all_units", **_cls(0.9), "per_session": _per_session(lin_all)},
        {"model": "logreg_selected_units", **_cls(0.9), "per_session": _per_session(lin_sel)},
        {"model": "logreg_trial_index", **_cls(0.35), "per_session": _per_session([0.35] * 6)},
    ]
    # DelayCAST Left/Right accuracy matches the linear decoder on the same units -> P1b non-inferior
    for row, l in zip(res["per_session"], lin_sel):
        row["balanced_accuracy_lr"] = l
    fc_rows = []
    for i, s in enumerate(SESSIONS):
        row = {"session": s}
        for r in REGIONS:
            row[f"deviance_explained_{r}"] = -0.05 - 0.01 * i
            row[f"deviance_explained_persistence_{r}"] = -0.50 - 0.02 * i
        fc_rows.append(row)
    res["forecast"] = {**{f"deviance_explained_{r}": -0.07 for r in REGIONS},
                       **{f"deviance_explained_persistence_{r}": -0.55 for r in REGIONS}, "per_session": fc_rows}
    res["csi"] = {"tau95_ms": 300.0, "tau95_median_ms": 300.0, "tau95_ci_ms": [200.0, 400.0], "tau95_logloss_ms": 300.0,
                  "tau95_logloss_ci_ms": [200.0, 500.0], "fraction": 0.95, "n_bootstrap": 100}
    res["tau95_linear_ms"] = 300.0
    windows = [(0.0, 400.0), (400.0, 800.0), (800.0, 1200.0)]
    res["temporal_occlusion"] = []
    for start, end in windows:
        last = end == 1200.0
        per = {s: (-0.20 - 0.01 * i if last else -0.02 + 0.005 * i) for i, s in enumerate(SESSIONS)}
        res["temporal_occlusion"].append({"window_start_ms": start, "window_end_ms": end,
                                          "delta_balanced_accuracy": float(np.mean(list(per.values()))),
                                          "delta_log_loss": 0.1 if last else 0.01, "delta_balanced_accuracy_lr": -0.2 if last else -0.02,
                                          "per_session": per})
    res["region_ablation"] = []
    for method in ("permute", "drop"):
        for r in REGIONS:
            hurt = -0.30 if r.startswith("ALM") else -0.05
            per = {s: hurt - 0.01 * i for i, s in enumerate(SESSIONS)}
            res["region_ablation"].append({"dropped_region": r, "method": method, "balanced_accuracy": pooled + hurt,
                                           "balanced_accuracy_lr": pooled + hurt, "delta_balanced_accuracy": hurt,
                                           "delta_balanced_accuracy_lr": hurt, "per_session": per})
    res["importance_agreement"] = {
        "importance_vs_score": {"mean_rho": 0.5, "median_rho": 0.55, "n_cells": 12, "n_positive": 11, "sign_test_p": 0.003},
        "gate_vs_score": {"mean_rho": -0.1, "median_rho": -0.1, "n_cells": 12, "n_positive": 5, "sign_test_p": 0.8},
        "gate_vs_importance": {"mean_rho": 0.1, "median_rho": 0.1, "n_cells": 12, "n_positive": 7, "sign_test_p": 0.4},
        "importance_vs_stability": {"mean_rho": 0.2, "median_rho": 0.2, "n_cells": 12, "n_positive": 8, "sign_test_p": 0.2},
    }
    return res


def _write_run(out: Path, kind: str, mode: str, res: dict, with_selection: bool = False) -> Path:
    d = out / "runs" / kind / mode / f"seed{res['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f)
    if with_selection:
        funnel = []
        for s in SESSIONS:
            rows = []
            for r in REGIONS:
                for u in range(10):
                    selected = u < 3
                    eligible = u < 7
                    rows.append({"region": r, "unit_index": u, "unit_id": u, "selected": selected, "eligible": eligible,
                                 "c_coupling": (u < 2) or (u == 8), "stability": 0.9 if selected else 0.3})
                funnel.append({"session": s, "region": r, "recorded": 10, "pass_floor": 9, "eligible": 7, "stable": 3, "selected": 3,
                               "K": 16, "filled_by_score": 0, "expected_false_selections_bound": 0.5, "n_fit_trials": 80,
                               "phi_SC": 0.1, "phi_SW": 0.8, "phi_SR": 0.3, "phi_CW": 0.0, "phi_CR": 0.1, "phi_WR": 0.2})
            pd.DataFrame(rows).to_csv(d / f"selection_{s.replace('/', '__')}.csv", index=False)
        pd.DataFrame(funnel).to_csv(d / "selection_funnel.csv", index=False)
    return d


def _write_summary(out: Path, phi_sw: float) -> None:
    rows = []
    for s in SESSIONS:
        row = {"session": s, "n_units": 40, "n_floor": 36, "n_eligible": 28, "n_selected": 12, "n_trials_used": 100}
        for r in REGIONS:
            row[f"sel_{r}"], row[f"tot_{r}"] = 3, 10
        row.update({"frac_selectivity": 0.3, "frac_coupling": 0.25, "frac_spectral": 0.1, "frac_ramp": 0.3, "frac_locus": 0.2,
                    "frac_ignore": 0.2, "median_stability_selected": 0.95, "median_onset_ms_selected": 150.0,
                    "frac_sustained_to_go_selected": 0.9, "null_median_stability_max": 0.1, "phi_SC": 0.1, "phi_SW": phi_sw,
                    "phi_SR": 0.3, "phi_CW": 0.0, "phi_CR": 0.1, "phi_WR": 0.2})
        rows.append(row)
    (out / "selection").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "selection" / "summary.csv", index=False)


def build_out_dir(root: Path, rate: list[float], phi_sw: float = 0.8) -> Path:
    out = root / "out"
    _write_run(out, "within", "criteria", make_results(CRIT, "criteria", full=True), with_selection=True)
    _write_run(out, "within", "rate", make_results(rate, "rate"))
    _write_run(out, "within", "random", make_results(RANDOM, "random"))
    _write_run(out, "negative_control", "criteria", make_results([0.34] * 6, "criteria", chance_p95=0.42, negative_control=True))
    _write_summary(out, phi_sw)
    return out


@pytest.fixture(scope="module")
def cfg():
    return load_config(None)


def _run_report(cfg, out: Path) -> tuple[Path, dict, str]:
    from delaycast.report import write_report

    md = write_report(cfg, out)
    rep = json.load(open(out / "report.json", encoding="utf-8"))
    return md, rep, md.read_text(encoding="utf-8")


def test_report_supported_scenario(cfg, tmp_path):
    out = build_out_dir(tmp_path, RATE_LOW)
    md, rep, text = _run_report(cfg, out)
    assert md == out / "REPORT.md" and md.is_file() and (out / "report.json").is_file()
    v = rep["verdicts"]
    assert set(v) == set(PREDICTIONS)
    assert v["P2"] == "supported"
    p2 = rep["predictions"]["P2"]
    for name in ("rate", "random"):
        c = p2["comparisons"][name]
        assert c["n_sessions"] == 6 and c["n_seeds"] == 1
        assert c["p"] < 0.05 and c["ci"][0] > 0 and c["verdict"] == "supported"
        assert [r["session"] for r in c["table"]] == sorted(SESSIONS)
    assert abs(p2["comparisons"]["rate"]["mean_diff"] - (np.mean(CRIT) - np.mean(RATE_LOW))) < 1e-9
    # negative control below its chance p95; non-inferiority of the selected-K decoders; the other tests
    assert v["P0"] == "supported"
    assert v["P1a"] == "supported" and v["P1b"] == "supported"
    assert rep["predictions"]["P1a"]["comparison"]["metric"] == "balanced_accuracy_lr"
    assert v["P3"] == "supported" and rep["predictions"]["P3"]["csi_pass"] and rep["predictions"]["P3"]["linear_tau95_pass"]
    assert v["P4"] == "supported"
    ce = rep["predictions"]["P4"]["coupling_enrichment"]
    assert ce["n_sessions"] == 6 and abs(ce["mean_frac_selected"] - 2 / 3) < 1e-9 and abs(ce["mean_frac_unselected"] - 0.0) < 1e-9
    assert v["P5a"] == "supported" and rep["predictions"]["P5a"]["comparison"]["method"] == "drop"
    assert v["P5b"] == "supported" and rep["predictions"]["P5b"]["n_ignore"] == 40 and not rep["predictions"]["P5b"]["confounded"]
    assert v["P6"] == "supported" and rep["predictions"]["P6"]["parts"]["gate_vs_score"] == "inconclusive"
    assert v["P7"] == "not run" and v["P8"] == "not run"
    # header
    h = rep["header"]
    assert h["n_sessions"] == 6 and h["n_sessions_by_dataset"] == {"A": 2, "B": 4}
    assert h["animals"]["A"] == ["Session1", "Session2"] and h["animals"]["B"] == ["sub-100001", "sub-100002", "sub-100003"]
    assert h["seeds"] == [0] and {r["run"] for r in h["runs"]} == {"criteria", "rate", "random", "negative_control/criteria"}
    assert h["test_trials_per_class"] == {"Ignore": 40, "Left": 60, "Right": 60}
    # selection summary + W-independence flag (phi_SW = 0.8 > 0.7)
    sel = rep["selection"]
    assert sel["w_independence"]["flag"] and "not independent" in sel["w_independence"]["message"]
    assert sel["per_region"]["ALM_L"] == {"recorded": 60, "selected": 18}
    assert sel["funnel_per_region"]["STR_R"]["selected"] == 18
    # the markdown mentions every prediction, the claim and the flag
    for pid in PREDICTIONS:
        assert f"| {pid}:" in text and f"### {pid}." in text
    assert "During the 1.2 s delay" in text and "W is not independent evidence in this dataset" in text
    assert "Wilcoxon p" in text and "bootstrap 95% CI" in text


def test_report_inconclusive_scenario(cfg, tmp_path):
    out = build_out_dir(tmp_path, RATE_MIXED, phi_sw=0.4)
    _, rep, text = _run_report(cfg, out)
    p2 = rep["predictions"]["P2"]
    c = p2["comparisons"]["rate"]
    assert c["n_sessions"] == 6 and c["verdict"] == "inconclusive" and c["p"] >= 0.05
    assert p2["comparisons"]["random"]["verdict"] == "supported"
    assert rep["verdicts"]["P2"] == "inconclusive"
    assert not rep["selection"]["w_independence"]["flag"]
    assert "**Verdict P2: inconclusive**" in text


def test_report_tolerates_empty_output_dir(cfg, tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    md, rep, text = _run_report(cfg, out)
    assert md.is_file()
    assert set(rep["verdicts"]) == set(PREDICTIONS) and set(rep["verdicts"].values()) == {"not run"}
    assert rep["header"]["n_sessions"] == 0 and rep["header"]["runs"] == []
    for pid in PREDICTIONS:
        assert f"### {pid}." in text
    json.dumps(rep)  # everything serialisable (no NaN leaked as float('nan'))


def test_report_too_few_sessions_is_not_testable(cfg, tmp_path):
    out = tmp_path / "out"
    crit = make_results(CRIT, "criteria")
    rate = make_results(RATE_LOW, "rate")
    crit["per_session"], rate["per_session"] = crit["per_session"][:2], rate["per_session"][:2]
    _write_run(out, "within", "criteria", crit)
    _write_run(out, "within", "rate", rate)
    _write_run(out, "within", "random", make_results(RANDOM, "random"))
    _, rep, _ = _run_report(cfg, out)
    assert rep["predictions"]["P2"]["comparisons"]["rate"]["verdict"] == "not testable"
    assert rep["predictions"]["P2"]["comparisons"]["rate"]["n_sessions"] == 2
    assert rep["verdicts"]["P2"] == "not testable"


def test_paired_test_rule():
    from delaycast.report import NOT_LOWER_MARGIN, paired_test, wilson_ci

    d = np.array([0.25, 0.23, 0.24, 0.28, 0.25, 0.21])
    r = paired_test(d, ">")
    assert r["verdict"] == "supported" and abs(r["p"] - 1 / 64) < 1e-9 and r["ci"][0] > 0 and r["n_sessions"] == 6
    assert paired_test(-d, "<")["verdict"] == "supported"
    assert paired_test(-d, ">")["verdict"] == "inconclusive"
    assert paired_test(np.zeros(6), ">")["verdict"] == "inconclusive"
    assert paired_test(d[:2], ">")["verdict"] == "not testable"
    # non-inferiority: tiny negative differences are fine, a real drop is not
    small = np.array([-0.005, 0.0, -0.01, 0.004, -0.002, 0.001])
    r = paired_test(small, "not_lower")
    assert r["verdict"] == "supported" and r["ci"][0] > -NOT_LOWER_MARGIN
    assert paired_test(small - 0.05, "not_lower")["verdict"] == "inconclusive"
    lo, hi = wilson_ci(24, 40)
    assert 0.44 < lo < 0.46 and 0.73 < hi < 0.75
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_compare_arms_seed_mean_and_missing_arm():
    from delaycast.report import compare_arms

    a = [make_results(CRIT, "criteria", seed=0), make_results([c + 0.02 for c in CRIT], "criteria", seed=1)]
    b = [make_results(RATE_LOW, "rate", seed=0)]
    c = compare_arms(a, b, "balanced_accuracy", ">", "criteria", "rate")
    assert c["n_seeds_a"] == 2 and c["n_seeds_b"] == 1 and c["n_seeds"] == 1
    assert abs(c["table"][0]["a"] - (CRIT[0] + 0.01)) < 1e-9  # mean over the two seeds first
    assert compare_arms(a, None, "balanced_accuracy", ">", "criteria", "rate")["verdict"] == "not run"
    # metric fallback: arms without Left/Right accuracy are compared on 3-class balanced accuracy
    for res in a + b:
        for row in res["per_session"]:
            row.pop("balanced_accuracy_lr")
    c = compare_arms(a, b, ("balanced_accuracy_lr", "balanced_accuracy"), ">", "criteria", "rate")
    assert c["metric"] == "balanced_accuracy" and c["verdict"] == "supported"
