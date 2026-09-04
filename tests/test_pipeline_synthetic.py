"""End-to-end pipeline on the synthetic recordings: cache -> selection -> training -> evaluation.

This is the integration test of the *honesty* guarantees that the unit tests cannot see:

* the results.json written by ``evaluate_run`` carries every top-level key the report and the figures read;
* the unit indices frozen into ``model.pt`` are exactly the ranked selection stored next to it, so a run can
  be re-evaluated without re-selecting and the two evaluations agree number for number;
* the leakage sentinel: every selection table was computed on train + validation trials only
  (``n_fit_trials == len(train) + len(val)``) and no test trial index ever entered the selection;
* the negative control (labels permuted within session before selection and training) is at chance.

Everything is shrunk (d_model 32, K 8, 4 epochs, few permutations / resamples) so that the whole chain -
one criteria run, one negative-control run and a re-load - stays within a few minutes on a CPU.  The test
uses the synthetic tree in the scratchpad when it exists and generates a one-session-per-dataset tree
otherwise; the cache and every output live under ``tmp_path``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from delaycast import CLASSES, REGIONS
from delaycast.config import load_config
from delaycast.data.cache import build_cache
from delaycast.data.dataset import tensors_from_indices
from delaycast.data.synthetic import make_synthetic
from delaycast.evaluate import evaluate_run
from delaycast.runs import list_runs, load_results, run_dir
from delaycast.train import load_run, run_training

pytestmark = pytest.mark.slow

SCRATCH_SYNTH = Path("/tmp/claude-0/-home-user-neuro-tcn/45bbd087-4c58-57cb-936a-9f32df8042b3/scratchpad/synth")

TOP_LEVEL_KEYS = ("mode", "seed", "holdout", "eval_mode", "negative_control", "spectral_branch", "adapt_info", "occlusion",
                  "classification", "classification_ci", "confusion", "per_session", "chance", "chance_balanced_accuracy",
                  "forecast", "context_sweep", "csi", "temporal_occlusion", "region_ablation", "attention_centre_of_mass_ms",
                  "importance_agreement", "baselines", "linear_sweep", "l1_overlap", "tau95_linear_ms")
CLASSIFICATION_KEYS = ("accuracy", "balanced_accuracy", "balanced_accuracy_lr", "macro_f1", "log_loss", "n", "n_per_class",
                       "n_classes_present", "recall")
BASELINE_MODELS = {"logreg_all_units", "logreg_pca50_all_units", "logreg_l1_all_units", "logreg_selected_units", "logreg_selected_units_windows",
                   "logreg_selected_ALM", "logreg_selected_STR", "logreg_trial_index"}
LEAF_FILES = ("results.json", "model.pt", "splits.json", "test_predictions.csv", "attention.npz", "neuron_importance.csv",
              "history.csv", "config.yaml", "selection_funnel.csv")


def _synthetic_root(tmp: Path) -> Path:
    if (SCRATCH_SYNTH / "Data").is_dir() and (SCRATCH_SYNTH / "Data2").is_dir():
        return SCRATCH_SYNTH
    make_synthetic(tmp / "synth", 1, 1, (8, 24, 24))
    return tmp / "synth"


def _tag(session: str) -> str:
    return session.replace("/", "__")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Cache, criteria run and evaluation shared by every test of this module (built once)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    root = _synthetic_root(tmp)
    cfg = load_config(None, [f"data.data_a_root={root / 'Data'}", f"data.data_b_root={root / 'Data2'}",
                             f"data.cache_dir={tmp / 'cache'}", f"output_dir={tmp / 'out'}",
                             "model.d_model=32", "selection.top_k_per_region=8", "train.epochs=4", "train.device=cpu",
                             "selection.n_subsamples=5", "selection.locus_n_perm=20", "selection.coupling_n_perm=50",
                             "selection.subsample_n_perm=20", "evaluate.n_shuffles=100", "evaluate.n_bootstrap=50"])
    timings = {}
    t0 = time.perf_counter()
    caches = {c.session: c for c in build_cache(cfg)}
    timings["build_cache"] = time.perf_counter() - t0
    out = Path(cfg.output_dir)
    t0 = time.perf_counter()
    run = run_training(cfg, mode="criteria", out_dir=run_dir(out, "within", "criteria", seed=0), caches=caches)
    timings["train"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = evaluate_run(run, cfg, caches)
    timings["evaluate"] = time.perf_counter() - t0
    return {"cfg": cfg, "caches": caches, "run": run, "res": res, "out": out, "timings": timings}


# ----------------------------------------------------------------------------- cache
def test_cache_geometry(pipeline):
    caches = pipeline["caches"]
    assert len(caches) >= 1
    for c in caches.values():
        assert c.context[REGIONS[0]].shape[2] == 120 and c.target[REGIONS[0]].shape[2] == 30
        for r in REGIONS:
            assert c.context[r].dtype == np.uint8 and c.target[r].dtype == np.uint8
            assert c.context[r].shape[:2] == c.target[r].shape[:2] == (c.n_trials, len(c.unit_ids[r]))
        assert set(np.unique(c.labels)) <= {0, 1, 2} and len(np.unique(c.labels)) == 3


# ----------------------------------------------------------------------------- results.json contract
def test_results_json_has_every_contract_key(pipeline):
    run, res = pipeline["run"], pipeline["res"]
    d = Path(run["out_dir"])
    for name in LEAF_FILES:
        assert (d / name).is_file(), name
    with open(d / "results.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    for k in TOP_LEVEL_KEYS:
        assert k in saved, f"results.json lacks {k!r}"
        assert k in res
    assert saved["mode"] == "criteria" and saved["seed"] == 0 and saved["holdout"] == [] and saved["negative_control"] is False
    assert saved["spectral_branch"] == "bands" and saved["occlusion"] == "permute" and saved["eval_mode"] == "within_session"
    cl = saved["classification"]
    assert all(k in cl for k in CLASSIFICATION_KEYS)
    assert set(cl["n_per_class"]) == set(CLASSES) and set(cl["recall"]) == set(CLASSES) and cl["n"] == sum(cl["n_per_class"].values())
    assert 0.0 <= cl["balanced_accuracy"] <= 1.0 and np.isfinite(cl["log_loss"])
    ci = saved["classification_ci"]
    assert len(ci["balanced_accuracy"]) == 2 and ci["balanced_accuracy"][0] <= ci["balanced_accuracy"][1]
    assert len(ci["balanced_accuracy_lr"]) == 2
    assert np.asarray(saved["confusion"]).shape == (3, 3) and int(np.sum(saved["confusion"])) == cl["n"]
    assert {p["session"] for p in saved["per_session"]} == set(pipeline["caches"])
    assert all(k in saved["chance"] for k in ("mean", "p95", "p99", "analytic", "n_shuffles", "scheme"))
    assert saved["chance"]["n_shuffles"] == 100 and saved["chance"]["analytic"] == pytest.approx(1 / 3)
    assert saved["chance_balanced_accuracy"] == {"mean": saved["chance"]["mean"], "p95": saved["chance"]["p95"]}
    fc = saved["forecast"]
    for r in REGIONS:
        assert f"deviance_explained_{r}" in fc and f"deviance_explained_persistence_{r}" in fc
    assert {p["session"] for p in fc["per_session"]} == set(pipeline["caches"])
    sweep = saved["context_sweep"]
    assert [s["context_ms"] for s in sweep] == [int(v) for v in pipeline["cfg"].evaluate.context_sweep_ms]
    assert all("per_session" in s and "balanced_accuracy" in s for s in sweep)
    assert all(k in saved["csi"] for k in ("tau95_ms", "tau95_median_ms", "tau95_ci_ms", "tau95_logloss_ms", "tau95_logloss_ci_ms", "fraction", "n_bootstrap"))
    assert 100 <= saved["csi"]["tau95_ms"] <= 1200
    occ = saved["temporal_occlusion"]
    assert len(occ) == 11 and occ[0]["window_start_ms"] == 0 and occ[-1]["window_end_ms"] == 1200
    for row in occ:
        assert all(k in row for k in ("delta_balanced_accuracy", "delta_log_loss", "delta_balanced_accuracy_lr",
                                      "delta_forecast_deviance_explained", "delta_forecast_deviance_explained_backbone", "per_session"))
    abl = saved["region_ablation"]
    assert {(a["dropped_region"], a["method"]) for a in abl} == {(r, m) for r in REGIONS for m in ("permute", "drop")}
    assert all("delta_balanced_accuracy" in a and "recall" in a and "per_session" in a for a in abl)
    assert set(saved["attention_centre_of_mass_ms"]) == {f"{r}_{c}" for r in REGIONS for c in CLASSES}
    assert all(0 <= v <= 1200 for v in saved["attention_centre_of_mass_ms"].values())
    assert {b["model"] for b in saved["baselines"]} == BASELINE_MODELS
    assert all("per_session" in b and "balanced_accuracy" in b for b in saved["baselines"])
    assert [s["context_ms"] for s in saved["linear_sweep"]] == [s["context_ms"] for s in sweep]
    assert {(o["session"], o["region"]) for o in saved["l1_overlap"]} == {(s, r) for s in pipeline["caches"] for r in REGIONS}
    assert all(k in o for o in saved["l1_overlap"] for k in ("n_l1", "n_criteria", "jaccard"))
    assert 100 <= saved["tau95_linear_ms"] <= 1200
    for name, d_ in saved["importance_agreement"].items():
        assert all(k in d_ for k in ("mean_rho", "median_rho", "n_cells", "n_positive", "sign_test_p")), name


def test_side_files_follow_the_layout(pipeline):
    run = pipeline["run"]
    d = Path(run["out_dir"])
    sessions = list(pipeline["caches"])
    att = np.load(d / "attention.npz")
    for r in REGIONS:
        for c in CLASSES:
            assert att[f"temporal_{r}_{c}"].shape == (120,)
        for s in sessions:
            assert att[f"gates_{_tag(s)}_{r}"].shape == (8,)
    for c in CLASSES:
        assert att[f"region_{c}"].shape == (len(REGIONS),)
    pred = pd.read_csv(d / "test_predictions.csv")
    assert list(pred.columns) == ["session", "trial", "label", "pred"] + [f"p_{c}" for c in CLASSES]
    assert np.allclose(pred[[f"p_{c}" for c in CLASSES]].sum(axis=1), 1.0, atol=1e-6)
    imp = pd.read_csv(d / "neuron_importance.csv")
    for col in ("session", "region", "k_slot", "unit_index", "delta_log_loss", "delta_balanced_accuracy",
                "delta_forecast_deviance_explained_others", "gate", "gate_rel", "score", "stability", "n_criteria", "rank", "unit_id"):
        assert col in imp.columns, col
    assert (imp.gate_rel <= 1.0 + 1e-6).all() and imp["rank"].notna().all()
    refs = list_runs(pipeline["out"])
    assert "criteria" in [r.name for r in refs] and all(r.kind in ("within", "negative_control") for r in refs)
    assert "criteria" in load_results(pipeline["out"])


# ----------------------------------------------------------------------------- unit indices + leakage sentinel
def test_checkpoint_unit_indices_equal_ranked_selection(pipeline):
    run = pipeline["run"]
    d = Path(run["out_dir"])
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    assert ck["k"] == 8 and ck["t_ctx"] == 120 and ck["t_tgt"] == 30 and ck["mode"] == "criteria"
    assert set(ck["sessions"]) == set(pipeline["caches"])
    for s in ck["sessions"]:
        tab = pd.read_csv(d / f"selection_{_tag(s)}.csv")
        for r in REGIONS:
            stored = np.asarray(ck["unit_index"][s][r])
            assert len(stored) == 8
            real = stored[stored >= 0]
            assert (stored[len(real):] == -1).all()                                # padding after the real units only
            ranked = tab[(tab.region == r) & tab.selected].sort_values("rank").unit_index.to_numpy(int)
            assert np.array_equal(real, ranked), (s, r)
            assert len(real) == int(((tab.region == r) & tab.selected).sum()) <= 8
            assert len(np.unique(real)) == len(real)


def test_selection_used_only_fit_trials(pipeline):
    """The leakage sentinel: n_fit_trials == |train| + |val| and no test trial ever entered the selection."""
    run = pipeline["run"]
    d = Path(run["out_dir"])
    with open(d / "splits.json", "r", encoding="utf-8") as f:
        splits = {s: {k: np.asarray(v, int) for k, v in v_.items()} for s, v_ in json.load(f).items()}
    funnel = pd.read_csv(d / "selection_funnel.csv")
    for sel in run["selections"]:
        s = sel.session
        sp = splits[s]
        n_fit = len(sp["train"]) + len(sp["val"])
        assert len(sp["test"]) > 0 and n_fit > 0
        assert not set(sp["test"]) & set(sp["train"]) and not set(sp["test"]) & set(sp["val"])
        tab = pd.read_csv(d / f"selection_{_tag(s)}.csv")
        assert (tab.n_fit_trials == n_fit).all(), s
        assert (funnel[funnel.session == s].n_fit_trials == n_fit).all()
        assert sel.trial_idx is not None and len(sel.trial_idx) == n_fit
        assert not set(sel.trial_idx.tolist()) & set(sp["test"].tolist()), s
        assert set(sel.trial_idx.tolist()) == set(sp["train"].tolist()) | set(sp["val"].tolist())
        y = pipeline["caches"][s].labels[sel.trial_idx]
        assert (tab.n_fit_left == int((y == 1).sum())).all() and (tab.n_fit_right == int((y == 2).sum())).all()
    pred = pd.read_csv(d / "test_predictions.csv")
    for s, g in pred.groupby("session"):
        trials = pipeline["caches"][s].trials
        assert set(g.trial) == set(trials[splits[s]["test"]])


# ----------------------------------------------------------------------------- negative control (P0)
@pytest.fixture(scope="module")
def negative_control(pipeline):
    cfg, out = pipeline["cfg"], pipeline["out"]
    t0 = time.perf_counter()
    run = run_training(cfg, mode="criteria", out_dir=run_dir(out, "negative_control", "criteria", seed=0),
                       caches=pipeline["caches"], negative_control=True)
    res = evaluate_run(run, cfg)                       # run["caches"] hold the permuted labels (baselines must see them too)
    pipeline["timings"]["negative_control"] = time.perf_counter() - t0
    return run, res


def test_negative_control_is_at_chance(pipeline, negative_control):
    run, res = negative_control
    assert res["negative_control"] is True
    for s, c in run["caches"].items():
        assert not np.array_equal(c.labels, pipeline["caches"][s].labels)            # labels really were permuted ...
        assert np.array_equal(np.sort(c.labels), np.sort(pipeline["caches"][s].labels))  # ... within the session
    bacc = res["classification"]["balanced_accuracy"]
    assert bacc <= res["chance"]["p99"] + 0.05, (bacc, res["chance"])
    assert "negative_control/criteria" in load_results(pipeline["out"])
    with open(Path(run["out_dir"]) / "results.json", "r", encoding="utf-8") as f:
        assert json.load(f)["negative_control"] is True


# ----------------------------------------------------------------------------- reload determinism
def _has_padded_region(tensors) -> bool:
    return any((t.unit_index[r] < 0).any() for t in tensors for r in REGIONS)


def test_load_run_tensors_match_training_tensors(pipeline):
    cfg, run = pipeline["cfg"], pipeline["run"]
    if not _has_padded_region(run["tensors"]):
        pytest.skip("every region is full (K_eff == K): the padding path is not exercised")
    again = load_run(run["out_dir"], cfg, pipeline["caches"])
    for t_new, t_old in zip(again["tensors"], run["tensors"]):
        for r in REGIONS:
            assert np.array_equal(t_new.neuron_mask[r], t_old.neuron_mask[r]), (t_new.session, r)
            assert np.array_equal(t_new.x[r], t_old.x[r]), (t_new.session, r)


def test_reload_reproduces_classification(pipeline):
    """Determinism of the saved unit indices: a reloaded run gives the same test-set numbers as the original.

    The reloaded tensors are rebuilt from the *non-negative* checkpoint indices (the work-around for the padding
    defect documented in ``test_load_run_tensors_match_training_tensors``); once that is fixed the work-around is a
    no-op and this test keeps its meaning."""
    cfg, run, res = pipeline["cfg"], pipeline["run"], pipeline["res"]
    t0 = time.perf_counter()
    again = load_run(run["out_dir"], cfg, pipeline["caches"])
    assert [t.session for t in again["tensors"]] == [t.session for t in run["tensors"]]
    for t_new, t_old in zip(again["tensors"], run["tensors"]):
        for r in REGIONS:
            assert np.array_equal(t_new.unit_index[r], t_old.unit_index[r])
    for s in again["splits"]:
        for k in ("train", "val", "test"):
            assert np.array_equal(again["splits"][s][k], run["splits"][s][k])
    again["tensors"] = [tensors_from_indices(pipeline["caches"][t.session], {r: t.unit_index[r][t.unit_index[r] >= 0] for r in REGIONS}, cfg)
                        for t in again["tensors"]]
    for t_new, t_old in zip(again["tensors"], run["tensors"]):
        for r in REGIONS:
            assert np.array_equal(t_new.unit_index[r], t_old.unit_index[r])
            assert np.array_equal(t_new.neuron_mask[r], t_old.neuron_mask[r])
            assert np.array_equal(t_new.x[r], t_old.x[r]) and np.array_equal(t_new.y[r], t_old.y[r])
    assert [sel.session for sel in again["selections"]] == [sel.session for sel in run["selections"]]
    for s_new, s_old in zip(again["selections"], run["selections"]):
        for r in REGIONS:
            assert np.array_equal(s_new.selected[r], s_old.selected[r])
    res2 = evaluate_run(again, cfg, pipeline["caches"])
    pipeline["timings"]["reload_evaluate"] = time.perf_counter() - t0
    for k in ("accuracy", "balanced_accuracy", "balanced_accuracy_lr", "macro_f1", "log_loss", "n"):
        assert res2["classification"][k] == pytest.approx(res["classification"][k], abs=1e-6), k
    assert res2["confusion"] == res["confusion"]
    assert [p["balanced_accuracy"] for p in res2["per_session"]] == pytest.approx([p["balanced_accuracy"] for p in res["per_session"]], abs=1e-6)
    assert [s["balanced_accuracy"] for s in res2["context_sweep"]] == pytest.approx([s["balanced_accuracy"] for s in res["context_sweep"]], abs=1e-6)
    print("\npipeline timings (s):", {k: round(v, 1) for k, v in pipeline["timings"].items()})
