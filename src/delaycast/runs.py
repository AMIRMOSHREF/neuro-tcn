"""Where runs live on disk and how they are loaded back.

Layout under ``output_dir`` (default ``outputs/delaycast``)::

    runs/within/<mode>[_<variant>]/seed<k>/            within-session split, one directory per seed
    runs/cross_session/<mode>/seed<k>/holdout_<session>/   leave-one-session-out (aggregate results.json one level up)
    runs/cross_dataset/<mode>/seed<k>/holdout_<A|B>/       train on one dataset, adapt+test on the other (aggregate one level up)
    runs/negative_control/<mode>/seed<k>/                  labels permuted within session before everything
    selection/                                             descriptive selection tables (all trials)
    figures/

``mode`` is the neuron set (criteria | rate | random); ``variant`` is a model ablation
(nospec | popmean; absent = full model).  Every leaf directory holds ``results.json``, ``model.pt``,
``splits.json``, ``test_predictions.csv``, ``attention.npz`` and ``neuron_importance.csv``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES

KINDS = ("within", "cross_session", "cross_dataset", "negative_control")


@dataclass(frozen=True)
class RunRef:
    kind: str            # within | cross_session | cross_dataset | negative_control
    mode: str            # criteria | rate | random
    variant: str         # "" | nospec | popmean
    seed: int
    path: Path           # directory holding results.json (aggregate for cross_* runs)

    @property
    def name(self) -> str:
        """Key used by figures / report: e.g. 'criteria', 'criteria_nospec', 'cross_dataset/criteria'."""
        base = self.mode + (f"_{self.variant}" if self.variant else "")
        return base if self.kind == "within" else f"{self.kind}/{base}"


def run_dir(out_dir: Path, kind: str, mode: str, variant: str = "", seed: int = 0, holdout_tag: str | None = None) -> Path:
    d = Path(out_dir) / "runs" / kind / (mode + (f"_{variant}" if variant else "")) / f"seed{int(seed)}"
    return d / f"holdout_{holdout_tag}" if holdout_tag else d


def _parse_mode_variant(name: str) -> tuple[str, str]:
    parts = name.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def list_runs(out_dir: Path) -> list[RunRef]:
    out = []
    root = Path(out_dir) / "runs"
    if not root.is_dir():
        return out
    for kind_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in KINDS):
        for mv in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            mode, variant = _parse_mode_variant(mv.name)
            for sd in sorted(p for p in mv.iterdir() if p.is_dir() and p.name.startswith("seed")):
                if (sd / "results.json").is_file():
                    out.append(RunRef(kind_dir.name, mode, variant, int(sd.name[4:]), sd))
    return out


def load_results(out_dir: Path) -> dict[str, list[dict]]:
    """name -> list of results.json dicts (one per seed, ascending seed)."""
    res: dict[str, list[dict]] = {}
    for ref in sorted(list_runs(out_dir), key=lambda r: (r.kind, r.mode, r.variant, r.seed)):
        with open(ref.path / "results.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("seed", ref.seed)
        d["_run_dir"] = str(ref.path)
        d["_name"] = ref.name
        res.setdefault(ref.name, []).append(d)
    return res


def aggregate_holdouts(parent: Path, cfg=None) -> dict | None:
    """Pool the per-holdout test predictions of a cross_* run into one results.json (chance level from the
    pooled predictions, never a hard-coded number); per-session metrics are kept from the leaves."""
    from .evaluate import chance_level, metrics, per_session_metrics
    parent = Path(parent)
    leaves = sorted(p for p in parent.glob("holdout_*") if (p / "results.json").is_file())
    if not leaves:
        return None
    preds, per_session, forecasts, leaf_results = [], [], [], {}
    for d in leaves:
        p = d / "test_predictions.csv"
        if p.is_file():
            preds.append(pd.read_csv(p))
        with open(d / "results.json", "r", encoding="utf-8") as f:
            r = json.load(f)
        leaf_results[d.name] = r
        per_session += r.get("per_session", [])
        forecasts += r.get("forecast", {}).get("per_session", [])
    if not preds:
        return None
    df = pd.concat(preds, ignore_index=True)
    y = df.label.to_numpy(int)
    prob = df[[f"p_{c}" for c in CLASSES]].to_numpy(float)
    logits = np.log(np.clip(prob, 1e-9, None))
    sessions = df.session.to_numpy()
    rng = np.random.default_rng(0)
    n_shuffles = int(cfg.evaluate.get_path("n_shuffles", 1000)) if cfg is not None else 1000
    first = next(iter(leaf_results.values()))
    agg = {k: first.get(k) for k in ("mode", "seed", "eval_mode", "negative_control", "spectral_branch", "occlusion")}
    agg.update({"aggregate_of": [d.name for d in leaves], "classification": metrics(y, logits),
                "per_session": per_session_metrics(y, logits, sessions), "chance": chance_level(y, logits.argmax(1), sessions, n_shuffles, rng)})
    agg["chance_balanced_accuracy"] = {"mean": agg["chance"]["mean"], "p95": agg["chance"]["p95"]}
    from sklearn.metrics import confusion_matrix
    agg["confusion"] = confusion_matrix(y, logits.argmax(1), labels=list(range(len(CLASSES)))).tolist()
    # mean of the leaf-level analyses that are comparable across holdouts
    for key in ("context_sweep", "temporal_occlusion", "region_ablation", "baselines", "linear_sweep"):
        vals = [r.get(key) for r in leaf_results.values() if r.get(key)]
        if vals:
            agg[key] = _mean_rows(vals)
    fc = {}
    for k in first.get("forecast", {}):
        if k != "per_session":
            fc[k] = float(np.nanmean([r["forecast"][k] for r in leaf_results.values() if k in r.get("forecast", {})]))
    fc["per_session"] = forecasts
    agg["forecast"] = fc
    csis = [r.get("csi", {}).get("tau95_ms") for r in leaf_results.values()]
    csis = [c for c in csis if c is not None]
    if csis:
        agg["csi"] = {"tau95_ms": float(np.median(csis)), "tau95_per_holdout": csis}
    df.to_csv(parent / "test_predictions.csv", index=False)
    with open(parent / "results.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return agg


def _mean_rows(list_of_rowlists: list[list[dict]]) -> list[dict]:
    """Element-wise mean of numeric fields across lists of row dicts with the same structure (per_session dicts merged)."""
    n = min(len(v) for v in list_of_rowlists)
    out = []
    for i in range(n):
        rows = [v[i] for v in list_of_rowlists]
        merged = {}
        for k in rows[0]:
            vals = [r.get(k) for r in rows]
            if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
                merged[k] = float(np.nanmean([v for v in vals if v is not None]))
            elif isinstance(vals[0], dict) and k == "per_session":
                merged[k] = {kk: vv for v in vals if isinstance(v, dict) for kk, vv in v.items()}
            elif isinstance(vals[0], list) and k == "per_session":
                merged[k] = [row for v in vals if isinstance(v, list) for row in v]
            else:
                merged[k] = vals[0]
        out.append(merged)
    return out
