"""Command-line entry point.  ``python -m delaycast <command> [--config ...] [--set key=value ...]``

Commands (see README for the full protocol):

  synth      write a small synthetic Data/Data2 tree for smoke tests
  inspect    list discovered sessions/trials and NPZ contents
  cache      bin every trial into per-session uint8 tensors (QC log with reasons)
  select     descriptive neuron selection on all trials (+ stability null), tables with reasons
  train      train + evaluate DelayCAST-Net: --modes criteria,rate,random --variants popmean,nospec --seeds 0,1,2
             --set train.eval_mode=within_session|cross_session|cross_dataset [--holdout ...] [--negative-control]
  evaluate   re-evaluate saved runs
  figures    Figures 1-4 (Fig 1 for the first or every session)
  figure1    Figure 1 for one specific NPZ trial file (--npz PATH)
  report     REPORT.md with the verdict on every prediction of the claim
  all        cache -> select -> train (within: modes + popmean control, seeds) -> cross_dataset -> negative control
             -> figures -> report   (``--quick``: one seed, within-session only)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES, REGIONS
from .config import Config, load_config

log = logging.getLogger("delaycast")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/delaycast.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set data.data_a_root=D:/Rodent/Data")
    p.add_argument("-v", "--verbose", action="store_true")


def _clone(cfg: Config, **overrides) -> Config:
    c = Config(copy.deepcopy(cfg.to_plain()))
    for k, v in overrides.items():
        c.set_path(k, v)
    return c


def _seeds(cfg: Config, args) -> list[int]:
    if getattr(args, "seeds", None):
        return [int(s) for s in str(args.seeds).split(",") if s.strip()]
    seeds = cfg.train.get_path("seeds", None)
    return [int(s) for s in seeds] if seeds else [int(cfg.train.seed)]


def _list(arg: str | None) -> list[str]:
    return [m.strip() for m in (arg or "").split(",") if m.strip()]


# ----------------------------------------------------------------------------- data commands
def cmd_synth(cfg: Config, args) -> None:
    from .data.synthetic import make_synthetic
    a, b = make_synthetic(args.out, n_sessions_a=args.sessions_a, n_sessions_b=args.sessions_b,
                          trials_per_class=tuple(args.trials_per_class), seed=int(cfg.train.seed))
    print(f"synthetic Dataset A -> {a}\nsynthetic Dataset B -> {b}")
    print("Run the pipeline on it with:\n"
          f"  python -m delaycast all --quick --set data.data_a_root={a} --set data.data_b_root={b} --set output_dir=outputs_synth --set data.cache_dir=cache_synth")


def cmd_inspect(cfg: Config, args) -> None:
    from .data.discovery import discover_all, summarize
    recs = discover_all(cfg)
    pd.set_option("display.width", 200)
    print(summarize(recs).to_string(index=False))
    if recs and args.npz_detail:
        data = np.load(recs[0].npz_path, allow_pickle=True)
        print(f"\nFirst NPZ: {recs[0].npz_path}")
        for k in data.files:
            arr = data[k]
            print(f"  {k:24s} shape={arr.shape} dtype={arr.dtype}")
        from .data.rasters import normalize_region
        regs, cnt = np.unique([normalize_region(r) or "unknown" for r in np.asarray(data["brain_region"]).astype(str)], return_counts=True)
        print("  regions:", dict(zip(regs, cnt)))


def cmd_cache(cfg: Config, args) -> None:
    from .data.cache import build_cache, cache_summary
    caches = build_cache(cfg, force=args.force)
    pd.set_option("display.width", 250)
    summ = cache_summary(caches)
    print(summ.to_string(index=False))
    if "MB" in summ:
        print(f"total in RAM: {summ.MB.sum():.0f} MB (uint8 counts)")


def cmd_select(cfg: Config, args) -> None:
    """Descriptive selection on ALL trials (never consumed by `train`, which re-selects on the training split)."""
    from .data.cache import load_cache
    from .features.selection import select_neurons, selection_summary
    caches = load_cache(cfg)
    out = Path(cfg.output_dir) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for c in caches:
        res = select_neurons(c, cfg, seed=int(cfg.train.seed))
        res.table.to_csv(out / f"{c.session.replace('/', '__')}.csv", index=False)
        results.append(res)
        log.info("%s: %s selected (%s)", c.session, res.n_selected(), "; ".join(f"{r}: {v}" for r, v in res.n_selected().items()))
    summ = selection_summary(results)
    summ.to_csv(out / "summary.csv", index=False)
    pd.concat([r.funnel for r in results]).to_csv(out / "funnel.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(f"\nper-unit tables with reasons written to {out}/ (descriptive: all trials; training runs re-select on their fit trials)")


# ----------------------------------------------------------------------------- training
def _dataset_of(session: str) -> str:
    return session.split("/", 1)[0]


def _train_one(cfg: Config, mode: str, variant: str, seed: int, kind: str, caches: dict, holdouts: dict[str, list[str]] | None,
               negative_control: bool) -> None:
    from .evaluate import evaluate_run
    from .runs import aggregate_holdouts, run_dir
    from .train import run_training
    cfg_v = _clone(cfg, **{"train.seed": seed})
    if variant == "nospec":
        cfg_v.set_path("model.spectral_branch", "none")
    elif variant == "popmean":
        cfg_v.set_path("model.spectral_branch", "popmean")
    elif variant:
        raise SystemExit(f"unknown variant {variant!r} (nospec | popmean)")
    if not holdouts:
        d = run_dir(cfg.output_dir, kind, mode, variant, seed)
        log.info("=== run %s/%s%s seed %d -> %s", kind, mode, f"_{variant}" if variant else "", seed, d)
        run = run_training(cfg_v, mode=mode, out_dir=d, caches=caches, negative_control=negative_control)
        evaluate_run(run, cfg_v, caches)
        return
    for tag, sessions in holdouts.items():
        d = run_dir(cfg.output_dir, kind, mode, variant, seed, holdout_tag=tag)
        log.info("=== run %s/%s%s seed %d holdout %s (%d sessions) -> %s", kind, mode, f"_{variant}" if variant else "", seed, tag, len(sessions), d)
        run = run_training(cfg_v, mode=mode, out_dir=d, caches=caches, holdout=sessions, negative_control=negative_control)
        evaluate_run(run, cfg_v, caches)
    agg = aggregate_holdouts(run_dir(cfg.output_dir, kind, mode, variant, seed), cfg_v)
    if agg:
        print(f"{kind}/{mode}{'_' + variant if variant else ''} seed {seed}: pooled balanced accuracy "
              f"{agg['classification']['balanced_accuracy']:.3f} (chance p95 {agg['chance']['p95']:.3f})")


def _holdout_plan(cfg: Config, caches: dict, args) -> tuple[str, dict[str, list[str]] | None]:
    """Returns (kind, {holdout_tag: [sessions]}) for the configured eval mode."""
    mode = str(cfg.train.eval_mode)
    if getattr(args, "negative_control", False):
        return "negative_control", None
    if mode == "within_session":
        return "within", None
    if mode == "cross_session":
        sessions = _list(getattr(args, "holdout", None)) or list(caches)
        missing = [s for s in sessions if s not in caches]
        if missing:
            raise SystemExit(f"unknown holdout session(s): {missing}; known: {list(caches)}")
        return "cross_session", {s.replace("/", "__"): [s] for s in sessions}
    if mode == "cross_dataset":
        by_ds: dict[str, list[str]] = {}
        for s in caches:
            by_ds.setdefault(_dataset_of(s), []).append(s)
        if len(by_ds) < 2:
            raise SystemExit("cross_dataset needs sessions from both Data (A/...) and Data2 (B/...)")
        return "cross_dataset", {ds: sess for ds, sess in sorted(by_ds.items())}
    raise SystemExit(f"unknown train.eval_mode {mode!r}")


def cmd_train(cfg: Config, args) -> None:
    from .train import get_caches
    caches = get_caches(cfg)
    kind, holdouts = _holdout_plan(cfg, caches, args)
    modes = _list(args.modes) or ["criteria"]
    variants = _list(getattr(args, "variants", None))
    for seed in _seeds(cfg, args):
        for mode in modes:
            _train_one(cfg, mode, "", seed, kind, caches, holdouts, args.negative_control)
            if mode == "criteria":
                for v in variants:
                    _train_one(cfg, mode, v, seed, kind, caches, holdouts, args.negative_control)


def cmd_evaluate(cfg: Config, args) -> None:
    from .evaluate import evaluate_run
    from .runs import aggregate_holdouts, list_runs
    from .train import get_caches, load_run
    caches = get_caches(cfg)
    refs = list_runs(cfg.output_dir)
    if not refs:
        sys.exit("no runs found - run `train` first")
    for ref in refs:
        leaves = sorted(p for p in ref.path.glob("holdout_*") if (p / "model.pt").is_file()) or [ref.path]
        for d in leaves:
            if not (d / "model.pt").is_file():
                continue
            cfg_v = _clone(cfg, **{"train.seed": ref.seed})
            if ref.variant == "nospec":
                cfg_v.set_path("model.spectral_branch", "none")
            elif ref.variant == "popmean":
                cfg_v.set_path("model.spectral_branch", "popmean")
            res = evaluate_run(load_run(d, cfg_v, caches), cfg_v, caches)
            print(f"{ref.name} seed {ref.seed} {d.name}: bal. acc {res['classification']['balanced_accuracy']:.3f}")
        if leaves[0] != ref.path:
            aggregate_holdouts(ref.path, cfg)


# ----------------------------------------------------------------------------- figures
def _selection_table_for(cfg: Config, session: str, caches: dict) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """Selection table shown in Figure 1: the criteria run's train-split table when it exists (what the model used),
    else the descriptive all-trial table from `select`, else computed now. Returns (table, note, importance|None)."""
    from .features.selection import select_neurons
    from .runs import list_runs
    tag = session.replace("/", "__")
    runs = [r for r in list_runs(cfg.output_dir) if r.kind == "within" and r.mode == "criteria" and not r.variant]
    imp = None
    for r in sorted(runs, key=lambda r: r.seed):
        p = r.path / f"selection_{tag}.csv"
        if p.is_file():
            tab = pd.read_csv(p)
            pi = r.path / "neuron_importance.csv"
            if pi.is_file():
                imp_all = pd.read_csv(pi)
                imp = imp_all[imp_all.session == session].copy() if "session" in imp_all else None
            src = f"training-split criteria of run {r.name} seed {r.seed}"
            return tab, src, imp
    p = Path(cfg.output_dir) / "selection" / f"{tag}.csv"
    if p.is_file():
        return pd.read_csv(p), "descriptive criteria on all trials (`delaycast select`)", None
    res = select_neurons(caches[session], cfg, seed=int(cfg.train.seed), n_null=0)
    return res.table, "descriptive criteria on all trials (computed now)", None


def _source_note(tab: pd.DataFrame, cfg: Config, src: str) -> str:
    if not len(tab):
        return src
    r = tab.iloc[0]
    n_sub = int(r.get("n_subsamples", cfg.selection.get_path("n_subsamples", 50)))
    return (f"{src}: n={int(r.n_fit_trials)} trials (I {int(r.n_fit_ignore)} / L {int(r.n_fit_left)} / R {int(r.n_fit_right)}); "
            f"{n_sub} stratified half-subsamples; BH-FDR q<{cfg.selection.fdr_q}; K={int(cfg.selection.top_k_per_region)} per region")


def _fig1_for_trial(cfg: Config, cache, ti: int | None, npz_path: Path, out_path: Path, caches: dict, qc_note: str = "") -> Path:
    from .figures import plot_raster_selection
    tab, src, imp = _selection_table_for(cfg, cache.session, caches)
    if ti is not None:
        trial_label = f"trial {int(cache.trials[ti])} - {CLASSES[int(cache.labels[ti])]}"
        fl = cache.meta.first_lick_s.iloc[ti] if "first_lick_s" in cache.meta else float("nan")
        if np.isfinite(fl):
            trial_label += f" - first lick +{fl:.2f} s after go"
    else:
        trial_label = f"{npz_path.name} (not in cache)"
    return plot_raster_selection(npz_path, cache, tab, cfg, out_path, trial_label=trial_label,
                                 source_note=_source_note(tab, cfg, src), importance=imp, qc_note=qc_note)


def cmd_figures(cfg: Config, args) -> None:
    from .figures import plot_attention, plot_results, plot_time_frequency
    from .runs import list_runs, load_results
    from .train import get_caches
    fig_dir = Path(cfg.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    caches = get_caches(cfg)
    cache_list = list(caches.values())

    targets = []
    if cfg.figures.raster_trial:
        p = Path(cfg.figures.raster_trial)
        for c in cache_list:
            m = c.meta[c.meta.npz_path.map(lambda s: Path(s).resolve() == p.resolve())]
            if len(m):
                targets.append((c, int(m.index[0])))
        if not targets:
            sys.exit(f"figures.raster_trial={p} is not part of the cached trials (use `figure1 --npz` for QC-dropped trials)")
    else:
        for c in (cache_list if args.all_sessions else cache_list[:1]):
            idx = np.where(c.labels == CLASSES.index("Left"))[0]
            targets.append((c, int(idx[0]) if len(idx) else 0))
    for c, ti in targets:
        tag = c.session.replace("/", "__")
        npz = Path(c.meta.npz_path.iloc[ti])
        p1 = _fig1_for_trial(cfg, c, ti, npz, fig_dir / f"fig1_raster_selection_{tag}.png", caches)
        tab, _, _ = _selection_table_for(cfg, c.session, caches)
        p2 = plot_time_frequency(c, tab, cfg, ti, fig_dir / f"fig2_time_frequency_{tag}.png")
        print(f"wrote {p1}\nwrote {p2}")

    results = load_results(cfg.output_dir)
    if not results:
        print("no results.json found yet - run `train` first to get fig3/fig4")
        return
    p4 = plot_results(results, cfg, fig_dir / "fig4_results.png")
    print(f"wrote {p4}")
    crit = [r for r in list_runs(cfg.output_dir) if r.kind == "within" and r.mode == "criteria" and not r.variant]
    if crit:
        ref = sorted(crit, key=lambda r: r.seed)[0]
        sels = {}
        for c in cache_list:
            p = ref.path / f"selection_{c.session.replace('/', '__')}.csv"
            if p.is_file():
                sels[c.session] = pd.read_csv(p)
        p3 = plot_attention(ref.path, results["criteria"][0], sels, cfg, fig_dir / "fig3_attention.png")
        print(f"wrote {p3}")


_SES_RE = re.compile(r"(sub-[^/\\]+_ses-[^/\\]+?)(?:_behavior.*)?$")


def _session_of_npz(npz: Path) -> str | None:
    """Session key from the on-disk layout: Data/<Session>/Rasters/<class>/x.npz or Data2/<sub>/<sub_ses...>/NPZ/<class>/x.npz."""
    parts = [p.name for p in npz.resolve().parents]
    if len(parts) >= 3 and parts[1].lower() == "rasters":
        return f"A/{parts[2]}"
    if len(parts) >= 3 and parts[1].upper() == "NPZ":
        m = _SES_RE.match(parts[2])
        return f"B/{m.group(1)}" if m else f"B/{parts[2].split('_behavior')[0]}"
    return None


def cmd_figure1(cfg: Config, args) -> None:
    from .data.cache import _cache_key
    from .data.discovery import parse_trial_number
    from .train import get_caches
    npz = Path(args.npz)
    if not npz.is_file():
        sys.exit(f"{npz} does not exist")
    caches = get_caches(cfg)
    session = args.session or _session_of_npz(npz)
    if session not in caches:
        sys.exit(f"could not map {npz} to a cached session (guessed {session!r}); known sessions: {list(caches)}; pass --session KEY")
    cache = caches[session]
    n = parse_trial_number(npz.stem)
    ti = None
    qc_note = ""
    where = np.flatnonzero(cache.trials == n) if n is not None else np.zeros(0, int)
    if len(where):
        ti = int(where[0])
    else:
        qc = Path(cfg.data.cache_dir) / _cache_key(cfg) / "qc_log.csv"
        reason = ""
        if qc.is_file():
            q = pd.read_csv(qc)
            hit = q[(q.session == session) & (q.trial == n)]
            reason = str(hit.reason.iloc[0]) if len(hit) and isinstance(hit.reason.iloc[0], str) else ""
        qc_note = f"trial {n} is not in the cache" + (f" (excluded by QC: {reason})" if reason else "")
    fig_dir = Path(cfg.output_dir) / "figures"
    out = fig_dir / f"fig1_raster_selection_{session.replace('/', '__')}_trial{n}.png"
    p = _fig1_for_trial(cfg, cache, ti, npz, out, caches, qc_note=qc_note)
    print(f"wrote {p} (+ .pdf)" + (f"\nnote: {qc_note}" if qc_note else ""))


def cmd_report(cfg: Config, args) -> None:
    from .report import write_report
    p = write_report(cfg, Path(cfg.output_dir))
    print(f"wrote {p}")
    txt = Path(p).read_text(encoding="utf-8")
    head = txt.split("\n## ", 2)
    print(head[0][:3000])


def cmd_all(cfg: Config, args) -> None:
    from .train import get_caches
    cmd_cache(cfg, argparse.Namespace(force=False))
    cmd_select(cfg, args)
    caches = get_caches(cfg)
    seeds = [int(cfg.train.seed)] if args.quick else _seeds(cfg, args)
    modes = _list(args.modes) or ["criteria", "rate", "random"]
    variants = [] if args.quick else _list(args.variants)
    for seed in seeds:
        for mode in modes:
            _train_one(cfg, mode, "", seed, "within", caches, None, False)
            if mode == "criteria":
                for v in variants:
                    _train_one(cfg, mode, v, seed, "within", caches, None, False)
    if not args.quick:
        datasets = {_dataset_of(s) for s in caches}
        if len(datasets) >= 2:
            cfg_x = _clone(cfg, **{"train.eval_mode": "cross_dataset"})
            _, holds = _holdout_plan(cfg_x, caches, argparse.Namespace(negative_control=False, holdout=None))
            for mode in ("criteria", "random"):
                _train_one(cfg_x, mode, "", seeds[0], "cross_dataset", caches, holds, False)
        _train_one(cfg, "criteria", "", seeds[0], "negative_control", caches, None, True)
    cmd_figures(cfg, argparse.Namespace(all_sessions=args.all_sessions))
    cmd_report(cfg, args)


# ----------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="delaycast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a small synthetic Data/Data2 tree for smoke tests"); _common(p)
    p.add_argument("--out", default="synthetic_data")
    p.add_argument("--sessions-a", type=int, default=2)
    p.add_argument("--sessions-b", type=int, default=2)
    p.add_argument("--trials-per-class", type=int, nargs=3, default=[12, 40, 40], metavar=("IGNORE", "LEFT", "RIGHT"))
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("inspect", help="list discovered sessions/trials and NPZ contents"); _common(p)
    p.add_argument("--npz-detail", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("cache", help="bin all trials into per-session tensors"); _common(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_cache)

    p = sub.add_parser("select", help="descriptive neuron selection on all trials (tables with reasons)"); _common(p)
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("train", help="train + evaluate DelayCAST-Net"); _common(p)
    p.add_argument("--modes", default="criteria", help="comma list of neuron sets: criteria,rate,random")
    p.add_argument("--variants", default="", help="comma list of model ablations of the criteria set: popmean,nospec")
    p.add_argument("--seeds", default=None, help="comma list of seeds (default: train.seeds from the config)")
    p.add_argument("--holdout", default=None, help="cross_session: comma list of held-out session keys (default: all, one at a time)")
    p.add_argument("--negative-control", action="store_true", help="permute labels within session before selection/training")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="re-evaluate every saved run"); _common(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("figures", help="make the publication figures"); _common(p)
    p.add_argument("--all-sessions", action="store_true", help="fig1/fig2 for every session (default: first)")
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("figure1", help="Figure 1 for one NPZ trial file"); _common(p)
    p.add_argument("--npz", required=True, help="path to the trial NPZ (e.g. .../Data/Session1/Rasters/Left/trial_331.npz)")
    p.add_argument("--session", default=None, help="session key if it cannot be inferred from the path (e.g. A/Session1)")
    p.set_defaults(fn=cmd_figure1)

    p = sub.add_parser("report", help="write REPORT.md with the verdict on every prediction"); _common(p)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("all", help="the whole protocol: cache -> select -> train -> cross-dataset -> negative control -> figures -> report"); _common(p)
    p.add_argument("--modes", default="criteria,rate,random")
    p.add_argument("--variants", default="popmean")
    p.add_argument("--seeds", default=None)
    p.add_argument("--quick", action="store_true", help="one seed, within-session only, no controls (smoke test)")
    p.add_argument("--all-sessions", action="store_true")
    p.set_defaults(fn=cmd_all)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:  # pragma: no cover
        pass
    cfg = load_config(args.config, args.overrides)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
