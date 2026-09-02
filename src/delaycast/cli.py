"""Command-line entry point.  ``python -m delaycast <command> [--config ...] [--set key=value ...]``"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES, REGIONS
from .config import Config, load_config

log = logging.getLogger("delaycast")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set data.data_a_root=D:/Rodent/Data")
    p.add_argument("-v", "--verbose", action="store_true")


def cmd_synth(cfg: Config, args) -> None:
    from .data.synthetic import make_synthetic
    a, b = make_synthetic(args.out, n_sessions_a=args.sessions_a, n_sessions_b=args.sessions_b,
                          trials_per_class=tuple(args.trials_per_class), seed=int(cfg.train.seed))
    print(f"synthetic Dataset A -> {a}\nsynthetic Dataset B -> {b}")
    print("Run the pipeline on it with:\n"
          f"  python -m delaycast all --set data.data_a_root={a} --set data.data_b_root={b} --set output_dir=outputs_synth --set data.cache_dir=cache_synth")


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
    print(cache_summary(caches).to_string(index=False))


def cmd_select(cfg: Config, args) -> None:
    from .data.cache import load_cache
    from .features.selection import select_neurons, selection_summary
    caches = load_cache(cfg)
    out = Path(cfg.output_dir) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for c in caches:
        res = select_neurons(c, cfg)
        res.table.to_csv(out / f"{c.session.replace('/', '__')}.csv", index=False)
        results.append(res)
    summ = selection_summary(results)
    summ.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(f"\nper-unit tables with reasons written to {out}/")


def _modes(args) -> list[str]:
    return [m.strip() for m in args.modes.split(",") if m.strip()]


def cmd_train(cfg: Config, args) -> None:
    from .data.cache import load_cache
    from .evaluate import evaluate_run
    from .train import run_training
    caches = {c.session: c for c in load_cache(cfg)}
    for mode in _modes(args):
        if cfg.train.eval_mode == "cross_session":
            sessions = list(caches)
            holds = [args.holdout] if args.holdout else sessions
            for h in holds:
                run = run_training(cfg, mode=mode, out_dir=Path(cfg.output_dir) / f"run_{mode}" / f"holdout_{h.replace('/', '__')}", holdout=h)
                evaluate_run(run, cfg, caches)
            _aggregate_cross_session(Path(cfg.output_dir) / f"run_{mode}")
        else:
            run = run_training(cfg, mode=mode)
            evaluate_run(run, cfg, caches)


def _aggregate_cross_session(run_dir: Path) -> None:
    rows, cms = [], np.zeros((3, 3))
    for d in sorted(run_dir.glob("holdout_*")):
        p = d / "results.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            rows.append({"holdout": d.name.replace("holdout_", ""), **r["classification"]})
            cms += np.asarray(r["confusion"])
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "cross_session_summary.csv", index=False)
        agg = {"classification": {"balanced_accuracy": float(df.balanced_accuracy.mean()), "accuracy": float(df.accuracy.mean()),
                                  "macro_f1": float(df.macro_f1.mean()), "n": int(df.n.sum())},
               "confusion": cms.astype(int).tolist(), "per_session": rows,
               "chance_balanced_accuracy": {"mean": 1 / 3, "p95": 0.45}}
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(df.to_string(index=False))


def cmd_evaluate(cfg: Config, args) -> None:
    from .data.cache import load_cache
    from .evaluate import evaluate_run
    from .train import load_run
    caches = {c.session: c for c in load_cache(cfg)}
    for mode in _modes(args):
        run_dir = Path(cfg.output_dir) / f"run_{mode}"
        res = evaluate_run(load_run(run_dir, cfg), cfg, caches)
        print(json.dumps({k: res[k] for k in ("classification", "chance_balanced_accuracy", "forecast")}, indent=1))


def cmd_figures(cfg: Config, args) -> None:
    from .data.cache import load_cache
    from .features.selection import select_neurons
    from .figures import plot_attention, plot_raster_selection, plot_results, plot_time_frequency
    from .figures.results_fig import load_results
    fig_dir = Path(cfg.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    caches = load_cache(cfg)
    selections = {c.session: select_neurons(c, cfg) for c in caches}

    # Fig 1 (+ Fig 2) for the requested trial or the first Left trial of every session.
    targets = []
    if cfg.figures.raster_trial:
        p = Path(cfg.figures.raster_trial)
        for c in caches:
            m = c.meta[c.meta.npz_path.map(lambda s: Path(s).resolve() == p.resolve())]
            if len(m):
                targets.append((c, int(m.index[0])))
        if not targets:
            sys.exit(f"figures.raster_trial={p} is not part of the cached trials")
    else:
        for c in (caches if args.all_sessions else caches[:1]):
            idx = np.where(c.labels == CLASSES.index("Left"))[0]
            targets.append((c, int(idx[0]) if len(idx) else 0))
    for c, ti in targets:
        tag = c.session.replace("/", "__")
        npz = c.meta.npz_path.iloc[ti]
        p1 = plot_raster_selection(npz, c, selections[c.session].table, cfg, fig_dir / f"fig1_raster_selection_{tag}.png",
                                   trial_label=f"trial {c.trials[ti]}, {CLASSES[c.labels[ti]]}")
        p2 = plot_time_frequency(c, selections[c.session].table, cfg, ti, fig_dir / f"fig2_time_frequency_{tag}.png")
        print(f"wrote {p1}\nwrote {p2}")

    results = load_results(cfg.output_dir)
    if results:
        p4 = plot_results(results, cfg, fig_dir / "fig4_results.png")
        print(f"wrote {p4}")
        att = Path(cfg.output_dir) / "run_criteria" / "attention.npz"
        if att.is_file():
            from .data.dataset import build_session_tensors
            tensors = {c.session: build_session_tensors(c, selections[c.session], cfg) for c in caches}
            unit_index = {s: t.unit_index for s, t in tensors.items()}
            p3 = plot_attention(att, {s: r.table for s, r in selections.items()}, unit_index, caches[0].bin_ms, cfg,
                                fig_dir / "fig3_attention.png")
            print(f"wrote {p3}")
    else:
        print("no results.json found yet - run `train` first to get fig3/fig4")


def cmd_all(cfg: Config, args) -> None:
    cmd_cache(cfg, argparse.Namespace(force=False))
    cmd_select(cfg, args)
    cmd_train(cfg, args)
    cmd_figures(cfg, argparse.Namespace(all_sessions=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="delaycast", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a small synthetic Data/Data2 tree for smoke tests"); _common(p)
    p.add_argument("--out", default="synthetic_data")
    p.add_argument("--sessions-a", type=int, default=2)
    p.add_argument("--sessions-b", type=int, default=2)
    p.add_argument("--trials-per-class", type=int, nargs=3, default=[10, 24, 24], metavar=("IGNORE", "LEFT", "RIGHT"))
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("inspect", help="list discovered sessions/trials and NPZ contents"); _common(p)
    p.add_argument("--npz-detail", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("cache", help="bin all trials into per-session tensors"); _common(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_cache)

    p = sub.add_parser("select", help="run the neuron-selection criteria"); _common(p)
    p.set_defaults(fn=cmd_select)

    for name, fn, hlp in (("train", cmd_train, "train + evaluate DelayCAST"), ("evaluate", cmd_evaluate, "re-evaluate a saved run")):
        p = sub.add_parser(name, help=hlp); _common(p)
        p.add_argument("--modes", default="criteria", help="comma list of neuron sets: criteria,rate,random")
        p.add_argument("--holdout", default=None, help="cross_session: evaluate a single held-out session key")
        p.set_defaults(fn=fn)

    p = sub.add_parser("figures", help="make the publication figures"); _common(p)
    p.add_argument("--all-sessions", action="store_true", help="fig1/fig2 for every session (default: first)")
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("all", help="cache -> select -> train (all modes) -> figures"); _common(p)
    p.add_argument("--modes", default="criteria,rate,random")
    p.add_argument("--holdout", default=None)
    p.set_defaults(fn=cmd_all)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.overrides)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
