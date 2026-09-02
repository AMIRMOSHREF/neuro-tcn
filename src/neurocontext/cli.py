from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import load_config
from .data import discover_trials, valid_trials
from .demo import generate_demo
from .figure import make_selection_figure
from .selection import rank_neurons
from .train import grouped_folds, train_fold


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", help="Path to Data (Session*/Rasters layout)")
    parser.add_argument("--data2-root", help="Path to Data2 (sub-*/session/NPZ layout)")
    parser.add_argument(
        "--metadata-csv", action="append", default=[],
        help="Behavioral/audited CSV; repeat for multiple files",
    )


def _records(args: argparse.Namespace):
    records = discover_trials(args.data_root, args.data2_root, args.metadata_csv)
    if not records:
        raise SystemExit("No trials found. Check root paths and expected folder layouts.")
    return records


def command_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    records = _records(args)
    valid = list(valid_trials(records, config))
    invalid_paths = sorted({str(r.npz_path) for r in records}.difference(str(r.npz_path) for r in valid))
    summary = {
        "discovered_trials": len(records),
        "valid_trials": len(valid),
        "invalid_trials": len(invalid_paths),
        "datasets": Counter(r.dataset for r in valid),
        "classes": Counter(r.label for r in valid),
        "groups": len({r.group for r in valid}),
        "invalid_paths": invalid_paths[:50],
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    records = list(valid_trials(_records(args), config))
    folds = grouped_folds(records, int(config["training"]["folds"]))
    requested = range(len(folds)) if args.fold == "all" else [int(args.fold)]
    metrics = [
        train_fold(records, config, args.output_dir, fold, args.device)
        for fold in requested
    ]
    print(json.dumps(metrics, indent=2))


def command_select(args: argparse.Namespace) -> None:
    records = _records(args)
    rows = rank_neurons(args.checkpoint, records, args.output, args.device)
    print(json.dumps({"ranked_units": len(rows), "selected_units": sum(r["selected"] for r in rows), "output": args.output}, indent=2))


def command_figure(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = make_selection_figure(args.npz, args.output, config, args.selection_csv)
    print(json.dumps({"figure": str(path), "pdf": str(path.with_suffix(".pdf"))}, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    paths = generate_demo(args.output_dir, load_config(args.config))
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurocontext",
        description="Delay-context response forecasting and action decoding",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo", help="Generate deterministic synthetic data")
    demo.add_argument("--config", default="config/default.yaml")
    demo.add_argument("--output-dir", default="demo")
    demo.set_defaults(func=command_demo)

    audit = subparsers.add_parser("audit", help="Validate trials and summarize coverage")
    audit.add_argument("--config", default="config/default.yaml")
    audit.add_argument("--output")
    _add_data_arguments(audit)
    audit.set_defaults(func=command_audit)

    train = subparsers.add_parser("train", help="Train one or all held-out-session folds")
    train.add_argument("--config", default="config/default.yaml")
    train.add_argument("--output-dir", default="results")
    train.add_argument("--fold", default="0", help="Fold number or 'all'")
    train.add_argument("--device", choices=["cpu", "cuda"])
    _add_data_arguments(train)
    train.set_defaults(func=command_train)

    select = subparsers.add_parser("select", help="Rank neurons from learned gates")
    select.add_argument("--checkpoint", required=True)
    select.add_argument("--output", default="results/neuron_selection.csv")
    select.add_argument("--device", choices=["cpu", "cuda"])
    _add_data_arguments(select)
    select.set_defaults(func=command_select)

    figure = subparsers.add_parser("figure", help="Render all versus selected raster panels")
    figure.add_argument("--config", default="config/default.yaml")
    figure.add_argument("--npz", required=True)
    figure.add_argument("--selection-csv")
    figure.add_argument("--output", default="results/neuron_selection.png")
    figure.set_defaults(func=command_figure)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
