"""Command-line entry points for the Part 1 data pipeline.

Commands:
    validate-config    Validate the project configuration.
    audit              Read, validate, clean and audit the source CSV.
    eda                Write exploratory tables and figures from the cleaned panel.
    folds              Reserve the future holdout and write temporal/spatial folds.
    sequence-coverage  Measure seven-day sequence availability against the folds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config, validate_config
from .data_audit import run_audit
from .eda import promote_figures, run_eda
from .sequence_audit import sequence_coverage_table
from .splits import save_folds


def _load(args: argparse.Namespace) -> tuple[dict, Path]:
    cfg, root = load_config(args.config)
    validate_config(cfg)
    return cfg, root


def _results_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["paths"]["results_root"] / cfg["project"].get("group_id", "")


def _read_clean(cfg: dict, root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "cleaned_daily.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned panel not found: {path}. Run `python -m pm25_project.cli audit` first."
        )
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def cmd_validate_config(args: argparse.Namespace) -> int:
    _load(args)
    print("Configuration valid")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    cfg, root = _load(args)
    clean, provenance = run_audit(cfg)
    cleaning = provenance["cleaning"]
    print(
        f"Cleaned {cleaning['input_rows']:,} -> {cleaning['clean_rows']:,} rows; "
        f"{provenance['station_count']} stations; "
        f"{provenance['date_min'][:10]} to {provenance['date_max'][:10]}"
    )
    print(f"Wrote {root / 'data' / 'processed' / 'cleaned_daily.csv'}")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    cfg, root = _load(args)
    frame = _read_clean(cfg, root)
    results = _results_dir(cfg, root)
    run_eda(frame, cfg)
    promote_figures(results / "figures", root / "figures")
    print(f"Wrote EDA tables under {results} and figures under {root / 'figures'}")
    return 0


def cmd_folds(args: argparse.Namespace) -> int:
    cfg, root = _load(args)
    frame = _read_clean(cfg, root)
    definitions = save_folds(frame, cfg)
    temporal = definitions["folds_temporal.json"]
    holdout = definitions["future_holdout.json"]
    print(
        f"Wrote {len(temporal['folds'])} temporal folds to {root / 'folds'}; "
        f"holdout starts {holdout['holdout_start'][:10]} "
        f"({holdout['n_holdout_dates']} dates, reserved)"
    )
    return 0


def cmd_sequence_coverage(args: argparse.Namespace) -> int:
    cfg, root = _load(args)
    frame = _read_clean(cfg, root)
    table = sequence_coverage_table(frame, cfg)
    results = _results_dir(cfg, root)
    print(table.head(12).to_string(index=False))
    print(f"Wrote {results / 'lookback_coverage.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pm25_project.cli",
        description="Part 1 PM2.5 data audit, EDA and fold preparation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "validate-config": (cmd_validate_config, "Validate src/configs/project.yaml."),
        "audit": (cmd_audit, "Read, validate, clean and audit the configured source CSV."),
        "eda": (cmd_eda, "Write exploratory tables and figures from the cleaned panel."),
        "folds": (cmd_folds, "Reserve the future holdout and write fold definitions."),
        "sequence-coverage": (cmd_sequence_coverage, "Check seven-day coverage against the folds."),
    }
    for name, (function, help_text) in commands.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument(
            "--config",
            default="src/configs/project.yaml",
            help="Path to project.yaml (default: src/configs/project.yaml).",
        )
        subparser.set_defaults(func=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
