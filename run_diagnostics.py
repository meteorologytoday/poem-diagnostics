#!/usr/bin/env python3
"""
run_diagnostics.py — entry point for POEM diagnostic plots.

Usage
-----
    python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
                              --modes annual JJA DJF \\
                              --diagnostics atmos ocean ice land_lpjml timeseries \\
                              --output-dir ./output

All arguments except --config have sensible defaults (see --help).
"""

import argparse
import sys
from pathlib import Path

import yaml

from src import data_loader
from src.diagnostics import atmos_2d, ocean_2d, ice_2d, land_lpjml, timeseries
from src.temporal_agg import SEASON_MONTHS, VALID_MODES

AVAILABLE_DIAGNOSTICS = ["atmos", "ocean", "ice", "land_lpjml", "timeseries"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate POEM climate model diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annual mean for all diagnostics
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml

  # JJA and DJF for atmosphere and ocean only
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --modes JJA DJF --diagnostics atmos ocean

  # Single calendar month (July = 7)
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --modes 7

  # Restrict to specific variables during development
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --diagnostics atmos --vars t_ref precip
""",
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["annual"],
        metavar="MODE",
        help=(
            "One or more temporal aggregation modes: 'annual', a season "
            "code (DJF MAM JJA SON), or a month number 1-12. "
            "Default: annual."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=AVAILABLE_DIAGNOSTICS,
        choices=AVAILABLE_DIAGNOSTICS,
        metavar="DIAG",
        help=(
            f"Diagnostic modules to run. Choose from: "
            f"{AVAILABLE_DIAGNOSTICS}. Default: all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Override the output directory from the config. "
            "Figures are written to <output-dir>/<experiment>/<mode>/<diagnostic>/."
        ),
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        metavar="VAR",
        help=(
            "Restrict plotting to these variable names. Applies to all "
            "selected diagnostic modules. Useful during development."
        ),
    )

    return parser.parse_args(argv)


def parse_mode(raw: str) -> str | int:
    """Convert a CLI mode string to the internal representation."""
    if raw == "annual" or raw in SEASON_MONTHS:
        return raw
    try:
        m = int(raw)
        if 1 <= m <= 12:
            return m
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid mode {raw!r}. Use 'annual', a season code, or 1-12."
    )


def apply_var_filter(config: dict, vars_filter: list[str] | None) -> dict:
    """Restrict variable lists in config to those in vars_filter."""
    if not vars_filter:
        return config
    import copy
    cfg = copy.deepcopy(config)
    diag = cfg.get("diagnostics", {})

    if "atmos" in diag:
        diag["atmos"]["vars"] = [v for v in diag["atmos"]["vars"] if v in vars_filter]

    if "ocean" in diag:
        diag["ocean"]["vars"] = [v for v in diag["ocean"]["vars"] if v in vars_filter]
        diag["ocean"]["depth_vars"] = [
            v for v in diag["ocean"].get("depth_vars", []) if v in vars_filter
        ]

    if "ice" in diag:
        diag["ice"]["vars"] = [v for v in diag["ice"]["vars"] if v in vars_filter]

    if "land_lpjml" in diag:
        for group in diag["land_lpjml"].get("monthly_vars", {}).values():
            group[:] = [v for v in group if v in vars_filter]
        for group in diag["land_lpjml"].get("annual_vars", {}).values():
            group[:] = [v for v in group if v in vars_filter]

    if "timeseries" in diag:
        diag["timeseries"]["ocean_scalar"] = [
            v for v in diag["timeseries"].get("ocean_scalar", []) if v in vars_filter
        ]
        diag["timeseries"]["atmos_global"] = [
            v for v in diag["timeseries"].get("atmos_global", []) if v in vars_filter
        ]

    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    # Load config
    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    with args.config.open() as fh:
        config = yaml.safe_load(fh)

    # Parse and validate modes
    modes = []
    for raw in args.modes:
        try:
            modes.append(parse_mode(raw))
        except argparse.ArgumentTypeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    # Determine output root
    out_cfg = config["output"]
    output_root = args.output_dir or Path(out_cfg["base_dir"])
    exp_name = config["experiment"]["name"]

    # Apply variable filter if requested
    config = apply_var_filter(config, args.vars)

    # Load all data (lazy)
    print(f"Loading data for experiment '{exp_name}' ...")
    try:
        data = data_loader.load_all(config)
    except FileNotFoundError as exc:
        print(f"Error loading data: {exc}", file=sys.stderr)
        return 1

    selected = set(args.diagnostics)

    # Run each mode × diagnostic combination
    for mode in modes:
        mode_str = str(mode)
        mode_dir = output_root / exp_name / mode_str
        print(f"\n── Mode: {mode_str} ──────────────────────────")

        if "atmos" in selected:
            if "atmos_month" in data:
                print("Running atmos diagnostics ...")
                atmos_2d.run(data["atmos_month"], config, mode, mode_dir)
            else:
                print("  [atmos] no atmos_month data loaded, skipping")

        if "ocean" in selected:
            if "ocean_month" in data:
                print("Running ocean diagnostics ...")
                ocean_2d.run(data["ocean_month"], config, mode, mode_dir)
            else:
                print("  [ocean] no ocean_month data loaded, skipping")

        if "ice" in selected:
            if "ice_month" in data:
                print("Running ice diagnostics ...")
                ice_2d.run(data["ice_month"], config, mode, mode_dir)
            else:
                print("  [ice] no ice_month data loaded, skipping")

        if "land_lpjml" in selected:
            if "lpjml" in data:
                print("Running LPJ-mL diagnostics ...")
                land_lpjml.run(data["lpjml"], config, mode, mode_dir)
            else:
                print("  [land_lpjml] no lpjml loader available, skipping")

        if "timeseries" in selected:
            if "ocean_scalar" in data and "atmos_month" in data:
                print("Running timeseries diagnostics ...")
                timeseries.run(
                    data["ocean_scalar"],
                    data["atmos_month"],
                    config,
                    mode,
                    mode_dir,
                )
            else:
                print("  [timeseries] missing ocean_scalar or atmos_month, skipping")

    print(f"\nDone. Output written to: {output_root / exp_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
