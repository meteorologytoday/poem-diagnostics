#!/usr/bin/env python3
"""
run_diagnostics.py — entry point for POEM diagnostic plots.

Usage
-----
    python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
                              --modes annual JJA DJF \\
                              --components atmos ocean ice land \\
                              --diag-types map_2d timeseries zonal_section

All arguments except --config have sensible defaults (see --help).
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

from src import data_loader
from src.diagnostics import atmos, ocean, ice, land_lpjml
from src.temporal_agg import SEASON_MONTHS

# Registry: component name → module with a HANDLERS dict and run() function.
COMPONENT_MODULES = {
    "atmos": atmos,
    "ocean": ocean,
    "ice": ice,
    "land": land_lpjml,
}

# All diag_types recognised across any component. A component silently
# skips types it does not support (not in its HANDLERS dict).
ALL_DIAG_TYPES = ["map_2d", "timeseries", "zonal_section"]

# Dataset keys each component reads from the data dict.
_COMPONENT_DATA_KEYS = {
    "atmos": ["atmos_month"],
    "ocean": ["ocean_month", "ocean_scalar"],
    "ice":   ["ice_month"],
    "land":  ["lpjml"],
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate POEM climate model diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annual mean, all components and diag-types
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml

  # JJA and DJF maps for atmosphere and ocean only
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --modes JJA DJF --components atmos ocean --diag-types map_2d

  # All timeseries diagnostics, annual
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --diag-types timeseries

  # Single variable during development
  python run_diagnostics.py --config config/minimal_setup_CTL.yaml \\
      --components atmos --diag-types map_2d --vars t_ref
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
            "Temporal aggregation modes: 'annual', season code "
            "(DJF MAM JJA SON), or month number 1-12. Default: annual."
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENT_MODULES),
        choices=list(COMPONENT_MODULES),
        metavar="COMPONENT",
        help=(
            f"Model components to process: {list(COMPONENT_MODULES)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--diag-types",
        nargs="+",
        default=ALL_DIAG_TYPES,
        choices=ALL_DIAG_TYPES,
        metavar="TYPE",
        help=(
            f"Diagnostic plot types: {ALL_DIAG_TYPES}. "
            "Default: all. Each component only produces types it supports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override the output base directory from the config.",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        metavar="VAR",
        help=(
            "Restrict to these variable names across all selected components "
            "and types. Useful during development."
        ),
    )
    return parser.parse_args(argv)


def parse_mode(raw: str) -> str | int:
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
    cfg = copy.deepcopy(config)
    diag = cfg.get("diagnostics", {})
    vset = set(vars_filter)

    for comp in ("atmos", "ocean", "ice"):
        comp_cfg = diag.get(comp, {})
        for dtype in ("map_2d", "zonal_section", "timeseries"):
            if dtype in comp_cfg and "vars" in comp_cfg[dtype]:
                comp_cfg[dtype]["vars"] = [
                    v for v in comp_cfg[dtype]["vars"] if v in vset
                ]

    land_cfg = diag.get("land", {})
    map2d = land_cfg.get("map_2d", {})
    for group in map2d.get("monthly_vars", {}).values():
        group[:] = [v for v in group if v in vset]
    for group in map2d.get("annual_vars", {}).values():
        group[:] = [v for v in group if v in vset]
    zs = land_cfg.get("zonal_section", {})
    if "vars" in zs:
        zs["vars"] = [v for v in zs["vars"] if v in vset]

    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    with args.config.open() as fh:
        config = yaml.safe_load(fh)

    modes = []
    for raw in args.modes:
        try:
            modes.append(parse_mode(raw))
        except argparse.ArgumentTypeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    output_root = args.output_dir or Path(config["output"]["base_dir"])
    exp_name = config["experiment"]["name"]
    config = apply_var_filter(config, args.vars)

    print(f"Loading data for experiment '{exp_name}' ...")
    try:
        data = data_loader.load_all(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading data: {exc}", file=sys.stderr)
        return 1

    selected_components = args.components
    selected_types = args.diag_types

    for mode in modes:
        mode_str = str(mode)
        mode_dir = output_root / exp_name / mode_str
        print(f"\n── Mode: {mode_str} ──────────────────────────")

        for comp_name in selected_components:
            module = COMPONENT_MODULES[comp_name]
            required_keys = _COMPONENT_DATA_KEYS[comp_name]
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(
                    f"  [{comp_name}] skipping — data not loaded: {missing}"
                )
                continue

            for diag_type in selected_types:
                if diag_type not in module.HANDLERS:
                    continue
                print(f"  Running {comp_name} / {diag_type} ...")
                module.run(diag_type, data, config, mode, mode_dir)

    print(f"\nDone. Output written to: {output_root / exp_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
