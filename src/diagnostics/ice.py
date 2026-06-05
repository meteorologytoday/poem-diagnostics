"""
Sea-ice diagnostic plots.

Supported diag_types:
  map_2d  — NH and SH polar stereographic maps per variable in
             config[ice][map_2d][vars]
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from src.temporal_agg import aggregate
from src import plot_utils


_UNITS: dict[str, str] = {
    "CN": "fraction",
    "HI": "m",
    "EXT": "fraction",
    "SST": "°C",
    "SSS": "psu",
}

_CLIM: dict[str, tuple[float, float]] = {
    "CN": (0.0, 1.0),
    "EXT": (0.0, 1.0),
}


def run(
    diag_type: str,
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    handler = HANDLERS.get(diag_type)
    if handler is None:
        return
    handler(data, config, mode, output_dir)


def _run_map_2d(
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    ds = data["ice_month"]
    grid_cfg = config["grids"]["ice"]
    diag_cfg = config["diagnostics"]["ice"].get("map_2d", {})
    out_cfg = config["output"]

    lat2d = ds[grid_cfg["lat"]].values
    lon2d = ds[grid_cfg["lon"]].values

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ice/map_2d] skipping '{varname}' — not in dataset")
            continue
        try:
            agg = aggregate(ds[varname], mode)
        except ValueError as exc:
            print(f"  [ice/map_2d] skipping '{varname}': {exc}")
            continue

        clim = _CLIM.get(varname, {})
        vmin = clim[0] if clim else None
        vmax = clim[1] if clim else None

        for hemisphere in diag_cfg.get("regions", ["NH", "SH"]):
            out_path = (
                output_dir / "ice" / "map_2d" / f"{varname}_{hemisphere}.{out_cfg['format']}"
            )
            plot_utils.polar_map(
                data=agg,
                lat=lat2d,
                lon=lon2d,
                hemisphere=hemisphere,
                title=f"Sea ice {varname} {hemisphere} — {mode}",
                output_path=out_path,
                units=_UNITS.get(varname, ""),
                vmin=vmin,
                vmax=vmax,
                n_levels=out_cfg.get("n_levels", 20),
                dpi=out_cfg["dpi"],
            )
            print(f"  [ice/map_2d] saved {out_path}")


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
}
