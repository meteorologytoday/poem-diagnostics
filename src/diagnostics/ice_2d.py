"""
Sea-ice 2-D diagnostic maps.

Produces polar stereographic maps for each variable and hemisphere
listed in the config. The ice grid is tripolar; GEOLAT and GEOLON are
2-D coordinate arrays stored as dataset variables (not coordinates).
"""

from pathlib import Path

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
    ds: xr.Dataset,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    """
    Generate polar sea-ice maps for all configured variables and regions.

    Parameters
    ----------
    ds:
        "ice_month" Dataset loaded by data_loader.load_all().
    config:
        Full experiment config dict.
    mode:
        Temporal aggregation mode passed to temporal_agg.aggregate().
    output_dir:
        Root output directory. Figures saved under output_dir / "ice".
    """
    grid_cfg = config["grids"]["ice"]
    lat_name = grid_cfg["lat"]
    lon_name = grid_cfg["lon"]
    diag_cfg = config["diagnostics"]["ice"]
    out_cfg = config["output"]

    lat2d = ds[lat_name].values
    lon2d = ds[lon_name].values

    for varname in diag_cfg["vars"]:
        if varname not in ds:
            print(f"  [ice] skipping '{varname}' — not in dataset")
            continue

        da = ds[varname]
        try:
            agg = aggregate(da, mode)
        except ValueError as exc:
            print(f"  [ice] skipping '{varname}': {exc}")
            continue

        units = _UNITS.get(varname, "")
        clim = _CLIM.get(varname, {})
        vmin = clim[0] if clim else None
        vmax = clim[1] if clim else None

        for hemisphere in diag_cfg.get("regions", ["NH", "SH"]):
            out_path = (
                output_dir / "ice" / f"{varname}_{hemisphere}.{out_cfg['format']}"
            )
            plot_utils.polar_map(
                data=agg,
                lat=lat2d,
                lon=lon2d,
                hemisphere=hemisphere,
                title=f"Sea ice {varname} {hemisphere} — {mode}",
                output_path=out_path,
                units=units,
                vmin=vmin,
                vmax=vmax,
                dpi=out_cfg["dpi"],
            )
            print(f"  [ice] saved {out_path}")
