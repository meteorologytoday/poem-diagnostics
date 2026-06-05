"""
Ocean 2-D diagnostic maps and zonal-mean depth sections.

Produces:
  - One global map per variable in config["diagnostics"]["ocean"]["vars"]
  - One zonal-mean latitude–depth section per variable in "depth_vars"

The ocean grid is tripolar; geographic coordinates are 2-D arrays
(geolat_t, geolon_t) stored as dataset coordinate variables.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src.temporal_agg import aggregate
from src import plot_utils


_UNITS: dict[str, str] = {
    "sst": "°C",
    "sss": "psu",
    "mld": "m",
    "swflx": "W m⁻²",
    "lw_heat": "W m⁻²",
    "sens_heat": "W m⁻²",
    "evap_heat": "W m⁻²",
    "temp": "°C",
    "salt": "psu",
}

_SYMMETRIC = {"lw_heat", "sens_heat", "evap_heat"}


def run(
    ds: xr.Dataset,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    """
    Generate ocean 2-D maps and depth sections for all configured variables.

    Parameters
    ----------
    ds:
        "ocean_month" Dataset loaded by data_loader.load_all().
    config:
        Full experiment config dict.
    mode:
        Temporal aggregation mode passed to temporal_agg.aggregate().
    output_dir:
        Root output directory. Figures saved under output_dir / "ocean".
    """
    grid_cfg = config["grids"]["ocean"]
    lat_name = grid_cfg["lat"]
    lon_name = grid_cfg["lon"]
    diag_cfg = config["diagnostics"]["ocean"]
    out_cfg = config["output"]

    # 2-D geographic coordinate arrays for the tripolar grid.
    lat2d = ds[lat_name].values
    lon2d = ds[lon_name].values

    # 2-D surface maps
    for varname in diag_cfg["vars"]:
        if varname not in ds:
            print(f"  [ocean] skipping '{varname}' — not in dataset")
            continue

        da = ds[varname]
        try:
            agg = aggregate(da, mode)
        except ValueError as exc:
            print(f"  [ocean] skipping '{varname}': {exc}")
            continue

        units = _UNITS.get(varname, "")
        symmetric = varname in _SYMMETRIC
        out_path = output_dir / "ocean" / f"{varname}.{out_cfg['format']}"

        plot_utils.global_map(
            data=agg,
            lat=lat2d,
            lon=lon2d,
            title=f"Ocean {varname} — {mode}",
            output_path=out_path,
            units=units,
            symmetric=symmetric,
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocean] saved {out_path}")

    # Zonal-mean depth sections
    depth_vars = diag_cfg.get("depth_vars", [])
    for varname in depth_vars:
        if varname not in ds:
            print(f"  [ocean/section] skipping '{varname}' — not in dataset")
            continue

        da = ds[varname]
        try:
            agg = aggregate(da, mode)
        except ValueError as exc:
            print(f"  [ocean/section] skipping '{varname}': {exc}")
            continue

        # Zonal mean over the x (xt_ocean) axis.
        zonal = agg.mean(dim="xt_ocean")

        # Recover 1-D latitude from the T-grid coordinate.
        lat1d = ds["yt_ocean"].values
        depth = ds["st_ocean"].values

        units = _UNITS.get(varname, "")
        symmetric = varname in _SYMMETRIC
        out_path = output_dir / "ocean" / f"{varname}_zonal_section.{out_cfg['format']}"

        plot_utils.zonal_section(
            data=zonal,
            lat=lat1d,
            depth=depth,
            title=f"Ocean {varname} zonal mean — {mode}",
            output_path=out_path,
            units=units,
            symmetric=symmetric,
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocean] saved {out_path}")
