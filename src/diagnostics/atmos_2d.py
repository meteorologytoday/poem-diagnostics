"""
Atmosphere 2-D diagnostic maps.

Produces one global map per variable listed in config["diagnostics"]["atmos"]["vars"].
Input is the "atmos_month" Dataset (lat × lon × time, standard lat-lon grid).
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src.temporal_agg import aggregate
from src import plot_utils


# Units for labelling colorbars. Variables absent from this dict get an
# empty label; add entries here as needed.
_UNITS: dict[str, str] = {
    "t_ref": "K",
    "t_surf": "K",
    "precip": "kg m⁻² s⁻¹",
    "slp": "Pa",
    "olr": "W m⁻²",
    "swdn_toa": "W m⁻²",
    "netrad_toa": "W m⁻²",
}

# Variables for which the colormap should be symmetric around zero.
_SYMMETRIC = {"netrad_toa", "lw_heat", "sens_heat"}


def run(
    ds: xr.Dataset,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    """
    Generate atmosphere 2-D maps for all configured variables.

    Parameters
    ----------
    ds:
        "atmos_month" Dataset loaded by data_loader.load_all().
    config:
        Full experiment config dict.
    mode:
        Temporal aggregation mode passed to temporal_agg.aggregate().
    output_dir:
        Root output directory for this experiment + mode combination.
        Figures are saved under output_dir / "atmos" / "{varname}.png".
    """
    grid_cfg = config["grids"]["atmos"]
    lat_name = grid_cfg["lat"]
    lon_name = grid_cfg["lon"]
    diag_cfg = config["diagnostics"]["atmos"]
    out_cfg = config["output"]

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    for varname in diag_cfg["vars"]:
        if varname not in ds:
            print(f"  [atmos] skipping '{varname}' — not in dataset")
            continue

        da = ds[varname]
        try:
            agg = aggregate(da, mode)
        except ValueError as exc:
            print(f"  [atmos] skipping '{varname}': {exc}")
            continue

        units = _UNITS.get(varname, "")
        symmetric = varname in _SYMMETRIC
        out_path = output_dir / "atmos" / f"{varname}.{out_cfg['format']}"

        plot_utils.global_map(
            data=agg,
            lat=lat,
            lon=lon,
            title=f"Atmos {varname} — {mode}",
            output_path=out_path,
            units=units,
            symmetric=symmetric,
            dpi=out_cfg["dpi"],
        )
        print(f"  [atmos] saved {out_path}")
