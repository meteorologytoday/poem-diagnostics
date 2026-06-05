"""
Atmosphere diagnostic plots.

Supported diag_types:
  map_2d      — global map per variable in config[atmos][map_2d][vars]
  timeseries  — area-weighted global-mean time series per variable
                in config[atmos][timeseries][vars]
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from src.temporal_agg import aggregate
from src import plot_utils


_UNITS: dict[str, str] = {
    "t_ref": "K",
    "t_surf": "K",
    "precip": "kg m⁻² s⁻¹",
    "slp": "Pa",
    "olr": "W m⁻²",
    "swdn_toa": "W m⁻²",
    "netrad_toa": "W m⁻²",
}

_SYMMETRIC = {"netrad_toa"}


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
    ds = data["atmos_month"]
    grid_cfg = config["grids"]["atmos"]
    diag_cfg = config["diagnostics"]["atmos"].get("map_2d", {})
    out_cfg = config["output"]
    year_label = data.get("year_label", "")

    lat = ds[grid_cfg["lat"]].values
    lon = ds[grid_cfg["lon"]].values

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [atmos/map_2d] skipping '{varname}' — not in dataset")
            continue
        try:
            agg = aggregate(ds[varname], mode)
        except ValueError as exc:
            print(f"  [atmos/map_2d] skipping '{varname}': {exc}")
            continue

        out_path = output_dir / "atmos" / "map_2d" / f"{varname}_{year_label}.{out_cfg['format']}"
        plot_utils.global_map(
            data=agg,
            lat=lat,
            lon=lon,
            title=f"Atmos {varname} — {mode} ({year_label})",
            output_path=out_path,
            units=_UNITS.get(varname, ""),
            symmetric=varname in _SYMMETRIC,
            n_levels=out_cfg.get("n_levels", 20),
            dpi=out_cfg["dpi"],
        )
        print(f"  [atmos/map_2d] saved {out_path}")


def _run_timeseries(
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    ds = data["atmos_month"]
    diag_cfg = config["diagnostics"]["atmos"].get("timeseries", {})
    out_cfg = config["output"]
    year_label = data.get("year_label", "")

    lat = ds["lat"]
    weights = np.cos(np.deg2rad(lat))
    weights = weights.where(weights > 0, other=0.0)

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [atmos/timeseries] skipping '{varname}' — not in dataset")
            continue
        da = ds[varname]
        if "pfull" in da.dims:
            print(f"  [atmos/timeseries] skipping '{varname}' — has vertical dim")
            continue

        global_mean = da.weighted(weights).mean(dim=["lat", "lon"])
        times = _to_numeric_years(global_mean)
        out_path = output_dir / "atmos" / "timeseries" / f"{varname}_{year_label}.{out_cfg['format']}"
        plot_utils.time_series(
            times=times,
            values=global_mean.values.astype(float),
            title=f"Atmos global mean {varname} ({year_label})",
            ylabel=_UNITS.get(varname, varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
        )
        print(f"  [atmos/timeseries] saved {out_path}")


def _to_numeric_years(da: xr.DataArray) -> np.ndarray:
    try:
        return np.array([v.year + (v.month - 1) / 12.0 for v in da.time.values])
    except AttributeError:
        return da.time.values.astype(float)


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
    "timeseries": _run_timeseries,
}
