"""
Ocean diagnostic plots.

Supported diag_types:
  map_2d         — global map per variable in config[ocean][map_2d][vars]
  zonal_section  — latitude-depth zonal mean per variable in
                   config[ocean][zonal_section][vars]
  timeseries     — pre-integrated global quantities from ocean_scalar.nc,
                   listed in config[ocean][timeseries][vars]
"""

from pathlib import Path
from typing import Callable

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
    "mass_total": "kg",
    "salt_total": "kg",
    "temp_total": "K·kg",
    "ke_tot": "J",
}

_SYMMETRIC = {"lw_heat", "sens_heat", "evap_heat"}


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
    ds = data["ocean_month"]
    grid_cfg = config["grids"]["ocean"]
    diag_cfg = config["diagnostics"]["ocean"].get("map_2d", {})
    out_cfg = config["output"]

    lat2d = ds[grid_cfg["lat"]].values
    lon2d = ds[grid_cfg["lon"]].values

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocean/map_2d] skipping '{varname}' — not in dataset")
            continue
        try:
            agg = aggregate(ds[varname], mode)
        except ValueError as exc:
            print(f"  [ocean/map_2d] skipping '{varname}': {exc}")
            continue

        out_path = output_dir / "ocean" / "map_2d" / f"{varname}.{out_cfg['format']}"
        plot_utils.global_map(
            data=agg,
            lat=lat2d,
            lon=lon2d,
            title=f"Ocean {varname} — {mode}",
            output_path=out_path,
            units=_UNITS.get(varname, ""),
            symmetric=varname in _SYMMETRIC,
            n_levels=out_cfg.get("n_levels", 20),
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocean/map_2d] saved {out_path}")


def _run_zonal_section(
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    ds = data["ocean_month"]
    diag_cfg = config["diagnostics"]["ocean"].get("zonal_section", {})
    out_cfg = config["output"]

    lat1d = ds["yt_ocean"].values
    depth = ds["st_ocean"].values

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocean/zonal_section] skipping '{varname}' — not in dataset")
            continue
        try:
            agg = aggregate(ds[varname], mode)
        except ValueError as exc:
            print(f"  [ocean/zonal_section] skipping '{varname}': {exc}")
            continue

        zonal = agg.mean(dim="xt_ocean")
        out_path = (
            output_dir / "ocean" / "zonal_section" / f"{varname}.{out_cfg['format']}"
        )
        plot_utils.zonal_section(
            data=zonal,
            lat=lat1d,
            depth=depth,
            title=f"Ocean {varname} zonal mean — {mode}",
            output_path=out_path,
            units=_UNITS.get(varname, ""),
            symmetric=varname in _SYMMETRIC,
            n_levels=out_cfg.get("n_levels", 20),
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocean/zonal_section] saved {out_path}")


def _run_timeseries(
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    ds = data["ocean_scalar"]
    diag_cfg = config["diagnostics"]["ocean"].get("timeseries", {})
    out_cfg = config["output"]

    for varname in diag_cfg.get("vars", []):
        if varname not in ds:
            print(f"  [ocean/timeseries] skipping '{varname}' — not in dataset")
            continue

        da = ds[varname].squeeze()
        times = _to_numeric_years(da)
        out_path = output_dir / "ocean" / "timeseries" / f"{varname}.{out_cfg['format']}"
        plot_utils.time_series(
            times=times,
            values=da.values.astype(float),
            title=f"Ocean scalar {varname}",
            ylabel=_UNITS.get(varname, varname),
            output_path=out_path,
            dpi=out_cfg["dpi"],
        )
        print(f"  [ocean/timeseries] saved {out_path}")


def _to_numeric_years(da: xr.DataArray) -> np.ndarray:
    try:
        return np.array([v.year + (v.month - 1) / 12.0 for v in da.time.values])
    except AttributeError:
        return da.time.values.astype(float)


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
    "zonal_section": _run_zonal_section,
    "timeseries": _run_timeseries,
}
