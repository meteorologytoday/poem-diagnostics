"""
LPJ-mL land model diagnostic plots.

Supported diag_types:
  map_2d         — global maps for monthly and annual LPJ-mL variables
                   listed in config[land][map_2d]
  zonal_section  — latitude-depth profiles for packed soil-layer variables
                   listed in config[land][zonal_section][vars]

Variables are loaded on demand via the lpjml loader callable; each
variable lives in a separate file, so loading is deferred until needed.

Time handling:
  Monthly variables (m-prefix): month indices derived via lpjml_month_coord().
  Annual variables: mode is always forced to "annual" — averaging annual
    output over a season would be meaningless.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from src.data_loader import lpjml_month_coord
from src.temporal_agg import aggregate
from src import plot_utils


_UNITS: dict[str, str] = {
    "mgpp": "g C m⁻² month⁻¹",
    "mnpp": "g C m⁻² month⁻¹",
    "mrh": "g C m⁻² month⁻¹",
    "agb": "g C m⁻²",
    "vegc": "g C m⁻²",
    "soilc": "g C m⁻²",
    "litc": "g C m⁻²",
    "firec": "g C m⁻²",
    "mevap": "mm month⁻¹",
    "mrunoff": "mm month⁻¹",
    "mdischarge": "m³ s⁻¹",
    "mpet": "mm month⁻¹",
    "malbedo": "fraction",
    "soil_surf_temp": "°C",
    "mswc1": "fraction",
    "mswc2": "fraction",
    "swe": "mm",
    "mburnt_area": "fraction",
    "msoiltemp": "°C",
    "soilc_layer": "g C m⁻²",
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
    loader = data["lpjml"]
    diag_cfg = config["diagnostics"]["land"].get("map_2d", {})
    grid_cfg = config["grids"]["lpjml"]
    out_cfg = config["output"]
    year_label = data.get("year_label", "")

    mask = _load_mask(loader, grid_cfg)

    monthly_groups: dict = diag_cfg.get("monthly_vars", {})
    for varnames in monthly_groups.values():
        for varname in varnames:
            _plot_monthly_map(varname, loader, mode, mask, out_cfg, year_label, output_dir)

    annual_groups: dict = diag_cfg.get("annual_vars", {})
    for varnames in annual_groups.values():
        for varname in varnames:
            _plot_annual_map(varname, loader, mode, mask, out_cfg, year_label, output_dir)


def _run_zonal_section(
    data: dict,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    loader = data["lpjml"]
    diag_cfg = config["diagnostics"]["land"].get("zonal_section", {})
    out_cfg = config["output"]
    year_label = data.get("year_label", "")

    for varname in diag_cfg.get("vars", []):
        _plot_soil_section(varname, loader, mode, out_cfg, year_label, output_dir)


# ── Per-variable helpers ──────────────────────────────────────────────────────

def _plot_monthly_map(
    varname: str,
    loader: Callable,
    mode: str | int,
    mask: np.ndarray | None,
    out_cfg: dict,
    year_label: str,
    output_dir: Path,
) -> None:
    try:
        da = loader(varname)
    except FileNotFoundError as exc:
        print(f"  [land/map_2d] skipping '{varname}': {exc}")
        return

    month_coord = lpjml_month_coord(da)
    try:
        agg = aggregate(da, mode, month_coord=month_coord)
    except ValueError as exc:
        print(f"  [land/map_2d] skipping '{varname}': {exc}")
        return

    values = np.where(mask, np.asarray(agg), np.nan) if mask is not None else np.asarray(agg)
    out_path = output_dir / "land" / "map_2d" / f"{varname}_{year_label}.{out_cfg['format']}"
    plot_utils.global_map(
        data=_wrap(values, da, varname),
        lat=da.lat.values,
        lon=da.lon.values,
        title=f"LPJ-mL {varname} — {mode} ({year_label})",
        output_path=out_path,
        units=_UNITS.get(varname, ""),
        n_levels=out_cfg.get("n_levels", 20),
        dpi=out_cfg["dpi"],
    )
    print(f"  [land/map_2d] saved {out_path}")


def _plot_annual_map(
    varname: str,
    loader: Callable,
    mode: str | int,
    mask: np.ndarray | None,
    out_cfg: dict,
    year_label: str,
    output_dir: Path,
) -> None:
    try:
        da = loader(varname)
    except FileNotFoundError as exc:
        print(f"  [land/map_2d] skipping '{varname}': {exc}")
        return

    if mode != "annual":
        print(
            f"  [land/map_2d] '{varname}' is annual-frequency; "
            f"ignoring mode '{mode}', using 'annual'"
        )
    agg = da.mean("time")

    values = np.where(mask, np.asarray(agg), np.nan) if mask is not None else np.asarray(agg)
    out_path = output_dir / "land" / "map_2d" / f"{varname}_{year_label}.{out_cfg['format']}"
    plot_utils.global_map(
        data=_wrap(values, da, varname),
        lat=da.lat.values,
        lon=da.lon.values,
        title=f"LPJ-mL {varname} — annual mean ({year_label})",
        output_path=out_path,
        units=_UNITS.get(varname, ""),
        n_levels=out_cfg.get("n_levels", 20),
        dpi=out_cfg["dpi"],
    )
    print(f"  [land/map_2d] saved {out_path}")


def _plot_soil_section(
    varname: str,
    loader: Callable,
    mode: str | int,
    out_cfg: dict,
    year_label: str,
    output_dir: Path,
) -> None:
    try:
        da = loader(varname)
    except FileNotFoundError as exc:
        print(f"  [land/zonal_section] skipping '{varname}': {exc}")
        return

    if "soil_layer" not in da.dims:
        print(
            f"  [land/zonal_section] skipping '{varname}' — "
            "no soil_layer dimension (check packed_vars config)"
        )
        return

    month_coord = lpjml_month_coord(
        xr.DataArray(da.time.values[: da.sizes["time"]], dims=["time"], name="time")
    )
    try:
        agg = aggregate(da, mode, month_coord=month_coord)
    except ValueError as exc:
        print(f"  [land/zonal_section] skipping '{varname}': {exc}")
        return

    zonal = agg.mean(dim="lon")
    n_layers = zonal.sizes["soil_layer"]
    out_path = output_dir / "land" / "zonal_section" / f"{varname}_{year_label}.{out_cfg['format']}"
    plot_utils.zonal_section(
        data=zonal,
        lat=da.lat.values,
        depth=np.arange(n_layers, dtype=float),
        title=f"LPJ-mL {varname} zonal mean — {mode} ({year_label})",
        output_path=out_path,
        units=_UNITS.get(varname, ""),
        n_levels=out_cfg.get("n_levels", 20),
        dpi=out_cfg["dpi"],
    )
    print(f"  [land/zonal_section] saved {out_path}")


def _load_mask(
    loader: Callable,
    grid_cfg: dict,
) -> np.ndarray | None:
    mask_var = grid_cfg.get("mask_var")
    if not mask_var:
        return None
    try:
        da = loader(mask_var)
        return np.asarray(da.isel(time=0)) != 0
    except (FileNotFoundError, KeyError):
        return None


def _wrap(
    values: np.ndarray,
    da: xr.DataArray,
    name: str,
) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["lat", "lon"],
        coords={"lat": da.lat.values, "lon": da.lon.values},
        name=name,
    )


HANDLERS: dict[str, Callable] = {
    "map_2d": _run_map_2d,
    "zonal_section": _run_zonal_section,
}
