"""
LPJ-mL land model diagnostic maps.

Produces:
  - Global maps for monthly-frequency variables (grouped by theme)
  - Global maps for annual-frequency variables
  - Zonal-mean soil-layer depth profiles for packed soil variables

Variables are loaded on demand via the lpjml loader callable rather than
all at once, because each variable lives in a separate file.

Time handling:
  Monthly variables (m-prefix): decoded month indices via lpjml_month_coord().
  Annual variables (no m-prefix): time axis represents years; seasonal
    aggregation is not applicable and "annual" is always used regardless
    of the requested mode. A warning is printed if the user requests
    a seasonal mode for an annual-frequency variable.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from src.data_loader import lpjml_month_coord, load_lpjml_var
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
}

# Annual-frequency variables — seasonal mode not applicable.
_ANNUAL_ONLY: frozenset[str] = frozenset([
    "agb", "vegc", "soilc", "litc", "firec", "flux_harvest",
    "aconv_loss_drain", "aconv_loss_evap",
])

# Soil-layer packed variables for which a depth profile is produced.
_SOIL_LAYER_VARS: frozenset[str] = frozenset(["msoiltemp", "soilc_layer"])


def run(
    lpjml_loader: Callable[[str], xr.DataArray],
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    """
    Generate LPJ-mL land model diagnostic maps.

    Parameters
    ----------
    lpjml_loader:
        Callable returned by data_loader.make_lpjml_loader(). Accepts a
        variable name and returns the corresponding xr.DataArray.
    config:
        Full experiment config dict.
    mode:
        Temporal aggregation mode. Seasonal modes are silently replaced
        by "annual" for annual-frequency variables.
    output_dir:
        Root output directory. Figures saved under output_dir / "land_lpjml".
    """
    diag_cfg = config["diagnostics"]["land_lpjml"]
    out_cfg = config["output"]
    grid_cfg = config["grids"]["lpjml"]

    # Load land mask and area weights once.
    mask, area = _load_mask_and_area(lpjml_loader, grid_cfg)

    # Monthly variables
    monthly_groups: dict[str, list[str]] = diag_cfg.get("monthly_vars", {})
    for group, varnames in monthly_groups.items():
        for varname in varnames:
            _plot_monthly(
                varname=varname,
                lpjml_loader=lpjml_loader,
                mode=mode,
                mask=mask,
                out_cfg=out_cfg,
                output_dir=output_dir,
            )

    # Annual variables — always use "annual" regardless of requested mode.
    annual_groups: dict[str, list[str]] = diag_cfg.get("annual_vars", {})
    for group, varnames in annual_groups.items():
        for varname in varnames:
            _plot_annual(
                varname=varname,
                lpjml_loader=lpjml_loader,
                mode=mode,
                mask=mask,
                out_cfg=out_cfg,
                output_dir=output_dir,
            )


# ── Internal per-variable helpers ─────────────────────────────────────────────

def _plot_monthly(
    varname: str,
    lpjml_loader: Callable,
    mode: str | int,
    mask: np.ndarray | None,
    out_cfg: dict,
    output_dir: Path,
) -> None:
    try:
        da = lpjml_loader(varname)
    except FileNotFoundError as exc:
        print(f"  [lpjml] skipping '{varname}': {exc}")
        return

    # Soil-layer packed variables: plot a zonal depth profile instead.
    if varname in _SOIL_LAYER_VARS:
        _plot_soil_section(da, varname, mode, out_cfg, output_dir)
        return

    month_coord = lpjml_month_coord(da)
    try:
        agg = aggregate(da, mode, month_coord=month_coord)
    except ValueError as exc:
        print(f"  [lpjml] skipping '{varname}': {exc}")
        return

    values = np.where(mask, np.asarray(agg), np.nan) if mask is not None else np.asarray(agg)
    lat = da.lat.values
    lon = da.lon.values
    units = _UNITS.get(varname, "")
    out_path = output_dir / "land_lpjml" / f"{varname}.{out_cfg['format']}"

    _make_data_array(values, lat, lon, varname)
    plot_utils.global_map(
        data=_make_data_array(values, lat, lon, varname),
        lat=lat,
        lon=lon,
        title=f"LPJ-mL {varname} — {mode}",
        output_path=out_path,
        units=units,
        dpi=out_cfg["dpi"],
    )
    print(f"  [lpjml] saved {out_path}")


def _plot_annual(
    varname: str,
    lpjml_loader: Callable,
    mode: str | int,
    mask: np.ndarray | None,
    out_cfg: dict,
    output_dir: Path,
) -> None:
    try:
        da = lpjml_loader(varname)
    except FileNotFoundError as exc:
        print(f"  [lpjml] skipping '{varname}': {exc}")
        return

    if mode != "annual":
        print(
            f"  [lpjml] '{varname}' is annual-frequency; "
            f"ignoring mode '{mode}' and using 'annual'"
        )

    agg = da.mean("time")

    values = np.where(mask, np.asarray(agg), np.nan) if mask is not None else np.asarray(agg)
    lat = da.lat.values
    lon = da.lon.values
    units = _UNITS.get(varname, "")
    out_path = output_dir / "land_lpjml" / f"{varname}.{out_cfg['format']}"

    plot_utils.global_map(
        data=_make_data_array(values, lat, lon, varname),
        lat=lat,
        lon=lon,
        title=f"LPJ-mL {varname} — annual mean",
        output_path=out_path,
        units=units,
        dpi=out_cfg["dpi"],
    )
    print(f"  [lpjml] saved {out_path}")


def _plot_soil_section(
    da: xr.DataArray,
    varname: str,
    mode: str | int,
    out_cfg: dict,
    output_dir: Path,
) -> None:
    """Produce a zonal-mean latitude–depth section for a soil-layer variable."""
    # da has dims (time, soil_layer, lat, lon) after unpacking.
    month_coord = lpjml_month_coord(
        # Reconstruct a 1-D time proxy from the packed time dim.
        xr.DataArray(
            da.time.values[: da.sizes["time"]],
            dims=["time"],
            name="time",
        )
    )
    try:
        agg = aggregate(da, mode, month_coord=month_coord)
    except ValueError as exc:
        print(f"  [lpjml] skipping '{varname}': {exc}")
        return

    # Zonal mean over longitude.
    zonal = agg.mean(dim="lon")
    n_layers = zonal.sizes["soil_layer"]
    lat = da.lat.values
    depth = np.arange(n_layers, dtype=float)

    units = _UNITS.get(varname, "")
    out_path = output_dir / "land_lpjml" / f"{varname}_zonal_section.{out_cfg['format']}"

    plot_utils.zonal_section(
        data=zonal,
        lat=lat,
        depth=depth,
        title=f"LPJ-mL {varname} zonal mean — {mode}",
        output_path=out_path,
        units=units,
        dpi=out_cfg["dpi"],
    )
    print(f"  [lpjml] saved {out_path}")


def _load_mask_and_area(
    lpjml_loader: Callable,
    grid_cfg: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the land mask and area arrays; return (None, None) if unavailable."""
    mask_var = grid_cfg.get("mask_var")
    area_var = grid_cfg.get("area_var")
    mask, area = None, None
    try:
        if mask_var:
            da_mask = lpjml_loader(mask_var)
            mask = np.asarray(da_mask.isel(time=0)) != 0
    except (FileNotFoundError, KeyError):
        pass
    try:
        if area_var:
            da_area = lpjml_loader(area_var)
            area = np.asarray(da_area.isel(time=0))
    except (FileNotFoundError, KeyError):
        pass
    return mask, area


def _make_data_array(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    name: str,
) -> xr.DataArray:
    return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name=name)
