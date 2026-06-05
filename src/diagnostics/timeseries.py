"""
Global time-series diagnostics.

Produces:
  - Time series from ocean_scalar.nc (pre-integrated global quantities)
  - Area-weighted global-mean time series for selected atmosphere variables

These plots span the full simulation period and are not subject to
temporal aggregation (the "mode" argument is accepted but unused here,
because averaging time series over a season would collapse them to a
scalar and defeat the purpose of the plot).
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src import plot_utils


_UNITS: dict[str, str] = {
    "mass_total": "kg",
    "salt_total": "kg",
    "temp_total": "K·kg",
    "ke_tot": "J",
    "t_ref": "K",
    "precip": "kg m⁻² s⁻¹",
    "olr": "W m⁻²",
}


def run(
    ds_ocean_scalar: xr.Dataset,
    ds_atmos: xr.Dataset,
    config: dict,
    mode: str | int,
    output_dir: Path,
) -> None:
    """
    Generate global time-series figures.

    Parameters
    ----------
    ds_ocean_scalar:
        "ocean_scalar" Dataset loaded by data_loader.load_all().
    ds_atmos:
        "atmos_month" Dataset for computing global-mean atmosphere fields.
    config:
        Full experiment config dict.
    mode:
        Accepted for API consistency but not used here; time series always
        span the full period.
    output_dir:
        Root output directory. Figures saved under output_dir / "timeseries".
    """
    diag_cfg = config["diagnostics"]["timeseries"]
    out_cfg = config["output"]

    # Ocean scalar time series — values are already globally integrated.
    for varname in diag_cfg.get("ocean_scalar", []):
        if varname not in ds_ocean_scalar:
            print(f"  [timeseries] skipping '{varname}' — not in ocean_scalar dataset")
            continue
        da = ds_ocean_scalar[varname].squeeze()
        _plot_series(da, varname, "ocean_scalar", out_cfg, output_dir)

    # Atmosphere global-mean time series — area-weighted average.
    lat_weights = _atmos_lat_weights(ds_atmos)
    for varname in diag_cfg.get("atmos_global", []):
        if varname not in ds_atmos:
            print(f"  [timeseries] skipping '{varname}' — not in atmos_month dataset")
            continue
        da = ds_atmos[varname]
        # Select surface/2-D field only (skip 3-D variables with pfull).
        if "pfull" in da.dims:
            print(
                f"  [timeseries] '{varname}' has a vertical dimension; "
                "skipping global mean (plot a profile instead)"
            )
            continue
        global_mean = da.weighted(lat_weights).mean(dim=["lat", "lon"])
        _plot_series(global_mean, varname, "atmos_global", out_cfg, output_dir)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _plot_series(
    da: xr.DataArray,
    varname: str,
    group: str,
    out_cfg: dict,
    output_dir: Path,
) -> None:
    times = _numeric_times(da)
    values = da.values.astype(float)
    units = _UNITS.get(varname, "")
    out_path = output_dir / "timeseries" / f"{group}_{varname}.{out_cfg['format']}"

    plot_utils.time_series(
        times=times,
        values=values,
        title=f"{group} — {varname}",
        ylabel=units or varname,
        output_path=out_path,
        dpi=out_cfg["dpi"],
    )
    print(f"  [timeseries] saved {out_path}")


def _atmos_lat_weights(ds: xr.Dataset) -> xr.DataArray:
    """Return cosine-of-latitude weights on the atmosphere grid."""
    lat = ds["lat"]
    weights = np.cos(np.deg2rad(lat))
    weights = weights.where(weights > 0, other=0.0)
    return weights


def _numeric_times(da: xr.DataArray) -> np.ndarray:
    """
    Return a numeric 1-D array suitable for the x-axis of a time series.

    For cftime/datetime64 axes, returns fractional years relative to the
    first time step. For raw numeric axes (e.g. from decode_times=False),
    returns the raw values as-is.
    """
    try:
        t = da.time.values
        # cftime objects support .year attribute.
        year0 = float(t[0].year)
        return np.array([v.year + (v.month - 1) / 12.0 for v in t])
    except AttributeError:
        pass
    # numpy datetime64
    try:
        t64 = da.time.values.astype("datetime64[M]")
        origin = t64[0].astype("datetime64[Y]").astype(float)
        return (t64 - t64[0]).astype(float) / 12.0
    except Exception:
        pass
    return da.time.values.astype(float)
