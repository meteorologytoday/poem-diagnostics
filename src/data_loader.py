"""
Data loading for POEM coupled-model and LPJ-mL land-model output.

Two distinct loading paths exist because the two model families use
different time-axis encodings:

  POEM components (atmos/ocean/ice/land)
    - One NetCDF file per epoch per component: {epoch}.{component}.nc
    - Epochs discovered by scanning history_dir
    - Loaded with xr.open_mfdataset (combine="by_coords", lazy)
    - decode_timedelta=False suppresses the xarray FutureWarning on
      the 'average_DT' variable present in POEM output

  LPJ-mL
    - One NetCDF file per variable per epoch: lpjml_{epoch}/{varname}.nc
    - Opened with decode_times=False because the time units
      ("years since 1901-1-1") use a non-standard encoding that cftime
      cannot decode directly
    - Some variables pack N levels (soil layers or PFTs) consecutively
      into the time axis; _unpack_lpjml reshapes these to
      (time, level, lat, lon)
    - Month indices for seasonal aggregation are derived via
      lpjml_month_coord(), which decodes the fractional-year time values
      assuming a 365-day (noleap) calendar
"""

import re
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr

# Day-of-year at the start of each month under a 365-day (noleap) calendar.
# Index 0 = January (DOY 0), index 11 = December (DOY 334).
_NOLEAP_MONTH_START_DOY = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    dtype=float,
)


# ── Epoch discovery ───────────────────────────────────────────────────────────

def discover_epochs(
    history_dir: str | Path,
    year_range: tuple[int, int] | None = None,
) -> list[str]:
    """
    Return a sorted list of epoch strings found in *history_dir*.

    Epochs are inferred from POEM output filenames of the form
    {epoch}.{component}.nc where epoch is a 10-digit date string
    (e.g. "0019070101"). The year is the first 4 digits of the epoch.

    Parameters
    ----------
    history_dir:
        Directory containing POEM history output files.
    year_range:
        Optional (start_year, end_year) tuple (both inclusive). Epochs
        whose year falls outside this range are excluded. When None, all
        discovered epochs are returned.
    """
    history_dir = Path(history_dir)
    pattern = re.compile(r"^(\d{10})\.\w+\.nc$")
    epochs: set[str] = set()
    for entry in history_dir.iterdir():
        m = pattern.match(entry.name)
        if m:
            epochs.add(m.group(1))
    if not epochs:
        raise FileNotFoundError(
            f"No POEM epoch files found in {history_dir}. "
            "Expected files named like '0019070101.atmos_month.nc'."
        )
    all_epochs = sorted(epochs)
    if year_range is None:
        return all_epochs
    start, end = year_range
    filtered = [e for e in all_epochs if start <= int(e[:4]) <= end]
    if not filtered:
        raise ValueError(
            f"No epochs found in year range [{start}, {end}]. "
            f"Available years: {sorted({int(e[:4]) for e in all_epochs})}"
        )
    return filtered


# ── POEM component loading ────────────────────────────────────────────────────

def load_poem_component(
    history_dir: str | Path,
    component_key: str,
    component_cfg: dict,
    epochs: list[str],
) -> xr.Dataset:
    """
    Load all epoch files for one POEM component into a single Dataset.

    Files are opened lazily with open_mfdataset and concatenated along
    the time coordinate. Missing epoch files are silently skipped so that
    partial runs load without error.

    Parameters
    ----------
    history_dir:
        Path to the model history directory.
    component_key:
        Human-readable name used in error messages (e.g. "atmos_month").
    component_cfg:
        Dict from config["data"]["components"][component_key]. Must
        contain "pattern" (str with {epoch} placeholder).
    epochs:
        Sorted list of epoch strings returned by discover_epochs().
    """
    history_dir = Path(history_dir)
    paths = []
    for epoch in epochs:
        p = history_dir / component_cfg["pattern"].format(epoch=epoch)
        if p.exists():
            paths.append(p)

    if not paths:
        raise FileNotFoundError(
            f"No files found for component '{component_key}' in {history_dir}."
        )

    # Use CFDatetimeCoder to force cftime objects — avoids out-of-range
    # numpy datetime64 errors for model years outside 1678–2261.
    # CFTimedeltaCoder with use_timedelta=False suppresses the FutureWarning
    # on the 'average_DT' variable present in POEM output.
    decode_timedelta = component_cfg.get("decode_timedelta", False)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    timedelta_coder = xr.coders.CFTimedeltaCoder() if decode_timedelta else False
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        decode_times=time_coder,
        decode_timedelta=timedelta_coder,
        data_vars="minimal",
        compat="no_conflicts",
    )


# ── LPJ-mL loading ───────────────────────────────────────────────────────────

def load_lpjml_var(
    history_dir: str | Path,
    lpjml_cfg: dict,
    varname: str,
    epochs: list[str],
) -> xr.DataArray:
    """
    Load a single LPJ-mL variable across all epochs.

    Epoch files are concatenated along their raw time axis. Packed
    variables (soil layers or PFTs encoded in the time axis) are
    reshaped to (time, level, lat, lon) automatically.

    Parameters
    ----------
    history_dir:
        Path to the model history directory.
    lpjml_cfg:
        Dict from config["data"]["lpjml"]. Must contain "subdir_pattern"
        and optionally "packed_vars".
    varname:
        Variable name, which must match the filename and the in-file
        variable name (e.g. "mgpp" → mgpp.nc contains variable "mgpp").
    epochs:
        Sorted list of epoch strings returned by discover_epochs().
    """
    history_dir = Path(history_dir)
    subdir_pattern: str = lpjml_cfg["subdir_pattern"]
    packed_vars: dict = lpjml_cfg.get("packed_vars", {})

    arrays: list[xr.DataArray] = []
    for epoch in epochs:
        subdir = history_dir / subdir_pattern.format(epoch=epoch)
        fpath = subdir / f"{varname}.nc"
        if not fpath.exists():
            raise FileNotFoundError(
                f"LPJ-mL variable file not found: {fpath}"
            )
        ds = xr.open_dataset(fpath, decode_times=False)
        if varname not in ds:
            raise KeyError(
                f"Variable '{varname}' not found in {fpath}. "
                f"Available: {list(ds.data_vars)}"
            )
        arrays.append(ds[varname])

    da = xr.concat(arrays, dim="time")

    if varname in packed_vars:
        da = _unpack_lpjml(da, **packed_vars[varname])

    return da


def make_lpjml_loader(
    history_dir: str | Path,
    lpjml_cfg: dict,
    epochs: list[str],
):
    """
    Return a callable ``loader(varname) -> xr.DataArray`` bound to the
    given history directory, config, and epochs. Diagnostic modules use
    this to load variables on demand without receiving the full config.
    """
    return partial(load_lpjml_var, history_dir, lpjml_cfg, epochs=epochs)


def lpjml_month_coord(da: xr.DataArray) -> np.ndarray:
    """
    Derive integer calendar months (1-12) for each time step of an
    LPJ-mL monthly DataArray opened with decode_times=False.

    LPJ-mL encodes time as fractional years since 1901-01-01 on a
    365-day (noleap) calendar. The fractional part of each value gives
    the day-of-year fraction, from which the month is recovered by
    comparing against the noleap month-start DOY table.

    This function assumes uniformly monthly output. Call it only on
    variables whose time axis represents monthly steps.
    """
    time_vals = da.time.values.astype(float)
    doy = (time_vals % 1.0) * 365.0
    # searchsorted with side='right' returns the number of month-start
    # boundaries that doy exceeds, which equals the 1-based month index.
    months = np.searchsorted(_NOLEAP_MONTH_START_DOY, doy, side="right").astype(int)
    # Clamp to [1, 12] as a safety measure against floating-point edge cases.
    return np.clip(months, 1, 12)


# ── Full load orchestration ───────────────────────────────────────────────────

def load_all(config: dict) -> dict:
    """
    Load all POEM components configured in *config* and build an lpjml
    loader callable. Returns a dict with the following keys:

      "atmos_month", "ocean_month", "ocean_scalar", "ice_month",
      "land_month"  — xr.Dataset (lazy, multi-epoch)
      "lpjml"       — callable(varname) -> xr.DataArray

    Components absent from the config are omitted from the result.
    """
    data_cfg = config["data"]
    history_dir = data_cfg["history_dir"]
    year_range = data_cfg.get("year_range")
    if year_range is not None:
        year_range = (int(year_range[0]), int(year_range[1]))
    epochs = discover_epochs(history_dir, year_range=year_range)

    result: dict = {}

    for key, comp_cfg in data_cfg.get("components", {}).items():
        result[key] = load_poem_component(history_dir, key, comp_cfg, epochs)

    lpjml_cfg = data_cfg.get("lpjml")
    if lpjml_cfg:
        result["lpjml"] = make_lpjml_loader(history_dir, lpjml_cfg, epochs)

    # Year range derived from the actual epochs used, available to all modules.
    y_start = epochs[0][:4]
    y_end = epochs[-1][:4]
    result["year_label"] = f"Y{y_start}" if y_start == y_end else f"Y{y_start}-Y{y_end}"

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unpack_lpjml(
    da: xr.DataArray,
    n_levels: int,
    dim_name: str,
) -> xr.DataArray:
    """
    Reshape a packed LPJ-mL DataArray from (N_levels * N_time, lat, lon)
    to (N_time, N_levels, lat, lon).

    LPJ-mL writes multi-level variables (soil layers, PFTs) by
    interleaving levels in the time axis:
      [lev0_t0, lev1_t0, ..., levN_t0, lev0_t1, lev1_t1, ...]
    """
    total = da.sizes["time"]
    if total % n_levels != 0:
        raise ValueError(
            f"Time axis length {total} is not divisible by "
            f"n_levels={n_levels} for packed variable '{da.name}'."
        )
    n_time = total // n_levels
    data = da.values.reshape(n_time, n_levels, da.sizes["lat"], da.sizes["lon"])
    return xr.DataArray(
        data,
        dims=["time", dim_name, "lat", "lon"],
        coords={"lat": da.lat.values, "lon": da.lon.values},
        name=da.name,
        attrs=da.attrs,
    )
