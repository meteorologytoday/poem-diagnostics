"""
Temporal aggregation over the time axis of an xarray DataArray.

Supports:
  - "annual"       : mean over all time steps
  - "DJF"/"MAM"/"JJA"/"SON" : mean over the named season
  - int 1-12       : mean over a single calendar month

POEM components carry a cftime/datetime64 time axis, so month extraction
uses xarray's .dt accessor. LPJ-mL data is opened with decode_times=False,
so callers must supply a precomputed month_coord array (see data_loader).
"""

import numpy as np
import xarray as xr

SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

VALID_MODES = {"annual"} | set(SEASON_MONTHS) | set(range(1, 13))


def aggregate(
    da: xr.DataArray,
    mode: str | int,
    month_coord: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Collapse the 'time' dimension of *da* according to *mode*.

    Parameters
    ----------
    da:
        Input DataArray with a 'time' dimension.
    mode:
        "annual" for the full-period mean; a season code ("DJF", "MAM",
        "JJA", "SON") for the seasonal climatology; or an integer 1-12
        for a single-month climatology.
    month_coord:
        Integer array (1-12) with one entry per time step. Required when
        da.time is not a datetime64/cftime axis — e.g. lpjml data opened
        with decode_times=False. Ignored when da.time supports .dt.month.

    Returns
    -------
    xr.DataArray
        Time-collapsed array preserving all other dimensions.

    Raises
    ------
    ValueError
        On an unrecognised mode or a missing month_coord for raw time axes.
    """
    if mode == "annual":
        return da.mean("time")

    months = _resolve_months(da, month_coord)

    if isinstance(mode, int):
        if mode not in range(1, 13):
            raise ValueError(f"Month index must be 1–12, got {mode!r}")
        mask = months == mode
    elif mode in SEASON_MONTHS:
        mask = np.isin(months, SEASON_MONTHS[mode])
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            f"Use 'annual', a season code {sorted(SEASON_MONTHS)}, or 1–12."
        )

    selected = da.isel(time=np.where(mask)[0])
    if selected.sizes["time"] == 0:
        raise ValueError(
            f"Mode {mode!r} matched no time steps. "
            "Check that month_coord covers the expected range."
        )
    return selected.mean("time")


def _resolve_months(
    da: xr.DataArray,
    month_coord: np.ndarray | None,
) -> np.ndarray:
    """Return an integer (1-12) array aligned with da.time."""
    if month_coord is not None:
        arr = np.asarray(month_coord, dtype=int)
        if arr.shape != (da.sizes["time"],):
            raise ValueError(
                f"month_coord length {arr.shape} does not match "
                f"time axis length {da.sizes['time']}"
            )
        return arr

    try:
        return da.time.dt.month.values.astype(int)
    except AttributeError:
        raise ValueError(
            "Cannot extract month from the time axis using .dt.month. "
            "Pass month_coord explicitly (required for lpjml data opened "
            "with decode_times=False)."
        )
