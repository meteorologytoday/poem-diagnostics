"""
Shared plotting utilities for POEM diagnostic figures.

All map figures are produced with cartopy. The helpers here handle:
  - cartopy projection construction
  - standard global and polar map layouts
  - colorbar attachment
  - figure persistence (directory creation + save)

Diagnostic modules should call these helpers rather than constructing
figures directly so that style changes propagate everywhere at once.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / SSH sessions
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

# Features added to every map by default.
_COASTLINES_KW = dict(linewidth=0.5, color="black")

# Default colormaps keyed by variable name. Diagnostic modules fall back
# to "viridis" for variables not listed here.
DEFAULT_CMAPS: dict[str, str] = {
    # atmosphere
    "t_ref": "RdBu_r",
    "t_surf": "RdBu_r",
    "netrad_toa": "RdBu_r",
    "slp": "viridis",
    "precip": "Blues",
    "olr": "YlOrRd_r",
    "swdn_toa": "YlOrRd",
    # ocean
    "sst": "RdBu_r",
    "sss": "PuBu",
    "temp": "RdBu_r",
    "salt": "PuBu",
    "mld": "viridis",
    "swflx": "YlOrRd",
    "lw_heat": "RdBu_r",
    "sens_heat": "RdBu_r",
    "evap_heat": "RdBu_r",
    # sea ice
    "CN": "Blues_r",
    "HI": "viridis",
    "EXT": "Blues_r",
    # land / lpjml
    "mgpp": "Greens",
    "mnpp": "Greens",
    "mrh": "YlOrRd",
    "agb": "YlGn",
    "vegc": "YlGn",
    "soilc": "YlOrBr",
    "litc": "YlOrBr",
    "firec": "YlOrRd",
    "mevap": "Blues",
    "mrunoff": "Blues",
    "mdischarge": "Blues",
    "mpet": "Blues",
    "malbedo": "Greys",
    "soil_surf_temp": "RdBu_r",
    "mswc1": "BrBG",
    "mswc2": "BrBG",
    "swe": "Blues",
    "mburnt_area": "YlOrRd",
}


# ── Projection factory ────────────────────────────────────────────────────────

def get_projection(name: str, **kwargs) -> ccrs.CRS:
    """
    Return a cartopy CRS by name.

    Supported names: "PlateCarree", "NorthPolarStereo", "SouthPolarStereo".
    Extra keyword arguments are forwarded to the cartopy constructor.
    """
    projections: dict[str, type] = {
        "PlateCarree": ccrs.PlateCarree,
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }
    if name not in projections:
        raise ValueError(
            f"Unknown projection {name!r}. "
            f"Choose from: {sorted(projections)}"
        )
    return projections[name](**kwargs)


# ── Map figure builders ───────────────────────────────────────────────────────

def global_map(
    data: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    dpi: int = 150,
) -> None:
    """
    Save a single global map of *data* on a PlateCarree projection.

    Parameters
    ----------
    data:
        2-D array (lat × lon) of values to plot.
    lat, lon:
        1-D or 2-D coordinate arrays matching the shape of *data*.
    title:
        Figure title string.
    output_path:
        Full path for the output PNG file. Parent directories are created
        automatically.
    cmap:
        Matplotlib colormap name. Defaults to DEFAULT_CMAPS or "viridis".
    units:
        Label appended to the colorbar.
    vmin, vmax:
        Color scale limits. When both are None they are derived from the
        2nd and 98th percentiles of non-NaN values.
    symmetric:
        Force a symmetric color scale around zero (overrides vmin/vmax).
    n_levels:
        Number of discrete color levels. Default: 20.
    dpi:
        Output resolution.
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": proj},
    )
    _add_map_features(ax)

    mesh = ax.pcolormesh(
        lon, lat, values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def polar_map(
    data: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    hemisphere: str,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    dpi: int = 150,
) -> None:
    """
    Save a polar stereographic map restricted to one hemisphere.

    Parameters
    ----------
    hemisphere:
        "NH" for North (NorthPolarStereo) or "SH" for South
        (SouthPolarStereo).
    All other parameters match global_map().
    """
    if hemisphere == "NH":
        proj = ccrs.NorthPolarStereo()
        extent = [-180, 180, 50, 90]
    elif hemisphere == "SH":
        proj = ccrs.SouthPolarStereo()
        extent = [-180, 180, -90, -50]
    else:
        raise ValueError(f"hemisphere must be 'NH' or 'SH', got {hemisphere!r}")

    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": proj},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    _add_map_features(ax)

    mesh = ax.pcolormesh(
        lon, lat, values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def zonal_section(
    data: xr.DataArray,
    lat: np.ndarray,
    depth: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    n_levels: int = 20,
    dpi: int = 150,
) -> None:
    """
    Save a latitude–depth section of the zonal mean of *data*.

    *data* is expected to have dimensions (depth, lat) after any prior
    averaging. The depth axis is plotted increasing downward.
    """
    values = np.asarray(data)
    cmap = cmap or DEFAULT_CMAPS.get(str(data.name), "viridis")
    vmin, vmax = _resolve_clim(values, vmin, vmax, symmetric)
    norm = _make_norm(cmap, vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        lat, depth, values,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth (dbar)")
    _add_colorbar(fig, ax, mesh, units)
    ax.set_title(title, fontsize=11)
    _save(fig, output_path, dpi)


def time_series(
    times: np.ndarray,
    values: np.ndarray | dict[str, np.ndarray],
    title: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 150,
) -> None:
    """Save a line plot of *values* against *times*.

    *values* may be a 1-D array (single line) or a dict mapping region
    names to 1-D arrays (one labelled line per region).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if isinstance(values, dict):
        for label, vals in values.items():
            ax.plot(times, vals, linewidth=1.0, label=label)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.plot(times, values, linewidth=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_norm(
    cmap: str,
    vmin: float,
    vmax: float,
    n_levels: int,
) -> mcolors.BoundaryNorm:
    bounds = np.linspace(vmin, vmax, n_levels + 1)
    return mcolors.BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap).N)


def _add_map_features(ax) -> None:
    ax.coastlines(**_COASTLINES_KW)
    ax.set_global()


def _add_colorbar(fig, ax, mesh, units: str) -> None:
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    if units:
        cbar.set_label(units, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _resolve_clim(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric: bool,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if vmin is None:
        vmin = float(np.percentile(finite, 2))
    if vmax is None:
        vmax = float(np.percentile(finite, 98))
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    return vmin, vmax


def _save(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
