# Developer Guide — POEM Diagnostic Plots

## Purpose

This codebase generates diagnostic figures for POEM coupled-model output
(atmosphere, ocean, sea ice) and LPJ-mL land model output. It is designed
to be flexible: any combination of temporal aggregation modes and diagnostic
modules can be requested on the command line without modifying source code.

---

## Repository layout

```
plotting_poem/
├── config/
│   └── minimal_setup_CTL.yaml   # experiment config (one file per experiment)
├── src/
│   ├── data_loader.py           # data loading for POEM and LPJ-mL
│   ├── temporal_agg.py          # temporal aggregation (annual / seasonal / monthly)
│   ├── plot_utils.py            # shared cartopy map and figure helpers
│   └── diagnostics/
│       ├── atmos_2d.py          # atmosphere 2-D maps
│       ├── ocean_2d.py          # ocean 2-D maps + depth sections
│       ├── ice_2d.py            # sea-ice polar maps
│       ├── land_lpjml.py        # LPJ-mL land model maps
│       └── timeseries.py        # global time series
├── run_diagnostics.py           # CLI entry point (argparse)
└── output/                      # generated figures (gitignored)
```

---

## CLI usage

```bash
python run_diagnostics.py --config config/minimal_setup_CTL.yaml \
                          --modes annual JJA DJF \
                          --diagnostics atmos ocean ice land_lpjml timeseries \
                          --output-dir ./output
```

| Argument | Default | Notes |
|---|---|---|
| `--config` | required | Path to the experiment YAML |
| `--modes` | `annual` | One or more of: `annual DJF MAM JJA SON` or integer `1`–`12` |
| `--diagnostics` | all | Subset of diagnostic modules to run |
| `--output-dir` | from config | Overrides `output.base_dir` in the YAML |
| `--vars` | all | Restrict to specific variable names (useful during development) |

### Output path structure

```
output/{experiment.name}/{mode}/{diagnostic}/{variable}.png
```

Running with a new mode only adds new files; it never overwrites figures
from other modes.

---

## Design principles

### 1. Aggregation is centralised

All temporal aggregation goes through `src/temporal_agg.aggregate()`.
Diagnostic modules never contain groupby or season-selection logic.
Adding a new aggregation mode (e.g. JJAS) means editing only `temporal_agg.py`.

### 2. Config vs CLI

- **Config (YAML)**: stable, experiment-specific facts — data paths, variable
  lists, grid metadata, output format.
- **CLI**: transient per-invocation choices — which modes, which modules,
  where to write. The config is committed to the repo; CLI arguments are not.

### 3. Diagnostic modules are independent

Each module under `src/diagnostics/` has a single `run()` function with
a consistent signature:

```python
def run(data, config, mode, output_dir): ...
```

Modules do not call each other. `run_diagnostics.py` is the only file that
knows all modules exist.

### 4. LPJ-mL data is loaded on demand

LPJ-mL has ~60 variables each in a separate file. The loader returns a
callable `lpjml_loader(varname) -> xr.DataArray` rather than loading
everything upfront. Diagnostic modules request variables individually.

---

## Two data loading paths

### POEM components (atmos / ocean / ice / land)

Files follow the naming pattern `{epoch}.{component}.nc`. Epochs are
10-digit date strings (e.g. `0019070101`) discovered by scanning the
history directory. All epochs for a component are opened lazily with
`xr.open_mfdataset(combine="by_coords")`.

`decode_timedelta=False` is set on every POEM open call to suppress an
xarray `FutureWarning` caused by the `average_DT` variable, which carries
a timedelta-like unit string but is not a timedelta coordinate.

### LPJ-mL

LPJ-mL files cannot be time-decoded by xarray or cftime because they use
`"years since 1901-1-1"` with a `noleap` calendar, which is not supported
by the underlying unit parser. All LPJ-mL files are opened with
`decode_times=False`.

Calendar months are recovered in `data_loader.lpjml_month_coord()` by
interpreting the fractional-year time values against a hard-coded 365-day
month-boundary table. This function is only valid for monthly-frequency
LPJ-mL output; do not call it on annual variables.

#### Packed variables

Some LPJ-mL variables encode multiple levels (soil layers or PFTs)
consecutively in the time axis:

```
layout: [lev0_t0, lev1_t0, ..., levN_t0, lev0_t1, lev1_t1, ...]
```

`data_loader._unpack_lpjml()` reshapes these to `(time, level, lat, lon)`.
The `n_levels` for each packed variable is declared in the config under
`data.lpjml.packed_vars`. **Verify `n_levels` against the data before
adding a new packed variable** — a wrong value produces a silent reshape
error that is hard to detect.

Known packed variables and their level counts (as of this run):

| Variable | n_levels | dim_name |
|---|---|---|
| `msoiltemp` | 6 | `soil_layer` |
| `soilc_layer` | 5 | `soil_layer` |
| `cftfrac` | 2 | `pft` |
| `pft_harvest_pft` | 32 | `pft` |

---

## Ocean grid note

The ocean grid is tripolar. Its geographic coordinates (`geolat_t`,
`geolon_t`) are 2-D arrays of shape `(yt_ocean, xt_ocean)` stored as
dataset coordinate variables (not as 1-D lat/lon vectors). Pass both 2-D
arrays to `plot_utils.global_map()` — `pcolormesh` accepts 2-D coordinate
arrays and will handle the non-rectangular grid correctly.

---

## How to add a new diagnostic variable

1. Add the variable name to the appropriate list in the YAML config
   (e.g. `diagnostics.atmos.vars`).
2. Optionally add a colormap entry to `plot_utils.DEFAULT_CMAPS` and a
   units string to the `_UNITS` dict in the relevant diagnostic module.
3. No code changes are required if the variable has the same spatial
   dimensions as existing variables in that component.

## How to add a new diagnostic module

1. Create `src/diagnostics/my_module.py` with a `run(data, config, mode, output_dir)` function.
2. Add `"my_module"` to `AVAILABLE_DIAGNOSTICS` in `run_diagnostics.py`.
3. Add the dispatch block inside `main()` in `run_diagnostics.py`.
4. Add the variable list to the config YAML under `diagnostics.my_module`.

## How to add a new temporal mode

1. Edit `src/temporal_agg.py` only.
2. If the new mode is a named season, add it to `SEASON_MONTHS`.
3. For a custom range (e.g. JJAS = June–September), extend the
   `elif` chain in `aggregate()`.

---

## Config reference

See `config/minimal_setup_CTL.yaml` for a fully annotated example. Key
sections:

| Section | Purpose |
|---|---|
| `experiment` | Name and description (used in output paths and titles) |
| `data.history_dir` | Root directory containing POEM epoch files and lpjml subdirs |
| `data.components` | Per-component filename pattern and decode options |
| `data.lpjml` | LPJ-mL subdir pattern, decode option, packed variable specs |
| `grids` | Lat/lon coordinate names and cartopy projection per component |
| `diagnostics` | Variable lists and options per diagnostic module |
| `output` | Output root, file format, and DPI |

---

## Dependencies

Required: `xarray`, `numpy`, `matplotlib`, `cartopy`, `PyYAML`.

The `pft_harvest_pft` packed variable assumes 32 PFTs. If the LPJ-mL
configuration changes the PFT count, update `n_levels` in the config.
