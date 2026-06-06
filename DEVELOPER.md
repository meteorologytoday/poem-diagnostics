# Developer Guide — POEM Diagnostic Plots

## Purpose

This codebase generates diagnostic figures for POEM coupled-model output
(atmosphere, ocean, sea ice) and LPJ-mL land model output. It is designed
to be flexible: any combination of temporal aggregation modes, model
components, and diagnostic types can be requested on the command line
without modifying source code.

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
│       ├── atmos.py             # atmosphere: map_2d, timeseries
│       ├── ocean.py             # ocean: map_2d, zonal_section, timeseries
│       ├── ice.py               # sea ice: map_2d
│       └── land_lpjml.py        # LPJ-mL land model: map_2d, zonal_section
├── run_diagnostics.py           # CLI entry point (argparse)
└── output/                      # generated figures (gitignored)
```

---

## CLI usage

```bash
python run_diagnostics.py --config config/minimal_setup_CTL.yaml \
                          --modes annual JJA DJF \
                          --components atmos ocean ice land \
                          --diag-types map_2d timeseries zonal_section
```

| Argument | Default | Notes |
|---|---|---|
| `--config` | required | Path to the experiment YAML |
| `--modes` | `annual` | One or more of: `annual DJF MAM JJA SON` or integer `1`–`12` |
| `--components` | all | `atmos ocean ice land` |
| `--diag-types` | all | `map_2d timeseries zonal_section` |
| `--history-dir` | from config | Overrides `data.history_dir` — point at a different experiment's output |
| `--exp-name` | from config | Overrides `experiment.name` — controls output subdirectory and plot titles |
| `--output-dir` | from config | Overrides `output.base_dir` in the YAML |
| `--year-range START END` | from config | Inclusive 4-digit year range (e.g. `--year-range 1907 2307`); overrides `data.year_range` |
| `--vars` | all | Restrict to specific variable names (useful during development) |

### Output path structure

```
output/{experiment.name}/{mode}/{component}/{diag_type}/{variable}.png
```

Running with a new mode or a new diag-type only adds files; it never
overwrites figures from other combinations.

---

## Design principles

### 1. Two independent selection axes

Components (`atmos`, `ocean`, `ice`, `land`) and diagnostic types
(`map_2d`, `timeseries`, `zonal_section`) are orthogonal dimensions.
Any combination can be requested at the CLI. A component silently skips
a diag-type it does not support — no error, no code change needed.

### 2. Aggregation is centralised

All temporal aggregation goes through `src/temporal_agg.aggregate()`.
Diagnostic modules never contain groupby or season-selection logic.
Adding a new aggregation mode (e.g. JJAS) means editing only `temporal_agg.py`.

### 3. Config vs CLI

- **Config (YAML)**: stable, experiment-specific facts — data paths, variable
  lists, grid metadata, output format.
- **CLI**: transient per-invocation choices — which modes, components, and
  diag-types to run. The config is committed to the repo; CLI arguments are not.

### 4. Component modules own their data

Each module under `src/diagnostics/` receives the full `data` dict and
extracts the datasets it needs. This keeps the module self-contained and
avoids fragile positional argument ordering:

```python
def run(diag_type: str, data: dict, config: dict, mode: str | int, output_dir: Path): ...
```

The module exposes a `HANDLERS` dict mapping diag-type names to internal
functions. `run_diagnostics.py` checks `HANDLERS` before calling `run()`.

### 5. LPJ-mL data is loaded on demand

LPJ-mL has ~60 variables each in a separate file. The loader returns a
callable `lpjml_loader(varname) -> xr.DataArray` rather than loading
everything upfront. The `land_lpjml` module requests variables individually.

---

## Supported diag-types per component

| Component | map_2d | timeseries | zonal_section |
|---|---|---|---|
| atmos | ✓ | ✓ | — |
| ocean | ✓ | ✓ | ✓ |
| ice | ✓ | — | — |
| land | ✓ | ✓ (global land mean) | ✓ (soil layers) |

A `—` means the component does not implement that type. Requesting it has
no effect.

---

## Two data loading paths

### POEM components (atmos / ocean / ice / land)

Files follow the naming pattern `{epoch}.{component}.nc`. Epochs are
10-digit strings of the form `00YYYYMMDD` (e.g. `0019070101` = year 1907,
month 01, day 01). The two leading zeros are a known artifact of the parent
model's output writer — they are not part of the year. The epoch year is
extracted as `epoch[2:6]` (characters 2–5), giving the 4-digit YYYY.
Epochs are discovered by scanning the history directory and filtered by the
`data.year_range` config field using this corrected year extraction.
All epochs for a component are opened lazily with
`xr.open_mfdataset(combine="by_coords")`.

`xr.coders.CFDatetimeCoder(use_cftime=True)` is used for time decoding
to handle model years that are outside the range of numpy datetime64.

`decode_timedelta=False` suppresses an xarray `FutureWarning` caused by
the `average_DT` variable, which carries a timedelta-like unit string but
is not a timedelta coordinate.

The optional `data.year_range: [start, end]` config field (inclusive)
filters which epochs are loaded. Omit it to load all epochs.

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

Known packed variables and their level counts (declared in `data.lpjml.packed_vars`):

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
arrays and handles the non-rectangular grid correctly.

---

## How to add a new diagnostic variable

1. Add the variable name to the appropriate list in the YAML config under
   `diagnostics.{component}.{diag_type}.vars`.
2. Optionally add a colormap entry to `plot_utils.DEFAULT_CMAPS` and a
   units string to the `_UNITS` dict in the component module.
3. No code changes required if the variable has the same spatial dimensions
   as existing variables in that component.

## How to add a new diag-type to an existing component

1. Write a `_run_{diag_type}()` function inside the component module.
2. Add it to that module's `HANDLERS` dict.
3. Add `{diag_type}` to `ALL_DIAG_TYPES` in `run_diagnostics.py`.
4. Add the variable list to the config under
   `diagnostics.{component}.{diag_type}`.

## How to add a new component

1. Create `src/diagnostics/my_component.py` with a `HANDLERS` dict and
   a `run(diag_type, data, config, mode, output_dir)` function.
2. Add it to `COMPONENT_MODULES` in `run_diagnostics.py`.
3. Add its required data keys to `_COMPONENT_DATA_KEYS` in `run_diagnostics.py`.
4. Add the config section under `diagnostics.my_component`.

## How to add a new temporal mode

1. Edit `src/temporal_agg.py` only.
2. If the new mode is a named season, add it to `SEASON_MONTHS`.
3. For a custom range (e.g. JJAS), extend the `elif` chain in `aggregate()`.

---

## Config reference

See `config/minimal_setup_CTL.yaml` for a fully annotated example. Key
sections:

| Section | Purpose |
|---|---|
| `experiment` | Name and description (used in output paths and plot titles) |
| `data.history_dir` | Root directory containing POEM epoch files and lpjml subdirs |
| `data.year_range` | Optional inclusive 4-digit model-year range (YYYY); omit to load all epochs |
| `data.components` | Per-component filename pattern and decode options |
| `data.lpjml` | LPJ-mL subdir pattern, decode option, packed variable specs |
| `grids` | Lat/lon coordinate names and cartopy projection per component |
| `diagnostics` | Two-layer structure: `{component}.{diag_type}.vars` |
| `output` | Output root, file format, and DPI |

---

## Dependencies

Required: `xarray`, `numpy`, `matplotlib`, `cartopy`, `PyYAML`.
