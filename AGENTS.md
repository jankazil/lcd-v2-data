# Agent Development Guide

## Project Scope and Isolation

**The project root is `.` relative to this `AGENTS.md`: the directory containing this file. This directory is the exclusive project boundary.** These instructions apply to the entire repository and remain valid when it is moved.

- Read, search, analyze, and edit project files only within this root. Scope commands explicitly to it.
- Do not inspect parent directories, sibling projects, other workspaces, or external locations for project context or instructions.
- Do not follow symbolic links, Git worktree pointers, submodule metadata, object alternates, or other Git references that lead outside the root. Check containment before following references.
- Do not infer architecture, requirements, or conventions from unrelated projects, previous sessions, or the current conversation. Verify project-specific claims against this repository.
- If required information, configuration, data, or tooling resides outside the boundary, report the dependency rather than accessing or inspecting it. Repository links to external documentation are references, not permission to retrieve it.
- Keep generated data, plots, fixtures, caches, and temporary project artifacts inside the root. Inspect script effects before execution; setup and publishing scripts can affect external environments and locations and must not expand this boundary.

## Purpose and Starting Points

`lcd-v2-data` is a Python toolkit and CLI for downloading NOAA NCEI Local Climatological Data version 2 and constructing full-hourly UTC weather time series. It selects individual GHCNh station IDs, U.S. states/territories, CONUS, or RTO/ISO regions and writes NetCDF datasets, with optional diagnostic plots.

Read [README.md](README.md) for the workflow, CLI flags, examples, and public API. Use [pyproject.toml](pyproject.toml), [Makefile](Makefile), and the implementation to resolve discrepancies in that documentation. Dependency and environment declarations are in [environment.yml](environment.yml); CI is in [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Repository Map and Data Flow

| Path | Responsibility |
| --- | --- |
| `src/lcd_data/build_lcd_dataset.py` | `run_build()` orchestrates the pipeline; `arg_parse()` and `main()` implement the CLI. |
| `src/lcd_data/stations.py` | `Stations` owns pandas metadata and an xarray hourly dataset; handles catalog parsing, selection, availability, CSV cleaning, interpolation, plotting, and NetCDF output. |
| `src/lcd_data/ncei.py` | NOAA URLs, station/year filenames, availability probes, download orchestration, and optional period-of-record API access. |
| `src/lcd_data/web.py` | HTTP availability checks, threaded downloads, retries, and ETag sidecars. |
| `src/lcd_data/region_codes.py`, `rto_iso.py` | Region selectors, packaged geometry loading, and RTO/ISO polygon merging. |
| `src/lcd_data/saturation.py` | Saturation vapor pressure and relative humidity calculations. |
| `src/lcd_data/__init__.py` | Distribution-based version lookup, with a local fallback. |
| `src/lcd_data/data/` | Bundled EIA GeoJSON and Census TIGER/Line state geometry and attribution. |
| `tests/test_test.py`, `src/lcd_data/test.py` | One placeholder test and its addition helper. |
| `scripts/` | Development setup and PyPI/Anaconda publishing scripts. |
| `plots/` | README sample image. |

Pipeline: load the GHCNh station catalog → select a station or spatial region → require an observation file for every requested year → save selected metadata → download CSVs unless offline → clean observations and interpolate each station → concatenate along `station` → write NetCDF.

`build_lcd_dataset` depends on `stations`, `region_codes`, and `ncei`; `stations` uses `ncei` and `saturation`; `ncei` delegates HTTP downloads to `web`; `region_codes` uses `rto_iso`. Region assets are resolved with `importlib.resources`, not the working directory.

## Environment and Commands

- Distribution name: `lcd-v2-data`; import package: `lcd_data`; source layout: `src/`.
- Python requirement: **3.12 or newer**. Build backend: setuptools (`setuptools>=77`, `wheel`).
- Runtime dependencies: geopandas, matplotlib, numpy, netcdf4, pandas, requests, shapely, timezonefinder, and xarray. The conda environment additionally includes h5netcdf and Jupyter tooling. No dependency lockfile is present.
- CI installs with `pip install -e ".[dev]"`; the dev extra declares pytest, pytest-cov, Ruff, and pre-commit. Installation changes the selected environment and may access package services; treat external environment requirements according to the isolation policy.
- `make setup-dev-env` runs `scripts/setup-dev-env.sh`. It writes `environment-dev.yml`, removes/recreates the `lcd-v2-data-dev` conda environment, replaces its user Jupyter kernel, installs the editable package, and installs Git hooks. It is not a harmless check and has effects outside the repository.

Run development commands from the project root in a prepared environment:

| Command | Effect |
| --- | --- |
| `python -m ruff check .` | Check lint; CI uses this without fixes. |
| `python -m ruff format --check .` | Check-only counterpart to the configured formatter. |
| `make test` or `pytest` | Run the configured `tests/` suite. |
| `make fmt` | Reformat Python files. |
| `make lint` | Lint with `--fix`; modifies files. |
| `make check` | Run formatting and fixing lint; does **not** run tests. |
| `python -m build` | Build distributions; used by the PyPI script and requires the separate `build` package. |
| `python -m twine check dist/*` | Validate built distributions; requires the separate `twine` package. |
| `build-lcd-dataset --help` | Show the installed CLI. |

See README CLI documentation for execution. Use repository-contained paths for `DATA_DIR` and optional `PLOT_DIR`. `--offline` requires a local `ghcnh-station-list.txt` and `LCD_<station_id>_<year>.csv` files for all requested years; no observation fixtures are bundled. `--refresh` forces downloading/reprocessing; an existing output NetCDF otherwise skips reconstruction after earlier pipeline steps.

`make clean` deletes caches and build artifacts. `make upload-pypi` and `make upload-anaconda` build and publish externally; do not use them as validation commands. Consult the scripts for their environment changes and cleanup behavior.

## Implementation Conventions and Processing Constraints

- Follow [.editorconfig](.editorconfig): UTF-8, LF, final newline, four-space indentation; YAML/JSON/TOML use two spaces.
- Ruff uses a 128-character line length, preserves quote style, and enables `E`, `F`, `I`, `UP`, `B`, and `SIM`, with explicit exceptions in `pyproject.toml`. Match surrounding names, type annotations, and docstrings rather than reformatting unrelated code.
- Station filters return new `Stations` instances; construction populates `self.time_series_hourly`. Preserve fixed-width catalog fields and their serialization order.
- CSV parsing relies on numeric column positions as well as field names. Preserve the LCDv2 schema assumptions, numeric coercion, selected temperature-field `+18.3` corrections, and metadata-derived coordinates/elevation unless deliberately changing those behaviors.
- Observation times are local **standard** time. The code uses the station timezone's offset on January 1, 2025 for all observations; it does not apply date-specific daylight saving offsets. Changes require explicit scientific validation.
- Cleaning excludes `BOGUS`, `SOD`, and `SOM` reports, resolves duplicate timestamps, and clears both temperature and dew point when temperature exceeds 60 °C. Interpolation rejects duplicate or non-monotonic times that remain.
- RH is derived from temperature/dew point; values outside 0–101% become missing and values between 100–101% are clipped to 100%. Wind uses `U = -speed*sin(direction)` and `V = -speed*cos(direction)`; calm wind with missing direction produces zero components.
- Hourly output spans January 1 00:00 through December 31 23:00 UTC for the inclusive year range. Default maximum interpolation gap is two hours. Preserve NaNs for unsupported gaps and exact observations at full hours, including the final observation.
- Main output variables are `T`, `Td`, `RH`, `U`, and `V` on `(time, station)`, with station metadata and variable attributes. NetCDF encoding uses float64 epoch seconds for time and float32 floating data variables; object data variables are omitted. Validate dimensions, units, missing values, and time encoding when changing output.
- Geographic filtering uses EPSG:4326 and `within`, which excludes polygon-boundary points. RTO merging treats PJM differently from the other regions; inspect `rto_iso.py` before changing filters.
- Preserve geospatial companion files, packaged resource paths, and both wheel package-data rules and `MANIFEST.in` source-distribution rules. `.gitattributes` assigns Git LFS to `.shp`, `.dbf`, `.shx`, and `.geojson`; an LFS pointer is not usable geometry. Read the data attribution files and [LICENSE](LICENSE) before redistributing assets.

## Configuration and External Interfaces

- Runtime configuration is through CLI arguments and function parameters; no application environment-variable configuration is declared in the Python source.
- `ncei.py` defines public NOAA GHCNh metadata URLs and LCDv2 CSV URLs under `https://www.ncei.noaa.gov/oa/`. Normal CLI downloads require no API token. The separate `get_period_of_record()` API uses a caller-supplied token with NOAA CDO `/cdo-web/api/v2/stations/`.
- Downloads compare server ETags with `<filename>.etag` sidecars. HTTP helpers can retry 1,200 times with three-second delays. Threaded download exceptions are printed and omitted from returned paths; a returned list does not establish complete download success. Mock HTTP in tests instead of waiting through retries or probing live NOAA services.
- Anaconda publishing uses `ANACONDA_API_TOKEN`, `ANACONDA_USER_NAME`, and optional `IMPORT_NAME_OVERRIDE`; consult `scripts/upload2anaconda.sh` for additional overrides. PyPI publishing prompts for `TWINE_PASSWORD` and sets `TWINE_USERNAME=__token__`. Keep credentials out of files and logs.

## Making and Reviewing Changes

1. Read the affected module, its callers, and applicable repository documentation. Keep changes within the requested scope and preserve unrelated work.
2. For behavioral changes, add focused tests under `tests/test_*.py` using `test_*` functions. Prefer small synthetic metadata/CSV inputs and mocked network responses. Exercise scientific edge cases, offline behavior, caching, and serialization when affected; keep test artifacts inside the repository.
3. Run appropriate lint, formatting checks, and tests when the environment supports them. Review auto-fixes explicitly. Report unavailable dependencies and checks that were not run; never equate the placeholder suite passing with pipeline correctness.
4. For packaging changes, inspect wheel/sdist data inclusion and perform a local resource-loading check when dependencies are available. Do not publish to test packaging.
5. Review the final diff for unintended source changes, generated data, credentials, or asset churn. Update README examples and public API descriptions when behavior changes. Report behavior, validation, limitations, and any manual checks still required.

## Known Gaps and Manual Verification

These are repository observations, not confirmed runtime failures:

- The only checked-in test asserts `add(2, 2) == 4`; downloading, scientific calculations, spatial selection, interpolation, CLI behavior, and NetCDF output lack checked-in regression coverage.
- README documents `run_build(..., plot_dir=None, ...)`, but the implementation requires an explicit `plot_dir` argument; pass `None` to disable plots. README also attributes download helpers to `ncei` that actually reside in `web`.
- README mentions `make type` and mypy settings, but no such target or mypy configuration exists. Ruff targets `py311` despite the package requiring Python >=3.12. CI specifies only Python `3.14.6`; supported-version coverage and availability of that exact interpreter need verification.
- CI checkout does not explicitly enable Git LFS retrieval. Verify actual geometry availability before geospatial or installed-package checks.
- Anaconda's default derived import name would be `lcd_v2_data`, while this package imports as `lcd_data`; verify/set `IMPORT_NAME_OVERRIDE=lcd_data` before an authorized release.
- `write_utc_hourly_netcdf()` loops over characters in `'time'` when attempting attribute cleanup. Review this path and test encoding conflicts if changing serialization.
- Live NOAA availability, upstream CSV compatibility, complete environment resolution, and end-to-end scientific correctness require validation beyond static repository inspection. Do not assume external services or installed dependencies have been verified.
