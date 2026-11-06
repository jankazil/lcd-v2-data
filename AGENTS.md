# Agent Development Guide

## Project Scope and Isolation

**The project root is the directory containing this `AGENTS.md` (`.` relative to this file). These instructions apply throughout that repository, including after it is moved.**

- Treat the repository root as the exclusive project boundary. Read, search, analyze, and modify project files only inside it. Run commands from the root unless an in-repository script requires another directory within the root.
- Do not inspect parent directories, sibling projects, unrelated workspaces, global configuration, installed dependency source, or other external locations for project context. Do not import assumptions from other projects or previous sessions.
- Check path containment before following symbolic links, Git worktree/submodule references, object alternates, configuration includes, or paths that may escape through `..`. Do not follow references outside the root.
- External URLs, paths, and credentials mentioned in repository files describe interfaces or dependencies; they do not authorize accessing resources outside the root. Report any required outside-root access instead of performing it.
- Keep development inputs, generated outputs, caches, fixtures, and temporary project files inside the root. Inspect scripts for filesystem, environment, network, or deployment effects before running them. Do not run workflows that cross the project boundary under these instructions.

## General Agent Instructions

- Base project-specific decisions on verified repository contents. Read the files relevant to the task; distinguish implementation facts, documentation claims, test results, and unverified assumptions.
- Before editing, inspect the relevant code and working-tree state. Make focused changes, preserve unrelated work, and retain established interfaces and data contracts unless the task requires otherwise.
- Use relevant checks that are safe in the available environment. Do not confuse a passing placeholder test, build, or lint check with end-to-end or scientific correctness. Report the checks actually run, failures, and verification gaps.
- Do not expose credentials, or perform publishing, deployment, destructive environment setup, or other unrelated side effects as routine development or validation.

## Project Overview and Repository Map

`lcd-v2-data` is a Python toolkit and CLI for downloading NOAA NCEI Local Climatological Data version 2 and constructing full-hourly UTC weather time series. It selects individual GHCNh station IDs, U.S. states/territories, CONUS, or RTO/ISO regions and writes NetCDF datasets, with optional diagnostic plots.

Read [README.md](README.md) for the workflow, CLI flags, examples, and public API. Use [pyproject.toml](pyproject.toml), [Makefile](Makefile), and the implementation to resolve discrepancies in that documentation. Dependency and environment declarations are in [environment.yml](environment.yml); CI is in [.github/workflows/ci.yml](.github/workflows/ci.yml).

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

## Architecture and Data Contracts

Pipeline: load the GHCNh station catalog → select a station or spatial region → require an observation file for every requested year → save selected metadata → download CSVs unless offline → clean observations and interpolate each station → concatenate along `station` → write NetCDF.

`build_lcd_dataset` depends on `stations`, `region_codes`, and `ncei`; `stations` uses `ncei` and `saturation`; `ncei` delegates HTTP downloads to `web`; `region_codes` uses `rto_iso`. Region assets are resolved with `importlib.resources`, not the working directory.

### Processing Constraints

- Station filters return new `Stations` instances; construction populates `self.time_series_hourly`. Preserve fixed-width catalog fields and their serialization order.
- CSV parsing relies on numeric column positions as well as field names. Preserve the LCDv2 schema assumptions, numeric coercion, selected temperature-field `+18.3` corrections, and metadata-derived coordinates/elevation unless deliberately changing those behaviors.
- Observation times are local **standard** time. The code uses the station timezone's offset on January 1, 2025 for all observations; it does not apply date-specific daylight saving offsets. Changes require explicit scientific validation.
- Cleaning excludes `BOGUS`, `SOD`, and `SOM` reports, resolves duplicate timestamps, and clears both temperature and dew point when temperature exceeds 60 °C. Interpolation rejects duplicate or non-monotonic times that remain.
- RH is derived from temperature/dew point; values outside 0–101% become missing and values between 100–101% are clipped to 100%. Wind uses `U = -speed*sin(direction)` and `V = -speed*cos(direction)`; calm wind with missing direction produces zero components.
- Hourly output spans January 1 00:00 through December 31 23:00 UTC for the inclusive year range. Default maximum interpolation gap is two hours. Preserve NaNs for unsupported gaps and exact observations at full hours, including the final observation.
- Main output variables are `T`, `Td`, `RH`, `U`, and `V` on `(time, station)`, with station metadata and variable attributes. NetCDF encoding uses float64 epoch seconds for time and float32 floating data variables; object data variables are omitted. Validate dimensions, units, missing values, and time encoding when changing output.
- Geographic filtering uses EPSG:4326 and `within`, which excludes polygon-boundary points. RTO merging treats PJM differently from the other regions; inspect `rto_iso.py` before changing filters.

## Environment and Setup

- Distribution name: `lcd-v2-data`; import package: `lcd_data`; source layout: `src/`.
- Python requirement: **3.12 or newer**. Build backend: setuptools (`setuptools>=77`, `wheel`).
- Runtime dependencies: geopandas, matplotlib, numpy, netcdf4, pandas, requests, shapely, timezonefinder, and xarray. The conda environment additionally includes h5netcdf and Jupyter tooling. No dependency lockfile is present.
- CI installs with `pip install -e ".[dev]"`; the dev extra declares pytest, pytest-cov, Ruff, and pre-commit. Installation changes the selected environment and may access package services; treat external environment requirements according to the isolation policy.
- `make setup-dev-env` runs `scripts/setup-dev-env.sh`. It writes `environment-dev.yml`, removes/recreates the `lcd-v2-data-dev` conda environment, replaces its user Jupyter kernel, installs the editable package, and installs Git hooks. It is not a harmless check and has effects outside the repository.

## Development Commands and Validation

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

## Coding Conventions

- Follow [.editorconfig](.editorconfig): UTF-8, LF, final newline, four-space indentation; YAML/JSON/TOML use two spaces.
- Ruff uses a 128-character line length, preserves quote style, and enables `E`, `F`, `I`, `UP`, `B`, and `SIM`, with explicit exceptions in `pyproject.toml`. Match surrounding names, type annotations, and docstrings rather than reformatting unrelated code.
- Preserve geospatial companion files, packaged resource paths, and both wheel package-data rules and `MANIFEST.in` source-distribution rules. `.gitattributes` assigns Git LFS to `.shp`, `.dbf`, `.shx`, and `.geojson`; an LFS pointer is not usable geometry. Read the data attribution files and [LICENSE](LICENSE) before redistributing assets.

## Configuration and External Interfaces

- Runtime configuration is through CLI arguments and function parameters; no application environment-variable configuration is declared in the Python source.
- `ncei.py` defines public NOAA GHCNh metadata URLs and LCDv2 CSV URLs under `https://www.ncei.noaa.gov/oa/`. Normal CLI downloads require no API token. The separate `get_period_of_record()` API uses a caller-supplied token with NOAA CDO `/cdo-web/api/v2/stations/`.
- Downloads compare server ETags with `<filename>.etag` sidecars. HTTP helpers can retry 1,200 times with three-second delays. Threaded download exceptions are printed and omitted from returned paths; a returned list does not establish complete download success. Mock HTTP in tests instead of waiting through retries or probing live NOAA services.
- Anaconda publishing uses `ANACONDA_API_TOKEN`, `ANACONDA_USER_NAME`, and optional `IMPORT_NAME_OVERRIDE`; consult `scripts/upload2anaconda.sh` for additional overrides. PyPI publishing prompts for `TWINE_PASSWORD` and sets `TWINE_USERNAME=__token__`. Keep credentials out of files and logs.

## Change and Review Workflow

1. Read the affected module, its callers, and applicable repository documentation. Keep changes within the requested scope and preserve unrelated work.
2. For behavioral changes, add focused tests under `tests/test_*.py` using `test_*` functions. Prefer small synthetic metadata/CSV inputs and mocked network responses. Exercise scientific edge cases, offline behavior, caching, and serialization when affected; keep test artifacts inside the repository.
3. Run appropriate lint, formatting checks, and tests when the environment supports them. Review auto-fixes explicitly. Report unavailable dependencies and checks that were not run; never equate the placeholder suite passing with pipeline correctness.
4. For packaging changes, inspect wheel/sdist data inclusion and perform a local resource-loading check when dependencies are available. Do not publish to test packaging.
5. Review the final diff for unintended source changes, generated data, credentials, or asset churn. Update README examples and public API descriptions when behavior changes. Report behavior, validation, limitations, and any manual checks still required.
6. **Maintain `AGENTS.md`:** When making changes that affect the project's architecture, interfaces, dependencies, workflows, conventions, or documented constraints, update the relevant sections of `AGENTS.md` in the same change. Keep revisions focused and based on verified repository contents. Do not modify `AGENTS.md` for changes that do not affect its accuracy or usefulness.

## Known Limitations and Verification Gaps

These are repository observations, not confirmed runtime failures:

- The only checked-in test asserts `add(2, 2) == 4`; downloading, scientific calculations, spatial selection, interpolation, CLI behavior, and NetCDF output lack checked-in regression coverage.
- README documents `run_build(..., plot_dir=None, ...)`, but the implementation requires an explicit `plot_dir` argument; pass `None` to disable plots. README also attributes download helpers to `ncei` that actually reside in `web`.
- README mentions `make type` and mypy settings, but no such target or mypy configuration exists. Ruff targets `py311` despite the package requiring Python >=3.12. CI specifies only Python `3.14.6`; supported-version coverage and availability of that exact interpreter need verification.
- CI checkout does not explicitly enable Git LFS retrieval. Verify actual geometry availability before geospatial or installed-package checks.
- Anaconda's default derived import name would be `lcd_v2_data`, while this package imports as `lcd_data`; verify/set `IMPORT_NAME_OVERRIDE=lcd_data` before an authorized release.
- `write_utc_hourly_netcdf()` loops over characters in `'time'` when attempting attribute cleanup. Review this path and test encoding conflicts if changing serialization.
- Live NOAA availability, upstream CSV compatibility, complete environment resolution, and end-to-end scientific correctness require validation beyond static repository inspection. Do not assume external services or installed dependencies have been verified.
