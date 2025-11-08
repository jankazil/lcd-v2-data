#!/usr/bin/env python

'''

Builds Local Climatological Data (LCD) datasets from NOAA NCEI observations for a specified U.S. state,
territory, RTO/ISO region, CONUS, or individual station. The module automates the complete workflow of
downloading station metadata and LCD observations, filtering stations by region and data availability,
constructing full-hourly UTC time series, saving results as NetCDF files, and optionally generating
diagnostic plots. It can be executed as a command-line tool or used programmatically through the
`run_build()` function.

Workflow:

1) Load the requested region geometry (RTO/ISO polygons or U.S. state boundaries) if a region is specified.
   If a station ID (GHCNh station identifier) is provided, geometry loading is skipped.
2) Retrieve the full LCD station catalog and spatially filter stations to the region,
   or select the specified station by ID.
3) Filter stations by data availability over [start_year, end_year].
4) Save the resulting station list to a given data directory (for regions) or proceed with the
   specified station (for individual station mode).
5) Download LCD observation files for the filtered station IDs. Files already present
   in the download directory and unchanged on the NOAA NCEI server are not re-downloaded.
   - Downloads can be run in parallel with the -n option.
   - The --offline flag disables network access and expects all required files to be present locally.
6) Construct full-hourly UTC time series for the selected region or station and period.
7) Optionally create plots of original and interpolated full-hourly UTC time series
   if a plot directory is provided (plot generation is slow).
8) Save the full-hourly UTC time series as a NetCDF file.

Output files:

- A text file listing the stations used (for regions).
- The downloaded LCD data files (unless offline mode is selected).
- Optional diagnostic plots, if a plot directory is specified.
- A NetCDF file containing the full-hourly UTC time series.

Assumptions:

- Network access to NOAA NCEI is available unless --offline is specified.

Example usage:

    build-lcd-dataset 2020 2025 USW00003017 /path/to/data -p /path/to/plots
    build-lcd-dataset 2022 2022 CAISO /path/to/data -n 32
    build-lcd-dataset 2021 2021 CO /path/to/data --offline
'''

import argparse
import os
import sys
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

import geopandas as gpd

from lcd_data import ncei, region_codes, rto_iso, stations


def run_build(
    start_year: int,
    end_year: int,
    region_name: str,
    data_dir: Path,
    plot_dir: Path,
    n_jobs: int,
    offline: bool = False,
    verbose: bool = False,
) -> Path:
    '''
    Download, process, and assemble NOAA NCEI Local Climatological Data (LCD) into a full-hourly UTC time
    series for a specified geographic region or individual station over a given range of years.

    Parameters
    ----------
    start_year : int
        Inclusive start year of the data range to process.
    end_year : int
        Inclusive end year of the data range to process.
    region_name : str
        Region selector, which may be a two-letter U.S. state or territory code, 'CONUS',
        an RTO/ISO code, or a specific station ID (GHCNh station identifier).
    data_dir : Path
        Directory where station lists, downloaded LCD files, and outputs will be stored. Created if it does not exist.
    plot_dir : Path
        Directory where diagnostic plots will be generated. If None, plots are not created.
    n_jobs : int
        Number of parallel download processes to use. If 1, downloads are performed serially.
    offline : bool, optional
        If True, operates without network access and expects all required files to be present locally. Default is False.
    verbose : bool, optional
        If True, prints detailed progress messages. Default is False.

    Returns
    -------
    Path
        Path to the generated NetCDF file containing the full-hourly UTC LCD time series for the selected region or station.

    Notes
    -----
    The function downloads or loads LCD station metadata, filters stations by spatial and temporal
    availability, downloads the corresponding LCD observation files (unless offline mode is enabled),
    constructs complete hourly UTC time series, and writes the resulting dataset to a NetCDF file.
    Optionally, it can generate plots of the original and interpolated time series for visual inspection.
    '''

    if verbose:
        if offline:
            print('Working offline. All required files must have been downloaded to ' + str(data_dir) + ' in a previous call.')
        else:
            print(
                'Working online. Will download files to '
                + str(data_dir)
                + ' unless they are already present and identical with their version on the NCEI server.'
            )

    # Create data directory unless it exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construct datetime objects from the start and end year
    start_date = datetime(year=start_year, month=1, day=1)
    end_date = datetime(year=end_year, month=12, day=31)

    #
    # Identify stations in the selected US state, territory, region, or handle an individual station
    #

    working_on_region = False

    if len(region_name) == 2:
        assert region_name in region_codes.us_states_territories, (
            'US state/territory code ' + region_name + ' is not available.'
        )
        working_on_region = True
        # Load US states shapefile directory from installed distribution using importlib.resources
        us_states_dir_res = files('lcd_data') / 'data' / 'CensusBureau' / 'US_states'
        with as_file(us_states_dir_res) as us_states_dir_path:
            us_states_shp_file = us_states_dir_path / 'tl_2024_us_state.shp'
            us_gdf = gpd.read_file(us_states_shp_file)
        region_gdf = us_gdf[us_gdf['STUSPS'].isin([region_name])]

    elif region_name in region_codes.rto_iso_regions:
        working_on_region = True
        # Load RTO/ISO region GeoJSON from installed distribution using importlib.resources
        rto_iso_geojson_res = files('lcd_data') / 'data' / 'EIA' / 'RTO_ISO_regions.geojson'
        with as_file(rto_iso_geojson_res) as rto_iso_geojson_path:
            region_gdf = rto_iso.region(rto_iso_geojson_path, region_name)

    elif region_name == region_codes.conus:
        working_on_region = True
        # Load US states shapefile directory from installed distribution using importlib.resources
        us_states_dir_res = files('lcd_data') / 'data' / 'CensusBureau' / 'US_states'
        with as_file(us_states_dir_res) as us_states_dir_path:
            us_states_shp_file = us_states_dir_path / 'tl_2024_us_state.shp'
            us_gdf = gpd.read_file(us_states_shp_file)
        exclude_codes = ['AK', 'HI', 'PR', 'GU', 'VI', 'AS', 'MP']
        region_gdf = us_gdf[~us_gdf['STUSPS'].isin(exclude_codes)]

    # File with list of all stations and metadata

    all_stations_file = data_dir / Path(os.path.basename(ncei.ghcnh_station_list_url))

    if offline:
        # Load the file from disk - it must have been downloaded previously to the data directory
        all_stations = stations.Stations.from_file(all_stations_file)
    else:
        # Load the file from the NCEI server and save it for later use
        all_stations = stations.Stations.from_url()
        all_stations.save_station_list(all_stations_file, verbose=verbose)

    # Determine if we are working on a region or an individual station

    if working_on_region:
        region_stations = all_stations.filter_by_region(region_gdf)
    else:
        region_stations = all_stations.filter_by_id(region_name)

    # Filter by data availability

    if offline:
        region_stations = region_stations.filter_by_data_availability_offline(data_dir, start_date, end_date, verbose=verbose)
    else:
        region_stations = region_stations.filter_by_data_availability_online(start_date, end_date, verbose=verbose)

    # Save the metadata file for these stations

    region_stations_file = data_dir / Path(region_name + '.' + str(start_date.year) + '-' + str(end_date.year) + '.txt')

    region_stations.save_station_list(region_stations_file)

    # Download LCD station data

    if not offline:
        _ = ncei.download_many(
            start_date.year, end_date.year, region_stations.ids(), data_dir, n_jobs=n_jobs, refresh=False, verbose=verbose
        )

    # Construct full-hourly UTC time series from the LCD station data

    region_stations.construct_hourly(
        data_dir, start_date.year, end_date.year, region=region_name, plot_dir=plot_dir, verbose=verbose
    )

    # Save full-hourly UTC time series as a netCDF file

    lcd_netcdf_file = data_dir / Path(region_name + '.' + str(start_date.year) + '-' + str(end_date.year) + '.nc')

    region_stations.write_utc_hourly_netcdf(lcd_netcdf_file, verbose=verbose)

    return lcd_netcdf_file


def arg_parse(argv=None):
    '''

    Command line argument parser.

    Parses command-line arguments and returns normalized values used by the script.

    Parameters
    ----------
    argv : list[str] or None
        Sequence of argument tokens to parse (excluding the program name). If None,
        arguments are taken from sys.argv[1:].

    Returns
    -------
    tuple[datetime, datetime, str, Path, Path | None, int | None, bool]
        start_date : datetime
            Inclusive start date constructed from `start_year` (January 1).

        end_date : datetime
            Inclusive end date constructed from `end_year` (December 31).

        region_name : str
            Region selector. One of a two-letter U.S. state or territory code
            (e.g., 'CA', 'PR'), the special region 'CONUS', an RTO/ISO region code
            {'ERCOT','CAISO','ISONE','NYISO','MISO','SPP','PJM'}, or an individual
            station ID (GHCNh station identifier).

        data_dir : Path
            Destination directory into which the station list, downloaded LCD files,
            and outputs will be written.

        plot_dir : Path | None
            Directory where plots of the original and the interpolated full-hourly UTC time
            series will be created. If None, plots are not generated. Creating plots is
            very slow.

        n_jobs : int | None
            Maximum number of parallel download workers. If None, downloads run
            single-threaded.

        offline : bool
            If True, work offline and expect all required inputs to be present in data_dir.

        verbose : bool
            If True, print information.

    Raises
    ------
    SystemExit
        If the provided arguments fail validation performed by argparse.
    '''

    code_description = (
        "Download NOAA NCEI Local Climatological Data (LCD) observations for stations located "
        "within a selected U.S. state, territory, the contiguous United States (CONUS), an "
        "RTO/ISO region, or for an individual station by station ID (GHCNh station identifier), "
        "over an inclusive range of years. The script filters stations spatially and by data "
        "availability (for regions) or selects the given station, saves the station list when "
        "applicable, downloads observations, constructs full-hourly UTC time series, optionally "
        "creates diagnostic plots, and writes a NetCDF file.\n\n"
        "Valid region or station arguments:\n\n"
        f"  - US states/territories: {', '.join(region_codes.us_states_territories)}\n\n"
        f"  - Special region: {region_codes.conus}\n\n"
        f"  - RTO/ISO regions: {', '.join(region_codes.rto_iso_regions)}\n\n"
        "  - Individual station: provide a station ID (GHCNh station identifier)\n\n"
        "Parallel downloads can be enabled with -n.\n\n"
        "LCD observation files already present in the download directory and unchanged on the NOAA NCEI server are not re-downloaded."
    )

    parser = argparse.ArgumentParser(description=code_description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mandatory arguments

    parser.add_argument('start_year', type=int, help='Start year of time range.')

    parser.add_argument('end_year', type=int, help='End year of time range (inclusive).')

    parser.add_argument(
        'region_name',
        type=str,
        help=(
            "Region or station selector. Use a two-letter U.S. state or territory code, 'CONUS', "
            "one of the RTO/ISO codes "
            f"({', '.join(region_codes.rto_iso_regions)}), or a station ID (GHCNh station identifier)."
        ),
    )

    parser.add_argument('data_dir', type=str, help='Directory path into which the data will be downloaded.')

    # Optional arguments

    parser.add_argument(
        '-n',
        '--n',
        type=int,
        help=(
            'Number of parallel download processes. n > 1 accelerates downloads significantly, '
            'but can result in network errors or in the server refusing to cooperate.'
        ),
    )

    parser.add_argument(
        '-o',
        '--offline',
        action='store_true',
        help=('Work offline. All required files must have been downloaded to data_dir in a previous call without this flag.'),
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help=('Print progress information.'),
    )

    parser.add_argument(
        '-p',
        '--plotdir',
        type=str,
        help=(
            'Directory where plots of the original and the interpolated full-hourly time series '
            'will be created. Very slow. If omitted, no plots are generated.'
        ),
    )

    args = parser.parse_args(argv)

    start_year = args.start_year
    end_year = args.end_year
    region_name = args.region_name
    data_dir = Path(args.data_dir)

    plot_dir = Path(args.plotdir) if args.plotdir is not None else None

    n_jobs = args.n

    offline = args.offline

    verbose = args.verbose

    return (start_year, end_year, region_name, data_dir, plot_dir, n_jobs, offline, verbose)


def main(argv=None):
    '''
    Command line interface entry point.
    '''

    (start_year, end_year, region_name, data_dir, plot_dir, n_jobs, offline, verbose) = arg_parse(
        argv if argv is not None else sys.argv[1:]
    )

    lcd_netcdf_file = run_build(start_year, end_year, region_name, data_dir, plot_dir, n_jobs, offline=offline, verbose=verbose)


if __name__ == '__main__':
    main()
