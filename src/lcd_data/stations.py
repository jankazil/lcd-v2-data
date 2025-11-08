import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

import geopandas as gpd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Point
from timezonefinder import TimezoneFinder

from lcd_data import ncei, saturation

matplotlib.use("Agg")  # Important to avoid runaway memory use upon creating plots repeatedly.


class Stations:
    '''

    Class for Local Climatological Data (LCD) station data.

    On construction, holds LCD station metadata. After calling time-series
    builders, will hold constructed hourly time series for select LCD
    observables.

    Metadata are read from a fixed-width station list compatible with the
    GHCNh file format ("ghcnh-station-list.txt"), either from a URL
    or from a local file. The GHCNh list is preferred because it exposes
    richer metadata; availability of LCD observations for a station must
    still be verified separately.

    '''

    #
    # LCD station metadata
    #

    meta_data: pd.DataFrame

    # LCD station metadata column names (variable names)

    meta_data_column_names = [
        'ID',
        'LAT',
        'LON',
        'ELEV',
        'ST',
        'NAME',
        'GCN_FLAG',
        'HCN_CRN_FLAG',
        'WMO_ID',
        'ICAO_ID',
    ]

    meta_data_column_long_names = [
        'Station ID',
        'Latitude',
        'Longitude',
        'Elevation above sea level',
        'US state',
        'Station name',
        'GCOS Surface Network (GSN) flag',
        'U.S. Historical Climatology Network (HCN) or U.S. Climate Reference Network (CRN) flag',
        'World Meteorological Organization (WMO) station number',
        'International Civil Aviation Organization (ICAO) identifier',
    ]

    meta_data_column_units = ['', 'degrees north', 'degrees east', 'm', '', '', '', '', '', '']

    meta_data_widths = [11, 9, 10, 7, 3, 31, 4, 4, 6, 5]

    #
    # LCD observations
    #

    time_series_hourly: xr.Dataset

    def __init__(self, meta_data: pd.DataFrame, time_series_hourly: xr.Dataset = None):
        '''

        Initialize a Stations object.

        Args:
            meta_data (pd.DataFrame):
                Station metadata in a DataFrame using the schema defined by
                "meta_data_column_names".
            time_series_hourly (xr.Dataset, optional):
                Optional xarray Dataset of hourly station time series aligned
                with the provided metadata. If omitted, no time series are set.

        '''

        self.meta_data = meta_data.copy(deep=True)

        if time_series_hourly is not None:
            self.time_series_hourly = time_series_hourly.copy(deep=True)
        else:
            self.time_series_hourly = None

        return

    @classmethod
    def from_url(cls) -> Self:
        '''

        Build a Stations instance by downloading the GHCNh-format station list
        ("ghcnh-station-list.txt") from its configured URL and parsing it.

        Returns:
            Stations:
                Instance populated with cleaned station metadata.

        Notes:
            The GHCNh list contains stations that may have very sparse
            or no observations in a given year.

        '''

        response = requests.get(ncei.ghcnh_station_list_url)
        response.raise_for_status()
        text_stream = StringIO(response.text)

        meta_data = cls._read_fwf(text_stream)

        return cls(meta_data)

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        '''

        Build a Stations instance from a local fixed-width file with the same
        structure as "ghcnh-station-list.txt".

        Args:
            file_path (Path):
                Path to the station-list file.

        Returns:
            Stations:
                Instance populated with cleaned station metadata.

        '''

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        meta_data = cls._read_fwf(file_path)

        return cls(meta_data)

    @classmethod
    def _read_fwf(cls, source) -> pd.DataFrame:
        '''
        Read a fixed-width station list into a DataFrame and apply metadata cleanup.

        Args:
            source (str | Path | file-like):
                File path or file-like object.

        Returns:
            pd.DataFrame:
                Cleaned station metadata (strings normalized, numerics parsed,
                invalid rows removed).
        '''

        df = pd.read_fwf(
            source,
            widths=cls.meta_data_widths,
            skiprows=0,
            names=cls.meta_data_column_names[0:10],
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )

        # Clean the data
        df = cls._clean_meta_data(df)

        return df

    @staticmethod
    def _clean_meta_data(meta_data: pd.DataFrame) -> pd.DataFrame:
        '''

        Normalize and validate a GHCNh-format LCD station metadata table.

        Operations performed:

          - Strip surrounding whitespace from all string fields.
          - Convert selected columns to pandas StringDtype with <NA> for blanks.
          - Parse LAT, LON, ELEV as numerics with coercion.
          - Drop rows with missing LAT/LON.
          - Drop rows with ELEV <= -999.
          - Drop rows whose NAME contains "BOGUS".

        Args:
            meta_data (pd.DataFrame):
                Raw station metadata as read from the fixed-width file.

        Returns:
            pd.DataFrame:
                Cleaned station metadata.

        '''

        # Strip whitespace everywhere
        meta_data = meta_data.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Columns by intended type
        string_cols = [
            'ID',
            'ST',
            'NAME',
            'GCN_FLAG',
            'HCN_CRN_FLAG',
            'WMO_ID',
            'ICAO_ID',
        ]
        numeric_cols = ['LAT', 'LON', 'ELEV']

        # Strings: turn '' into <NA> and use StringDtype
        for col in string_cols:
            meta_data[col] = meta_data[col].replace('', pd.NA).astype('string')

        # Numerics: parse and coerce invalid/blank to NaN
        for col in numeric_cols:
            meta_data[col] = pd.to_numeric(meta_data[col], errors='coerce')

        # Reset index
        meta_data = meta_data.reset_index(drop=True)

        # Filter out rows with latitude or longitude holding a NaN, reset index and drop old index

        meta_data = meta_data.dropna(subset=['LAT', 'LON']).reset_index(drop=True)

        # Filter out rows with elevation <= -999, reset index and drop old index

        meta_data = meta_data[meta_data['ELEV'] > -999].reset_index(drop=True)

        # Filter out rows with 'BOGUS' in the station name, reset index and drop old index

        meta_data = meta_data[~meta_data['NAME'].str.contains('BOGUS', na=False)].reset_index(drop=True)

        return meta_data

    def filter_period_of_record_exists(self, token: str, verbose: bool = False, n_jobs: int = 16) -> Self:
        '''
        Keep only stations that have a nonempty period of record in NCEI CDO.

        For each station ID (GHCNh station identifier) in the metadata, queries
        the CDO API for mindate and maxdate. Stations without both dates are dropped.

        Args:
            token (str):
                NOAA CDO API token (https://www.ncei.noaa.gov/cdo-web/token).
            verbose (bool):
                If True, print progress messages.
            n_jobs (int):
                Number of concurrent API requests.

        Returns:
            Stations:
                New instance with metadata filtered to stations that report a
                period of record.
        '''
        if verbose:
            print()
            print('Filtering out stations for which there is no period of record', flush=True)
            print()

        filtered_rows = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_row = {
                executor.submit(ncei.get_period_of_record, row['ID'], token): row for _, row in self.meta_data.iterrows()
            }

            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    result = future.result()
                    if result['mindate'] is not None and result['maxdate'] is not None:
                        filtered_rows.append(row)
                        if verbose:
                            print(
                                'Including station',
                                row['ID'],
                                row['NAME'],
                                'POR =',
                                result['mindate'],
                                '-',
                                result['maxdate'],
                                flush=True,
                            )
                    else:
                        if verbose:
                            print('Excluding station', row['ID'], row['NAME'], '(no period of record)', flush=True)
                except Exception as e:
                    if verbose:
                        print(f"Error retrieving station {row['ID']}: {e}", flush=True)

        meta_data = pd.DataFrame(filtered_rows, columns=self.meta_data.columns)
        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def filter_by_coordinates(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> Self:
        '''

        Spatial filter by inclusive latitude/longitude bounds.

        Args:
            min_lat (float): Minimum latitude in degrees north.
            max_lat (float): Maximum latitude in degrees north.
            min_lon (float): Minimum longitude in degrees east.
            max_lon (float): Maximum longitude in degrees east.

        Returns:
            Stations:
                New instance containing only stations within the box.

        '''

        # Filter by latitude and longitude

        meta_data = self.meta_data[
            (self.meta_data['LAT'] >= min_lat)
            & (self.meta_data['LAT'] <= max_lat)
            & (self.meta_data['LON'] >= min_lon)
            & (self.meta_data['LON'] <= max_lon)
        ]

        # Reset row index

        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def filter_by_region(self, region_gdf: gpd.GeoDataFrame) -> Self:
        '''

        Spatial filter by a GeoPandas geometry (WGS84).

        Args:
            region_gdf (GeoPandas.GeoDataFrame):
                One or more geometries that define the selection region.
                Reprojected to EPSG:4326 if needed.

        Returns:
            Stations:
                New instance containing only stations whose points lie within
                the union of the region geometries.

        '''

        # Ensure the coordinate reference system (CRS) is WGS84 (latitude/longitude)
        if region_gdf.crs != 'EPSG:4326':
            region_gdf = region_gdf.to_crs('EPSG:4326')

        # Create GeoDataFrame from self.meta_data with the WGS84 (latitude/longitude)
        # coordinate reference system (CRS)
        meta_data_gdf = gpd.GeoDataFrame(
            self.meta_data,
            geometry=[Point(lon, lat) for lon, lat in zip(self.meta_data['LON'], self.meta_data['LAT'], strict=False)],
            crs='EPSG:4326',
        )

        # Combine geometries in region_gdf into a single multipolygon
        area = region_gdf.unary_union

        # Filter stations that fall within area
        filtered_meta_data_gdf = meta_data_gdf[meta_data_gdf.geometry.within(area)]

        # Construct a plain pandas DataFrame by dropping the geometry column
        meta_data = pd.DataFrame(filtered_meta_data_gdf.drop(columns='geometry'))

        # Reset row index
        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def filter_by_data_availability_online(
        self, start_time: datetime, end_time: datetime, n_jobs: int = 16, verbose: bool = False
    ) -> Self:
        '''

        Keep only stations for which all LCD files needed for the given
        period are present on the NCEI server.

        This checks server-side file listings for every year in the inclusive
        [start_time.year, end_time.year] range. It does not download files.

        Args:
            start_time (datetime):
                Start of the period of interest (only the year is used).
            end_time (datetime):
                End of the period of interest (only the year is used).
            n_jobs (int):
                Maximum number of parallel URL requests.
            verbose (bool):
                If True, print progress messages.

        Returns:
            Stations:
                New instance with metadata filtered to stations whose yearly
                LCD files are all available for the period.

        '''

        # Force time zone to be UTC
        start_time = start_time.replace(tzinfo=UTC)
        end_time = end_time.replace(tzinfo=UTC)

        # Get the URLs of all LCD data files on the NCEI web server:

        all_file_urls = ncei.lcd_data_urls(self.ids(), start_time.year, end_time.year, n_jobs=n_jobs, verbose=verbose)

        #
        # Construct a new dataframe
        #

        if verbose:
            print()
            print(
                'Filtering out stations for which not all files with observations are available on the NCEI server in the time range',
                str(start_time.year),
                'to',
                str(end_time.year),
                flush=True,
            )
            print()

        filtered_rows = []

        unavailable_urls = []

        for _, row in self.meta_data.iterrows():
            observations_files_available = True

            for year in range(start_time.year, end_time.year + 1):
                url = ncei.lcd_data_url(year, row['ID'])
                if url not in all_file_urls:
                    observations_files_available = False
                    unavailable_urls.append(url)

            if observations_files_available:
                filtered_rows.append(row)
                if verbose:
                    print('Including station', row['ID'], row['NAME'], flush=True)
            else:
                if verbose:
                    print(
                        'Excluding station',
                        row['ID'],
                        row['NAME'],
                        '(not all files with observations in the requested time range are available for download)',
                        flush=True,
                    )
                    for url in unavailable_urls:
                        print('Unavailable: ', url)

        # Construct a new DataFrame from the selected rows
        meta_data = pd.DataFrame(filtered_rows, columns=self.meta_data.columns)

        # Reset row index

        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def filter_by_data_availability_offline(
        self, data_dir: Path, start_time: datetime, end_time: datetime, verbose: bool = False
    ) -> Self:
        '''

        Keep only stations for which all LCD files in the given period
        are present in the given data directory.

        This checks local file listings for every year in the inclusive
        [start_time.year, end_time.year] range. It does not download files.

        Args:
            data_dir (Path):
                Directory containing the LCD files named as on the NCEI LCD server.
            start_time (datetime):
                Start of the period of interest (only the year is used).
            end_time (datetime):
                End of the period of interest (only the year is used).
            verbose (bool):
                If True, print progress messages.

        Returns:
            Stations:
                New instance with metadata filtered to stations whose yearly
                LCD files are all available for the period.

        '''

        # Force time zone to be UTC
        start_time = start_time.replace(tzinfo=UTC)
        end_time = end_time.replace(tzinfo=UTC)

        #
        # Construct a new dataframe
        #

        if verbose:
            print()
            print(
                'Filtering out stations for which not all files with observations are available in '
                + str(data_dir)
                + ' in the time range',
                str(start_time.year),
                'to',
                str(end_time.year),
                flush=True,
            )
            print()

        filtered_rows = []

        unavailable_files = []

        for _, row in self.meta_data.iterrows():
            observations_files_available = True

            for year in range(start_time.year, end_time.year + 1):
                file_path = data_dir / ncei.lcd_data_file_name(year, row['ID'])
                if not file_path.exists():
                    observations_files_available = False
                    unavailable_files.append(file_path)

            if observations_files_available:
                filtered_rows.append(row)
                if verbose:
                    print('Including station', row['ID'], row['NAME'], flush=True)
            else:
                if verbose:
                    print(
                        'Excluding station',
                        row['ID'],
                        row['NAME'],
                        '(not all files with observations in the requested time range are available locally)',
                        flush=True,
                    )
                    for file_path in unavailable_files:
                        print('Unavailable: ', file_path)

        # Construct a new DataFrame from the selected rows
        meta_data = pd.DataFrame(filtered_rows, columns=self.meta_data.columns)

        # Reset row index

        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def filter_by_id(self, station_id: str) -> Self:
        '''

        Filter by station ID (GHCNh station identifier)

        Args:
            station_id (str): GHCNh station identifier.

        Returns:
            Stations:
                New instance containing only stations with the given ID

        '''

        if station_id not in self.ids():
            raise ValueError(
                station_id
                + ' is not an available station ID (GHCNh station identifier). For a full list station IDs, see '
                + ncei.lcd_station_list_url
            )

        # Filter by station ID (GHCNh station identifier)

        meta_data = self.meta_data[(self.meta_data['ID'] == station_id)]

        # Reset row index

        meta_data = meta_data.reset_index(drop=True)

        return Stations(meta_data)

    def ids(self) -> list[str]:
        '''

        Return station IDs (GHCNh station identifiers).

        Returns:
            list[str]:
                Station IDs (GHCNh station identifiers) as strings.

        '''

        # Create list of station IDs
        station_ids = list(self.meta_data['ID'].values)

        return station_ids

    def save_station_list(self, file_path: Path, verbose: bool = False):
        '''
        Write the station list to a fixed-width file compatible with
        "ghcnh-station-list.txt".

        Args:
            file_path (Path):
                Destination path. Parent directories are created if missing.
        '''

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in self.meta_data.iterrows():
                # Safe getter: returns empty string if value is NaN
                def safe_get(value, fmt=None):
                    if pd.isna(value):
                        return '' if fmt is None else fmt.format('')
                    return value if fmt is None else fmt.format(value)

                row_strs = [
                    f"{safe_get(row['ID']):<11}",
                    f"{safe_get(row['LAT'], '{:+9.4f}')}",
                    f"{safe_get(row['LON'], '{:+10.4f}')}",
                    f"{safe_get(row['ELEV'], '{:+7.1f}')}",
                    f"{safe_get(row['ST']):>3}",
                    f"{' ' + safe_get(row['NAME']):<31}",
                    f"{safe_get(row['GCN_FLAG']):>4}",
                    f"{safe_get(row['HCN_CRN_FLAG']):>4}",
                    f"{safe_get(row['WMO_ID']):>6}",
                    f"{safe_get(row['ICAO_ID']):>5}",
                ]

                f.write("".join(row_strs) + '\n')

        if verbose:
            print()
            print('Created the stations metadata file', file_path)

        return

    def read_station_observations(
        self, data_dir: Path, start_year: int, end_year: int, station_id: str, verbose: bool = False
    ) -> tuple[pd.DataFrame, dict, dict]:
        '''
        Load Local Climatological Data (LCD) files for one station over an
        inclusive year range and return a cleaned, time-indexed table plus
        column metadata.

        The function locates files in the data_dir directory for "station_id"
        and the years "start_year" through "end_year". Each file is read,
        essential type conversions are applied, rows with no observations,
        with invalid report types, or with duplicated times are removed,
        and all years are concatenated into a single DataFrame indexed by
        UTC timestamps.

        Processing performed:
          - Verifies each expected file exists; raises "ValueError" if a file
            is missing.
          - Reads files as text with comma separation and double-quoted fields;
            empty strings become NaN; non-converted fields remain strings.
          - Converts known time fields in the file from Local Standard Time (without
            Daylight Saving Time) to UTC timezone-aware "datetime64[ns, UTC]" objects
          - Parses known numeric fields to "float" with invalid entries set to
            NaN.
          - Adjusts select temperature-like columns that are stored as
            degrees Celsius with a 18.3 C base so that values are returned in
            true degrees Celsius.
          - Sets the "DATE" column as the index ("DatetimeIndex" in UTC) and
            drops the original "DATE" column.
          - Filters out rows where "REPORT_TYPE" is "BOGUS", "SOD"
            (summary of day), or "SOM" (summary of month).
          - Drops temperature and dew point temperature when temperature > 60 C.
            Such extreme values are indoubtedly non-representative, and may be caused,
            e.g., by aircraft exhaust. Still, non-representative values < 60 C
            may remain in the data.
          - Sets station latitude, longitude, and elevation to values from station metadata
          - Removes duplicated times
          - Computes hourly relative humidity when both dry-bulb temperature
            and dew-point temperature are available. Values outside [0, 101]
            are set to NaN. Values between 100 and 101 are set to 100. The
            upper bound of 101 is used to accommodate small dew point temperature
            measurement errors.

        Args:
            data_dir (Path):
                Directory containing the LCD files named as on the NCEI LCD server.
            start_year (int):
                First year to include (inclusive).
            end_year (int):
                Last year to include (inclusive).
            station_id (str):
                GHCNh station identifier.
            verbose : bool, optional
                If True, print progress information to stdout. Defaults to False.

        Returns:
            pd.DataFrame:
                Observations for the station across the requested years,
                indexed by UTC timestamp ("pandas.DatetimeIndex"). Numeric and
                datetime columns are parsed where applicable. Rows with
                non-observational report types are removed. Includes the derived
                "HourlyRelativeHumidity" column where computable.
            dict:
                "long_names" mapping of column name to human-readable
                description for the returned columns, including the derived
                relative humidity.
            dict:
                "units" mapping of column name to unit string (for example,
                "C", "hPa", "m/s") or "None" if not applicable.

        Notes:
            The function preserves the general LCD column layout but performs
            light cleaning and harmonization so that timestamps are UTC, numeric
            fields are floats, and a small set of temperature columns are
            corrected to true degrees Celsius.
        '''

        # Local files for the given station and the year range
        local_files = ncei.lcd_data_file_paths(start_year, end_year, [station_id], data_dir)

        #
        # Loop over the files, and read tand process the data
        #

        if verbose:
            print()
            print('Reading data for station', station_id)
            print()

        dfs = []

        for file_path in local_files:
            if not file_path.exists():
                raise ValueError(str(file_path) + ' does not exist.')

            # Read the file as text

            with open(file_path) as f:
                df = pd.read_csv(
                    f,
                    sep=',',  # Comma-separated
                    header=0,  # First row contains column names
                    quotechar='"',  # Handle values in double quotes
                    na_values=[''],  # Treat empty strings as NaN
                    dtype=str,  # Read all data as strings for now
                )

                # Convert time columns to datetime objects, check if conversion failed

                time_column_index = [1, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]

                for ii in time_column_index:
                    original_column = df.iloc[:, ii].copy()

                    converted_column = pd.to_datetime(original_column, errors='coerce')

                    df.iloc[:, ii] = converted_column

                    # Find rows where conversion failed (NaT after conversion)

                    failed = original_column[converted_column.isna() & original_column.notna()]
                    if not failed.empty:
                        print(f'Failed to convert time string in column {ii} to datetime object. Problematic time string(s):')
                        print(failed)

                # Convert Local Standard Time (without Daylight Saving Time) to UTC

                tf = TimezoneFinder()

                # Use coordinates from meta data to determine the time zone

                station_mask = self.meta_data['ID'] == station_id

                lat_meta, lon_meta, elev_meta = self.meta_data.loc[station_mask, ['LAT', 'LON', 'ELEV']].iloc[0]
                station = self.meta_data.loc[station_mask, 'NAME'].iloc[0]

                tzname = tf.timezone_at(lat=float(lat_meta), lng=float(lon_meta))
                if tzname is None:
                    raise ValueError("Could not determine timezone from the provided coordinates.")

                if verbose:
                    print(
                        'STATION:',
                        station,
                        '  LAT =',
                        lat_meta,
                        '  LON =',
                        lon_meta,
                        '  ELEV =',
                        elev_meta,
                        '  TIME ZONE:',
                        tzname,
                        '  FILE:',
                        file_path,
                    )

                tz = ZoneInfo(tzname)

                # Offset of Local Standard Time (without Daylight Saving Time) relative to UTC (utc_offset = LST - UTC)
                utc_offset = datetime(2025, 1, 1, 0, 0, tzinfo=tz).utcoffset()
                if utc_offset is None:
                    raise ValueError(f"Could not obtain UTC offset for timezone: {tzname}")

                offset_td = pd.to_timedelta(utc_offset)

                for ii in time_column_index:
                    col_label = df.columns[ii]

                    # Work on a temporary datetime Series (naive LST)
                    converted = pd.to_datetime(df[col_label], errors='coerce')

                    # Only rows with parsable timestamp
                    mask = converted.notna()

                    # Set time to UTC = LST - utc_offset, then localize to UTC
                    utc_vals = (converted - offset_td).dt.tz_localize('UTC')

                    # Assign back only where valid; this keeps original values elsewhere
                    df.loc[mask, col_label] = utc_vals[mask]

                # Convert columns that contain numbers to floating point values

                column_index = [
                    2,
                    3,
                    4,
                    8,
                    9,
                    10,
                    11,
                    13,
                    15,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    62,
                    64,
                    66,
                    67,
                    69,
                    70,
                    71,
                    72,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    117,
                    121,
                    122,
                ]

                for ii in column_index:
                    df.iloc[:, ii] = pd.to_numeric(df.iloc[:, ii], errors='coerce').astype(float)

                # Convert columns that contain degrees Celsius with 18.3 degree base to degree Celsius

                column_index = [33, 34, 35, 54, 55, 81, 82, 84, 85, 88, 89]

                for ii in column_index:
                    df.iloc[:, ii] = df.iloc[:, ii] + 18.3

                # Set latitude, longitude, and elevation to values from metadata
                df.iloc[:, 2] = lat_meta
                df.iloc[:, 3] = lon_meta
                df.iloc[:, 4] = elev_meta

                dfs.append(df)

        df_output = pd.concat(dfs, ignore_index=True)

        # Set the 'DATE' column as index and drop the old 'DATE' column
        df_output = df_output.set_index(pd.DatetimeIndex(df_output['DATE'], name='DATE'))
        df_output = df_output.drop(columns='DATE')

        # Remove rows in which the column 'REPORT_TYPE' is 'BOGUS'
        df_output = df_output[df_output['REPORT_TYPE'] != 'BOGUS']

        # Remove rows in which the column 'REPORT_TYPE' is 'SOD' (summary of day)
        df_output = df_output[df_output['REPORT_TYPE'] != 'SOD']

        # Remove rows in which the column 'REPORT_TYPE' is 'SOM' (summary of month)
        df_output = df_output[df_output['REPORT_TYPE'] != 'SOM']

        # Regularize temperature to not exceed 60C - this eliminates extreme values, e.g. caused by aircraft exhaust
        mask = df_output['HourlyDryBulbTemperature'] > 60
        df_output.loc[mask, 'HourlyDryBulbTemperature'] = np.nan
        df_output.loc[mask, 'HourlyDewPointTemperature'] = np.nan

        # Remove time duplicates by keeping the first duplicated time at which not all
        # selected observables are NaNs, for each duplicated time

        observables = ['HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyWindSpeed']

        if df_output.index.has_duplicates:
            dup_mask = df_output.index.duplicated(keep=False)
            duplicate_index_set = set(df_output.index[dup_mask])

            row_numbers_remove = []

            for index in duplicate_index_set:
                # Absolute row numbers for this index (in original order)
                row_numbers_index = np.flatnonzero(df_output.index == index)

                # DataFrame consisting of rows with this index, only selected observables
                row_numbers_df = df_output.loc[index, observables]

                # Mask: True where at least one of the observables is NOT NaN
                mask = row_numbers_df.notna().any(axis=1).to_numpy()

                # Positions (absolute) with at least one non-NaN
                candidates = row_numbers_index[mask]

                if candidates.size > 0:
                    keep = candidates[:1]  # Keep the first non-NaN row
                else:
                    keep = np.array([], dtype=int)  # Empty array to drop all

                # Remove all rows in this group except the kept one
                to_remove = np.setdiff1d(row_numbers_index, keep, assume_unique=True)

                row_numbers_remove.extend(to_remove.tolist())

            # Info for user

            if verbose:
                print()
                print('Removing duplicated times:')
                for ii in row_numbers_remove:
                    print(file_path, df_output.index[ii])

            # Convert row numbers to index labels and drop duplicated rows

            mask = np.ones(len(df_output), dtype=bool)
            mask[row_numbers_remove] = False
            df_output = df_output[mask]

        # Create a dictionary that translates the column name to long name

        long_names_list = [
            'GHCNh station identifier',
            'Latitude',
            'Longitude',
            'Altitude above sea level',
            'Station long name',
            'Code (see GHCNh documentation)',
            'Code (see GHCNh documentation)',
            'Altimeter setting',
            'Dew point temperature (at ~ 2 m)',
            'Temperature (at ~ 2 m)',
            'Precipitation (1 h)',
            'Up to 3 weather codes (see appendix)',
            'Pressure change',
            'Pressure tendency code',
            'Relative humidity',
            'Up to 3 sky cover codes and cloud base height',
            'Sea level pressure',
            'Station pressure',
            'Visibility',
            'Wet bulb temperature',
            'Wind direction (at ~ 10 m)',
            'Wind gust speed (at ~ 10 m)',
            'Wind speed (at ~ 10 m)',
            'Local sunrise time',
            'Local sunset time',
            'Daily average dew point temperature',
            'Daily average dry bulb temperature',
            'Daily average relative humidity',
            'Daily average sea level pressure',
            'Daily average station pressure',
            'Daily average wet bulb temperature',
            'Daily average wind speed (at ~ 10 m)',
            'Daily cooling degree days',
            'Daily departure from normal average temperature',
            'Daily heating degree days',
            'Daily maximum dry bulb temperature',
            'Daily minimum dry bulb temperature',
            'Daily peak wind direction (at ~ 10 m)',
            'Daily peak wind speed (at ~ 10 m)',
            'Daily precipitation',
            'Daily snow depth',
            'Daily snowfall',
            'Daily sustained wind direction (at ~ 10 m)',
            'Daily sustained wind speed (at ~ 10 m)',
            'Daily weather codes',
            'Monthly average relative humidity',
            'Number of days with precipitation > 0.01',
            'Number of days with precipitation > 0.10',
            'Number of days with temperature > 32 F',
            'Number of days with temperature > 90 F',
            'Number of days with temperature < 0 C',
            'Number of days with temperature < 32 F',
            'Monthly departure from normal average temperature',
            'Monthly departure from normal cooling degree days',
            'Monthly departure from normal heating degree days',
            'Monthly departure from normal maximum temperature',
            'Monthly departure from normal minimum temperature',
            'Monthly departure from normal precipitation',
            'Monthly average dew point temperature',
            'Monthly greatest precipitation',
            'Date of monthly greatest precipitation',
            'Monthly greatest snow depth',
            'Date of monthly greatest snow depth',
            'Monthly greatest snowfall',
            'Date of monthly greatest snowfall',
            'Monthly maximum sea level pressure value',
            'Date of monthly max sea level pressure',
            'Time of monthly max sea level pressure',
            'Monthly maximum temperature',
            'Monthly mean temperature',
            'Monthly minimum sea level pressure value',
            'Date of monthly min sea level pressure',
            'Time of monthly min sea level pressure',
            'Monthly minimum temperature',
            'Monthly average sea level pressure',
            'Monthly average station pressure',
            'Monthly total liquid precipitation',
            'Monthly total snowfall',
            'Monthly average wet bulb temperature',
            'Monthly average wind speed (at ~ 10 m)',
            'Cooling degree days season-to-date',
            'Monthly cooling degree days',
            'Number of days with snowfall in month',
            'Heating degree days season-to-date',
            'Monthly heating degree days',
            'Number of days with thunderstorms in month',
            'Number of days with heavy fog in month',
            'Normals cooling degree days',
            'Normals heating degree days',
            'End date/time for 5-minute precipitation',
            'End date/time for 10-minute precipitation',
            'End date/time for 15-minute precipitation',
            'End date/time for 20-minute precipitation',
            'End date/time for 30-minute precipitation',
            'End date/time for 45-minute precipitation',
            'End date/time for 60-minute precipitation',
            'End date/time for 80-minute precipitation',
            'End date/time for 100-minute precipitation',
            'End date/time for 120-minute precipitation',
            'End date/time for 150-minute precipitation',
            'End date/time for 180-minute precipitation',
            '5-minute precipitation value',
            '10-minute precipitation value',
            '15-minute precipitation value',
            '20-minute precipitation value',
            '30-minute precipitation value',
            '45-minute precipitation value',
            '60-minute precipitation value',
            '80-minute precipitation value',
            '100-minute precipitation value',
            '120-minute precipitation value',
            '150-minute precipitation value',
            '180-minute precipitation value',
            'Remarks',
            'Backup station direction',
            'Backup station distance',
            'Backup station distance unit',
            'Backup elements',
            'Backup station elevation',
            'Backup equipment',
            'Backup station latitude',
            'Backup station longitude',
            'Backup station name',
            'Wind equipment change date',
        ]

        long_names = {}

        for name, long_name in zip(list(df_output.columns), long_names_list, strict=False):
            long_names[name] = long_name

        # Create a dictionary that translates the column name to units

        units_list = [
            '',
            '° N',
            '° E',
            'm',
            '',
            '',
            '',
            'hPa',
            'C',
            'C',
            'mm',
            'weather code(s)',
            'hPa',
            'code',
            'percent',
            'oktas + cloud base height',
            'hPa',
            'hPa',
            'km',
            'C',
            '° (1-360)',
            'm/s',
            'm/s',
            'LST (HH:mm)',
            'LST (HH:mm)',
            'C',
            'C',
            'percent',
            'hPa',
            'hPa',
            'C',
            'm/s',
            'C',
            'C',
            'C',
            'C',
            'C',
            '(1-360)',
            'm/s',
            'mm',
            'mm',
            'mm',
            '° (1-360)',
            'm/s',
            '2-digit codes',
            'percent',
            'count',
            'count',
            'count',
            'count',
            'count',
            'count',
            'C',
            'C',
            'C',
            'C',
            'C',
            'mm',
            'C',
            'mm',
            'DD-DD',
            'mm',
            'DD-DD',
            'mm',
            'DD-DD',
            'hPa',
            'DD',
            'HHmm',
            'C',
            'C',
            'hPa',
            'DD',
            'HHmm',
            'C',
            'hPa',
            'hPa',
            'mm',
            'mm',
            'C',
            'm/s',
            'C',
            'C',
            'count',
            'C',
            'C',
            'count',
            'count (visibility < 1/4 mile)',
            'C',
            'C',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'datetime',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            '',
            '',
            '',
            'miles',
            '',
            '',
            '',
            'decimal degrees',
            'decimal degrees',
            '',
            '',
        ]

        units = {}

        for name, unit in zip(list(df_output.columns), units_list, strict=False):
            units[name] = unit

        #
        # Calculate relative humidity where both temperature and dew point temperature are available
        #

        mask = df_output['HourlyDewPointTemperature'].notna() & df_output['HourlyDryBulbTemperature'].notna()

        # Column for relative humidity

        rh_name = 'HourlyRelativeHumidity'
        long_names[rh_name] = 'Relative humidity (at ~ 2 m)'
        units[rh_name] = '%'

        # Initialize with NaNs

        df_output[rh_name] = np.nan

        # Calculate RH where temperature and dry bulk temperature are available

        df_output.loc[mask, rh_name] = df_output.loc[mask, ['HourlyDryBulbTemperature', 'HourlyDewPointTemperature']].apply(
            lambda s: saturation.rh(s['HourlyDryBulbTemperature'], s['HourlyDewPointTemperature']), axis=1
        )

        # RH outside the [0,101] range indicates a problem with the temperature/dew point temperature.
        # Set these values to NaN. Set values between 100 and 101 to 100. The upper bound of 101
        # is used to accommodate small dew point temperature measurement errors.

        mask = df_output[rh_name].between(0, 101)
        df_output.loc[~mask, rh_name] = np.nan

        mask = df_output[rh_name].between(100, 101)
        df_output.loc[mask, rh_name] = 100

        return df_output, long_names, units

    def write_utc_hourly_netcdf(self, file_path: Path, verbose: bool = False):
        '''
        Writes the xarray self.time_series_hourly into a netCDF file.

        Args:
            file_path (Path): Path to the netCDF file. The file will be overwritten if it exists.
        verbose : bool, optional
            If True, print progress information to stdout. Defaults to False.
        '''

        # Remove attribute/encoding conflicts with any units set previously on time-like variables
        for name in 'time':
            if name in self.time_series_hourly.variables:
                # Remove conflicting 'units' or 'calendar' attributes if present
                for key in ('units', 'calendar'):
                    if key in self.time_series_hourly[name].attrs:
                        del self.time_series_hourly[name].attrs[key]

        # Build encoding

        encoding = {
            "time": {
                "dtype": "float64",
                "units": "seconds since 1970-01-01T00:00:00Z",
                "calendar": "proleptic_gregorian",
            },
            **{
                var: {"dtype": "float32"}
                for var in self.time_series_hourly.data_vars
                if np.issubdtype(self.time_series_hourly[var].dtype, np.floating)
            },
        }

        # Identify any variables in the xarray that hold objects (such as datatime objects)
        object_vars = [name for name, var in self.time_series_hourly.data_vars.items() if var.dtype == object]

        # Save to netCDF skipping any variables that are objects
        if object_vars:
            self.time_series_hourly.drop_vars(object_vars).to_netcdf(file_path, encoding=encoding)
        else:
            self.time_series_hourly.to_netcdf(file_path, encoding=encoding)

        if verbose:
            print()
            print('Created the full-hourly UTC time series file', file_path)
            print()

        return

    def construct_hourly(
        self,
        data_dir: Path,
        start_year: int,
        end_year: int,
        region: str = 'unspecified',
        max_interpolation_interval_h: float = 2,
        plot_dir: Path = None,
        verbose: bool = False,
    ) -> xr.Dataset:
        '''
        Build regular, full-hourly UTC time series for each station in "self.meta_data"
        by interpolating Local Climatological Data (LCD) observations to the hourly grid,
        for the following observables:

            "T"         (from "HourlyDryBulbTemperature")
            "Td"        (from "HourlyDewPointTemperature")
            "RH"        (from "RelativeHumidity")
            "windspeed" (from "HourlyWindSpeed")

        Parameters
        ----------
        data_dir : Path
            Directory containing LCD files.
        start_year : int
            First calendar year to include (inclusive).
        end_year : int
            Last calendar year to include (inclusive).
        region : str, optional
            Region label to store as a global attribute in each station dataset. Default is
            "unspecified".
        max_interpolation_interval_h : float, optional
            Maximum gap, in hours, across which interpolation is permitted. Passed through
            to "interpolate_to_full_hour". Default is "2".
        plot_dir : Path | None, optional
            Directory in which to save comparison plots produced by "interpolate_to_full_hour".
            If the directory does not exist, it will be created. If "None" (default), no plots
            are written. Generating plots can be slow for many stations and years.
        verbose : bool, optional
            If True, print progress information to stdout. Defaults to False.

        Returns
        -------
        xr.Dataset
            The hourly dataset concatenated across all successfully processed stations on the
            "station" dimension. The same object is also stored on "self.time_series_hourly".

        Notes
        -----
        - Interpolation is performed by "interpolate_to_full_hour" using nearest valid neighbors
          and linear interpolation on a UTC hourly grid.
        - Only stations that produce a non-"None" dataset are included in the concatenation.
        - The concatenation requires that non-concatenated variables and attributes match exactly;
          conflicting global attributes are dropped per "combine_attrs='drop_conflicts'".
        - If no stations yield data, concatenation will fail; ensure that at least one station has
          valid observations in the requested period.

        Examples
        --------
        Construct hourly time series for all known stations from 2020 through 2022:

            ds = self.construct_hourly(
                data_dir=Path("/data/lcd"),
                start_year=2020,
                end_year=2022,
                region="CONUS",
                max_interpolation_interval_h=2.0,
                plot_dir=None,
            )
        '''

        var_names_lcd = ['HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyWindSpeed']
        var_names = ['T', 'Td', 'RH', 'windspeed']

        ds_stations = []

        assert len(self.ids()) > 0, 'No stations in dataset. Aborting.'

        for station_id in self.ids():
            # Interpolate time series of the observables

            ds = self.interpolate_to_full_hour(
                data_dir,
                start_year,
                end_year,
                station_id,
                var_names_lcd,
                var_names,
                region=region,
                max_interpolation_interval_h=max_interpolation_interval_h,
                plot_dir=plot_dir,
                verbose=verbose,
            )

            if ds is not None:
                ds_stations.append(ds)

        assert len(ds_stations) > 0, 'Not enough data. Aborting.'

        # Concatenate the individual station datasets into one dataset

        self.time_series_hourly = xr.concat(
            ds_stations,
            dim='station',  # Concatenate along the 'station' dimension (stacks datasets by station)
            data_vars='minimal',  # Only include data variables that are identical across datasets
            coords='minimal',  # Include only coordinates that are identical across datasets
            compat='equals',  # Require that non-concatenated variables and attributes match exactly
            combine_attrs='drop_conflicts',  # Keep global attributes that agree; drop any that conflict
        )

        return

    def interpolate_to_full_hour(
        self,
        data_dir: Path,
        start_year: int,
        end_year: int,
        station_id: str,
        var_names_lcd: list[str],
        var_names_output: list[str],
        region: str = 'unspecified',
        max_interpolation_interval_h: float = 2,
        plot_dir: Path = None,
        verbose: bool = False,
    ) -> xr.Dataset | None:
        '''
        Interpolate one or more Local Climatological Data (LCD) observables to a regular hourly
        UTC grid for a single station using nearest valid neighbors and linear interpolation.

        The procedure:

          1. Read LCD observations for the specified station and year range.
          2. Build an hourly UTC time coordinate covering the range [start_year-01-01 00:00, end_year-12-31 23:00].
          3. For each full hour, search outward from the hour to find the nearest non-NaN values
             on the left and right within the "max_interpolation_interval_h" (in hours).
             If both sides are found, linearly interpolate to the hour; otherwise assign NaN.
          4. Package the hourly series and station metadata into an xarray Dataset.

        This method is suitable for instantaneous observables (for example, temperature, dew point,
        wind speed). It must not be used for accumulated or interval-total quantities
        (for example, precipitation totals).

        Parameters
        ----------
        data_dir : Path
            Directory containing previously downloaded LCD files.
        start_year : int
            First calendar year to include (inclusive).
        end_year : int
            Last calendar year to include (inclusive).
        station_id : str
            GHCN station identifier used by LCD.
        var_names_lcd : list[str]
            LCD variable names to interpolate. Examples include:

                HourlyDryBulbTemperature
                HourlyDewPointTemperature
                RelativeHumidity
                HourlyWindSpeed

        var_names_output : list[str]
            Output variable names to assign in the result. Must be the same length as
            "var_names_lcd" and correspond positionally (one output name per input name).
            Examples include:

                T
                Td
                RH
                windspeed

        region : str, optional
            Free-form region label to store in global attributes. Default is "unspecified".
        max_interpolation_interval_h : float, optional
            Maximum temporal gap, in hours, that is permitted for interpolation across a missing
            interval. Default is "2".
        plot_dir : Path | None, optional
            Directory in which to save time-series comparison plots. If the directory does not
            exist, it will be created. If "None" (default), no plots are written. Generating
            plots is comparatively slow for large datasets.
        verbose : bool, optional
            If True, print progress information to stdout. Defaults to False.

        Returns
        -------
        xr.Dataset | None
            Dataset with:

              - A regular hourly "time" coordinate spanning the requested years.
              - One data variable per entry in "var_names_output" with dimensions
                "('time', 'station')" containing the interpolated series.
              - A "station" coordinate of length 1.
              - A "UTC" variable holding Python "datetime" objects for convenience.
              - Station metadata variables (latitude, longitude, elevation, name).
              - Global attributes describing source, processing, and region.

            Returns "None" if there are no valid data for the requested variables
            in the requested period.

        Raises
        ------
        ValueError
            If the input time index contains duplicate timestamps, which prevents safe
            interpolation.
        ValueError
            If the input time index is not monotonically increasing.

        Notes
        -----
        - Interpolation occurs only when non-NaN neighbors exist on both sides of a target hour
          within "max_interpolation_interval_h". Otherwise the result is NaN.
        - If an LCD observation falls exactly on a full hour and is non-NaN, that value is used
          directly without interpolation.
        - The function prints basic progress messages and information about any detected issues
          with duplicate or non-monotonic time indices.
        - When "plot_dir" is provided, one PNG is written per year and variable showing the
          original irregular LCD series and the interpolated hourly series for the station.

        Examples
        --------
        Interpolate dry-bulb temperature and wind speed to hourly resolution:

            ds = self.interpolate_to_full_hour(
                data_dir=Path("/data/lcd"),
                start_year=2020,
                end_year=2022,
                station_id="USW00023174",
                var_names_lcd=["HourlyDryBulbTemperature", "HourlyWindSpeed"],
                var_names_output=["T", "windspeed"],
                region="CONUS",
                max_interpolation_interval_h=2.0,
                plot_dir=None,
            )

        '''

        #
        # Read the LCD data
        #

        df, long_names, units = self.read_station_observations(data_dir, start_year, end_year, station_id, verbose=verbose)

        local_files = ncei.lcd_data_file_paths(start_year, end_year, [station_id], data_dir)

        # Check for time duplicates

        if df.index.has_duplicates:
            print('The dataframe constructed from the files')
            print()

            for local_file in local_files:
                print(local_file)

            print()
            print('contains duplicate times:')

            # Show all rows for duplicated timestamps (including the first occurrences)
            dup_all_mask = df.index.duplicated(keep=False)
            print(df[dup_all_mask].sort_index())

            # Count how many times each timestamp appears
            print(df.index.value_counts().sort_values(ascending=False).head(10))

            # How many duplicated labels total (beyond the unique set)
            print(df.index.duplicated().sum())

            raise ValueError('Cannot safely interpolate because dataframe index contains duplicates.')

        # Check if time is not decreasing (allows equal times)

        if not df.index.is_monotonic_increasing:
            print('In the dataframe constructed from the files')
            print()

            for local_file in local_files:
                print(local_file)

            print()
            print('time is not monotonically increasing')

            raise ValueError('Cannot safely interpolate because of non-monotonic time.')

        #
        # Construct an xarray dataset containing time information only
        #

        # Construct full hour UTC times for the given range of years, as a list of datetime objects

        start_time = datetime(year=start_year, month=1, day=1, hour=0, minute=0, second=0)
        start_time = start_time.replace(tzinfo=UTC)

        end_time = datetime(year=end_year, month=12, day=31, hour=23, minute=0, second=0)
        end_time = end_time.replace(tzinfo=UTC)

        time_step = timedelta(hours=1)

        hours_utc_datetime = []

        time = start_time

        while time <= end_time:
            hours_utc_datetime.append(time)
            time += time_step

        #
        # If there are no data in the dataset, return None
        #

        if len(df) == 0:
            if verbose:
                print('No data for station ' + station_id + '. Skipping.')
            return None

        #
        # If there are no valid values for latitude and longitude in the dataset, return None
        #

        if df['LATITUDE'].isna().all() or df['LONGITUDE'].isna().all():
            if verbose:
                print('No valid latitude/longitude for station ' + station_id + '. Skipping.')
            return None

        #
        # If there are no valid values for the requested variables in the dataset, return None
        #

        if df[var_names_lcd].isna().all().all():
            if verbose:
                print('No valid data for the requested variables for station ' + station_id + '. Skipping.')
            return None

        # Convert to numpy array of numpy.datetime64 objects
        hours_utc_datetime64 = pd.to_datetime(hours_utc_datetime).values

        # Initialize the empty xarray dataset

        ds = xr.Dataset(coords={'time': hours_utc_datetime64, 'station': [df['STATION'].values[0]]})

        # Add a variable holding the UTC time as datetime objects

        ds['UTC'] = ('time', np.array(hours_utc_datetime, dtype='O'))
        ds['UTC'].attrs['long_name'] = 'UTC'
        ds['UTC'].attrs['units'] = ''

        # Add variables that are a function of station only

        ds['LAT'] = ('station', [df['LATITUDE'].values[0]])
        ds['LAT'].attrs['long_name'] = long_names['LATITUDE']
        ds['LAT'].attrs['units'] = units['LATITUDE']

        ds['LON'] = ('station', [df['LONGITUDE'].values[0]])
        ds['LON'].attrs['long_name'] = long_names['LONGITUDE']
        ds['LON'].attrs['units'] = units['LONGITUDE']

        ds['ELEV'] = ('station', [df['ELEVATION'].values[0]])
        ds['ELEV'].attrs['long_name'] = long_names['ELEVATION']
        ds['ELEV'].attrs['units'] = units['ELEVATION']

        if pd.isna(df['NAME'].values[0]):
            ds['STATION_NAME'] = ('station', ['NO NAME STATION'])
        else:
            ds['STATION_NAME'] = ('station', [df['NAME'].values[0]])

        ds['STATION_NAME'].attrs['long_name'] = long_names['NAME']
        ds['STATION_NAME'].attrs['units'] = units['NAME']

        # Add global attributes

        ds.attrs['name'] = 'LCD'
        ds.attrs['long_name'] = 'Local Climatological Data'
        ds.attrs['description'] = (
            'Local Climatological Data (LCD) are summaries of climatological conditions from airport and other prominent weather stations managed by NWS, FAA, and DOD.'
        )
        ds.attrs['source'] = 'National Centers for Environmental Information (NCEI)'
        ds.attrs['URL'] = ncei.lcd_url
        ds.attrs['processed_with'] = 'https://github.com/jankazil/lcd-data'
        ds.attrs['region'] = region

        #
        # Construct full hour time series
        #

        # Find times in the LCD time series that bound the full hour

        times_l = df.index[:-1]
        times_r = df.index[1:]

        index_l = []
        index_r = []

        for hour in hours_utc_datetime:
            mask = (hour >= times_l) & (hour < times_r)

            if mask.any():
                index = np.where(mask)[0][0]
                index_l.append(index)
                index_r.append(index + 1)
            else:
                index_l.append(None)
                index_r.append(None)

        #
        # Interpolate the LCD time series to the full hour for each requested variable
        #

        hour_n = len(hours_utc_datetime)

        max_interpolation_interval_s = 3600 * max_interpolation_interval_h

        # Loop over observables

        if verbose:
            print()

        for var_name_lcd, var_name_output in zip(var_names_lcd, var_names_output, strict=False):
            if verbose:
                print(
                    'Constructing full-hourly time series for the station '
                    + station_id
                    + ' for the time range '
                    + str(start_year)
                    + '-'
                    + str(end_year)
                    + ' for '
                    + var_name_lcd
                )

            timeseries_hourly = np.full(hour_n, np.nan)

            # Loop over full hours

            for hour_i in range(hour_n):
                # If there is a full-hourly datapoint in the LCD time series, use it without interpolation
                if (
                    index_l[hour_i] is not None
                    and hours_utc_datetime[hour_i] == df.index[index_l[hour_i]]
                    and not np.isnan(df[var_name_lcd].iloc[index_l[hour_i]])
                ):
                    timeseries_hourly[hour_i] = df[var_name_lcd].iloc[index_l[hour_i]]

                    continue

                # Only proceed if we have valid left and right indices around the current full hour

                if index_l[hour_i] is not None and index_r[hour_i] is not None:
                    # Search leftwards for the nearest non-NaN observable value within the maximum interpolation interval

                    ii = index_l[hour_i]
                    value_l = np.nan
                    time_l = np.nan
                    time_delta_seconds_l = np.nan

                    while ii >= 0:
                        if np.abs((hours_utc_datetime[hour_i] - df.index[ii]).total_seconds()) > max_interpolation_interval_s:
                            break
                        if not np.isnan(df[var_name_lcd].iloc[ii]):
                            time_l = df.index[ii]
                            time_delta_seconds_l = np.abs((hours_utc_datetime[hour_i] - time_l).total_seconds())
                            value_l = df[var_name_lcd].iloc[ii]
                            break
                        ii -= 1

                    # Search rightwards for the nearest non-NaN observable value within the maximum interpolation interval

                    ii = index_r[hour_i]
                    value_r = np.nan
                    time_r = np.nan
                    #                    time_delta_seconds_r = np.nan

                    # Only proceed if a valid observation was found on the left in the interpolation interval

                    if not np.isnan(time_delta_seconds_l):
                        while ii < len(df.index):
                            if (
                                np.abs((hours_utc_datetime[hour_i] - df.index[ii]).total_seconds())
                                > max_interpolation_interval_s - time_delta_seconds_l
                            ):
                                break
                            if not np.isnan(df[var_name_lcd].iloc[ii]):
                                time_r = df.index[ii]
                                #                                time_delta_seconds_r = np.abs((hours_utc_datetime[hour_i] - time_r).total_seconds())
                                value_r = df[var_name_lcd].iloc[ii]
                                break
                            ii += 1

                    # If both left and right observable values are valid, perform (linear) interpolation

                    if not np.isnan(value_l) and not np.isnan(value_r):
                        timeseries_hourly[hour_i] = (
                            value_l
                            + (hours_utc_datetime[hour_i] - time_l).total_seconds()
                            * (value_r - value_l)
                            / (time_r - time_l).total_seconds()
                        )

            # We still need to check whether the last full hour coincides
            # with the last time in LCD time series, as the above algorithm
            # misses that last hour

            if hours_utc_datetime[-1] == df.index[-1]:
                timeseries_hourly[-1] = df[var_name_lcd].iloc[-1]

            # Add the interpolated observable value to the dataset, as both a function of time and station

            ds[var_name_output] = (('time', 'station'), timeseries_hourly[:, np.newaxis])
            ds[var_name_output].attrs['long_name'] = long_names[var_name_lcd]
            ds[var_name_output].attrs['units'] = units[var_name_lcd]
            ds[var_name_output].attrs['max_interpolation_interval_h'] = max_interpolation_interval_h
            ds[var_name_output].attrs['processing'] = (
                'Hourly values interpolated from values at reported station times, '
                + 'with a maximum interpolation inverval of '
                + str(np.round(max_interpolation_interval_h, 2))
                + ' h'
            )

            #
            # Create plot comparing the original and interpolated time series
            #

            if plot_dir is not None:
                plot_dir.mkdir(parents=True, exist_ok=True)

                for year in range(df.index[0].year, df.index[-1].year + 1):
                    for station_index in range(len(ds.coords['station'])):
                        int_max_int = str(np.round(ds[var_name_output].attrs['max_interpolation_interval_h'], 2))

                        # Construct plot title

                        station_name = ds['STATION_NAME'].values[station_index]
                        station_id = ds.coords['station'].values[station_index]
                        station_lat = str(ds['LAT'].values[station_index]) + ' ' + ds['LAT'].attrs['units']
                        station_lon = str(ds['LON'].values[station_index]) + ' ' + ds['LON'].attrs['units']
                        station_elev = str(ds['ELEV'].values[station_index]) + ' ' + ds['ELEV'].attrs['units']

                        title = (
                            station_name
                            + ' (Station ID: '
                            + station_id
                            + ', lat = '
                            + station_lat
                            + ', lon = '
                            + station_lon
                            + ', elevation = '
                            + str(station_elev)
                            + ')'
                            + ', data source: Local Climatological Data (LCD), National Centers for Environmental Information (NCEI), processed with https://github.com/jankazil/lcd-data'
                        )

                        # Plotting
                        fig, ax = plt.subplots(figsize=(100, 4))
                        plt.plot(
                            df.index,
                            df[var_name_lcd],
                            label='LCD original time series',
                            linewidth=1,
                            marker='o',
                            markersize=2,
                            color='black',
                        )
                        plt.plot(
                            ds.time,
                            ds[var_name_output][:, station_index],
                            label='LCD interpolated hourly time series, ' + int_max_int + ' h maximum interpolation interval',
                            linewidth=0.2,
                            marker='o',
                            markersize=0.5,
                            color='red',
                        )

                        # Major ticks once a month
                        ax.xaxis.set_major_locator(mdates.MonthLocator())

                        # Format the tick labels, e.g. 'Jan 2025', 'Feb 2025'
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

                        # Time range
                        start_date = datetime(year=year, month=1, day=1)
                        end_date = datetime(year=year + 1, month=1, day=1)
                        plt.xlim(start_date, end_date)

                        # Value range
                        if 'elative humidity' in ds[var_name_output].attrs['long_name']:
                            plt.ylim(0, 100)

                        # Annotations
                        plt.xlabel('Date')
                        plt.ylabel(ds[var_name_output].attrs['long_name'] + ' (' + ds[var_name_output].attrs['units'] + ')')
                        plt.title(title)

                        if 'elative humidity' in ds[var_name_output].attrs['long_name']:
                            ax.legend(loc='lower left')
                        else:
                            ax.legend(loc='upper left')

                        # Save as PNG
                        plot_file = plot_dir / Path(
                            station_id + '.' + str(year) + '.' + var_name_output + '.' + int_max_int + 'h.png'
                        )
                        plt.savefig(plot_file, bbox_inches='tight', dpi=600)
                        plt.close()

                        if verbose:
                            print('Created the plot ' + str(plot_file))

                        # Attempts to prevent run-away memory use in repeated calls (thanks to the Python language not having been designed for high performance applications)
                        del fig, ax
                        gc.collect()

        return ds
