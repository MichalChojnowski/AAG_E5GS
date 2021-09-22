import pandas as pd
import numpy as np
import json
import pathlib
import time
import uuid
import json
import urllib3
import logging
import os

from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

import configparser

from cartoframes import read_carto, to_carto, create_table_from_query
from cartoframes.data.observatory import Enrichment, Variable, Dataset
from cartoframes.auth import Credentials, set_default_credentials
from cartoframes.data.clients import SQLClient
from carto.sql import BatchSQLClient
from carto.auth import NonVerifiedAPIKeyAuthClient
from carto.exceptions import CartoException

from longitude.core.data_sources.carto_async import CartoAsyncDataSource

from datetime import datetime

from geopandas import GeoDataFrame
from typing import Union

from constants.global_constants import (
    meters_in_mile,
    miles_in_metre,
    urb_buffers,
    ENRICHMENT_PATH,
    PROJECT_DIR,
)

config = configparser.ConfigParser()
config.read(PROJECT_DIR + "credentials.ini")

# Configuring logs
logging_level = logging.getLevelName("INFO")
logging_format = "[%(asctime)s] (%(process)d) {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
loggin_datefmt = "%Y-%m-%dT%H:%M:%SZ"
logging.basicConfig(
    level=logging_level, format=logging_format, datefmt=loggin_datefmt
)
logger = logging.getLogger("vantage")
logger.setLevel(logging_level)


class IsochroneType:
    CAR = "car"
    WALk = "walk"


class RouteType:
    CAR = "car"
    WALk = "walk"


class RouteModeType:
    SHORT = "shortest"
    FAST = "fastest"


class GeometryType:
    POINTS = "points"
    POLYGONS = "polygons"


class CartoFramesModel:
    def __init__(self):
        self.__user = config["carto"]["user"]
        self.__api_key = config["carto"]["api_key"]
        self.__get_auth_client(self.__user, self.__api_key)
        self.NOW = datetime.now().date().strftime("%Y%m%d")
        self.SQL = SQLClient()
        self.BATCH = self.__authenticate_batch_api(self.__user, self.__api_key)
        self.creds = Credentials(self.__user, self.__api_key)
        self.ds = CartoAsyncDataSource(
            config["carto"]["user"],
            config["carto"]["api_key"],
        )
        self.vars = self.get_vars()

    @staticmethod
    def __get_auth_client(carto_user, api_key):
        set_default_credentials(
            username=carto_user,
            api_key=api_key,
            base_url=f"https://{carto_user}.carto.com/",
        )

    @staticmethod
    def __authenticate_batch_api(carto_user, api_key):
        auth = NonVerifiedAPIKeyAuthClient(
            api_key=api_key, base_url=f"https://{carto_user}.carto.com/"
        )
        return BatchSQLClient(auth)

    def check_status(self, query, verbose=False, wait=0.75):
        job_id = query["job_id"]
        check_status = {"status": "None"}

        while check_status["status"] != "done":
            readJob = self.BATCH.read(job_id)
            check_status["status"] = readJob["status"]
            time.sleep(wait)
            if check_status["status"] == "failed":
                log.error(readJob)
                raise CartoException
        if verbose:
            log.debug(readJob)
        return readJob

    def batch(self, sql, verbose=False, wait=0.75):
        query = self.BATCH.create(sql)
        return self.check_status(query, verbose, wait)

    def _sanitize_quotes(self, value):
        """
        Private method to sanitize quotes for SQL queries
        """
        return value.replace("'", "''")

    def _get_uuid(self):
        return str(uuid.uuid1()).replace("-", "_")

    async def generic_sql(self, query: str):
        try:
            return await self.ds.query(self._sanitize_quotes(query))
        except Exception as err:
            raise err

    def get_geopandas_from_query(self, sql: str):
        """
        Read a table or a SQL query from the CARTO account
        and return a GeoDataFrame.
        The geometry field need to be called as "the_geom"

        Parameters:

        - sql: SQL query

        Return:

        - geopandas.GeoDataFrame
        """
        return read_carto(sql)

    def update_rows(
        self,
        table: str,
        identificator_key: str,
        columns_to_update: list,
        dataframe: Union[pd.DataFrame, GeoDataFrame],
    ):
        """
        Update values in CARTO account tables

        Parameters:

        - table: name of the CARTO table to update
        - column_to_update: name of the field to update
        - identificator_key: name of the unique key
            to locate the correct row to update
        - columns_to_update: list with the name of the columns to update
        - dataframe: Dataframe or GeoDataframe

        Return:

        - boolean True|False
        """
        aux_table_name = "_dgn_" + str(uuid.uuid1()).replace("-", "_")
        try:

            if not (
                isinstance(dataframe, GeoDataFrame)
                or isinstance(dataframe, pd.DataFrame)
            ):
                raise Exception("input params error")

            log.info("Processing data")
            to_carto(
                dataframe, aux_table_name, cartodbfy=False, log_enabled=False
            )

            setter = ",".join(
                map(
                    lambda column: f"{column} = aux.{column}",
                    columns_to_update,
                )
            )
            query = f"""UPDATE {table} t
                SET {setter}
                FROM {aux_table_name} aux
                WHERE t.{identificator_key} = aux.{identificator_key}"""

            log.info("Updating data ...")
            self.SQL.execute(query)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.info("Updated !")
            return True
        except Exception as e:
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.error(e)
            return False

    def insert_rows(
        self,
        table: str,
        columns: list,
        dataframe: Union[pd.DataFrame, GeoDataFrame],
    ):
        """
        Insert rows in CARTO account tables.

        Parameters:

        - table: name of the CARTO table to update
        - columns: array with the name of the field to append
        - dataframe: Dataframe or GeoDataframe

        Return:

        - boolean True|False
        """
        # we need the 3 variables filled
        if not columns:
            return False

        aux_table_name = "_dgn_" + str(uuid.uuid1()).replace("-", "_")
        try:

            if not (
                isinstance(dataframe, GeoDataFrame)
                or isinstance(dataframe, pd.DataFrame)
            ):
                raise Exception("input params error")

            log.info("Processing data")
            to_carto(
                dataframe, aux_table_name, cartodbfy=False, log_enabled=False
            )

            fields = ",".join(columns)

            query = f"""INSERT INTO {table} ({fields})
                SELECT {fields}
                FROM {aux_table_name} aux"""

            log.info("Inserting data ...")
            self.SQL.execute(query)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.info("Inserted !")

            return True
        except Exception as e:
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.error(e)
            return False

    def get_isochrones(
        self,
        geo_dataframe: Union[pd.DataFrame, GeoDataFrame],
        minutes: int,
        type: IsochroneType,
    ):
        """
        Get isochrone from each row in geodataframe.

        Parameters:

        - geo_dataframe: geodataframe.
          Must contain latitude and longitude columns
        - minutes: minutes
        - type: IsochroneType.CAR | IsochroneType.WALK

        Return:

        - DeoDataframe
        """
        aux_table_name = "_dgn_" + str(uuid.uuid1()).replace("-", "_")

        if not isinstance(geo_dataframe, GeoDataFrame):
            raise Exception("input params error")

        if "latitude" not in list(
            geo_dataframe.columns
        ) or "longitude" not in list(geo_dataframe.columns):
            raise Exception("input params error")

        log.info("Processing data")

        if "the_geom" in list(geo_dataframe.columns):
            geo_dataframe = geo_dataframe.drop(columns=["the_geom"])

        if "geometry" in list(geo_dataframe.columns):
            geo_dataframe = geo_dataframe.drop(columns=["geometry"])

        to_carto(
            geo_dataframe, aux_table_name, cartodbfy=False, log_enabled=False
        )

        query = f"""
            select * from {aux_table_name} c
            CROSS JOIN LATERAL (
            SELECT isochrone.the_geom the_geom
                FROM
                cdb_isochrone (
                ST_SetSRID(ST_MakePoint(c.longitude, c.latitude) ,4326),
                '{type}',
                ARRAY[{minutes} * 60]::integer[]
                ) isochrone
            ) iso"""

        log.info("Calculating isochrones ...")

        try:
            results = read_carto(query)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.info("Calculated !")
            return results
        except Exception as e:
            log.debug(e)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")

    def get_routes(
        self,
        dataframe: Union[pd.DataFrame, GeoDataFrame],
        route_type: RouteType,
        mode_type: RouteModeType,
    ):
        """
        Get route between coordinates

        Parameters:

        - dataframe: pandas Dataframe or geoDataframe contain latitude_from,
            longitude_from, latitude_to, longitude_to columns in 4326
        - route_type: RouteType.CAR | RouteType.WALK
        - mode_type: RouteModeType.SHORT | RouteModeType.FAST

        Return:

        - GeoDataframe copy with route, miles and seconds
        """

        aux_table_name = "_dgn_" + str(uuid.uuid1()).replace("-", "_")
        if not (
            isinstance(dataframe, GeoDataFrame)
            or isinstance(dataframe, pd.DataFrame)
        ):
            raise Exception("input params error")

        if (
            "latitude_from" not in list(dataframe.columns)
            or "longitude_from" not in list(dataframe.columns)
            or "longitude_to" not in list(dataframe.columns)
            or "latitude_to" not in list(dataframe.columns)
        ):
            raise Exception("input params error")

        log.info("Processing data")
        if "the_geom" in list(dataframe.columns):
            dataframe = dataframe.drop(columns=["the_geom"])

        if "geometry" in list(dataframe.columns):
            dataframe = dataframe.drop(columns=["geometry"])

        to_carto(dataframe, aux_table_name, cartodbfy=False, log_enabled=False)

        if route_type == RouteType.CAR:
            query = f"""
                select * from {aux_table_name} c
                CROSS JOIN LATERAL (
                    SELECT
                        route.shape as the_geom,
                        round((route.length * {miles_in_metre})::numeric, 2) AS miles,
                        route.duration AS seconds
                    FROM
                    cdb_route_point_to_point(
                        ST_SetSRID(ST_MakePoint(c.longitude_from, c.latitude_from) ,4326),
                        ST_SetSRID(ST_MakePoint(c.longitude_to, c.latitude_to) ,4326),
                        '{route_type}',
                        ARRAY ['mode_type={mode_type}'] :: text []
                    ) route
                ) subquery"""
        else:
            query = f"""
                select * from {aux_table_name} c
                CROSS JOIN LATERAL (
                    SELECT
                        route.shape as the_geom,
                        round((route.length * {miles_in_metre})::numeric, 2) AS miles,
                        route.duration AS seconds
                    FROM
                    cdb_route_point_to_point(
                        ST_SetSRID(ST_MakePoint(c.longitude_from, c.latitude_from) ,4326),
                        ST_SetSRID(ST_MakePoint(c.longitude_to, c.latitude_to) ,4326),
                        '{route_type}'
                    ) route
                ) subquery"""

        log.info("Calculating routes ...")

        try:
            results = read_carto(query)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.info("Calculated !")
            return results
        except Exception as e:
            log.debug(e)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")

    def get_buffers(
        self, geo_dataframe: Union[pd.DataFrame, GeoDataFrame], miles: int
    ):
        """
        Get buffer from each row in geodataframe. Return original geodataframe
        with a new buffer column

        Parameters:

        - geo_dataframe: geodataframe. Must contain latitude and longitude columns
        - miles: miles

        Return:

        - DeoDataframe
        """
        aux_table_name = "_dgn_" + str(uuid.uuid1()).replace("-", "_")

        if not isinstance(geo_dataframe, GeoDataFrame):
            raise Exception("input params error")

        if "latitude" not in list(
            geo_dataframe.columns
        ) or "longitude" not in list(geo_dataframe.columns):
            raise Exception("input params error")

        log.info("Processing data")
        if "the_geom" in list(geo_dataframe.columns):
            geo_dataframe = geo_dataframe.drop(columns=["the_geom"])

        if "geometry" in list(geo_dataframe.columns):
            geo_dataframe = geo_dataframe.drop(columns=["geometry"])

        to_carto(
            geo_dataframe, aux_table_name, cartodbfy=False, log_enabled=False
        )

        query = f"""
            select * from {aux_table_name} c
            CROSS JOIN LATERAL (
                SELECT
                ST_BUFFER(
                    ST_SetSRID(ST_MakePoint(c.longitude, c.latitude),4326)::geography, {miles} * {meters_in_mile}
                )::geometry AS the_geom
            ) buff"""
        log.info("Calculating buffers ...")

        try:
            results = read_carto(query)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")
            log.info("Calculated !")
            return results
        except Exception as e:
            log.debug(e)
            self.SQL.execute(f"DROP TABLE IF EXISTS {aux_table_name}")

    def enrichment_variables(
        self,
        geo_dataframe: Union[pd.DataFrame, GeoDataFrame],
        variables_name: list,
        type_geometry: GeometryType = GeometryType.POLYGONS,
    ):
        """
        Enrich polygon only with variables needed

        Parameters:
        - geo_dataframe: GeoDataFrame to enrich
        - variables_name: array name of the variable's dataset

        Return:

        - GeoDataFrame
        """

        if not isinstance(geo_dataframe, GeoDataFrame):
            return None
        try:
            enrichment = Enrichment()
            variables = []
            for value in variables_name:
                variables.append(Variable.get(value))

            if type_geometry in [GeometryType.POLYGONS]:
                enriched_data = enrichment.enrich_polygons(
                    geo_dataframe, variables=variables, geom_col="the_geom"
                )
            elif type_geometry in [GeometryType.POINTS]:
                enriched_data = enrichment.enrich_points(
                    geo_dataframe, variables=variables, geom_col="the_geom"
                )
            else:
                enriched_data = geo_dataframe.copy()
            return enriched_data
        except Exception as e:
            log.error(e)
            return None

    def enrichment_dataset(
        self,
        geo_dataframe: Union[pd.DataFrame, GeoDataFrame],
        dataset_name: str,
    ):
        """
        Enrich polygon with all variables from dataset

        Parameters:
        - geo_dataframe: GeoDataFrame to enrich
        - dataset_name: name of the dataset

        Return:

        - GeoDataFrame
        """

        if not isinstance(geo_dataframe, GeoDataFrame):
            return None
        try:
            enrichment = Enrichment()
            enriched_data = enrichment.enrich_polygons(
                geo_dataframe,
                variables=Dataset.get(dataset_name).variables,
                geom_col="the_geom",
            )
            return enriched_data
        except Exception as e:
            log.error(e)
            return None

    def get_vars(self):

        PATH = pathlib.Path(PROJECT_DIR)
        with open(
            PATH.joinpath("src/enrichment", "enrichment_dbs.json")
        ) as jsonfile:
            enrich_dbs = json.load(jsonfile)

        with open(
            PATH.joinpath("src/constants", "models_variables.json")
        ) as jsonfile:
            models_variables = json.load(jsonfile)

        d = {**enrich_dbs, **models_variables}

        return d

    def make_pct_cols(self, gdf: GeoDataFrame, vars) -> GeoDataFrame:
        """Convert raw enrichment columns to percentage variables
        Parameters:
            gdf (GeoDataFrame): Carto-format geopandas dataframe with polygon geometry
            vars (dict): Dictionary in the format of str:tuple, where the format is
                'new_column':('numerator_column', 'divisor_column'). The variable configuration
                can be edited in enrichment_vars.py
        Returns:
            gdf (GeoDataFrame): Original dataframe with percentage columns calculated
        """

        # Calculate percentage columns
        for new_col, [num, div] in vars.items():
            gdf[new_col] = gdf[num].values / gdf[div].values

        return gdf

    def make_buffer(
        self,
        carto_tbl: str,
        buffers: Union[dict, int, float],
        cat_col=None,
        creds: Credentials = None,
        units="mi",
        calc_area=True,
        read_table=True,
        output_name: str = None,
    ) -> GeoDataFrame:
        """Create radius buffer around stores to create straight-line trade areas

        Parameters:
            carto_tbl (str): Carto table with locations to create trade area buffer
            buffers (dict, int): Radius buffer used to create trade area.  Provide a dictionary
                with key:value pairs for varying radii by category, or provide an int
                or float object for consistent radii
            cat_col (str): Category column in carto_tbl to create varying buffers
            creds (Credentials): Carto credentials object
            units (str): Set unit for buffer column signifier
            calc_area (bool): If true, adds an area column in our dataset with the catchment area
            read_table (bool): If true, reads table into local environment for enrichment.
                If false, creates a carto table in the provided account with the name output_name
            output_name (str): Name of table to be created if read_table is false.
                Defaults to carto_tbl + '_catchment'

        Returns:
            carto_df (GeoDataFrame): GeoDataFrame with trade area buffer
        """

        if not isinstance(buffers, (dict, int, float)):
            raise TypeError(
                "buffer input must be either a dictionary, int, or float"
            )

        if isinstance(buffers, (int, float)):
            if calc_area:
                sql = """
                        WITH temp AS (
                            SELECT
                                cartodb_id
                                , ST_SetSRID(ST_Buffer(the_geom::geography,
                                    {} * {})::geometry,4326) the_geom
                            FROM
                                {})
                        SELECT
                            t.*
                            , ST_AREA(t.the_geom::geography) * 0.00000038610 AS area
                        FROM
                            temp t """.format(
                    meters_in_mile, buffers, carto_tbl
                )

            else:
                sql = """
                        SELECT
                            cartodb_id
                            , ST_SetSRID(ST_Buffer(the_geom::geography,
                                {} * {})::geometry,4326) the_geom
                        FROM
                            {}""".format(
                    meters_in_mile, buffers, carto_tbl
                )

            self.buf_col = str(buffers) + units

        if isinstance(buffers, dict):
            # Create dynamic case when clause based on dictionary
            cw = ["CASE"]
            for i in buffers.items():
                s = "WHEN {} = '{}' THEN {}".format(cat_col, i[0], i[1])
                cw.append(s)
            cw.append("END")
            cw = " ".join(cw)

            # Create dynamic case when for buffer column
            cw_b = ["CASE"]
            for i in buffers.items():
                s = "WHEN {} = '{}' THEN '{}_{}'".format(
                    cat_col, i[0], i[1], units
                )
                cw_b.append(s)
            cw_b.append("END")
            cw_b = " ".join(cw_b)

            # Insert dynamic case when directly into SQL statement
            if calc_area:
                sql = """
                        WITH temp AS (
                            SELECT
                                cartodb_id
                                , ST_SetSRID(ST_Buffer(the_geom::geography,
                                    {} * {})::geometry,4326) the_geom
                                , {} as buffer
                            FROM
                                {})
                        SELECT
                            t.*
                            , ST_AREA(t.the_geom::geography) * 0.00000038610 AS area
                        FROM
                            temp t """.format(
                    meters_in_mile, cw, cw_b, carto_tbl
                )

            else:
                # Insert dynamic case when directly into SQL statement
                sql = """
                        SELECT
                            cartodb_id
                            , ST_SetSRID(ST_Buffer(the_geom::geography,
                                {} * {})::geometry,4326) the_geom
                            , {} as buffer
                        FROM
                            {}""".format(
                    meters_in_mile, cw, cw_b, carto_tbl
                )

            self.buf_col = None

        if read_table:
            carto_df = read_carto(sql, creds)

        else:
            if output_name is None:
                output_name = f"dgn_{self._get_uuid()}_catchment"
            carto_df = output_name
            create_table_from_query(
                sql, output_name, credentials=creds, if_exists="replace"
            )

        return carto_df

    def std_var_enrich(
        self,
        df: GeoDataFrame,
        dataset: str,
        incl_perc: bool = False,
        incl_catchment: bool = False,
        do_credentials: Credentials = None,
    ) -> GeoDataFrame:
        """Enrich trade area polygons with standard whitespace variables
        Paramters:
            df (GeoDataFrame): Carto-format geopandas dataframe with polygon geometry,
                the dataframe must have a column named "the_geom" with a shapely
                polygon in each row. This is the standard format for a trade area
                calculation from the Carto isolines data service
            do_credentials (Credntials): Carto credntials object with a username and
                API key that allows Data Observatory v2 access
            incl_prec (bool): Whether or not to include additional columns with
                calculated percentages

        Returns:
            enrich (GeoDataFrame): Original dataframe with enriched demographic
                variables appended
        """
        # Validate user inputs
        if not isinstance(df, GeoDataFrame) and "the_geom" in df.columns:
            raise TypeError(
                "First input must be a GeoDataFrame with column named the_geom"
            )

        if not isinstance(
            df.loc[df.index[0], "the_geom"], (MultiPolygon, Polygon)
        ):
            raise TypeError("Dataframe geometry must be a polygon")

        if not isinstance(dataset, str):
            raise ValueError(
                "Must supply a string input dataset to guide enrichment"
            )

        if do_credentials is None:
            do_credentials = self.creds

        d = self.vars[dataset]

        # Retrieve standard American Community Survey variables
        enrichment = Enrichment(credentials=do_credentials)

        print(
            "Downloading standard variables from Carto...", end="", flush=True
        )
        enrich = enrichment.enrich_polygons(
            dataframe=df, variables=d["VAR_IDS"]
        )
        print("complete")

        # Aggregate population and income brackets
        print(
            "Aggregating data and calculating percentages...",
            end="",
            flush=True,
        )
        for new_col, old_col_list in d["VARS_TO_CONSOLIDATE"].items():
            enrich[new_col] = enrich[old_col_list].values.sum(axis=1)
            enrich.drop(old_col_list, axis=1, inplace=True)

        # Calculate percentage columns
        if incl_perc:
            enrich = self.make_pct_cols(enrich, d["PERC_COLS_TO_CALC"])

        # Append catchment information to column names
        if incl_catchment and self.buf_col is not None:
            col_list = [col for col in enrich if col not in df]
            col_rep = [col + "_" + str(self.buf_col) for col in col_list]
            enrich.columns = list(df) + col_rep

        print("complete")

        return enrich

    def poi_density(
        self,
        carto_tbl: str,
        buffers: Union[dict, int, float],
        cat_col=None,
        poi_tbl: str = "competitor_locations",
        poi_cat: str = None,
        priority_pois: str = None,
        incl_catchment: bool = False,
        creds: Credentials = None,
        units="mi",
    ) -> GeoDataFrame:
        """Finds locations within catchment areas

        Parameters:
            carto_tbl (str): Carto table with locations to create trade area buffer
            buffers (dict, int): Radius buffer used to create trade area.  Provide a dictionary
                with key:value pairs for varying radii by category, or provide an int
                or float object for consistent radii
            cat_col (str): Category column in carto_tbl to create varying buffers
            poi_tbl (str): Carto table with point of interest locations
            poi_cat (str): Category column in poi_tbl to use to aggregate poi counts
            priority_pois (str): Carto lookup table with priority brands
            creds (Credentials): Carto credentials object, if null defaults to instantiated default credentials
            units (str): Set unit for buffer column signifier

        Returns:
            carto_df (GeoDataFrame): Geodataframe with trade area buffer
        """

        if not isinstance(buffers, (dict, int, float)):
            raise TypeError(
                "buffer input must be either a dictionary, int, or float"
            )

        if creds is None:
            creds = self.creds

        carto_tbl_catchment = self.make_buffer(
            carto_tbl=carto_tbl,
            buffers=buffers,
            cat_col=cat_col,
            creds=creds,
            units=units,
            calc_area=True,
            read_table=False,
        )

        if priority_pois is not None:
            sql = f"""
                    WITH buffers as (
                        SELECT
                            *
                        FROM
                            {carto_tbl_catchment}),

                    filtered_pois as(
                        SELECT
                            p.*
                        FROM
                            {poi_tbl} p
                        INNER JOIN
                            {priority_pois} pp
                        ON
                            p.{poi_cat} = pp.{poi_cat}
                    ),

                    locations as(
                        SELECT
                            b.*
                            , p.cartodb_id comp_id
                            , p.{poi_cat}
                        FROM
                            buffers b
                        LEFT JOIN
                            filtered_pois p
                        ON
                            ST_INTERSECTS(p.the_geom, b.the_geom))

                    SELECT
                        l.cartodb_id
                        , l.{poi_cat}
                        , l.area
                        , COUNT(l.comp_id) as count
                    FROM
                        locations l
                    GROUP BY
                        1,2,3"""

        else:
            sql = f"""
                        WITH buffers as (
                            SELECT
                                *
                            FROM
                                {carto_tbl_catchment}),

                        locations as(
                            SELECT
                                b.*
                                , p.cartodb_id comp_id
                                , p.{poi_cat}
                            FROM
                                buffers b
                            LEFT JOIN
                                {poi_tbl} p
                            ON
                                ST_INTERSECTS(p.the_geom, b.the_geom))

                        SELECT
                            l.cartodb_id
                            , l.{poi_cat}
                            , l.area
                            , COUNT(l.comp_id) as count
                        FROM
                            locations l
                        GROUP BY
                            1,2,3"""

        carto_df = read_carto(sql, credentials=creds)

        # Drop na values from poi_cat
        na_n = carto_df[poi_cat].isna().sum()
        n = carto_df.shape[0]

        print(
            f"{na_n} null rows found in {poi_cat} column, representing {np.round((na_n/n)*100, 2)}% of total rows...",
            end="",
            flush=True,
        )
        carto_df.dropna(subset=[poi_cat], inplace=True)
        print(f"dropped {na_n} rows")

        # Create density column
        carto_df["density"] = carto_df["count"] / carto_df["area"]

        # Pivot from long-to-wide
        carto_df_c = carto_df.pivot(
            index=["cartodb_id", "area"], columns=poi_cat, values="count"
        )

        carto_df_d = carto_df.pivot(
            index=["cartodb_id", "area"], columns=poi_cat, values="density"
        )

        if priority_pois is not None:
            carto_df_c["priority_total"] = carto_df_c.sum(axis=1)
        else:
            carto_df_c["total"] = carto_df_c.sum(axis=1)

        # Rename columns
        carto_df_c.columns = [col + "_count" for col in carto_df_c]
        carto_df_d.columns = [col + "_density" for col in carto_df_d]

        # Merge datasets
        carto_df = carto_df_c.merge(
            carto_df_d, left_index=True, right_index=True
        )

        # Reset index and fillna
        carto_df.fillna(0, inplace=True)
        carto_df.reset_index(inplace=True)

        # Add total density column
        if priority_pois is not None:
            carto_df["priority_total_density"] = (
                carto_df["priority_total_count"] / carto_df["area"]
            )
        else:
            carto_df["total_density"] = (
                carto_df["total_count"] / carto_df["area"]
            )

        # Append catchment information to column names
        if incl_catchment and self.buf_col is not None:
            key_cols = ["cartodb_id", "area"]
            col_list = [col for col in carto_df if col not in key_cols]
            col_rep = [col + "_" + str(self.buf_col) for col in col_list]
            carto_df.columns = key_cols + col_rep

        carto_df.columns = carto_df.columns.str.lower()

        # as we have got a table with an unique name, we need to delete it.
        self.SQL.execute(f"DROP TABLE IF EXISTS {carto_tbl_catchment}")
        return carto_df

    def find_nearest(
        self,
        store_tbl: str,
        poi_tbl: str,
        self_flag: str = False,
        num_nearest: int = 1,
        cat_col: str = None,
        cat_val: str = None,
        creds: Credentials = None,
    ):
        """Finds nearest poi to locations

        Parameters:
            store_tbl (str): Carto table with store locations
            poi_tbl (str): Carto table with point of interest locations
            num_nearest (int): Number of nearest locations to find
            cat_col (str): Category column in poi_tbl to use to filter
                to specific brands or categories (e.g. 'category')
            cat_val (str): Category value in poi_tbl to filter to specific brands or categories (e.g. 'McDonalds')
            max_dist (float): Maximum distance to use when crafting variables
            creds (Credentials): Carto credentials object, if null defaults to instantiated default credentials

        Returns:
            carto_df (GeoDataFrame): Geodataframe with nearest poi information appended
        """

        if creds is None:
            creds = self.creds

        if cat_col is not None and cat_val is not None:
            where_clause = (
                f"""WHERE UPPER({cat_col}) like UPPER('{cat_val}')"""
            )
            cat_clause = f""", n.{cat_col}"""
            cat_clause_2 = f""", p.{cat_col}"""
        else:
            where_clause = ""
            cat_clause = ""
            cat_clause_2 = ""

        if store_tbl != poi_tbl:
            sql = f"""
                    SELECT
                        s.cartodb_id
                        , n.cartodb_id as nearest_id
                        , n.ranked
                        {cat_clause}
                        , n.closest_dist_miles
                    FROM
                        {store_tbl} s
                    CROSS JOIN LATERAL (
                        SELECT
                            cartodb_id
                            , the_geom
                            {cat_clause_2}
                            , st_distance(geography(s.the_geom), geography(p.the_geom))/{meters_in_mile}
                                as closest_dist_miles
                            , row_number() over(order by s.the_geom_webmercator <-> the_geom_webmercator) AS ranked
                        FROM
                            {poi_tbl} p
                        {where_clause}
                        LIMIT {num_nearest}
                    ) n
                    """
            col_name = f"avg_dist_{num_nearest}_poi"
            if cat_val is not None:
                col_name = col_name + f"_{cat_val}"

            if self_flag:
                col_name = f"avg_dist_{num_nearest}_store"

        else:
            sql = f"""
                    SELECT
                        s.cartodb_id
                        , n.cartodb_id as nearest_id
                        , n.ranked
                        , n.closest_dist_miles
                    FROM
                        {store_tbl} s
                    CROSS JOIN LATERAL (
                        SELECT
                            cartodb_id
                            , the_geom
                            , st_distance(geography(s.the_geom), geography(p.the_geom))/{meters_in_mile}
                                as closest_dist_miles
                            , row_number() over(order by s.the_geom_webmercator <-> the_geom_webmercator) AS ranked
                        FROM
                            {store_tbl} p
                        WHERE
                            s.cartodb_id != p.cartodb_id
                        LIMIT {num_nearest}
                    ) n
                    """
            col_name = f"avg_dist_{num_nearest}_store"

        carto_df = read_carto(sql, credentials=creds)

        carto_df = carto_df.groupby("cartodb_id")["closest_dist_miles"].mean()
        carto_df.name = col_name
        carto_df = carto_df.reset_index()
        carto_df.columns = carto_df.columns.str.lower()

        return carto_df

    def standardize_cols(self, df: pd.DataFrame, db_id: str) -> pd.DataFrame:
        """Standardizes columns leveraging instructions in enrichment_dbs

        Args:
            df (pd.DataFrame): Dataframe with columns to enrich
            db_id (str): ID of the database in enrichment_dbs.json
        """
        d = self.get_vars()
        d = d[db_id]

        df = self.make_pct_cols(df, d["PERC_COLS_TO_CALC"])

        return df


if __name__ == "__main__":
    pass
    # c = CartoFramesModel().batch('SELECT * FROM __seeds', verbose=True)