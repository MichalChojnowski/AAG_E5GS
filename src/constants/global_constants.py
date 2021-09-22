#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains all of the global constants for other modules in the repo.

"""
import os

# Define standard paths
if os.name == "posix":
    PROJECT_DIR = (
        "/"
        + os.path.join(*str(os.path.realpath(__file__)).split("/")[:-3])
        + "/"
    )
else:  # assume windows
    PROJECT_DIR = ("/").join(
        str(os.path.realpath(__file__)).split("\\")[:-2]
    ) + "/"


# Path to enrichment templates
ENRICHMENT_PATH = PROJECT_DIR + "src/enrichment/"

# Path to documents
MLFLOW_PATH = PROJECT_DIR + "src/docs/mlruns/"

# Geospatial constants
meters_in_mile = 1609.34
miles_in_metre = 0.000621371

# Mileage buffers by urbanicity
urb_buffers = {"Urban": 1, "Sub-urban": 3, "Rural": 5, "Other": 5}

# Model features
auv_features = {
    "features": [
        "cartodb_id",
        "the_geom",
        "store_id",
        "latitude",
        "longitude",
        "area",
        "buffer",
    ],
    "categorical": [],
    "target": "target_var",
}
