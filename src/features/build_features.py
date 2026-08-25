import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# BASELINE FEATURES
# vmax
# mslp
# delta_vmax_6h: Wind speed intensity change over the last 6h
# delta_vmax_12h: Wind speed intensity change over the last 12h
# latitude, longtitude
# translation speed: Wind speed (km/h) between consecutive points
# storm_age_hours: Hours since the storm was first recorded
# month
# day_of_year

# Circumference of storm
def point_distance_km(latitude_1, longitude_1, latitude_2, longitude_2):
    # The Earth's radius (km)
    earth_radius = 6371.0
    
    # Calculate change in latitude
    diff_lat = radians(latitude_2 - latitude_1)
    
    # Calculate change in longitude
    diff_long = radians(longitude_2 - longitude_1)
    
    # Haversine formula: Calculates how far apart 2 points are along the Earth's curved surface
    angular_separation = (sin(diff_lat / 2)**2 + cos(radians(latitude_1)) * cos(radians(latitude_2)) * sin(diff_long / 2)**2)
    
    # Calculate distance (km)
    distance = 2 * earth_radius * atan2(sqrt(angular_separation), sqrt(1 - angular_separation))
    
    return distance


def extract_features(df):
    df = df.sort_values(["storm_id", "datetime"]).copy()

    # Persistence features
    df["vmax_prev_6h"]  = df.groupby("storm_id")["vmax"].shift(1)
    df["vmax_prev_12h"] = df.groupby("storm_id")["vmax"].shift(2)

    df["delta_vmax_6h"]  = df["vmax"] - df["vmax_prev_6h"]
    df["delta_vmax_12h"] = df["vmax"] - df["vmax_prev_12h"]

    # Translation speed (km/h) 
    df["lat_prev"] = df.groupby("storm_id")["latitude"].shift(1)
    df["lon_prev"] = df.groupby("storm_id")["longitude"].shift(1)

    df["translation_speed"] = df.apply(
        lambda r: point_distance_km(r["lat_prev"], r["lon_prev"], r["latitude"], r["longitude"]) / 6.0
        if pd.notna(r["lat_prev"]) else np.nan,
        axis=1
    )

    # Storm age 
    df["storm_age_hours"] = (
        df.groupby("storm_id")["datetime"]
          .transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
    )

    # Seasonality 
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear

    return df