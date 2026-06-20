"""
data_cleaning.py
Functions for loading, inspecting, cleaning, and summarizing the
HydroSense-Kenya raw datasets.
"""

import numpy as np
import pandas as pd


def load_datasets(weather_path, soil_path, params_path):
    """Load the three raw csv files, treating 'NA' and '' as missing."""
    weather = pd.read_csv(weather_path, na_values=["NA", ""])
    soil = pd.read_csv(soil_path, na_values=["NA", ""])
    params = pd.read_csv(params_path, na_values=["NA", ""])
    weather["date"] = pd.to_datetime(weather["date"])
    soil["timestamp"] = pd.to_datetime(soil["timestamp"])
    return weather, soil, params


def report_missing(df):
    """Return a Series with the count of missing values per column."""
    return df.isna().sum()


def flag_outliers_iqr(series, k=1.5):
    """
    Identify outliers in a numeric series using the IQR rule.
    Returns a boolean mask the same length as `series`.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def clean_weather(weather, temp_max_plausible=40.0):
    """
    Clean the weather dataframe:
      - interpolate missing rainfall and humidity values (time-ordered)
      - flag / cap implausible temperature spikes (e.g. sensor fault > 40C)
    Returns a new cleaned dataframe; original is left untouched.
    """
    df = weather.copy().sort_values("date").reset_index(drop=True)

    # Implausible temperature sensor fault -> treat as missing then interpolate
    df.loc[df["temperature_c"] > temp_max_plausible, "temperature_c"] = np.nan

    for col in ["rainfall_mm", "humidity_pct", "temperature_c"]:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    return df


def clean_soil(soil, tank_capacity_liters=6000):
    """
    Clean the soil sensor dataframe:
      - interpolate missing soil_moisture_pct per zone
      - flag tank_level readings above a plausible physical capacity as faults
        and replace with the per-zone median
      - flag pump_flow_lpm == 0 with sensor_status == 'CHECK' as a fault and
        replace with the per-zone median flow
    """
    df = soil.copy().sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    # Tank level sensor fault (value far above plausible capacity)
    fault_mask = df["tank_level_liters"] > tank_capacity_liters
    for zone in df["zone_id"].unique():
        zmask = df["zone_id"] == zone
        median_val = df.loc[zmask & ~fault_mask, "tank_level_liters"].median()
        df.loc[zmask & fault_mask, "tank_level_liters"] = median_val

    # Pump flow fault flagged by sensor_status == CHECK
    flow_fault = df["sensor_status"] == "CHECK"
    for zone in df["zone_id"].unique():
        zmask = df["zone_id"] == zone
        median_flow = df.loc[zmask & ~flow_fault, "pump_flow_lpm"].median()
        df.loc[zmask & flow_fault, "pump_flow_lpm"] = median_flow

    # Missing soil moisture: interpolate per zone
    df["soil_moisture_pct"] = (
        df.groupby("zone_id")["soil_moisture_pct"]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    return df


def merge_clean_dataset(weather_clean, soil_clean, params):
    """Merge cleaned weather and soil data with crop-zone parameters into one tidy table."""
    soil_clean = soil_clean.copy()
    soil_clean["date"] = soil_clean["timestamp"].dt.normalize()
    merged = soil_clean.merge(weather_clean, on="date", how="left")
    merged = merged.merge(params, on="zone_id", how="left")
    return merged


def summary_statistics(df, columns):
    """Return descriptive statistics (count, mean, std, min, max) for given columns."""
    return df[columns].describe().T