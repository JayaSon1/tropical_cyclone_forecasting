import pandas as pd

from config import settings
from src.features.time_utils import value_at

# Label scope (AUDIT.md §6 decisions 1, 4, 5): P(RI | the storm stays a tropical cyclone
# over water for the next 24 h). The exclusions use future information (status and
# landfall at t+24), so metrics are within this scope, not operational.


# For each row, is there an L record in [t, t+24 h] in the same storm?
# L records are often off-synoptic, so they are taken from the raw records.
def landfall_in_next_24h(df, raw):
    landfalls = raw.loc[raw["record_id"] == "L", ["storm_id", "datetime"]]
    landfalls = landfalls.rename(columns={"datetime": "landfall_time"})
    pairs = df[["storm_id", "datetime"]].reset_index().merge(landfalls, on="storm_id")
    in_window = (pairs["landfall_time"] >= pairs["datetime"]) & (
        pairs["landfall_time"] <= pairs["datetime"] + pd.Timedelta(hours=24)
    )
    return df.index.isin(pairs.loc[in_window, "index"])


# Keep only synoptic times (00/06/12/18 UTC), so t+24 and the lags are true time offsets
def synoptic_rows(raw):
    is_synoptic = (raw["datetime"].dt.hour % 6 == 0) & (raw["datetime"].dt.minute == 0)
    df = raw[is_synoptic].reset_index(drop=True)

    duplicated = df.duplicated(["storm_id", "datetime"], keep=False)
    if duplicated.any():
        storms = sorted(df.loc[duplicated, "storm_id"].unique())
        raise ValueError(f"duplicate synoptic timestamps in storms: {storms}")
    return df


def ri_labels(raw):

    # Landfall markers come from the raw L records, before off-synoptic rows are dropped
    df = synoptic_rows(raw)
    df["is_landfall"] = df["record_id"] == "L"
    df["landfall_next_24h"] = landfall_in_next_24h(df, raw)

    # Wind speed and status at exactly t+24 h within the same storm (NaN if no record)
    df["vmax_24h"] = pd.to_numeric(value_at(df, 24, "vmax"))
    df["status_24h"] = value_at(df, 24, "status")

    # Calculate the change in wind speed over 24 hours
    df["delta_vmax_24h"] = df["vmax_24h"] - df["vmax"]

    # Calculate which entries need to be excluded
    # Future observations must exist
    future_bool = df["vmax_24h"].notna()

    # Current observations must exist
    current_bool = df["vmax"].notna()

    # Tropical / subtropical at t and at t+24
    tropical_bool = df["status"].isin(settings.TROPICAL_STATUSES) & df["status_24h"].isin(settings.TROPICAL_STATUSES)

    landfall_bool = ~df["landfall_next_24h"]

    # Extract valid observations
    valid_observations = future_bool & current_bool & landfall_bool & tropical_bool

    # If the change is at least 30 kt in 24 h (Kaplan and DeMaria, 2003) -> Rapid Intensification
    df["RI"] = pd.NA
    df.loc[valid_observations, "RI"] = (
        df.loc[valid_observations, "delta_vmax_24h"] >= 30
    ).astype(int)

    return df
