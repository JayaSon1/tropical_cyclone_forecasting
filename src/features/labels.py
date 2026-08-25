import pandas as pd

# For each row, look ahead up to 4 steps inside the same storm
def landfall_in_next_24h(group):
    # Need to remove the current observation too
    current = group["is_landfall"]
    
    # rolling window of the next 4 rows 
    future_landfall = (
        group["is_landfall"]
        .shift(-1).fillna(False)
        | group["is_landfall"].shift(-2).fillna(False)
        | group["is_landfall"].shift(-3).fillna(False)
        | group["is_landfall"].shift(-4).fillna(False)
    )
    return future_landfall | current
    
def ri_labels(df):
    
    # Make a copy of the df
    df = df.copy()
    
    # Extract containing the wind speed (vmax) 24 hours into the future for each storm
    # Looks 4 rows forwards: .shift(-4)
    # For every observation, find the vmax value 4 observations later (4 × 6 h: 24 h) within the same storm
    df["vmax_24h"] = df.groupby("storm_id")["vmax"].shift(-4)   
    
    # Calculate the change in wind speed over 24 hours
    df["delta_vmax_24h"] = df["vmax_24h"] - df["vmax"]
    
    # Mark every landfall record
    df["is_landfall"] = (df["record_id"] == "L")
    
    df["landfall_next_24h"] = (
        df.groupby("storm_id", group_keys=False)
            .apply(landfall_in_next_24h)
    )
        
    # Calculate which entries need to be excluded
    # Future observations must exist
    future_bool = df["vmax_24h"].notna()
    
    # Current observations must exist
    current_bool = df["vmax"].notna()
    
    # Keep only tropical / subtropical statuses
    tropical_bool = df["status"].isin(["TD", "TS", "HU", "SD", "SS"])
    
    landfall_bool = ~df["landfall_next_24h"]
    
    # Extract valid observations
    valid_observations = future_bool & current_bool & landfall_bool & tropical_bool
    
    # If the change is more than 30 (Maria and Kaplan, 2003) -> Rapid Intensification
    df["RI"] = pd.NA
    df.loc[valid_observations, "RI"] = (
        df.loc[valid_observations, "delta_vmax_24h"] >= 30
    ).astype(int)
    
    # print("df head: ", df.head())
    return df