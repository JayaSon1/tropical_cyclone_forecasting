import pandas as pd


# Value of a column at exactly t + hours within the same storm, aligned to df's rows.
# NaN where the storm has no record at that time. Requires unique (storm_id, datetime).
def value_at(df, hours, col):
    shifted = df[["storm_id", "datetime", col]].copy()
    shifted["datetime"] = shifted["datetime"] - pd.Timedelta(hours=hours)
    matched = df[["storm_id", "datetime"]].merge(
        shifted, on=["storm_id", "datetime"], how="left", validate="one_to_one"
    )
    return pd.Series(matched[col].to_numpy(), index=df.index, name=col)
