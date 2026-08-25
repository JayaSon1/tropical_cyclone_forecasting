from pathlib import Path

from data.parse_hurdat2 import parse_hurdat2
from src.features.labels import ri_labels

if __name__ == "__main__":
    
    # Parse file
    print("Parsing HURCAT2 File")
    df = parse_hurdat2("data/raw/hurdat2-1851-2025-02272026.txt")
    
    # Save df
    print("Saving HURCAT2 File")
    df.to_parquet("data/processed/hurdat2_raw.parquet", index=False)
    
    # Extract RI labelled observations
    df = ri_labels(df)
    
    # Keep only rows with a valid label
    ri_observations = df.dropna(subset=["RI"]).copy()
    ri_observations["RI"] = ri_observations["RI"].astype(int)
    
    print("Total valid observations:", len(ri_observations))
    print("\nClass balance:")
    print(ri_observations["RI"].value_counts())
    print(ri_observations["RI"].value_counts(normalize=True).round(3))
    
    # RI label sanity checks
    # How many RI events did we find?
    print("Number of RI cases:", ri_observations["RI"].sum())

    # Look at a few known RI storms (examples)
    print(ri_observations[ri_observations["RI"] == 1][["storm_id", "name", "datetime", "vmax", "vmax_24h", "delta_vmax_24h"]].head(10))

    # Confirm no landfall records slipped through
    print("Landfalls in labelled set:", (ri_observations["record_id"] == "L").sum())  # should be 0 or very low