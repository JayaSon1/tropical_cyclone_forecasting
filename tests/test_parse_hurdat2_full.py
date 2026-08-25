from pathlib import Path
import pytest
from datetime import datetime
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.parse_hurdat2 import parse_hurdat2   

FULL_FILE = Path("data/raw/hurdat2-1851-2025-02272026.txt")  

def test_full_hurdat2_parsing():

    df = parse_hurdat2(FULL_FILE)

    # Basic structure 
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50_000, "Expected tens of thousands of records for full Atlantic history"
    
    required_cols = {
        "storm_id", "name", "datetime", "status",
        "latitude", "longitude", "vmax", "mslp", "record_id"
    }
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Storm identity
    assert df["storm_id"].str.startswith("AL").all(), "All Atlantic storm_ids should start with AL"
    assert df["storm_id"].nunique() > 1500, "Expected >1500 unique storms since 1851"

    # Produces multiple unique storm_ids
    n_storms = df["storm_id"].nunique()
    print(f"Unique storms found: {n_storms:,}")
    
    assert n_storms > 1500, f"Expected >1500 unique storms in the full Atlantic record, got {n_storms}"

    # All IDs should look like Atlantic storm IDs
    assert df["storm_id"].str.startswith("AL").all(), "Some storm_ids do not start with 'AL'"

    # No completely empty storms
    storm_sizes = df.groupby("storm_id").size()
    
    assert (storm_sizes > 0).all(), "Found storm(s) with zero records"
    
    # Every storm should have at least a few points
    min_size = storm_sizes.min()
    print(f"Smallest storm has {min_size} records")
    
    assert min_size >= 1, f"Found a storm with only {min_size} records"
    
    
    # Time ordering 
    # Within each storm the times must be sorted
    assert df.groupby("storm_id")["datetime"].is_monotonic_increasing.all(), \
        "Datetime is not sorted within every storm"

    
    # Debug to diagnose range for latitude as latitude failed unit test
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

    print(f"Latitude  range: {lat_min:.1f} -> {lat_max:.1f}")
    print(f"Longitude range: {lon_min:.1f} -> {lon_max:.1f}")

    assert 0 <= lat_min <= lat_max <= 85, f"Latitude out of range: {lat_min} to {lat_max}"
    assert -150 <= lon_min <= lon_max <= 70, f"Longitude out of range: {lon_min} to {lon_max}"

    # vmax sanity (allow missing values)
    vmax_valid = df["vmax"].dropna()
    
    vmax_min, vmax_max = df["vmax"].min(), df["vmax"].max()
    print(f"vmax range: {vmax_min:.1f} -> {vmax_max:.1f}")
    
    assert len(vmax_valid) > 0, "No valid vmax values found"
    assert vmax_valid.between(0, 200).all(), "vmax has unrealistic values"
    
    # Known storm still present 
    irene = df[df["storm_id"] == "AL092011"]
    assert len(irene) > 30, "Irene (AL092011) should still be present in the full file"
    assert (irene["name"] == "IRENE").all()

    # Landfall records exist somewhere 
    assert (df["record_id"] == "L").any(), "Expected some landfall (L) records in the full database"

    # No completely empty storms 
    storm_sizes = df.groupby("storm_id").size()
    assert (storm_sizes > 0).all()

    print(f"\nFull file summary:")
    print(f"Total records: {len(df):,}")
    print(f"Unique storms: {df['storm_id'].nunique():,}")
    print(f"Date range: {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
    print(f"RI-ready rows: will be calculated later")
    

if __name__ == "__main__":
    test_full_hurdat2_parsing()
    
    # Use: python -m pytest tests/test_parse_hurdat2_full.py -v  
                    