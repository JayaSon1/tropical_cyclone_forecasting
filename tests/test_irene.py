from pathlib import Path
import pytest
from datetime import datetime
import pandas as pd

from data.parse_hurdat2 import parse_hurdat2

TEST_FILE = Path("data/raw/test_irene_2011.txt")

def test_irene_basic_parsing():
    df = parse_hurdat2(TEST_FILE)

    # Basic shape
    assert isinstance(df, pd.DataFrame), "df is not a DataFrame"
    assert len(df) > 0, "df is empty"
    assert set(["storm_id", "name", "datetime", "status", "latitude", "longitude", "vmax", "mslp"]).issubset(df.columns), "Not all columns are in df"

    # Storm identity
    assert (df["storm_id"] == "AL092011").all(), "Storm ID is incorrect"
    assert (df["name"] == "IRENE").all(), "Storm name is incorrect"

    # Sorted by time
    assert df["datetime"].is_monotonic_increasing, "Date is not monotonically increasing"

    # First record (known values from HURDAT2 documentation)
    first = df.iloc[0]
    assert first["datetime"] == datetime(2011, 8, 21, 0, 0), " Datetime is wrong"
    assert first["status"] == "TS", "Status is wrong"
    assert abs(first["latitude"] - 15.0) < 0.01, "Latitude is wrong"
    assert abs(first["longitude"] - (-59.0)) < 0.01, "Longitude is wrong"
    assert first["vmax"] == 45, "vmax is wrong"
    assert first["mslp"] == 1006, "mslp is wrong"

    # Landfall record exists
    assert (df["record_id"] == "L").any(), "Expected at least one landfall (L) record for Irene"

    # Latitude / longitude signs
    assert (df["latitude"] > 0).all(), "Latitude sign is wrong"          # Northern Hemisphere
    assert (df["longitude"] < 0).all(), "Longitude sign is wrong"            # Western Hemisphere
    

if __name__ == "__main__":
    
    # Use: python -m pytest tests/test_irene.py -v
    
    # Parse file
    test_irene_basic_parsing
    