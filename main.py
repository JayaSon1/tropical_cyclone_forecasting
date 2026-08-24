from pathlib import Path

from data.parse_hurdat2 import parse_hurdat2

if __name__ == "__main__":
    
    # Parse file
    print("Parsing HURCAT2 File")
    df = parse_hurdat2("data/raw/hurdat2-1851-2025-02272026.txt")
    
    # Save df
    print("Saving HURCAT2 File")
    df.to_parquet("data/processed/hurdat2_raw.parquet", index=False)
    
   