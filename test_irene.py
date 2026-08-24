from pathlib import Path

from data.parse_hurdat2 import parse_hurdat2


if __name__ == "__main__":
    
    # Parse file
    print("Parsing test_irene_2011 File")
    df = parse_hurdat2("data/raw/test_irene_2011.txt")
    
    # Save df
    print("Saving test_irene_2011 File")
    df.to_parquet("data/processed/test_irene_2011_raw.parquet", index=False)
    
    