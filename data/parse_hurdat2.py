# Turn the HURDAT2 (Atlantic) file into a clean pandas DataFrame with one row per 6-hour observation

# --- Format ---
# Header Line: storm_id (basin, ATCF number, year), name, no. data lines that follow
# Data lines: date (yyyymmdd), time (hhmm), record identifier, status, latitude, longitude, maximum sustained wind (kt), minimum central pressure (mb), wind radii (post 2024)

# --- Record Identifier Codes ---
# L: Landfall
# I: Intensity Peak
# P: Pressure minimum
# Blank: Regular 6-hourly point

# --- Status Codes ---
# TD, TS, HU: Tropical
# EX: Extratropical
# SD, SS: Subtropical
# LO, WV, DB: Other

import pandas as pd
from pathlib import Path
from datetime import datetime

# Convert HURDAT2 numeric field, treating common missing codes as None
def handle_none(value: str):
    
    value = value.strip()
    
    if value in ("", "-99", "-999", " "):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_hurdat2(filepath):
    
    # Define structure of file
    rows = []
    current_storm_id = None
    current_name = None
    
    # Open file
    print("Opening File")
    with open(filepath, "r") as f:
        # Iterate through each line of the file
        for line in f:
            
            # Remove whitespace
            line = line.strip()
            
            if not line:
                continue
            
            # Access parts of the data entry
            parts = [p.strip() for p in line.split(",")]
            # print("Parts: ", parts)
            
            # Process Header line
            # Start with the basin code (AL, EP, CP)
            if parts[0].startswith(("AL", "EP", "CP")) and len(parts) <= 4:
                # Access storm ID and name
                current_storm_id = parts[0]
                current_name = parts[1]
                continue
            
            # Process Data Line
            if current_storm_id is None:
                continue
            
            date_str = parts[0]    
            time_str = parts[1].zfill(4)    
        
            
            if parts[2]:
                record_id = parts[2]  
            else:
                record_id = None
                
            status = parts[3]
            
            latitude_str = parts[4]
            longitude_str = parts[5]
            
            # Convert latitude and longitude
            latitude = float(latitude_str[:-1]) * (1 if latitude_str.endswith("N") else -1)
            longitude = float(longitude_str[:-1]) * (-1 if longitude_str.endswith("W") else 1)

            
            if parts[6]:
                vmax = int(parts[6]) 
            else:
                vmax = None
                            
            if parts[7] not in ("", "-999"):
                mslp = int(parts[7]) 
            else:
                mslp = None 
                
            # Convert date 
            date = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
                          
            # Handle none code
            vmax = handle_none(parts[6])
            mslp = handle_none(parts[7])
                            
            # Append data entry
            rows.append({
                "storm_id": current_storm_id,
                "name": current_name,
                "datetime": date,
                "record_id": record_id,
                "status": status,
                "latitude": latitude,
                "longitude": longitude,
                "vmax": vmax,
                "mslp": mslp,
            })

    print("DataFrame created")
    df = pd.DataFrame(rows)
    df = df.sort_values(["storm_id", "datetime"]).reset_index(drop=True)
    return df       
            
            
