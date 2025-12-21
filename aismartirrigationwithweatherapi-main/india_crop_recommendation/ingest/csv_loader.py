"""
CSV Data Loader for State Soil Moisture Files
Reads CSV files from states.csv/ folder and standardizes columns

Usage:
    from ingest.csv_loader import load_all_states, load_state
    
    # Load single state
    df = load_state("Maharashtra")
    
    # Load all states
    df = load_all_states()
"""
import os
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Column mapping from raw CSV to standard names
COLUMN_MAP = {
    "Date": "date",
    "State Name": "state",
    "DistrictName": "district",
    "Average Soilmoisture Level (at 15cm)": "sm_level_15cm",
    "Average SoilMoisture Volume (at 15cm)": "sm_volume_15cm",
    "Aggregate Soilmoisture Percentage (at 15cm)": "sm_pct_agg_15cm",
    "Volume Soilmoisture percentage (at 15cm)": "sm_pct_vol_15cm",
}

# Standard output columns
STANDARD_COLS = ["date", "state", "district", "soil_moisture_pct", "sm_volume", "year", "month", "day"]

# State name normalization
STATE_ALIASES = {
    "MAHARASHTRA": "Maharashtra",
    "GUJARAT": "Gujarat", 
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "TAMILNADU": "Tamil Nadu",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "UTTARPRADESH": "Uttar Pradesh",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "WESTBENGAL": "West Bengal",
    "WEST BENGAL": "West Bengal",
    "ANDHRAPRADESH": "Andhra Pradesh",
    "ANDHRA PRADESH": "Andhra Pradesh",
    "HIMACHALPRADESH": "Himachal Pradesh",
    "HIMACHAL PRADESH": "Himachal Pradesh",
}

def get_csv_folder() -> Path:
    """Get path to states.csv folder."""
    return Path(__file__).parent.parent / "states.csv"


def list_available_states() -> List[str]:
    """List all states with CSV data available."""
    csv_folder = get_csv_folder()
    files = glob.glob(str(csv_folder / "sm_*.csv"))
    states = []
    for f in files:
        name = Path(f).stem  # sm_Maharashtra_2020
        parts = name.split("_")
        if len(parts) >= 2:
            state_raw = parts[1]
            states.append(STATE_ALIASES.get(state_raw.upper(), state_raw))
    return sorted(set(states))


def load_state(state: str, year: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV data for a single state.
    
    Args:
        state: State name (case-insensitive)
        year: Optional year filter
    
    Returns:
        DataFrame with standardized columns
    """
    csv_folder = get_csv_folder()
    
    # Find matching file (fuzzy match on state name)
    state_lower = state.lower().replace(" ", "")
    pattern = f"sm_*{year or '*'}.csv" if year else "sm_*.csv"
    files = glob.glob(str(csv_folder / pattern))
    
    matching = []
    for f in files:
        fname_lower = Path(f).stem.lower()
        if state_lower in fname_lower.replace("_", ""):
            matching.append(f)
    
    if not matching:
        log.warning(f"No CSV files found for state: {state}")
        return pd.DataFrame(columns=STANDARD_COLS)
    
    dfs = []
    for fpath in matching:
        try:
            df = _load_single_csv(fpath)
            dfs.append(df)
            log.info(f"Loaded {len(df)} rows from {Path(fpath).name}")
        except Exception as e:
            log.error(f"Error loading {fpath}: {e}")
    
    if not dfs:
        return pd.DataFrame(columns=STANDARD_COLS)
    
    return pd.concat(dfs, ignore_index=True)


def load_all_states(year: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV data for all available states.
    
    Args:
        year: Optional year filter
        
    Returns:
        Combined DataFrame with all states
    """
    csv_folder = get_csv_folder()
    pattern = f"sm_*{year}.csv" if year else "sm_*.csv"
    files = glob.glob(str(csv_folder / pattern))
    
    if not files:
        log.warning(f"No CSV files found in {csv_folder}")
        return pd.DataFrame(columns=STANDARD_COLS)
    
    dfs = []
    for fpath in files:
        try:
            df = _load_single_csv(fpath)
            dfs.append(df)
            log.info(f"Loaded {len(df)} rows from {Path(fpath).name}")
        except Exception as e:
            log.error(f"Error loading {fpath}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total: {len(combined)} rows from {len(dfs)} files")
    return combined


def _load_single_csv(filepath: str) -> pd.DataFrame:
    """Load and standardize a single CSV file."""
    df = pd.read_csv(filepath)
    
    # Rename columns
    df = df.rename(columns=COLUMN_MAP)
    
    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
    
    # Normalize state names
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip().str.upper().map(
            lambda x: STATE_ALIASES.get(x, x.title() if x != "NAN" else "Unknown")
        )
    
    # Normalize district names
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title()
        df["district"] = df["district"].replace("Nan", "Unknown")
    
    # Create primary soil moisture column (use volumetric %)
    if "sm_pct_vol_15cm" in df.columns:
        df["soil_moisture_pct"] = df["sm_pct_vol_15cm"]
    elif "sm_pct_agg_15cm" in df.columns:
        df["soil_moisture_pct"] = df["sm_pct_agg_15cm"]
    else:
        df["soil_moisture_pct"] = np.nan
    
    # Keep volume for reference
    if "sm_volume_15cm" in df.columns:
        df["sm_volume"] = df["sm_volume_15cm"]
    else:
        df["sm_volume"] = np.nan
    
    # Handle missing/invalid values
    df["soil_moisture_pct"] = df["soil_moisture_pct"].replace(0, np.nan)
    df = df.dropna(subset=["date"])
    
    return df[STANDARD_COLS]


def get_district_stats(state: str) -> pd.DataFrame:
    """Get statistics per district for a state."""
    df = load_state(state)
    if df.empty:
        return pd.DataFrame()
    
    stats = df.groupby("district").agg({
        "soil_moisture_pct": ["mean", "std", "min", "max", "count"],
        "date": ["min", "max"]
    }).round(2)
    
    stats.columns = ["sm_mean", "sm_std", "sm_min", "sm_max", "count", "date_min", "date_max"]
    return stats.reset_index()


def get_monthly_avg(state: Optional[str] = None) -> pd.DataFrame:
    """Get monthly average soil moisture."""
    df = load_state(state) if state else load_all_states()
    if df.empty:
        return pd.DataFrame()
    
    return df.groupby(["state", "month"]).agg({
        "soil_moisture_pct": "mean"
    }).round(2).reset_index()


def export_to_parquet(output_path: str, year: Optional[int] = None):
    """Export all CSV data to Parquet format."""
    df = load_all_states(year)
    if df.empty:
        log.warning("No data to export")
        return
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    log.info(f"Exported {len(df)} rows to {output}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="CSV Soil Moisture Data Loader")
    parser.add_argument("--state", "-s", help="State name to load")
    parser.add_argument("--year", "-y", type=int, help="Year filter")
    parser.add_argument("--list", "-l", action="store_true", help="List available states")
    parser.add_argument("--stats", action="store_true", help="Show district statistics")
    parser.add_argument("--export", "-e", help="Export to Parquet file")
    parser.add_argument("--head", "-n", type=int, default=10, help="Number of rows to show")
    
    args = parser.parse_args()
    
    if args.list:
        states = list_available_states()
        print(f"Available states ({len(states)}):")
        for s in states:
            print(f"  - {s}")
    elif args.export:
        export_to_parquet(args.export, args.year)
    elif args.stats and args.state:
        stats = get_district_stats(args.state)
        print(f"\n{args.state} District Statistics:")
        print(stats.to_string(index=False))
    elif args.state:
        df = load_state(args.state, args.year)
        print(f"\n{args.state} Data ({len(df)} total rows):")
        print(df.head(args.head).to_string(index=False))
        print(f"\nSoil Moisture Stats:")
        print(df["soil_moisture_pct"].describe().round(2))
    else:
        df = load_all_states(args.year)
        print(f"\nAll States Data ({len(df)} total rows):")
        print(df.head(args.head).to_string(index=False))
        print(f"\nPer-State Summary:")
        print(df.groupby("state")["soil_moisture_pct"].agg(["count", "mean", "std"]).round(2))
