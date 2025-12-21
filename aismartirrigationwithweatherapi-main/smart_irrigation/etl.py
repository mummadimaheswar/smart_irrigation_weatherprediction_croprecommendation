"""ETL pipeline: normalize, transform, and store curated data."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .config import RAW_DIR, CURATED_DIR, OUTPUT, STATES

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_timestamps(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Convert to UTC datetime, handle various formats."""
    if col not in df.columns:
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                col = c
                break
    
    df = df.copy()
    df["datetime"] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["week"] = df["datetime"].dt.isocalendar().week
    return df.dropna(subset=["datetime"])


def normalize_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize state/district names."""
    df = df.copy()
    
    # State normalization
    state_map = {
        "assam": "ASSAM", "punjab": "PUNJAB", "maharashtra": "MAHARASHTRA",
        "karnataka": "KARNATAKA", "uttar pradesh": "UTTAR PRADESH",
        "up": "UTTAR PRADESH", "mh": "MAHARASHTRA", "pb": "PUNJAB",
    }
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.lower().map(
            lambda x: state_map.get(x, x.upper() if x else None)
        )
    
    # District normalization
    if "district" in df.columns:
        df["district"] = df["district"].str.strip().str.upper()
    
    return df


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent units: moisture 0-1, rain mm, temp C."""
    df = df.copy()
    
    # Soil moisture: convert % to fraction
    if "soil_moisture" in df.columns:
        if df["soil_moisture"].max() > 1.5:
            df["soil_moisture"] = df["soil_moisture"] / 100.0
    
    # Temperature: convert F to C if needed
    if "temp" in df.columns:
        if df["temp"].median() > 50:  # Likely Fahrenheit
            df["temp"] = (df["temp"] - 32) * 5/9
    
    # Humidity: ensure 0-100
    if "humidity" in df.columns:
        if df["humidity"].max() <= 1:
            df["humidity"] = df["humidity"] * 100
    
    # Rain: ensure mm (some sources use cm)
    if "rain" in df.columns:
        if df["rain"].max() < 10 and df["rain"].sum() > 0:
            df["rain"] = df["rain"] * 10  # cm to mm
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Flag or clip values outside physical bounds."""
    df = df.copy()
    
    bounds = {
        "soil_moisture": (0, 1),
        "temp": (-50, 60),
        "humidity": (0, 100),
        "wind_speed": (0, 100),
        "rain": (0, 500),
        "rain_1h": (0, 200),
        "pressure": (800, 1100),
        "ndvi": (-1, 1),
    }
    
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                log.warning(f"{col}: {mask.sum()} values outside [{lo}, {hi}]")
                df.loc[mask, col] = np.nan
    
    return df


def impute_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """Fill missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)
    elif method == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill(limit=3)
    elif method == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily."""
    if "datetime" not in df.columns:
        return df
    
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    
    agg_rules = {
        "temp": ["mean", "min", "max"],
        "humidity": "mean",
        "wind_speed": "mean",
        "rain_1h": "sum",
        "rain": "sum",
        "pressure": "mean",
        "clouds": "mean",
        "soil_moisture": "mean",
        "ndvi": "mean",
    }
    
    # Filter to existing columns
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    group_cols = ["date"]
    for col in ["state", "district"]:
        if col in df.columns:
            group_cols.append(col)
    
    daily = df.groupby(group_cols).agg(agg_rules).reset_index()
    
    # Flatten multi-level columns
    daily.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else c[0] if isinstance(c, tuple) else c
        for c in daily.columns
    ]
    
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# MERGE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def merge_weather_soil(weather_df: pd.DataFrame, soil_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather and soil data on date + location."""
    if weather_df.empty or soil_df.empty:
        return weather_df if not weather_df.empty else soil_df
    
    # Ensure date columns
    for df in [weather_df, soil_df]:
        if "date" not in df.columns and "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
    
    merge_keys = ["date"]
    for col in ["state", "district"]:
        if col in weather_df.columns and col in soil_df.columns:
            merge_keys.append(col)
    
    return pd.merge(weather_df, soil_df, on=merge_keys, how="outer", suffixes=("", "_soil"))


def merge_ndvi(base_df: pd.DataFrame, ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge NDVI (typically 16-day) with daily data using forward fill."""
    if ndvi_df.empty:
        return base_df
    
    ndvi_df = ndvi_df.copy()
    ndvi_df["date"] = pd.to_datetime(ndvi_df["date"]).dt.date
    
    merge_keys = ["date"]
    for col in ["state", "district"]:
        if col in base_df.columns and col in ndvi_df.columns:
            merge_keys.append(col)
    
    merged = pd.merge(base_df, ndvi_df[merge_keys + ["ndvi"]], on=merge_keys, how="left")
    
    # Forward fill NDVI (satellite data is sparse)
    if "ndvi" in merged.columns:
        merged["ndvi"] = merged.groupby(["state", "district"])["ndvi"].ffill(limit=20)
    
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def save_curated(df: pd.DataFrame, name: str, partition_by: Optional[list] = None):
    """Save curated data in configured format."""
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    
    partition_by = partition_by or OUTPUT.partition_by
    
    if OUTPUT.format == "parquet":
        path = CURATED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy" if OUTPUT.compress else None)
        log.info(f"Saved {len(df)} rows to {path}")
    
    elif OUTPUT.format == "csv":
        path = CURATED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info(f"Saved {len(df)} rows to {path}")
    
    elif OUTPUT.format == "postgres":
        if OUTPUT.postgres_uri:
            from sqlalchemy import create_engine
            engine = create_engine(OUTPUT.postgres_uri)
            df.to_sql(name, engine, if_exists="append", index=False)
            log.info(f"Inserted {len(df)} rows to postgres table {name}")
        else:
            log.error("Postgres URI not configured")


def load_curated(name: str) -> pd.DataFrame:
    """Load curated data."""
    if OUTPUT.format == "parquet":
        path = CURATED_DIR / f"{name}.parquet"
    elif OUTPUT.format == "csv":
        path = CURATED_DIR / f"{name}.csv"
    else:
        log.error(f"Unsupported format: {OUTPUT.format}")
        return pd.DataFrame()
    
    if not path.exists():
        log.warning(f"Curated data not found: {path}")
        return pd.DataFrame()
    
    if OUTPUT.format == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# ETL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class ETLPipeline:
    """End-to-end ETL pipeline."""
    
    def __init__(self):
        self.weather_frames = []
        self.soil_frames = []
        self.ndvi_frames = []
    
    def load_raw(self, pattern: str = "*.parquet") -> "ETLPipeline":
        """Load all raw data files."""
        for f in RAW_DIR.glob(pattern):
            try:
                df = pd.read_parquet(f)
                name = f.stem.lower()
                
                if "weather" in name:
                    self.weather_frames.append(df)
                elif "soil" in name:
                    self.soil_frames.append(df)
                elif "ndvi" in name:
                    self.ndvi_frames.append(df)
                else:
                    log.info(f"Unknown data type: {name}")
            except Exception as e:
                log.warning(f"Failed to load {f}: {e}")
        
        log.info(f"Loaded {len(self.weather_frames)} weather, "
                 f"{len(self.soil_frames)} soil, {len(self.ndvi_frames)} NDVI files")
        return self
    
    def transform(self) -> pd.DataFrame:
        """Apply all transformations and merge."""
        # Concatenate each type
        weather = pd.concat(self.weather_frames, ignore_index=True) if self.weather_frames else pd.DataFrame()
        soil = pd.concat(self.soil_frames, ignore_index=True) if self.soil_frames else pd.DataFrame()
        ndvi = pd.concat(self.ndvi_frames, ignore_index=True) if self.ndvi_frames else pd.DataFrame()
        
        # Normalize each
        for df in [weather, soil, ndvi]:
            if not df.empty:
                df = normalize_timestamps(df)
                df = normalize_geo(df)
                df = normalize_units(df)
                df = validate_ranges(df)
        
        # Aggregate weather to daily
        if not weather.empty and "datetime" in weather.columns:
            weather = aggregate_daily(weather)
        
        # Merge all
        result = weather if not weather.empty else pd.DataFrame()
        if not soil.empty:
            result = merge_weather_soil(result, soil) if not result.empty else soil
        if not ndvi.empty:
            result = merge_ndvi(result, ndvi) if not result.empty else ndvi
        
        # Impute
        if not result.empty:
            result = impute_missing(result)
        
        return result
    
    def run(self, output_name: str = "irrigation_dataset") -> pd.DataFrame:
        """Run full pipeline."""
        self.load_raw()
        result = self.transform()
        
        if not result.empty:
            save_curated(result, output_name)
        
        return result
