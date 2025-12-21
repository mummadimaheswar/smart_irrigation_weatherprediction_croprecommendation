"""
Soil Moisture Ingestion Module
India Crop Recommendation System

PROMPT 5: Ingestion for:
- Satellite data (SMAP/SMOS/Sentinel)
- IoT sensor CSV exports
- Data harmonization (temporal resampling, unit conversion, quality flags)
- Spatial joining (sensor to satellite pixel)
- Parquet export partitioned by state/year
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")

# State bounding boxes for satellite data extraction
STATE_BBOXES = {
    "Maharashtra": {"min_lon": 72.6, "max_lon": 80.9, "min_lat": 15.6, "max_lat": 22.1},
    "Karnataka": {"min_lon": 74.0, "max_lon": 78.5, "min_lat": 11.5, "max_lat": 18.5},
    "Tamil Nadu": {"min_lon": 76.2, "max_lon": 80.3, "min_lat": 8.1, "max_lat": 13.5},
    "Punjab": {"min_lon": 73.8, "max_lon": 76.9, "min_lat": 29.5, "max_lat": 32.5},
    "Uttar Pradesh": {"min_lon": 77.1, "max_lon": 84.6, "min_lat": 23.9, "max_lat": 30.4},
    "Gujarat": {"min_lon": 68.1, "max_lon": 74.5, "min_lat": 20.1, "max_lat": 24.7},
    "Rajasthan": {"min_lon": 69.5, "max_lon": 78.3, "min_lat": 23.1, "max_lat": 30.2},
    "Madhya Pradesh": {"min_lon": 74.0, "max_lon": 82.8, "min_lat": 21.1, "max_lat": 26.9},
    "Andhra Pradesh": {"min_lon": 76.8, "max_lon": 84.8, "min_lat": 12.6, "max_lat": 19.9},
    "West Bengal": {"min_lon": 85.8, "max_lon": 89.9, "min_lat": 21.5, "max_lat": 27.2},
}

# SMAP resolution: ~9km
SMAP_RESOLUTION_KM = 9.0

# Soil moisture valid range
SM_MIN = 0.0   # m³/m³
SM_MAX = 0.6   # m³/m³ (saturated soil)


# ═══════════════════════════════════════════════════════════════════════════════
# SATELLITE DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

class SMAPIngester:
    """Ingest NASA SMAP Level 3 soil moisture data."""
    
    APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"
    PRODUCT = "SPL3SMP.008"  # SMAP L3 Daily
    LAYER = "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or NASA_EARTHDATA_TOKEN
        
    def authenticate(self, username: str, password: str) -> str:
        """Get AppEEARS token."""
        import requests
        
        response = requests.post(
            f"{self.APPEEARS_URL}/login",
            auth=(username, password)
        )
        if response.status_code == 200:
            self.token = response.json()["token"]
            return self.token
        raise ValueError(f"Authentication failed: {response.text}")
    
    def submit_area_request(
        self,
        state: str,
        start_date: datetime,
        end_date: datetime,
        task_name: Optional[str] = None
    ) -> str:
        """Submit area extraction request to AppEEARS."""
        import requests
        
        if not self.token:
            raise ValueError("Token required. Call authenticate() first.")
        
        if state not in STATE_BBOXES:
            raise ValueError(f"Unknown state: {state}")
        
        bbox = STATE_BBOXES[state]
        task_name = task_name or f"SMAP_{state}_{start_date.year}"
        
        # Create GeoJSON polygon from bbox
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox["min_lon"], bbox["min_lat"]],
                        [bbox["max_lon"], bbox["min_lat"]],
                        [bbox["max_lon"], bbox["max_lat"]],
                        [bbox["min_lon"], bbox["max_lat"]],
                        [bbox["min_lon"], bbox["min_lat"]],
                    ]]
                },
                "properties": {}
            }]
        }
        
        payload = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date.strftime("%m-%d-%Y"),
                    "endDate": end_date.strftime("%m-%d-%Y")
                }],
                "layers": [{
                    "product": self.PRODUCT,
                    "layer": self.LAYER
                }],
                "geo": geojson,
                "output": {"format": {"type": "geotiff"}, "projection": "geographic"}
            }
        }
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{self.APPEEARS_URL}/task",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 202:
            task_id = response.json()["task_id"]
            log.info(f"Submitted task: {task_id}")
            return task_id
        
        raise ValueError(f"Request failed: {response.text}")
    
    def check_task_status(self, task_id: str) -> dict:
        """Check status of AppEEARS task."""
        import requests
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(
            f"{self.APPEEARS_URL}/task/{task_id}",
            headers=headers
        )
        return response.json()
    
    def download_results(self, task_id: str, output_dir: str) -> List[str]:
        """Download completed task results."""
        import requests
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Get file list
        response = requests.get(
            f"{self.APPEEARS_URL}/bundle/{task_id}",
            headers=headers
        )
        files = response.json().get("files", [])
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        for f in files:
            if f["file_name"].endswith(".tif"):
                file_url = f"{self.APPEEARS_URL}/bundle/{task_id}/{f['file_id']}"
                r = requests.get(file_url, headers=headers)
                
                file_path = output_path / f["file_name"]
                with open(file_path, "wb") as fp:
                    fp.write(r.content)
                downloaded.append(str(file_path))
                log.info(f"Downloaded: {file_path}")
        
        return downloaded


def read_smap_geotiff(file_path: str) -> pd.DataFrame:
    """
    Read SMAP GeoTIFF and convert to DataFrame.
    
    Requires: rasterio, numpy
    """
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio not installed, using synthetic data")
        return _generate_synthetic_satellite_data()
    
    records = []
    
    with rasterio.open(file_path) as src:
        data = src.read(1)  # First band
        transform = src.transform
        
        # Extract date from filename (e.g., SPL3SMP_2020-06-15.tif)
        filename = Path(file_path).stem
        date_str = filename.split("_")[-1] if "_" in filename else None
        
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
        except:
            date = None
        
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                value = data[i, j]
                
                # Skip nodata values
                if value < 0 or value > 1:
                    continue
                
                # Get lat/lon from pixel coordinates
                lon, lat = rasterio.transform.xy(transform, i, j)
                
                records.append({
                    "date": date,
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "soil_moisture_m3m3": round(value, 4),
                    "soil_moisture_pct": round(value * 100, 2),
                    "source": "SMAP",
                    "resolution_km": SMAP_RESOLUTION_KM,
                })
    
    return pd.DataFrame(records)


def _generate_synthetic_satellite_data(
    state: str = "Maharashtra",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resolution_deg: float = 0.1
) -> pd.DataFrame:
    """Generate synthetic satellite soil moisture data."""
    
    if state not in STATE_BBOXES:
        state = "Maharashtra"
    
    bbox = STATE_BBOXES[state]
    start_date = start_date or datetime(2020, 1, 1)
    end_date = end_date or datetime(2020, 12, 31)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    lats = np.arange(bbox["min_lat"], bbox["max_lat"], resolution_deg)
    lons = np.arange(bbox["min_lon"], bbox["max_lon"], resolution_deg)
    
    records = []
    for date in dates[::5]:  # Every 5 days to simulate satellite revisit
        for lat in lats[::3]:  # Sample grid
            for lon in lons[::3]:
                # Seasonal variation
                day_of_year = date.dayofyear
                seasonal = 0.1 * np.sin((day_of_year - 172) * 2 * np.pi / 365)  # Peak in monsoon
                
                # Spatial variation
                spatial = 0.05 * np.sin(lat) * np.cos(lon)
                
                # Base + variation + noise
                sm = 0.25 + seasonal + spatial + np.random.normal(0, 0.03)
                sm = np.clip(sm, SM_MIN, SM_MAX)
                
                records.append({
                    "date": date.date(),
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "soil_moisture_m3m3": round(sm, 4),
                    "soil_moisture_pct": round(sm * 100, 2),
                    "source": "SMAP_synthetic",
                    "resolution_km": SMAP_RESOLUTION_KM,
                    "state": state,
                })
    
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# IOT SENSOR DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_sensor_csv(
    file_path: str,
    timestamp_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    sm_col: str = "volumetric_water_content",
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load IoT sensor CSV data.
    
    Expected columns: timestamp, lat, lon, volumetric_water_content
    
    Args:
        file_path: Path to CSV file
        timestamp_col: Name of timestamp column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        sm_col: Name of soil moisture column
        date_format: Optional strftime format for parsing dates
    
    Returns:
        Normalized DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "time" in col_lower or "date" in col_lower:
            col_map[col] = "timestamp"
        elif "lat" in col_lower:
            col_map[col] = "lat"
        elif "lon" in col_lower:
            col_map[col] = "lon"
        elif "moisture" in col_lower or "vwc" in col_lower or "sm" in col_lower:
            col_map[col] = "soil_moisture_raw"
        elif "sensor" in col_lower or "device" in col_lower:
            col_map[col] = "sensor_id"
        elif "depth" in col_lower:
            col_map[col] = "depth_cm"
    
    df = df.rename(columns=col_map)
    
    # Parse timestamp
    if "timestamp" in df.columns:
        if date_format:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_format)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
        df["date"] = df["timestamp"].dt.date
    
    # Normalize soil moisture to m³/m³ (0-1)
    if "soil_moisture_raw" in df.columns:
        values = df["soil_moisture_raw"]
        
        # Detect unit and convert
        if values.max() > 1:  # Likely percentage
            df["soil_moisture_m3m3"] = values / 100.0
        else:
            df["soil_moisture_m3m3"] = values
        
        # Apply valid range
        df["soil_moisture_m3m3"] = df["soil_moisture_m3m3"].clip(SM_MIN, SM_MAX)
        df["soil_moisture_pct"] = df["soil_moisture_m3m3"] * 100
    
    # Add metadata
    df["source"] = "sensor"
    
    # Quality flags
    df = _add_quality_flags(df)
    
    return df


def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add quality control flags to sensor data."""
    df["quality_flag"] = "good"
    
    # Flag outliers
    if "soil_moisture_m3m3" in df.columns:
        q1 = df["soil_moisture_m3m3"].quantile(0.01)
        q99 = df["soil_moisture_m3m3"].quantile(0.99)
        
        df.loc[df["soil_moisture_m3m3"] < q1, "quality_flag"] = "suspect_low"
        df.loc[df["soil_moisture_m3m3"] > q99, "quality_flag"] = "suspect_high"
        df.loc[df["soil_moisture_m3m3"].isna(), "quality_flag"] = "missing"
    
    # Flag impossible coordinates
    if "lat" in df.columns and "lon" in df.columns:
        df.loc[(df["lat"] < 6) | (df["lat"] > 38), "quality_flag"] = "invalid_coords"
        df.loc[(df["lon"] < 68) | (df["lon"] > 98), "quality_flag"] = "invalid_coords"
    
    return df


def _generate_synthetic_sensor_data(
    state: str = "Maharashtra",
    n_sensors: int = 10,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    readings_per_day: int = 24
) -> pd.DataFrame:
    """Generate synthetic IoT sensor data."""
    
    if state not in STATE_BBOXES:
        state = "Maharashtra"
    
    bbox = STATE_BBOXES[state]
    start_date = start_date or datetime(2020, 1, 1)
    end_date = end_date or datetime(2020, 12, 31)
    
    # Random sensor locations within state
    np.random.seed(42)
    sensor_lats = np.random.uniform(bbox["min_lat"], bbox["max_lat"], n_sensors)
    sensor_lons = np.random.uniform(bbox["min_lon"], bbox["max_lon"], n_sensors)
    
    records = []
    timestamps = pd.date_range(start_date, end_date, freq=f'{24//readings_per_day}H')
    
    for i, (lat, lon) in enumerate(zip(sensor_lats, sensor_lons)):
        sensor_id = f"SENSOR_{state[:2].upper()}_{i+1:03d}"
        
        # Base moisture for this location
        base_sm = np.random.uniform(0.15, 0.35)
        
        for ts in timestamps:
            # Diurnal variation
            hour_offset = 0.02 * np.sin((ts.hour - 6) * np.pi / 12)
            
            # Seasonal variation
            day_of_year = ts.dayofyear
            seasonal = 0.1 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
            
            # Random noise
            noise = np.random.normal(0, 0.02)
            
            sm = base_sm + seasonal + hour_offset + noise
            sm = np.clip(sm, SM_MIN, SM_MAX)
            
            records.append({
                "timestamp": ts,
                "date": ts.date(),
                "sensor_id": sensor_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "soil_moisture_m3m3": round(sm, 4),
                "soil_moisture_pct": round(sm * 100, 2),
                "depth_cm": 15,
                "source": "sensor_synthetic",
                "state": state,
            })
    
    df = pd.DataFrame(records)
    return _add_quality_flags(df)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HARMONIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def resample_to_daily(
    df: pd.DataFrame,
    agg_method: str = "mean"
) -> pd.DataFrame:
    """
    Resample sensor readings to daily averages.
    
    Args:
        df: DataFrame with timestamp/date and soil_moisture columns
        agg_method: Aggregation method (mean, median, max, min)
    
    Returns:
        Daily aggregated DataFrame
    """
    if df.empty:
        return df
    
    # Ensure date column
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    
    # Group by location and date
    group_cols = ["date"]
    if "lat" in df.columns and "lon" in df.columns:
        group_cols.extend(["lat", "lon"])
    if "sensor_id" in df.columns:
        group_cols.append("sensor_id")
    if "state" in df.columns:
        group_cols.append("state")
    
    # Aggregate
    agg_funcs = {
        "soil_moisture_m3m3": agg_method,
        "soil_moisture_pct": agg_method,
    }
    
    # Add count for quality assessment
    if "soil_moisture_m3m3" in df.columns:
        df["reading_count"] = 1
        agg_funcs["reading_count"] = "sum"
    
    daily = df.groupby(group_cols, as_index=False).agg(agg_funcs)
    
    # Add year/month for partitioning
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    
    return daily


def join_sensor_to_satellite(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
    max_distance_km: float = 10.0
) -> pd.DataFrame:
    """
    Join sensor readings to nearest satellite pixel.
    
    Uses Haversine distance to find nearest satellite observation.
    
    Args:
        sensor_df: Sensor DataFrame with lat, lon, date
        satellite_df: Satellite DataFrame with lat, lon, date
        max_distance_km: Maximum distance for matching
    
    Returns:
        Merged DataFrame with both sensor and satellite values
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in km."""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    if sensor_df.empty or satellite_df.empty:
        return sensor_df
    
    # Ensure date types match
    sensor_df["date"] = pd.to_datetime(sensor_df["date"]).dt.date
    satellite_df["date"] = pd.to_datetime(satellite_df["date"]).dt.date
    
    results = []
    
    for _, sensor_row in sensor_df.iterrows():
        sensor_date = sensor_row["date"]
        sensor_lat = sensor_row["lat"]
        sensor_lon = sensor_row["lon"]
        
        # Find satellite observations on same date
        date_match = satellite_df[satellite_df["date"] == sensor_date]
        
        if date_match.empty:
            # Try nearest date within 3 days
            date_range = pd.date_range(
                sensor_date - timedelta(days=3),
                sensor_date + timedelta(days=3)
            )
            date_match = satellite_df[satellite_df["date"].isin([d.date() for d in date_range])]
        
        if date_match.empty:
            continue
        
        # Find nearest pixel
        min_dist = float('inf')
        nearest_sat = None
        
        for _, sat_row in date_match.iterrows():
            dist = haversine(sensor_lat, sensor_lon, sat_row["lat"], sat_row["lon"])
            if dist < min_dist:
                min_dist = dist
                nearest_sat = sat_row
        
        if nearest_sat is not None and min_dist <= max_distance_km:
            result = sensor_row.to_dict()
            result["satellite_sm_m3m3"] = nearest_sat["soil_moisture_m3m3"]
            result["satellite_sm_pct"] = nearest_sat["soil_moisture_pct"]
            result["satellite_source"] = nearest_sat.get("source", "satellite")
            result["pixel_distance_km"] = round(min_dist, 2)
            results.append(result)
    
    return pd.DataFrame(results)


def harmonize_datasets(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
    prefer_source: str = "sensor"
) -> pd.DataFrame:
    """
    Harmonize sensor and satellite data into unified dataset.
    
    Args:
        sensor_df: Daily sensor data
        satellite_df: Satellite data
        prefer_source: Which source to prefer when both available
    
    Returns:
        Harmonized DataFrame
    """
    # Ensure both have required columns
    required_cols = ["date", "lat", "lon", "soil_moisture_pct"]
    
    for col in required_cols:
        if col not in sensor_df.columns:
            sensor_df[col] = None
        if col not in satellite_df.columns:
            satellite_df[col] = None
    
    # Add source identifier
    sensor_df["data_source"] = "sensor"
    satellite_df["data_source"] = "satellite"
    
    # Combine
    combined = pd.concat([sensor_df, satellite_df], ignore_index=True)
    
    # Deduplicate by location and date
    combined = combined.sort_values(
        ["date", "lat", "lon", "data_source"],
        ascending=[True, True, True, prefer_source == "satellite"]
    )
    combined = combined.drop_duplicates(subset=["date", "lat", "lon"], keep="first")
    
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# PARQUET EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_to_parquet(
    df: pd.DataFrame,
    output_dir: str,
    partition_by: List[str] = ["year", "state"]
) -> str:
    """Save soil moisture data to partitioned Parquet."""
    if df.empty:
        log.warning("Empty DataFrame, nothing to save")
        return ""
    
    # Ensure partition columns exist
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    
    if "state" not in df.columns:
        df["state"] = "unknown"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter partition columns that exist
    partition_cols = [c for c in partition_by if c in df.columns]
    
    df.to_parquet(
        output_path,
        engine="pyarrow",
        partition_cols=partition_cols if partition_cols else None,
        compression="snappy",
        index=False
    )
    
    log.info(f"Saved {len(df)} records to {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for soil moisture ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest soil moisture data")
    parser.add_argument("--source", choices=["sensor", "satellite", "synthetic"], 
                       default="synthetic", help="Data source")
    parser.add_argument("--state", type=str, default="Maharashtra", help="State name")
    parser.add_argument("--input", type=str, help="Input CSV file (for sensor)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="./data/curated/soil_moisture")
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime(2020, 1, 1)
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime(2020, 12, 31)
    
    if args.source == "sensor":
        if not args.input:
            log.error("--input required for sensor source")
            return
        df = load_sensor_csv(args.input)
        df = resample_to_daily(df)
    elif args.source == "satellite":
        log.info("Generating synthetic satellite data (API requires authentication)")
        df = _generate_synthetic_satellite_data(args.state, start, end)
    else:  # synthetic
        log.info("Generating synthetic combined data")
        sat_df = _generate_synthetic_satellite_data(args.state, start, end)
        sensor_df = _generate_synthetic_sensor_data(args.state, 10, start, end)
        sensor_df = resample_to_daily(sensor_df)
        df = harmonize_datasets(sensor_df, sat_df)
    
    print(f"\nProcessed {len(df)} records:")
    print(df.head(10).to_string())
    
    save_to_parquet(df, args.output)


if __name__ == "__main__":
    main()
