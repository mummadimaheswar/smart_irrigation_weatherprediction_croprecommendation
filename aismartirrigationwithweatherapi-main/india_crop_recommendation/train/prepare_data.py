"""
Training Data Preparation
Merges CSV soil moisture data with synthetic weather to create ML training dataset

Usage:
    python -m india_crop_recommendation.train.prepare_data --output data/training.parquet
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Crop assignment rules based on conditions
CROP_RULES = {
    "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "months": [6,7,8,9]},
    "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "months": [10,11,12,1,2]},
    "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "months": [6,7,8,9]},
    "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "months": [4,5,6,7]},
    "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "months": [1,2,3,10,11,12]},
    "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "months": [6,7,8]},
    "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "months": [6,7,8]},
    "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "months": [10,11,12]},
    "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "months": [10,11,12,1]},
    "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "months": [10,11,12,1,2]},
}

# State coordinates for weather simulation
STATE_COORDS = {
    "Maharashtra": (19.7, 75.7), "Gujarat": (22.3, 71.2), "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2), "Tamil Nadu": (11.1, 78.7), "Telangana": (18.1, 79.0),
    "Uttar Pradesh": (26.8, 80.9), "Uttarakhand": (30.1, 79.3), "West Bengal": (22.9, 87.8),
    "Andhra Pradesh": (15.9, 79.7), "Himachal Pradesh": (31.1, 77.2),
}


def load_csv_data() -> pd.DataFrame:
    """Load all CSV soil moisture data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest.csv_loader import load_all_states
    return load_all_states()


def add_synthetic_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic weather features based on state and month."""
    np.random.seed(42)
    
    # Temperature varies by latitude and month
    def get_temp(row):
        lat = STATE_COORDS.get(row["state"], (20, 78))[0]
        month = row["month"]
        # Base temp decreases with latitude
        base = 35 - (lat - 10) * 0.5
        # Seasonal variation
        if month in [12, 1, 2]:
            base -= 10
        elif month in [3, 4, 5]:
            base += 5
        return base + np.random.normal(0, 3)
    
    df["temp_mean_c"] = df.apply(get_temp, axis=1)
    df["temp_min_c"] = df["temp_mean_c"] - np.random.uniform(5, 10, len(df))
    df["temp_max_c"] = df["temp_mean_c"] + np.random.uniform(5, 10, len(df))
    
    # Rainfall - higher in monsoon months
    def get_rainfall(row):
        month = row["month"]
        if month in [6, 7, 8, 9]:  # Monsoon
            return np.random.exponential(100)
        elif month in [10, 11]:  # Post-monsoon
            return np.random.exponential(30)
        else:  # Dry
            return np.random.exponential(10)
    
    df["precip_mm"] = df.apply(get_rainfall, axis=1)
    
    # Humidity correlates with rainfall and moisture
    df["humidity_pct"] = 40 + df["soil_moisture_pct"] * 0.5 + np.random.normal(0, 10, len(df))
    df["humidity_pct"] = df["humidity_pct"].clip(20, 100)
    
    # Wind speed
    df["wind_speed_ms"] = np.random.exponential(3, len(df))
    
    # NDVI (vegetation index) - correlates with moisture
    df["ndvi"] = 0.2 + df["soil_moisture_pct"] / 200 + np.random.normal(0, 0.1, len(df))
    df["ndvi"] = df["ndvi"].clip(0, 1)
    
    # Coordinates
    df["lat"] = df["state"].map(lambda s: STATE_COORDS.get(s, (20, 78))[0])
    df["lon"] = df["state"].map(lambda s: STATE_COORDS.get(s, (20, 78))[1])
    
    # Derived features
    df["precip_7d_sum"] = df["precip_mm"] * 7 * np.random.uniform(0.5, 1.5, len(df))
    df["temp_7d_mean"] = df["temp_mean_c"] + np.random.normal(0, 1, len(df))
    df["gdd_base10"] = np.maximum(df["temp_mean_c"] - 10, 0) * 30  # Growing degree days
    df["water_deficit_mm"] = np.maximum(50 - df["soil_moisture_pct"] - df["precip_mm"] / 10, 0)
    
    return df


def assign_crop_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign crop labels based on conditions."""
    
    def best_crop(row):
        sm = row["soil_moisture_pct"]
        temp = row["temp_mean_c"]
        month = row["month"]
        
        scores = {}
        for crop, rules in CROP_RULES.items():
            score = 0
            # Soil moisture fit
            if rules["sm_min"] <= sm <= rules["sm_max"]:
                mid = (rules["sm_min"] + rules["sm_max"]) / 2
                score += 30 * (1 - abs(sm - mid) / (rules["sm_max"] - rules["sm_min"]))
            # Temperature fit
            if rules["temp_min"] <= temp <= rules["temp_max"]:
                mid = (rules["temp_min"] + rules["temp_max"]) / 2
                score += 30 * (1 - abs(temp - mid) / (rules["temp_max"] - rules["temp_min"]))
            # Season fit
            if month in rules["months"]:
                score += 40
            
            scores[crop] = score
        
        return max(scores, key=scores.get)
    
    df["crop_name"] = df.apply(best_crop, axis=1)
    return df


def prepare_training_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """Full pipeline: load CSV → add weather → assign labels → save."""
    log.info("Loading CSV soil moisture data...")
    df = load_csv_data()
    
    if df.empty:
        log.warning("No CSV data found, generating synthetic data")
        df = generate_fully_synthetic(5000)
    else:
        log.info(f"Loaded {len(df)} records from CSV")
        
        # Add weather
        log.info("Adding synthetic weather features...")
        df = add_synthetic_weather(df)
        
        # Assign crops
        log.info("Assigning crop labels...")
        df = assign_crop_labels(df)
    
    # Clean
    df = df.dropna(subset=["soil_moisture_pct", "temp_mean_c", "crop_name"])
    
    # Rename for model
    df = df.rename(columns={"soil_moisture_pct": "soil_moisture_pct"})
    
    log.info(f"Final dataset: {len(df)} samples, {df['crop_name'].nunique()} crops")
    log.info(f"Crop distribution:\n{df['crop_name'].value_counts()}")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        log.info(f"Saved to {output_path}")
    
    return df


def generate_fully_synthetic(n_samples: int = 5000) -> pd.DataFrame:
    """Generate fully synthetic training data when no CSV available."""
    np.random.seed(42)
    
    states = list(STATE_COORDS.keys())
    
    data = {
        "date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        "state": np.random.choice(states, n_samples),
        "district": [f"District_{i % 50}" for i in range(n_samples)],
        "month": np.random.randint(1, 13, n_samples),
    }
    
    df = pd.DataFrame(data)
    df["year"] = df["date"].dt.year
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    
    # Soil moisture
    df["soil_moisture_pct"] = np.random.uniform(10, 60, n_samples)
    df["sm_volume"] = df["soil_moisture_pct"] * 20
    
    # Add weather
    df = add_synthetic_weather(df)
    
    # Assign crops
    df = assign_crop_labels(df)
    
    return df


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("-o", "--output", default="data/training.parquet", help="Output path")
    parser.add_argument("--synthetic", action="store_true", help="Use fully synthetic data")
    parser.add_argument("-n", "--samples", type=int, default=5000, help="Synthetic samples")
    
    args = parser.parse_args()
    
    if args.synthetic:
        df = generate_fully_synthetic(args.samples)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"✅ Generated {len(df)} synthetic samples → {args.output}")
    else:
        prepare_training_data(args.output)
