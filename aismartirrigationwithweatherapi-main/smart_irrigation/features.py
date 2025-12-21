"""Feature engineering: rainfall, ET, soil moisture, phenology, soil type."""
import numpy as np
import pandas as pd
from typing import Optional

from .config import CROP_PARAMS, SOIL_TYPES

# ─────────────────────────────────────────────────────────────────────────────
# RAINFALL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def rainfall_features(df: pd.DataFrame, rain_col: str = "rain_sum") -> pd.DataFrame:
    """Compute short and long-term rainfall statistics."""
    df = df.copy()
    
    # Find rain column
    if rain_col not in df.columns:
        for c in df.columns:
            if "rain" in c.lower():
                rain_col = c
                break
    
    if rain_col not in df.columns:
        return df
    
    # Sort by location + date
    sort_cols = ["date"]
    group_cols = []
    for c in ["state", "district"]:
        if c in df.columns:
            sort_cols.insert(0, c)
            group_cols.append(c)
    
    df = df.sort_values(sort_cols)
    
    # Rolling windows
    windows = {"3d": 3, "7d": 7, "14d": 14, "30d": 30}
    
    for name, days in windows.items():
        col = f"rain_{name}"
        if group_cols:
            df[col] = df.groupby(group_cols)[rain_col].transform(
                lambda x: x.rolling(days, min_periods=1).sum()
            )
        else:
            df[col] = df[rain_col].rolling(days, min_periods=1).sum()
    
    # Days since last rain
    df["rain_binary"] = (df[rain_col] > 0.1).astype(int)
    if group_cols:
        df["days_since_rain"] = df.groupby(group_cols)["rain_binary"].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )
    else:
        df["days_since_rain"] = df["rain_binary"].groupby(
            (df["rain_binary"] != df["rain_binary"].shift()).cumsum()
        ).cumcount()
    
    # Rain intensity categories
    df["rain_intensity"] = pd.cut(
        df[rain_col],
        bins=[-np.inf, 0.1, 2.5, 7.5, 35, np.inf],
        labels=["none", "light", "moderate", "heavy", "extreme"]
    )
    
    return df.drop(columns=["rain_binary"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# EVAPOTRANSPIRATION PROXIES
# ─────────────────────────────────────────────────────────────────────────────

def et_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ET0 proxies and crop ETc."""
    df = df.copy()
    
    # Get column names (handle aggregated naming)
    temp_col = next((c for c in df.columns if "temp" in c.lower() and "mean" in c.lower()), 
                    next((c for c in df.columns if "temp" in c.lower()), None))
    temp_max = next((c for c in df.columns if "temp" in c.lower() and "max" in c.lower()), None)
    temp_min = next((c for c in df.columns if "temp" in c.lower() and "min" in c.lower()), None)
    humidity_col = next((c for c in df.columns if "humid" in c.lower()), None)
    wind_col = next((c for c in df.columns if "wind" in c.lower()), None)
    
    # Hargreaves ET0 (simplified - needs lat for full accuracy)
    if temp_col and temp_max and temp_min:
        ra = 15.0  # Approximate extraterrestrial radiation MJ/m2/day
        df["et0_hargreaves"] = 0.0023 * ra * (df[temp_col] + 17.8) * np.sqrt(
            np.maximum(df[temp_max] - df[temp_min], 0.1)
        )
    elif temp_col:
        # Very rough proxy
        df["et0_proxy"] = np.maximum(0, (df[temp_col] - 10) * 0.15)
    
    # Atmospheric demand proxy (VPD-like)
    if temp_col and humidity_col:
        # Saturation vapor pressure
        es = 0.6108 * np.exp(17.27 * df[temp_col] / (df[temp_col] + 237.3))
        ea = es * df[humidity_col] / 100
        df["vpd"] = np.maximum(0, es - ea)  # Vapor pressure deficit
    
    # Wind stress factor
    if wind_col:
        df["wind_stress"] = np.clip(df[wind_col] / 5.0, 0, 2)  # Normalized 0-2
    
    # Rolling ET
    et_col = "et0_hargreaves" if "et0_hargreaves" in df.columns else "et0_proxy"
    if et_col in df.columns:
        df["et_7d"] = df[et_col].rolling(7, min_periods=1).sum()
        df["et_14d"] = df[et_col].rolling(14, min_periods=1).sum()
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOIL MOISTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def soil_moisture_features(df: pd.DataFrame, moisture_col: str = "soil_moisture") -> pd.DataFrame:
    """Compute soil moisture statistics and trends."""
    df = df.copy()
    
    if moisture_col not in df.columns:
        for c in df.columns:
            if "moisture" in c.lower() or "vwc" in c.lower():
                moisture_col = c
                break
    
    if moisture_col not in df.columns:
        return df
    
    # Group columns
    group_cols = [c for c in ["state", "district"] if c in df.columns]
    
    # Rolling statistics
    windows = [3, 7, 14]
    for w in windows:
        if group_cols:
            df[f"sm_mean_{w}d"] = df.groupby(group_cols)[moisture_col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"sm_std_{w}d"] = df.groupby(group_cols)[moisture_col].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        else:
            df[f"sm_mean_{w}d"] = df[moisture_col].rolling(w, min_periods=1).mean()
            df[f"sm_std_{w}d"] = df[moisture_col].rolling(w, min_periods=1).std()
    
    # Trend (slope of last 7 days)
    def calc_slope(x):
        if len(x) < 3:
            return 0
        return np.polyfit(range(len(x)), x, 1)[0]
    
    if group_cols:
        df["sm_trend_7d"] = df.groupby(group_cols)[moisture_col].transform(
            lambda x: x.rolling(7, min_periods=3).apply(calc_slope, raw=True)
        )
    else:
        df["sm_trend_7d"] = df[moisture_col].rolling(7, min_periods=3).apply(calc_slope, raw=True)
    
    # Deficit from typical
    df["sm_deficit"] = df[f"sm_mean_14d"] - df[moisture_col]
    
    # Categories
    df["sm_category"] = pd.cut(
        df[moisture_col],
        bins=[0, 0.15, 0.20, 0.30, 0.40, 1.0],
        labels=["critical", "low", "adequate", "optimal", "saturated"]
    )
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHENOLOGY / CROP STAGE
# ─────────────────────────────────────────────────────────────────────────────

def phenology_features(df: pd.DataFrame, crop: str = "wheat") -> pd.DataFrame:
    """Add crop stage and growth window features."""
    df = df.copy()
    
    params = CROP_PARAMS.get(crop, CROP_PARAMS.get("wheat"))
    sowing_months = params["sowing_months"]
    growing_days = params["growing_days"]
    
    # Estimate days after sowing (DAS)
    # Assume sowing at start of sowing window
    if "month" in df.columns:
        df["in_sowing_window"] = df["month"].isin(sowing_months).astype(int)
        
        # Simple DAS estimation based on month
        def estimate_das(row):
            month = row["month"]
            day = row.get("day", 15)
            
            # Find closest sowing month
            if month in sowing_months:
                return day
            
            for sm in sowing_months:
                if month > sm:
                    das = (month - sm) * 30 + day
                    if das <= growing_days:
                        return das
            return -1  # Not in growing season
        
        df["das_estimated"] = df.apply(estimate_das, axis=1)
    
    # Crop stage based on DAS
    if "das_estimated" in df.columns:
        kc_stages = params["kc_stages"]
        stage_days = [growing_days * 0.15, growing_days * 0.35, 
                      growing_days * 0.75, growing_days]
        
        def get_stage(das):
            if das < 0:
                return "off_season"
            if das < stage_days[0]:
                return "initial"
            if das < stage_days[1]:
                return "development"
            if das < stage_days[2]:
                return "mid"
            if das <= stage_days[3]:
                return "late"
            return "harvest"
        
        df["crop_stage"] = df["das_estimated"].apply(get_stage)
        
        # Stage-specific Kc
        stage_kc = {"initial": kc_stages[0], "development": kc_stages[1],
                    "mid": kc_stages[2], "late": kc_stages[3], 
                    "off_season": 0.5, "harvest": 0.3}
        df["kc"] = df["crop_stage"].map(stage_kc)
    
    # Critical windows (stress sensitivity)
    if "crop_stage" in df.columns:
        critical_stages = {"development", "mid"}  # Most sensitive
        df["critical_window"] = df["crop_stage"].isin(critical_stages).astype(int)
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOIL TYPE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def soil_type_features(df: pd.DataFrame, soil_type: str = "loam") -> pd.DataFrame:
    """Add soil type properties."""
    df = df.copy()
    
    # If soil type column exists, use it; otherwise use default
    if "soil_type" not in df.columns:
        df["soil_type"] = soil_type
    
    # Map soil properties
    df["awc"] = df["soil_type"].map(lambda x: SOIL_TYPES.get(x, {}).get("awc", 0.15))
    df["infiltration_rate"] = df["soil_type"].map(
        lambda x: SOIL_TYPES.get(x, {}).get("infiltration_mm_hr", 15)
    )
    df["drainage"] = df["soil_type"].map(
        lambda x: SOIL_TYPES.get(x, {}).get("drainage", "moderate")
    )
    
    # Irrigation efficiency factors
    drainage_eff = {"poor": 0.7, "moderate": 0.85, "good": 0.9, "excessive": 0.75}
    df["irrigation_efficiency"] = df["drainage"].map(drainage_eff)
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# WATER BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def water_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute water balance: inputs - outputs."""
    df = df.copy()
    
    # Find columns
    rain_col = next((c for c in df.columns if "rain" in c.lower() and "sum" in c.lower()),
                    next((c for c in df.columns if "rain" in c.lower()), None))
    et_col = next((c for c in df.columns if c.startswith("et0") or c.startswith("et_")), None)
    
    if rain_col and et_col:
        # Daily balance
        df["water_balance"] = df[rain_col] - df[et_col]
        
        # Cumulative deficit
        group_cols = [c for c in ["state", "district"] if c in df.columns]
        if group_cols:
            df["cum_deficit_7d"] = df.groupby(group_cols)["water_balance"].transform(
                lambda x: x.rolling(7, min_periods=1).sum()
            )
            df["cum_deficit_14d"] = df.groupby(group_cols)["water_balance"].transform(
                lambda x: x.rolling(14, min_periods=1).sum()
            )
        else:
            df["cum_deficit_7d"] = df["water_balance"].rolling(7, min_periods=1).sum()
            df["cum_deficit_14d"] = df["water_balance"].rolling(14, min_periods=1).sum()
    
    # Irrigation need indicator
    if "soil_moisture" in df.columns and "cum_deficit_7d" in df.columns:
        df["irrigation_urgency"] = (
            (df["soil_moisture"] < 0.22).astype(int) * 2 +
            (df["cum_deficit_7d"] < -20).astype(int)
        )
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def engineer_all_features(
    df: pd.DataFrame,
    crop: str = "wheat",
    soil_type: str = "loam"
) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    
    df = rainfall_features(df)
    df = et_features(df)
    df = soil_moisture_features(df)
    df = phenology_features(df, crop=crop)
    df = soil_type_features(df, soil_type=soil_type)
    df = water_balance_features(df)
    
    # Drop rows with too many NaNs
    thresh = len(df.columns) * 0.5
    df = df.dropna(thresh=int(thresh))
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns for modeling."""
    exclude = {
        "date", "datetime", "timestamp", "state", "district", "year", 
        "irrigation_needed", "target", "label"
    }
    
    features = []
    for col in df.columns:
        if col.lower() in exclude:
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            features.append(col)
    
    return features
