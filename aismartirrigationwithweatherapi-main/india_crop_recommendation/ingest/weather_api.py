"""
Weather API Ingestion Module
India Crop Recommendation System

PROMPT 4: Production-ready weather API integration with:
- Configurable API key via environment variables
- fetch_historical_weather() and fetch_forecast()
- Exponential backoff for rate limiting
- Normalized pandas DataFrame output
- CLI for Parquet export
"""
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Keys from environment
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY", "")

# API Endpoints
OPENWEATHER_HISTORICAL_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
VISUALCROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

# Rate limits
OPENWEATHER_RATE_LIMIT = 60  # requests per minute (free tier)
VISUALCROSSING_RATE_LIMIT = 1000  # records per day (free tier)

# State coordinates (centroids)
STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# RETRY/BACKOFF UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def exponential_backoff(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate delay with exponential backoff."""
    delay = min(base * (2 ** attempt), max_delay)
    return delay


def request_with_retry(url: str, params: dict, max_retries: int = 5) -> Optional[dict]:
    """Make HTTP request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                delay = exponential_backoff(attempt)
                log.warning(f"Rate limited. Waiting {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
            elif response.status_code == 401:
                log.error("Invalid API key")
                return None
            else:
                log.warning(f"HTTP {response.status_code}: {response.text[:100]}")
                time.sleep(exponential_backoff(attempt))
                
        except requests.exceptions.Timeout:
            log.warning(f"Timeout (attempt {attempt+1})")
            time.sleep(exponential_backoff(attempt))
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed: {e}")
            time.sleep(exponential_backoff(attempt))
    
    log.error(f"All {max_retries} attempts failed for {url}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# OPENWEATHERMAP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_historical_weather_openweather(
    state: str,
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical weather from OpenWeatherMap.
    
    Note: Free tier only supports 5 days back. Paid plan required for more.
    """
    api_key = api_key or OPENWEATHER_API_KEY
    if not api_key:
        log.error("OPENWEATHER_API_KEY not set")
        return pd.DataFrame()
    
    if state not in STATE_COORDS:
        log.error(f"Unknown state: {state}")
        return pd.DataFrame()
    
    lat, lon = STATE_COORDS[state]
    records = []
    current_date = start_date
    
    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        params = {
            "lat": lat,
            "lon": lon,
            "dt": timestamp,
            "appid": api_key,
            "units": "metric"
        }
        
        data = request_with_retry(OPENWEATHER_HISTORICAL_URL, params)
        
        if data and "data" in data:
            for hour in data["data"]:
                records.append({
                    "date": current_date.date(),
                    "state": state,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": datetime.fromtimestamp(hour.get("dt", 0), tz=timezone.utc),
                    "temp_c": hour.get("temp"),
                    "feels_like_c": hour.get("feels_like"),
                    "humidity_pct": hour.get("humidity"),
                    "pressure_hpa": hour.get("pressure"),
                    "wind_m_s": hour.get("wind_speed"),
                    "clouds_pct": hour.get("clouds"),
                    "weather": hour.get("weather", [{}])[0].get("main", ""),
                })
        
        # Rate limiting
        time.sleep(1.1)  # ~60 req/min
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = _aggregate_to_daily(df, state, lat, lon)
    
    return df


def fetch_forecast_openweather(
    state: str,
    days: int = 5,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """Fetch weather forecast (5-day, 3-hour intervals)."""
    api_key = api_key or OPENWEATHER_API_KEY
    if not api_key:
        log.error("OPENWEATHER_API_KEY not set")
        return pd.DataFrame()
    
    if state not in STATE_COORDS:
        log.error(f"Unknown state: {state}")
        return pd.DataFrame()
    
    lat, lon = STATE_COORDS[state]
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40
    }
    
    data = request_with_retry(OPENWEATHER_FORECAST_URL, params)
    
    if not data or "list" not in data:
        return pd.DataFrame()
    
    records = []
    for item in data["list"]:
        records.append({
            "date": datetime.fromtimestamp(item["dt"], tz=timezone.utc).date(),
            "state": state,
            "lat": lat,
            "lon": lon,
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "temp_c": item["main"]["temp"],
            "temp_min_c": item["main"]["temp_min"],
            "temp_max_c": item["main"]["temp_max"],
            "humidity_pct": item["main"]["humidity"],
            "pressure_hpa": item["main"]["pressure"],
            "wind_m_s": item["wind"]["speed"],
            "clouds_pct": item["clouds"]["all"],
            "precip_mm": item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0),
            "weather": item["weather"][0]["main"] if item["weather"] else "",
        })
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = _aggregate_to_daily(df, state, lat, lon)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALCROSSING FUNCTIONS (Better for historical data)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_historical_weather_visualcrossing(
    state: str,
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical weather from VisualCrossing.
    Better for historical data - 50+ years available.
    """
    api_key = api_key or VISUALCROSSING_API_KEY
    if not api_key:
        log.warning("VISUALCROSSING_API_KEY not set, using synthetic data")
        return _generate_synthetic_historical(state, start_date, end_date)
    
    if state not in STATE_COORDS:
        log.error(f"Unknown state: {state}")
        return pd.DataFrame()
    
    lat, lon = STATE_COORDS[state]
    location = f"{lat},{lon}"
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = f"{VISUALCROSSING_URL}/{location}/{start_str}/{end_str}"
    params = {
        "key": api_key,
        "unitGroup": "metric",
        "include": "days",
        "contentType": "json"
    }
    
    data = request_with_retry(url, params)
    
    if not data or "days" not in data:
        return pd.DataFrame()
    
    records = []
    for day in data["days"]:
        records.append({
            "date": datetime.strptime(day["datetime"], "%Y-%m-%d").date(),
            "state": state,
            "lat": lat,
            "lon": lon,
            "t_min_c": day.get("tempmin"),
            "t_max_c": day.get("tempmax"),
            "t_mean_c": day.get("temp"),
            "precip_mm": day.get("precip", 0) or 0,
            "humidity_pct": day.get("humidity"),
            "wind_m_s": day.get("windspeed", 0) / 3.6 if day.get("windspeed") else None,  # km/h to m/s
            "solar_rad_wm2": day.get("solarradiation"),
            "uv_index": day.get("uvindex"),
            "conditions": day.get("conditions", ""),
        })
    
    return pd.DataFrame(records)


def fetch_forecast_visualcrossing(
    state: str,
    days: int = 15,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """Fetch weather forecast from VisualCrossing (up to 15 days)."""
    api_key = api_key or VISUALCROSSING_API_KEY
    if not api_key:
        log.warning("VISUALCROSSING_API_KEY not set, using synthetic data")
        return _generate_synthetic_forecast(state, days)
    
    if state not in STATE_COORDS:
        log.error(f"Unknown state: {state}")
        return pd.DataFrame()
    
    lat, lon = STATE_COORDS[state]
    location = f"{lat},{lon}"
    
    url = f"{VISUALCROSSING_URL}/{location}/next{days}days"
    params = {
        "key": api_key,
        "unitGroup": "metric",
        "include": "days",
        "contentType": "json"
    }
    
    data = request_with_retry(url, params)
    
    if not data or "days" not in data:
        return pd.DataFrame()
    
    records = []
    for day in data["days"]:
        records.append({
            "date": datetime.strptime(day["datetime"], "%Y-%m-%d").date(),
            "state": state,
            "lat": lat,
            "lon": lon,
            "t_min_c": day.get("tempmin"),
            "t_max_c": day.get("tempmax"),
            "t_mean_c": day.get("temp"),
            "precip_mm": day.get("precip", 0) or 0,
            "humidity_pct": day.get("humidity"),
            "wind_m_s": day.get("windspeed", 0) / 3.6 if day.get("windspeed") else None,
            "precip_prob": day.get("precipprob", 0),
            "conditions": day.get("conditions", ""),
        })
    
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_historical_weather(
    state: str,
    start_date: datetime,
    end_date: datetime,
    source: str = "visualcrossing"
) -> pd.DataFrame:
    """
    Unified interface for fetching historical weather.
    
    Args:
        state: Indian state name
        start_date: Start date
        end_date: End date
        source: "openweather" or "visualcrossing"
    
    Returns:
        DataFrame with columns: date, state, lat, lon, t_min_c, t_max_c, 
        t_mean_c, precip_mm, humidity_pct, wind_m_s
    """
    if source == "openweather":
        return fetch_historical_weather_openweather(state, start_date, end_date)
    else:
        return fetch_historical_weather_visualcrossing(state, start_date, end_date)


def fetch_forecast(
    state: str,
    days: int = 7,
    source: str = "visualcrossing"
) -> pd.DataFrame:
    """
    Unified interface for fetching weather forecast.
    
    Args:
        state: Indian state name
        days: Number of days to forecast
        source: "openweather" or "visualcrossing"
    
    Returns:
        DataFrame with forecast data
    """
    if source == "openweather":
        return fetch_forecast_openweather(state, min(days, 5))
    else:
        return fetch_forecast_visualcrossing(state, min(days, 15))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_to_daily(df: pd.DataFrame, state: str, lat: float, lon: float) -> pd.DataFrame:
    """Aggregate hourly/3-hourly data to daily."""
    if df.empty:
        return df
    
    daily = df.groupby("date").agg({
        "temp_c": ["min", "max", "mean"] if "temp_c" in df.columns else [],
        "humidity_pct": "mean",
        "wind_m_s": "mean",
        "precip_mm": "sum" if "precip_mm" in df.columns else [],
    }).reset_index()
    
    # Flatten column names
    daily.columns = ["_".join(col).strip("_") for col in daily.columns]
    
    daily["state"] = state
    daily["lat"] = lat
    daily["lon"] = lon
    
    # Rename columns
    rename_map = {
        "temp_c_min": "t_min_c",
        "temp_c_max": "t_max_c",
        "temp_c_mean": "t_mean_c",
        "humidity_pct_mean": "humidity_pct",
        "wind_m_s_mean": "wind_m_s",
        "precip_mm_sum": "precip_mm",
    }
    daily = daily.rename(columns=rename_map)
    
    return daily


def _generate_synthetic_historical(state: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate synthetic historical data when API not available."""
    import numpy as np
    
    if state not in STATE_COORDS:
        return pd.DataFrame()
    
    lat, lon = STATE_COORDS[state]
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Base temperature varies by latitude (cooler in north)
    base_temp = 30 - (lat - 15) * 0.5
    
    records = []
    for date in dates:
        # Seasonal variation
        day_of_year = date.dayofyear
        seasonal_offset = 10 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
        
        t_mean = base_temp + seasonal_offset + np.random.normal(0, 2)
        t_min = t_mean - np.random.uniform(4, 8)
        t_max = t_mean + np.random.uniform(4, 8)
        
        # Monsoon effect (June-September)
        is_monsoon = 6 <= date.month <= 9
        precip = np.random.exponential(15) if is_monsoon else np.random.exponential(2)
        humidity = np.random.uniform(70, 95) if is_monsoon else np.random.uniform(40, 70)
        
        records.append({
            "date": date.date(),
            "state": state,
            "lat": lat,
            "lon": lon,
            "t_min_c": round(t_min, 1),
            "t_max_c": round(t_max, 1),
            "t_mean_c": round(t_mean, 1),
            "precip_mm": round(max(0, precip), 1),
            "humidity_pct": round(humidity, 1),
            "wind_m_s": round(np.random.uniform(1, 5), 1),
        })
    
    return pd.DataFrame(records)


def _generate_synthetic_forecast(state: str, days: int) -> pd.DataFrame:
    """Generate synthetic forecast data when API not available."""
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days)
    return _generate_synthetic_historical(state, start, end)


# ═══════════════════════════════════════════════════════════════════════════════
# PARQUET EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_to_parquet(
    df: pd.DataFrame,
    output_dir: str,
    partition_by: List[str] = ["year", "state"]
) -> str:
    """Save DataFrame to partitioned Parquet."""
    if df.empty:
        log.warning("Empty DataFrame, nothing to save")
        return ""
    
    # Add year column if not present
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save partitioned
    df.to_parquet(
        output_path,
        engine="pyarrow",
        partition_cols=partition_by,
        compression="snappy",
        index=False
    )
    
    log.info(f"Saved {len(df)} records to {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for weather data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch weather data for Indian states")
    parser.add_argument("--state", type=str, required=True, help="State name (e.g., Maharashtra)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Forecast days (default: 7)")
    parser.add_argument("--mode", choices=["historical", "forecast"], default="forecast")
    parser.add_argument("--source", choices=["openweather", "visualcrossing"], default="visualcrossing")
    parser.add_argument("--output", type=str, default="./data/curated/weather", help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "historical":
        if not args.start or not args.end:
            parser.error("--start and --end required for historical mode")
        
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        
        log.info(f"Fetching historical weather for {args.state} from {start} to {end}")
        df = fetch_historical_weather(args.state, start, end, args.source)
    else:
        log.info(f"Fetching {args.days}-day forecast for {args.state}")
        df = fetch_forecast(args.state, args.days, args.source)
    
    if df.empty:
        log.error("No data retrieved")
        return
    
    print(f"\nRetrieved {len(df)} records:")
    print(df.head(10).to_string())
    
    # Save to Parquet
    save_to_parquet(df, args.output)


if __name__ == "__main__":
    main()
