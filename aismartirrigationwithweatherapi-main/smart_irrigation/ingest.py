"""Multi-source data ingestion: weather APIs, government data, satellite, sensors."""
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import requests
import pandas as pd

from .config import SOURCES, RAW_DIR, STATES, DISTRICTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# WEATHER API CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

class OpenWeatherClient:
    """OpenWeatherMap One Call API 3.0 client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.base_url = SOURCES.openweather_url
    
    def fetch_forecast(self, lat: float, lon: float, exclude: str = "minutely") -> dict:
        """Fetch 48h hourly + 8 day daily forecast."""
        if not self.api_key:
            log.warning("No OpenWeather API key")
            return {}
        
        resp = requests.get(self.base_url, params={
            "lat": lat, "lon": lon, "appid": self.api_key,
            "units": "metric", "exclude": exclude
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def fetch_historical(self, lat: float, lon: float, dt: int) -> dict:
        """Fetch historical data for a specific timestamp."""
        if not self.api_key:
            return {}
        
        url = f"{self.base_url}/timemachine"
        resp = requests.get(url, params={
            "lat": lat, "lon": lon, "dt": dt,
            "appid": self.api_key, "units": "metric"
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()


class VisualCrossingClient:
    """Visual Crossing Weather API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VISUALCROSSING_API_KEY")
        self.base_url = SOURCES.visualcrossing_url
    
    def fetch_range(self, lat: float, lon: float, start: str, end: str) -> dict:
        """Fetch weather data for date range (YYYY-MM-DD format)."""
        if not self.api_key:
            log.warning("No VisualCrossing API key")
            return {}
        
        location = f"{lat},{lon}"
        url = f"{self.base_url}/{location}/{start}/{end}"
        resp = requests.get(url, params={
            "key": self.api_key, "unitGroup": "metric",
            "include": "days,hours", "contentType": "json"
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# GOVERNMENT DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

class SoilMoistureLoader:
    """Load soil moisture data from CSV/government sources."""
    
    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """Load soil moisture CSV with flexible column detection."""
        df = pd.read_csv(path)
        
        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        
        # Find date column
        date_cols = [c for c in df.columns if "date" in c or "time" in c]
        if date_cols:
            df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        
        # Find moisture column
        moisture_cols = [c for c in df.columns if "moisture" in c or "vwc" in c]
        if moisture_cols:
            df["soil_moisture"] = pd.to_numeric(df[moisture_cols[0]], errors="coerce")
            if df["soil_moisture"].max() > 1.5:
                df["soil_moisture"] /= 100.0
        
        return df
    
    @staticmethod
    def load_directory(path: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
        """Load all CSVs from directory and concatenate."""
        path = Path(path)
        frames = []
        for f in path.glob(pattern):
            try:
                frames.append(SoilMoistureLoader.load_csv(f))
            except Exception as e:
                log.warning(f"Failed to load {f}: {e}")
        
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class CropStatisticsLoader:
    """Load crop statistics from government portals."""
    
    @staticmethod
    def parse_agmarknet(html_or_json: str) -> pd.DataFrame:
        """Parse Agmarknet price/area data."""
        # Placeholder - actual implementation depends on data format
        return pd.DataFrame()
    
    @staticmethod
    def load_crop_calendar(state: str) -> dict:
        """Get sowing/harvesting windows by state and crop."""
        # State-specific crop calendars
        calendars = {
            "PUNJAB": {
                "wheat": {"sowing": (10, 11), "harvest": (3, 4)},
                "rice": {"sowing": (5, 6), "harvest": (10, 11)},
            },
            "MAHARASHTRA": {
                "cotton": {"sowing": (5, 6), "harvest": (11, 12)},
                "soybean": {"sowing": (6, 7), "harvest": (10, 11)},
            },
        }
        return calendars.get(state, {})


# ─────────────────────────────────────────────────────────────────────────────
# SATELLITE DATA
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteDataClient:
    """Fetch NDVI and other satellite-derived indices."""
    
    def __init__(self, modis_token: Optional[str] = None):
        self.modis_url = SOURCES.modis_ndvi
        self.token = modis_token or os.getenv("MODIS_TOKEN")
    
    def fetch_ndvi(self, lat: float, lon: float, start: str, end: str, product: str = "MOD13Q1") -> pd.DataFrame:
        """Fetch MODIS NDVI time series."""
        if not self.token:
            log.warning("No MODIS token - returning empty")
            return pd.DataFrame()
        
        url = f"{self.modis_url}/{product}/subset"
        params = {
            "latitude": lat, "longitude": lon,
            "startDate": start, "endDate": end,
            "kmAboveBelow": 0, "kmLeftRight": 0
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            records = []
            for subset in data.get("subset", []):
                records.append({
                    "date": subset.get("calendar_date"),
                    "ndvi": subset.get("data", [0])[0] * 0.0001,  # Scale factor
                })
            return pd.DataFrame(records)
        except Exception as e:
            log.error(f"MODIS fetch failed: {e}")
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED INGESTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoPoint:
    state: str
    district: str
    lat: float
    lon: float

# Major district coordinates (approximate centroids)
GEO_COORDS = {
    ("ASSAM", "KAMRUP"): (26.14, 91.67),
    ("ASSAM", "NAGAON"): (26.35, 92.68),
    ("PUNJAB", "LUDHIANA"): (30.90, 75.85),
    ("PUNJAB", "AMRITSAR"): (31.63, 74.87),
    ("MAHARASHTRA", "PUNE"): (18.52, 73.86),
    ("MAHARASHTRA", "NASHIK"): (19.99, 73.79),
    ("KARNATAKA", "BELGAUM"): (15.85, 74.50),
    ("KARNATAKA", "MYSORE"): (12.30, 76.64),
    ("UTTAR PRADESH", "LUCKNOW"): (26.85, 80.95),
    ("UTTAR PRADESH", "VARANASI"): (25.32, 82.99),
}


class DataIngester:
    """Unified data ingestion from all sources."""
    
    def __init__(self):
        self.weather = OpenWeatherClient()
        self.visualcrossing = VisualCrossingClient()
        self.satellite = SatelliteDataClient()
        self.soil = SoilMoistureLoader()
        self.crop_stats = CropStatisticsLoader()
        
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    def ingest_weather(self, state: str, district: str) -> pd.DataFrame:
        """Fetch and store weather data for location."""
        coords = GEO_COORDS.get((state.upper(), district.upper()))
        if not coords:
            log.warning(f"No coordinates for {state}/{district}")
            return pd.DataFrame()
        
        lat, lon = coords
        
        # Try OpenWeather first
        data = self.weather.fetch_forecast(lat, lon)
        if data and "hourly" in data:
            records = []
            for h in data["hourly"]:
                records.append({
                    "timestamp": datetime.utcfromtimestamp(h["dt"]),
                    "temp": h.get("temp"),
                    "humidity": h.get("humidity"),
                    "pressure": h.get("pressure"),
                    "wind_speed": h.get("wind_speed"),
                    "clouds": h.get("clouds"),
                    "rain_1h": h.get("rain", {}).get("1h", 0),
                    "state": state,
                    "district": district,
                    "source": "openweather",
                })
            
            df = pd.DataFrame(records)
            self._save_raw(df, f"weather_{state}_{district}")
            return df
        
        return pd.DataFrame()
    
    def ingest_soil_moisture(self, path: str | Path) -> pd.DataFrame:
        """Load soil moisture from CSV files."""
        df = self.soil.load_csv(path)
        if not df.empty:
            self._save_raw(df, f"soil_{Path(path).stem}")
        return df
    
    def ingest_ndvi(self, state: str, district: str, start: str, end: str) -> pd.DataFrame:
        """Fetch NDVI for location and date range."""
        coords = GEO_COORDS.get((state.upper(), district.upper()))
        if not coords:
            return pd.DataFrame()
        
        lat, lon = coords
        df = self.satellite.fetch_ndvi(lat, lon, start, end)
        if not df.empty:
            df["state"] = state
            df["district"] = district
            self._save_raw(df, f"ndvi_{state}_{district}")
        return df
    
    def ingest_all(self, soil_path: Optional[str] = None) -> dict:
        """Run full ingestion pipeline."""
        results = {"weather": [], "soil": None, "ndvi": []}
        
        # Weather for all configured locations
        for (state, district), coords in GEO_COORDS.items():
            if state in STATES:
                try:
                    df = self.ingest_weather(state, district)
                    if not df.empty:
                        results["weather"].append(df)
                except Exception as e:
                    log.error(f"Weather ingest failed for {state}/{district}: {e}")
        
        # Soil moisture
        if soil_path:
            results["soil"] = self.ingest_soil_moisture(soil_path)
        
        # NDVI (last 90 days)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for (state, district), coords in GEO_COORDS.items():
            if state in STATES:
                try:
                    df = self.ingest_ndvi(state, district, start, end)
                    if not df.empty:
                        results["ndvi"].append(df)
                except Exception as e:
                    log.error(f"NDVI ingest failed for {state}/{district}: {e}")
        
        return results
    
    def _save_raw(self, df: pd.DataFrame, name: str):
        """Save raw data with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RAW_DIR / f"{name}_{ts}.parquet"
        df.to_parquet(path, index=False)
        log.info(f"Saved {len(df)} rows to {path}")
