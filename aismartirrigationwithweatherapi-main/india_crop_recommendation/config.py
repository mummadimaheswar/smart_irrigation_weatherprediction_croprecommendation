"""
India Crop Recommendation System - Configuration
Scope: All Indian States, 2015-2024
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# GEOGRAPHY
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli",
    "Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

# State centroids (lat, lon) for API calls
STATE_COORDS: Dict[str, tuple] = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Uttar Pradesh": (26.8467, 80.9462),
    "West Bengal": (22.9868, 87.8550),
    "Odisha": (20.9517, 85.0985),
    "Jharkhand": (23.6102, 85.2799),
    "Chhattisgarh": (21.2787, 81.8661),
    "Uttarakhand": (30.0668, 79.0193),
    "Himachal Pradesh": (31.1048, 77.1734),
}

YEARS = list(range(2015, 2025))  # 2015-2024

# ─────────────────────────────────────────────────────────────────────────────
# CROPS
# ─────────────────────────────────────────────────────────────────────────────

CROPS = [
    # Cereals
    "rice", "wheat", "maize", "bajra", "jowar", "ragi", "barley",
    # Pulses
    "chickpea", "pigeonpea", "lentil", "greengram", "blackgram",
    # Oilseeds
    "groundnut", "mustard", "soybean", "sunflower", "sesame",
    # Cash crops
    "cotton", "sugarcane", "jute", "tobacco",
    # Fruits/Vegetables
    "potato", "onion", "tomato", "banana", "mango"
]

CROP_SEASONS = {
    "kharif": ["rice", "maize", "bajra", "jowar", "cotton", "groundnut", "soybean", "pigeonpea"],
    "rabi": ["wheat", "barley", "chickpea", "lentil", "mustard", "potato", "onion"],
    "zaid": ["greengram", "blackgram", "sunflower", "vegetables"]
}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# S3-style paths (local simulation)
S3_BUCKET = "india-crop-data"
S3_RAW = f"s3://{S3_BUCKET}/raw"
S3_CURATED = f"s3://{S3_BUCKET}/curated"
S3_PROCESSED = f"s3://{S3_BUCKET}/processed"

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS (from environment)
# ─────────────────────────────────────────────────────────────────────────────

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY", "")
NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "india_crops")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

POSTGRES = PostgresConfig()

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    name: str
    url: str
    data_type: str
    access_method: str
    format: str
    rate_limit: Optional[str] = None
    auth: Optional[str] = None
    notes: str = ""

DATA_SOURCES: List[DataSource] = [
    # Soil Moisture
    DataSource("NASA SMAP", "https://nsidc.org/data/smap", "soil_moisture", 
               "API/bulk", "HDF5", "100 req/hr", "NASA Earthdata", "9km resolution, L3 daily"),
    DataSource("ESA SMOS", "https://smos-diss.eo.esa.int", "soil_moisture",
               "FTP bulk", "NetCDF", None, "ESA account", "Global 40km"),
    DataSource("ISRO Bhuvan", "https://bhuvan.nrsc.gov.in", "soil_moisture",
               "WMS/download", "GeoTIFF", None, "Free registration", "India-specific"),
    
    # Weather
    DataSource("OpenWeatherMap", "https://api.openweathermap.org", "weather",
               "REST API", "JSON", "60 req/min free", "API key", "Historical + forecast"),
    DataSource("VisualCrossing", "https://www.visualcrossing.com", "weather",
               "REST API", "JSON/CSV", "1000 req/day free", "API key", "Good historical"),
    DataSource("IMD", "https://mausam.imd.gov.in", "weather",
               "Manual/API", "CSV/XLS", None, "Request access", "Official India Met"),
    
    # Crop Data
    DataSource("data.gov.in", "https://data.gov.in", "crop_statistics",
               "API/download", "CSV/JSON", "3 req/sec", "API key", "Official govt data"),
    DataSource("ICRISAT", "http://data.icrisat.org", "crop_statistics",
               "Download", "CSV", None, "Free", "District-level crop data"),
    DataSource("FAO", "https://www.fao.org/faostat", "crop_statistics",
               "API/download", "CSV", None, "Free", "Country-level, reliable"),
    
    # Satellite/NDVI
    DataSource("MODIS", "https://modis.gsfc.nasa.gov", "ndvi",
               "AppEEARS API", "GeoTIFF", None, "NASA Earthdata", "250m-1km, 16-day"),
    DataSource("Sentinel-2", "https://scihub.copernicus.eu", "ndvi",
               "API", "SAFE/GeoTIFF", None, "ESA account", "10m resolution"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_COLUMNS = {
    # Time
    "date": "DATE",
    "year": "INTEGER",
    "month": "INTEGER",
    "season": "VARCHAR(10)",  # kharif/rabi/zaid
    
    # Geography
    "state": "VARCHAR(50)",
    "district": "VARCHAR(100)",
    "lat": "DECIMAL(9,6)",
    "lon": "DECIMAL(9,6)",
    
    # Soil Moisture
    "soil_moisture_pct": "DECIMAL(5,2)",  # 0-100%
    "soil_moisture_source": "VARCHAR(20)",  # sensor/satellite
    
    # Weather
    "temp_min_c": "DECIMAL(5,2)",
    "temp_max_c": "DECIMAL(5,2)",
    "temp_mean_c": "DECIMAL(5,2)",
    "precip_mm": "DECIMAL(8,2)",
    "humidity_pct": "DECIMAL(5,2)",
    "wind_speed_ms": "DECIMAL(5,2)",
    
    # Crop
    "crop": "VARCHAR(50)",
    "area_ha": "DECIMAL(12,2)",
    "yield_kg_per_ha": "DECIMAL(10,2)",
    "production_tonnes": "DECIMAL(14,2)",
    
    # Derived
    "ndvi": "DECIMAL(5,4)",  # -1 to 1
    "spi": "DECIMAL(5,2)",   # Standardized Precip Index
    "pet_mm": "DECIMAL(8,2)",  # Potential ET
    "water_deficit_mm": "DECIMAL(8,2)",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    target_col: str = "crop"
    feature_cols: List[str] = field(default_factory=lambda: [
        "soil_moisture_pct", "temp_mean_c", "precip_mm", "humidity_pct",
        "ndvi", "water_deficit_mm", "month", "lat", "lon"
    ])
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

MODEL_CONFIG = ModelConfig()

# ─────────────────────────────────────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 4
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

API_CONFIG = APIConfig()
