"""Project configuration: scope, outputs, and evaluation metrics."""
from pathlib import Path
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# SCOPE
# ─────────────────────────────────────────────────────────────────────────────
STATES = ["ASSAM", "PUNJAB", "MAHARASHTRA", "KARNATAKA", "UTTAR PRADESH"]
YEARS = list(range(2018, 2026))
CROPS = ["rice", "wheat", "maize", "cotton", "sugarcane", "soybean"]

# District mapping (state -> list of priority districts)
DISTRICTS = {
    "ASSAM": ["KAMRUP", "NAGAON", "SONITPUR", "JORHAT", "DIBRUGARH"],
    "PUNJAB": ["LUDHIANA", "AMRITSAR", "PATIALA", "JALANDHAR", "BATHINDA"],
    "MAHARASHTRA": ["PUNE", "NASHIK", "NAGPUR", "AHMEDNAGAR", "SOLAPUR"],
    "KARNATAKA": ["BELGAUM", "MYSORE", "DHARWAD", "SHIMOGA", "TUMKUR"],
    "UTTAR PRADESH": ["LUCKNOW", "ALLAHABAD", "VARANASI", "AGRA", "MEERUT"],
}

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OutputConfig:
    format: str = "parquet"  # csv, parquet, postgres
    postgres_uri: str = ""   # postgresql://user:pass@host:5432/db
    compress: bool = True
    partition_by: list = field(default_factory=lambda: ["state", "year"])

OUTPUT = OutputConfig()

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricsConfig:
    # Classification metrics for irrigation decision
    primary: str = "f1_weighted"
    secondary: list = field(default_factory=lambda: ["precision", "recall", "accuracy"])
    
    # Regression metrics for soil moisture prediction
    regression: list = field(default_factory=lambda: ["rmse", "mae", "r2"])
    
    # Business metrics
    water_savings_pct: float = 0.0  # Target % reduction in water use
    yield_impact_pct: float = 0.0   # Yield change from recommendations

METRICS = MetricsConfig()

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataSourceConfig:
    # Weather APIs
    openweather_url: str = "https://api.openweathermap.org/data/3.0/onecall"
    visualcrossing_url: str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    imd_url: str = "https://mausam.imd.gov.in/api"  # If available
    
    # Government portals
    agmarknet_url: str = "https://agmarknet.gov.in"
    soil_health_url: str = "https://soilhealth.dac.gov.in"
    
    # Satellite data
    modis_ndvi: str = "https://modis.ornl.gov/rst/api/v1"
    sentinel_hub: str = "https://services.sentinel-hub.com"

SOURCES = DataSourceConfig()

# ─────────────────────────────────────────────────────────────────────────────
# CROP PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CROP_PARAMS = {
    "rice": {
        "kc_stages": [0.5, 1.05, 1.2, 0.9],  # initial, dev, mid, late
        "root_depth_cm": 40,
        "vwc_critical": 0.30,
        "vwc_optimal": 0.40,
        "growing_days": 120,
        "sowing_months": [6, 7],  # June-July (Kharif)
    },
    "wheat": {
        "kc_stages": [0.3, 0.95, 1.15, 0.4],
        "root_depth_cm": 150,
        "vwc_critical": 0.18,
        "vwc_optimal": 0.28,
        "growing_days": 140,
        "sowing_months": [10, 11],  # Oct-Nov (Rabi)
    },
    "maize": {
        "kc_stages": [0.3, 0.9, 1.2, 0.6],
        "root_depth_cm": 180,
        "vwc_critical": 0.16,
        "vwc_optimal": 0.26,
        "growing_days": 100,
        "sowing_months": [6, 7, 1, 2],  # Kharif + Rabi
    },
    "cotton": {
        "kc_stages": [0.35, 0.9, 1.15, 0.7],
        "root_depth_cm": 150,
        "vwc_critical": 0.15,
        "vwc_optimal": 0.25,
        "growing_days": 180,
        "sowing_months": [4, 5],
    },
    "sugarcane": {
        "kc_stages": [0.4, 1.0, 1.25, 0.75],
        "root_depth_cm": 200,
        "vwc_critical": 0.20,
        "vwc_optimal": 0.32,
        "growing_days": 365,
        "sowing_months": [1, 2, 10],
    },
    "soybean": {
        "kc_stages": [0.4, 0.8, 1.15, 0.5],
        "root_depth_cm": 100,
        "vwc_critical": 0.18,
        "vwc_optimal": 0.28,
        "growing_days": 100,
        "sowing_months": [6, 7],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SOIL TYPES
# ─────────────────────────────────────────────────────────────────────────────
SOIL_TYPES = {
    "clay": {"awc": 0.20, "infiltration_mm_hr": 5, "drainage": "poor"},
    "clay_loam": {"awc": 0.18, "infiltration_mm_hr": 10, "drainage": "moderate"},
    "loam": {"awc": 0.17, "infiltration_mm_hr": 15, "drainage": "good"},
    "sandy_loam": {"awc": 0.12, "infiltration_mm_hr": 25, "drainage": "good"},
    "sand": {"awc": 0.08, "infiltration_mm_hr": 50, "drainage": "excessive"},
    "silt_loam": {"awc": 0.19, "infiltration_mm_hr": 12, "drainage": "moderate"},
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    baseline: str = "rules"  # rules, decision_tree
    ensemble: str = "xgboost"  # xgboost, lightgbm, random_forest
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

MODEL = ModelConfig()

# ─────────────────────────────────────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit: int = 100  # requests per minute

API = APIConfig()

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SchedulerConfig:
    weather_interval_hours: int = 6
    etl_interval_hours: int = 24
    model_retrain_days: int = 30

SCHEDULER = SchedulerConfig()
