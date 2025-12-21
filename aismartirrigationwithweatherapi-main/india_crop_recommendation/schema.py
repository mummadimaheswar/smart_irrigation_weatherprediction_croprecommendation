"""
Schema Design & DDL
India Crop Recommendation System

PROMPT 3: Canonical dataset schema with Parquet partitioning and Postgres DDL
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL SCHEMA DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_DEFINITION = """
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CANONICAL DATASET SCHEMA                                                      │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ COLUMN               │ TYPE            │ UNIT        │ EXAMPLE                                                  │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 📅 TEMPORAL                                                                                                      │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ date                 │ DATE            │ -           │ 2020-06-15                                               │
│ year                 │ INT             │ -           │ 2020                                                     │
│ month                │ INT             │ 1-12        │ 6                                                        │
│ week                 │ INT             │ 1-53        │ 24                                                       │
│ season               │ VARCHAR(10)     │ -           │ kharif                                                   │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 🗺️ GEOGRAPHY                                                                                                     │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ state_code           │ VARCHAR(5)      │ -           │ MH                                                       │
│ state_name           │ VARCHAR(50)     │ -           │ Maharashtra                                              │
│ district_code        │ VARCHAR(10)     │ -           │ MH001                                                    │
│ district_name        │ VARCHAR(100)    │ -           │ Pune                                                     │
│ latitude             │ DECIMAL(9,6)    │ degrees     │ 18.520430                                                │
│ longitude            │ DECIMAL(9,6)    │ degrees     │ 73.856744                                                │
│ elevation_m          │ DECIMAL(7,2)    │ meters      │ 560.00                                                   │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 🌡️ WEATHER                                                                                                       │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ temp_min_c           │ DECIMAL(5,2)    │ °C          │ 24.50                                                    │
│ temp_max_c           │ DECIMAL(5,2)    │ °C          │ 35.20                                                    │
│ temp_mean_c          │ DECIMAL(5,2)    │ °C          │ 29.85                                                    │
│ precip_mm            │ DECIMAL(8,2)    │ mm          │ 12.50                                                    │
│ humidity_pct         │ DECIMAL(5,2)    │ %           │ 78.50                                                    │
│ wind_speed_ms        │ DECIMAL(5,2)    │ m/s         │ 3.20                                                     │
│ solar_rad_wm2        │ DECIMAL(8,2)    │ W/m²        │ 245.00                                                   │
│ et_ref_mm            │ DECIMAL(6,2)    │ mm/day      │ 5.20                                                     │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 💧 SOIL MOISTURE                                                                                                 │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ soil_moisture_pct    │ DECIMAL(5,2)    │ % vol       │ 32.50                                                    │
│ soil_moisture_source │ VARCHAR(20)     │ -           │ SMAP                                                     │
│ soil_temp_c          │ DECIMAL(5,2)    │ °C          │ 28.00                                                    │
│ soil_type            │ VARCHAR(30)     │ -           │ clay_loam                                                │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 🌱 VEGETATION                                                                                                    │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ ndvi                 │ DECIMAL(5,4)    │ -1 to 1     │ 0.6500                                                   │
│ evi                  │ DECIMAL(5,4)    │ -1 to 1     │ 0.4200                                                   │
│ lai                  │ DECIMAL(5,2)    │ m²/m²       │ 3.50                                                     │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 🌾 CROP DATA                                                                                                     │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ crop_name            │ VARCHAR(50)     │ -           │ rice                                                     │
│ crop_season          │ VARCHAR(10)     │ -           │ kharif                                                   │
│ area_ha              │ DECIMAL(12,2)   │ hectares    │ 15000.00                                                 │
│ yield_kg_per_ha      │ DECIMAL(10,2)   │ kg/ha       │ 2850.00                                                  │
│ production_tonnes    │ DECIMAL(14,2)   │ tonnes      │ 42750.00                                                 │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 📊 DERIVED FEATURES                                                                                              │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ water_deficit_mm     │ DECIMAL(8,2)    │ mm          │ -7.30                                                    │
│ gdd_base10           │ DECIMAL(8,2)    │ degree-days │ 19.85                                                    │
│ spi_30d              │ DECIMAL(5,2)    │ -           │ -0.45                                                    │
│ precip_7d_sum        │ DECIMAL(8,2)    │ mm          │ 45.00                                                    │
│ temp_7d_mean         │ DECIMAL(5,2)    │ °C          │ 28.50                                                    │
├──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┤
│ 🔖 METADATA                                                                                                      │
├──────────────────────┬─────────────────┬─────────────┬──────────────────────────────────────────────────────────┤
│ created_at           │ TIMESTAMP       │ UTC         │ 2024-01-15 10:30:00                                      │
│ updated_at           │ TIMESTAMP       │ UTC         │ 2024-01-15 10:30:00                                      │
│ data_quality_flag    │ VARCHAR(10)     │ -           │ verified                                                 │
│ source_id            │ VARCHAR(50)     │ -           │ visualcrossing_2020                                      │
└──────────────────────┴─────────────────┴─────────────┴──────────────────────────────────────────────────────────┘

SAMPLE ROW:
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
date=2020-06-15, year=2020, month=6, week=24, season=kharif,
state_code=MH, state_name=Maharashtra, district_code=MH001, district_name=Pune, 
latitude=18.520430, longitude=73.856744, elevation_m=560.00,
temp_min_c=24.50, temp_max_c=35.20, temp_mean_c=29.85, precip_mm=12.50, humidity_pct=78.50,
wind_speed_ms=3.20, solar_rad_wm2=245.00, et_ref_mm=5.20,
soil_moisture_pct=32.50, soil_moisture_source=SMAP, soil_temp_c=28.00, soil_type=clay_loam,
ndvi=0.6500, evi=0.4200, lai=3.50,
crop_name=rice, crop_season=kharif, area_ha=15000.00, yield_kg_per_ha=2850.00, production_tonnes=42750.00,
water_deficit_mm=-7.30, gdd_base10=19.85, spi_30d=-0.45, precip_7d_sum=45.00, temp_7d_mean=28.50,
created_at=2024-01-15 10:30:00, updated_at=2024-01-15 10:30:00, data_quality_flag=verified, source_id=vc_2020
"""

# ═══════════════════════════════════════════════════════════════════════════════
# POSTGRES DDL
# ═══════════════════════════════════════════════════════════════════════════════

POSTGRES_DDL = """
-- ============================================================================
-- INDIA CROP RECOMMENDATION SYSTEM - DATABASE SCHEMA
-- ============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- REFERENCE TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS ref_states (
    state_code VARCHAR(5) PRIMARY KEY,
    state_name VARCHAR(50) NOT NULL UNIQUE,
    centroid_lat DECIMAL(9,6),
    centroid_lon DECIMAL(9,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ref_districts (
    district_code VARCHAR(10) PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    state_code VARCHAR(5) REFERENCES ref_states(state_code),
    centroid_lat DECIMAL(9,6),
    centroid_lon DECIMAL(9,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ref_crops (
    crop_id SERIAL PRIMARY KEY,
    crop_name VARCHAR(50) NOT NULL UNIQUE,
    crop_type VARCHAR(30),  -- cereal, pulse, oilseed, cash, fruit, vegetable
    primary_season VARCHAR(10),  -- kharif, rabi, zaid
    water_requirement_mm DECIMAL(8,2),
    optimal_temp_min_c DECIMAL(5,2),
    optimal_temp_max_c DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MAIN CURATED TABLE (PARTITIONED BY YEAR)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crop_data (
    id BIGSERIAL,
    
    -- Temporal
    date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    week INT,
    season VARCHAR(10),
    
    -- Geography
    state_code VARCHAR(5) NOT NULL,
    state_name VARCHAR(50) NOT NULL,
    district_code VARCHAR(10),
    district_name VARCHAR(100),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    elevation_m DECIMAL(7,2),
    
    -- Weather
    temp_min_c DECIMAL(5,2),
    temp_max_c DECIMAL(5,2),
    temp_mean_c DECIMAL(5,2),
    precip_mm DECIMAL(8,2),
    humidity_pct DECIMAL(5,2),
    wind_speed_ms DECIMAL(5,2),
    solar_rad_wm2 DECIMAL(8,2),
    et_ref_mm DECIMAL(6,2),
    
    -- Soil Moisture
    soil_moisture_pct DECIMAL(5,2),
    soil_moisture_source VARCHAR(20),
    soil_temp_c DECIMAL(5,2),
    soil_type VARCHAR(30),
    
    -- Vegetation
    ndvi DECIMAL(5,4),
    evi DECIMAL(5,4),
    lai DECIMAL(5,2),
    
    -- Crop
    crop_name VARCHAR(50),
    crop_season VARCHAR(10),
    area_ha DECIMAL(12,2),
    yield_kg_per_ha DECIMAL(10,2),
    production_tonnes DECIMAL(14,2),
    
    -- Derived
    water_deficit_mm DECIMAL(8,2),
    gdd_base10 DECIMAL(8,2),
    spi_30d DECIMAL(5,2),
    precip_7d_sum DECIMAL(8,2),
    temp_7d_mean DECIMAL(5,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_quality_flag VARCHAR(10) DEFAULT 'raw',
    source_id VARCHAR(50),
    
    PRIMARY KEY (id, year)
) PARTITION BY RANGE (year);

-- Create partitions for each year (2015-2024)
CREATE TABLE crop_data_2015 PARTITION OF crop_data FOR VALUES FROM (2015) TO (2016);
CREATE TABLE crop_data_2016 PARTITION OF crop_data FOR VALUES FROM (2016) TO (2017);
CREATE TABLE crop_data_2017 PARTITION OF crop_data FOR VALUES FROM (2017) TO (2018);
CREATE TABLE crop_data_2018 PARTITION OF crop_data FOR VALUES FROM (2018) TO (2019);
CREATE TABLE crop_data_2019 PARTITION OF crop_data FOR VALUES FROM (2019) TO (2020);
CREATE TABLE crop_data_2020 PARTITION OF crop_data FOR VALUES FROM (2020) TO (2021);
CREATE TABLE crop_data_2021 PARTITION OF crop_data FOR VALUES FROM (2021) TO (2022);
CREATE TABLE crop_data_2022 PARTITION OF crop_data FOR VALUES FROM (2022) TO (2023);
CREATE TABLE crop_data_2023 PARTITION OF crop_data FOR VALUES FROM (2023) TO (2024);
CREATE TABLE crop_data_2024 PARTITION OF crop_data FOR VALUES FROM (2024) TO (2025);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX idx_crop_data_date ON crop_data (date);
CREATE INDEX idx_crop_data_state ON crop_data (state_code);
CREATE INDEX idx_crop_data_district ON crop_data (district_code);
CREATE INDEX idx_crop_data_crop ON crop_data (crop_name);
CREATE INDEX idx_crop_data_state_date ON crop_data (state_code, date);
CREATE INDEX idx_crop_data_geo ON crop_data (latitude, longitude);

-- ============================================================================
-- WEATHER TABLE (DAILY)
-- ============================================================================

CREATE TABLE IF NOT EXISTS weather_daily (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state_code VARCHAR(5) NOT NULL,
    district_code VARCHAR(10),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    
    temp_min_c DECIMAL(5,2),
    temp_max_c DECIMAL(5,2),
    temp_mean_c DECIMAL(5,2),
    precip_mm DECIMAL(8,2),
    humidity_pct DECIMAL(5,2),
    wind_speed_ms DECIMAL(5,2),
    solar_rad_wm2 DECIMAL(8,2),
    
    source VARCHAR(30),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (date, state_code, district_code)
);

-- ============================================================================
-- SOIL MOISTURE TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS soil_moisture (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state_code VARCHAR(5) NOT NULL,
    district_code VARCHAR(10),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    
    soil_moisture_pct DECIMAL(5,2),
    depth_cm INT DEFAULT 5,
    source VARCHAR(20),  -- SMAP, SMOS, sensor
    quality_flag VARCHAR(10),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (date, latitude, longitude, source)
);

-- ============================================================================
-- CROP STATISTICS TABLE (YEARLY)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crop_statistics (
    id BIGSERIAL PRIMARY KEY,
    year INT NOT NULL,
    state_code VARCHAR(5) NOT NULL,
    district_code VARCHAR(10),
    crop_name VARCHAR(50) NOT NULL,
    season VARCHAR(10),
    
    area_ha DECIMAL(12,2),
    yield_kg_per_ha DECIMAL(10,2),
    production_tonnes DECIMAL(14,2),
    
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (year, state_code, district_code, crop_name, season)
);

-- ============================================================================
-- MODEL PREDICTIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id UUID,
    
    -- Input
    state_code VARCHAR(5),
    district_code VARCHAR(10),
    target_date DATE,
    soil_moisture_pct DECIMAL(5,2),
    
    -- Output
    recommended_crop_1 VARCHAR(50),
    confidence_1 DECIMAL(5,4),
    recommended_crop_2 VARCHAR(50),
    confidence_2 DECIMAL(5,4),
    recommended_crop_3 VARCHAR(50),
    confidence_3 DECIMAL(5,4),
    
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_latest_weather AS
SELECT DISTINCT ON (state_code, district_code)
    state_code, district_code, date, temp_mean_c, precip_mm, humidity_pct
FROM weather_daily
ORDER BY state_code, district_code, date DESC;

CREATE OR REPLACE VIEW v_crop_summary AS
SELECT 
    year, state_code, crop_name,
    SUM(area_ha) as total_area_ha,
    AVG(yield_kg_per_ha) as avg_yield,
    SUM(production_tonnes) as total_production
FROM crop_statistics
GROUP BY year, state_code, crop_name;
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FOLDER LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

FOLDER_LAYOUT = """
india_crop_recommendation/
│
├── data/
│   │
│   ├── raw/                              # Original unprocessed data
│   │   ├── weather/
│   │   │   ├── openweathermap/
│   │   │   │   └── year=2020/
│   │   │   │       └── state=MH/
│   │   │   │           └── weather_MH_2020.json
│   │   │   └── visualcrossing/
│   │   │       └── year=2020/
│   │   │           └── state=MH/
│   │   │               └── weather_MH_2020.csv
│   │   │
│   │   ├── soil_moisture/
│   │   │   ├── smap/
│   │   │   │   └── year=2020/
│   │   │   │       └── SPL3SMP_2020_india.h5
│   │   │   ├── sensors/
│   │   │   │   └── year=2020/
│   │   │   │       └── state=MH/
│   │   │   │           └── sensor_data_MH_2020.csv
│   │   │   └── bhuvan/
│   │   │       └── year=2020/
│   │   │           └── bhuvan_sm_2020.tif
│   │   │
│   │   ├── crop_statistics/
│   │   │   ├── data_gov_in/
│   │   │   │   └── crop_stats_2015_2024.json
│   │   │   ├── icrisat/
│   │   │   │   └── icrisat_district_crops.csv
│   │   │   └── ministry/
│   │   │       └── annual_reports/
│   │   │           └── 2020_crop_stats.xlsx
│   │   │
│   │   └── ndvi/
│   │       └── modis/
│   │           └── year=2020/
│   │               └── MOD13Q1_2020_india.tif
│   │
│   ├── curated/                          # Cleaned, deduplicated, validated
│   │   ├── weather/
│   │   │   └── year=2020/
│   │   │       └── state=MH/
│   │   │           └── weather_daily.parquet
│   │   │
│   │   ├── soil_moisture/
│   │   │   └── year=2020/
│   │   │       └── state=MH/
│   │   │           └── soil_moisture_daily.parquet
│   │   │
│   │   ├── crop_statistics/
│   │   │   └── year=2020/
│   │   │       └── crop_stats.parquet
│   │   │
│   │   └── ndvi/
│   │       └── year=2020/
│   │           └── state=MH/
│   │               └── ndvi_16day.parquet
│   │
│   └── processed/                        # Feature-engineered, model-ready
│       ├── features/
│       │   └── year=2020/
│       │       └── state=MH/
│       │           └── features.parquet
│       │
│       ├── training/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       │
│       └── inference/
│           └── latest_features.parquet
│
├── models/
│   ├── lightgbm_v1.0.joblib
│   ├── random_forest_v1.0.joblib
│   ├── feature_importance.csv
│   └── model_metrics.json
│
├── logs/
│   ├── etl/
│   │   └── 2024-01-15_etl.log
│   ├── training/
│   │   └── 2024-01-15_training.log
│   └── api/
│       └── 2024-01-15_api.log
│
└── outputs/
    ├── reports/
    │   └── evaluation_report.html
    └── exports/
        └── crop_data_export.csv

PARQUET PARTITIONING STRATEGY:
─────────────────────────────
- Level 1: year (2015-2024)
- Level 2: state (MH, KA, TN, ...)
- Compression: snappy
- Row groups: ~100MB each

Example path: data/curated/weather/year=2020/state=MH/weather_daily.parquet

S3 PATHS (if using cloud):
──────────────────────────
s3://india-crop-data/raw/weather/openweathermap/year=2020/state=MH/
s3://india-crop-data/curated/weather/year=2020/state=MH/
s3://india-crop-data/processed/features/year=2020/state=MH/
s3://india-crop-models/lightgbm_v1.0.joblib
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PYARROW SCHEMA (for Parquet)
# ═══════════════════════════════════════════════════════════════════════════════

PYARROW_SCHEMA = """
import pyarrow as pa

CROP_DATA_SCHEMA = pa.schema([
    # Temporal
    ('date', pa.date32()),
    ('year', pa.int16()),
    ('month', pa.int8()),
    ('week', pa.int8()),
    ('season', pa.string()),
    
    # Geography
    ('state_code', pa.string()),
    ('state_name', pa.string()),
    ('district_code', pa.string()),
    ('district_name', pa.string()),
    ('latitude', pa.float64()),
    ('longitude', pa.float64()),
    ('elevation_m', pa.float32()),
    
    # Weather
    ('temp_min_c', pa.float32()),
    ('temp_max_c', pa.float32()),
    ('temp_mean_c', pa.float32()),
    ('precip_mm', pa.float32()),
    ('humidity_pct', pa.float32()),
    ('wind_speed_ms', pa.float32()),
    ('solar_rad_wm2', pa.float32()),
    ('et_ref_mm', pa.float32()),
    
    # Soil Moisture
    ('soil_moisture_pct', pa.float32()),
    ('soil_moisture_source', pa.string()),
    ('soil_temp_c', pa.float32()),
    ('soil_type', pa.string()),
    
    # Vegetation
    ('ndvi', pa.float32()),
    ('evi', pa.float32()),
    ('lai', pa.float32()),
    
    # Crop
    ('crop_name', pa.string()),
    ('crop_season', pa.string()),
    ('area_ha', pa.float64()),
    ('yield_kg_per_ha', pa.float32()),
    ('production_tonnes', pa.float64()),
    
    # Derived
    ('water_deficit_mm', pa.float32()),
    ('gdd_base10', pa.float32()),
    ('spi_30d', pa.float32()),
    ('precip_7d_sum', pa.float32()),
    ('temp_7d_mean', pa.float32()),
    
    # Metadata
    ('created_at', pa.timestamp('us')),
    ('updated_at', pa.timestamp('us')),
    ('data_quality_flag', pa.string()),
    ('source_id', pa.string()),
])
"""


def create_postgres_tables(connection_string: str):
    """Execute DDL to create all tables."""
    import psycopg2
    
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    # Split and execute DDL statements
    for statement in POSTGRES_DDL.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                cur.execute(statement)
            except Exception as e:
                print(f"Error executing: {statement[:50]}... - {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database tables created successfully!")


def create_folder_structure(base_path: str):
    """Create the folder structure for local development."""
    from pathlib import Path
    
    folders = [
        "data/raw/weather/openweathermap",
        "data/raw/weather/visualcrossing",
        "data/raw/soil_moisture/smap",
        "data/raw/soil_moisture/sensors",
        "data/raw/crop_statistics/data_gov_in",
        "data/raw/crop_statistics/icrisat",
        "data/raw/ndvi/modis",
        "data/curated/weather",
        "data/curated/soil_moisture",
        "data/curated/crop_statistics",
        "data/curated/ndvi",
        "data/processed/features",
        "data/processed/training",
        "data/processed/inference",
        "models",
        "logs/etl",
        "logs/training",
        "logs/api",
        "outputs/reports",
        "outputs/exports",
    ]
    
    base = Path(base_path)
    for folder in folders:
        (base / folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Created folder structure at {base_path}")


if __name__ == "__main__":
    print(SCHEMA_DEFINITION)
    print("\n" + "="*80 + "\n")
    print(POSTGRES_DDL)
    print("\n" + "="*80 + "\n")
    print(FOLDER_LAYOUT)
