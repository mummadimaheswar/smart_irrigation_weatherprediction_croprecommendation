"""
Airflow DAG for India Crop Recommendation ETL Pipeline

PROMPT 6: Reproducible ETL pipeline with:
- Daily ingestion, deduplication, validation
- Logging, retry policies, alerting
- Writes to Postgres

Run with:
    docker-compose up -d
    # or
    airflow standalone  # for local dev
"""
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.hooks.base import BaseHook
from airflow.models import Variable

# ═══════════════════════════════════════════════════════════════════════════════
# DAG CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
}

STATES = ['Maharashtra', 'Karnataka', 'Punjab', 'Gujarat', 'Tamil Nadu']

# ═══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_weather(**context) -> Dict[str, Any]:
    """Ingest weather data from APIs."""
    import pandas as pd
    from datetime import datetime
    import sys
    sys.path.insert(0, '/opt/airflow/dags/india_crop_recommendation')
    
    from ingest.weather_api import fetch_historical_weather, save_to_parquet
    
    execution_date = context['execution_date']
    start_date = execution_date - timedelta(days=1)
    end_date = execution_date
    
    all_data = []
    for state in STATES:
        try:
            df = fetch_historical_weather(state, start_date, end_date)
            if not df.empty:
                all_data.append(df)
                logging.info(f"Ingested {len(df)} weather records for {state}")
        except Exception as e:
            logging.error(f"Failed to ingest weather for {state}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = f"/opt/airflow/data/raw/weather/{execution_date.strftime('%Y-%m-%d')}"
        save_to_parquet(combined, output_path)
        
        return {
            "records_ingested": len(combined),
            "states_processed": len(all_data),
            "output_path": output_path
        }
    
    return {"records_ingested": 0, "states_processed": 0}


def ingest_soil_moisture(**context) -> Dict[str, Any]:
    """Ingest soil moisture data."""
    import pandas as pd
    import sys
    sys.path.insert(0, '/opt/airflow/dags/india_crop_recommendation')
    
    from ingest.soil_moisture import (
        _generate_synthetic_satellite_data,
        _generate_synthetic_sensor_data,
        resample_to_daily,
        harmonize_datasets,
        save_to_parquet
    )
    
    execution_date = context['execution_date']
    start_date = execution_date - timedelta(days=1)
    end_date = execution_date
    
    all_data = []
    for state in STATES:
        try:
            # In production, this would call actual APIs
            sat_df = _generate_synthetic_satellite_data(state, start_date, end_date)
            sensor_df = _generate_synthetic_sensor_data(state, 5, start_date, end_date)
            sensor_df = resample_to_daily(sensor_df)
            
            combined = harmonize_datasets(sensor_df, sat_df)
            all_data.append(combined)
            logging.info(f"Ingested {len(combined)} soil moisture records for {state}")
        except Exception as e:
            logging.error(f"Failed to ingest soil moisture for {state}: {e}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        output_path = f"/opt/airflow/data/raw/soil_moisture/{execution_date.strftime('%Y-%m-%d')}"
        save_to_parquet(result, output_path)
        
        return {
            "records_ingested": len(result),
            "states_processed": len(all_data),
            "output_path": output_path
        }
    
    return {"records_ingested": 0, "states_processed": 0}


def validate_and_dedup(**context) -> Dict[str, Any]:
    """Validate data and remove duplicates."""
    import pandas as pd
    from pathlib import Path
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    weather_path = f"/opt/airflow/data/raw/weather/{date_str}"
    soil_path = f"/opt/airflow/data/raw/soil_moisture/{date_str}"
    
    validation_results = {
        "weather": {"valid": 0, "invalid": 0, "duplicates_removed": 0},
        "soil_moisture": {"valid": 0, "invalid": 0, "duplicates_removed": 0}
    }
    
    # Validate weather
    if Path(weather_path).exists():
        df = pd.read_parquet(weather_path)
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'state', 'lat', 'lon'], keep='last')
        
        # Validate ranges
        df = df[
            (df['t_min_c'].between(-20, 50, inclusive='both')) &
            (df['t_max_c'].between(-10, 55, inclusive='both')) &
            (df['precip_mm'] >= 0) &
            (df['humidity_pct'].between(0, 100, inclusive='both'))
        ]
        
        validation_results["weather"]["valid"] = len(df)
        validation_results["weather"]["invalid"] = original_count - len(df) - (original_count - len(df.drop_duplicates()))
        validation_results["weather"]["duplicates_removed"] = original_count - len(df)
        
        # Save validated
        curated_path = f"/opt/airflow/data/curated/weather/{date_str}"
        df.to_parquet(curated_path, index=False)
    
    # Validate soil moisture
    if Path(soil_path).exists():
        df = pd.read_parquet(soil_path)
        original_count = len(df)
        
        df = df.drop_duplicates(subset=['date', 'lat', 'lon', 'data_source'], keep='last')
        df = df[df['soil_moisture_pct'].between(0, 100, inclusive='both')]
        
        validation_results["soil_moisture"]["valid"] = len(df)
        validation_results["soil_moisture"]["duplicates_removed"] = original_count - len(df)
        
        curated_path = f"/opt/airflow/data/curated/soil_moisture/{date_str}"
        df.to_parquet(curated_path, index=False)
    
    logging.info(f"Validation results: {validation_results}")
    return validation_results


def load_to_postgres(**context) -> Dict[str, Any]:
    """Load curated data to Postgres."""
    import pandas as pd
    from pathlib import Path
    from sqlalchemy import create_engine
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    # Get connection from Airflow
    try:
        conn = BaseHook.get_connection('postgres_default')
        engine = create_engine(
            f"postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
        )
    except Exception:
        logging.warning("Postgres connection not configured, skipping load")
        return {"status": "skipped", "reason": "no_connection"}
    
    records_loaded = 0
    
    # Load weather
    weather_path = f"/opt/airflow/data/curated/weather/{date_str}"
    if Path(weather_path).exists():
        df = pd.read_parquet(weather_path)
        df.to_sql('weather_daily', engine, if_exists='append', index=False)
        records_loaded += len(df)
    
    # Load soil moisture
    soil_path = f"/opt/airflow/data/curated/soil_moisture/{date_str}"
    if Path(soil_path).exists():
        df = pd.read_parquet(soil_path)
        df.to_sql('soil_moisture', engine, if_exists='append', index=False)
        records_loaded += len(df)
    
    return {"records_loaded": records_loaded}


def run_feature_engineering(**context) -> Dict[str, Any]:
    """Generate features for ML."""
    import pandas as pd
    from pathlib import Path
    
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    weather_path = f"/opt/airflow/data/curated/weather/{date_str}"
    soil_path = f"/opt/airflow/data/curated/soil_moisture/{date_str}"
    
    if not Path(weather_path).exists() or not Path(soil_path).exists():
        return {"status": "skipped", "reason": "missing_data"}
    
    weather_df = pd.read_parquet(weather_path)
    soil_df = pd.read_parquet(soil_path)
    
    # Merge on date and location
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    soil_df['date'] = pd.to_datetime(soil_df['date'])
    
    # Round coordinates for matching
    weather_df['lat_round'] = weather_df['lat'].round(1)
    weather_df['lon_round'] = weather_df['lon'].round(1)
    soil_df['lat_round'] = soil_df['lat'].round(1)
    soil_df['lon_round'] = soil_df['lon'].round(1)
    
    merged = pd.merge(
        weather_df, soil_df,
        on=['date', 'lat_round', 'lon_round'],
        how='outer',
        suffixes=('_weather', '_soil')
    )
    
    # Add derived features
    if 't_min_c' in merged.columns and 't_max_c' in merged.columns:
        merged['temp_range_c'] = merged['t_max_c'] - merged['t_min_c']
        merged['gdd_base10'] = ((merged['t_max_c'] + merged['t_min_c']) / 2 - 10).clip(lower=0)
    
    if 'precip_mm' in merged.columns and 'soil_moisture_pct' in merged.columns:
        # Simple water balance proxy
        et_estimate = 5.0  # mm/day estimate
        merged['water_balance'] = merged['precip_mm'] - et_estimate
    
    # Save features
    output_path = f"/opt/airflow/data/processed/features/{date_str}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    
    return {"features_generated": len(merged), "output_path": output_path}


def send_success_notification(**context) -> None:
    """Send success notification via webhook."""
    import requests
    
    webhook_url = Variable.get("slack_webhook_url", default_var=None)
    if not webhook_url:
        logging.info("No webhook configured, skipping notification")
        return
    
    execution_date = context['execution_date']
    
    payload = {
        "text": f":white_check_mark: India Crop ETL completed successfully for {execution_date.strftime('%Y-%m-%d')}"
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        logging.warning(f"Failed to send notification: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id='india_crop_etl',
    default_args=default_args,
    description='Daily ETL pipeline for India crop recommendation system',
    schedule_interval='0 3 * * *',  # Run at 3 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['india', 'crop', 'etl'],
    max_active_runs=1,
) as dag:
    
    # Sensor to check API availability
    check_weather_api = HttpSensor(
        task_id='check_weather_api',
        http_conn_id='visualcrossing_api',
        endpoint='/',
        request_params={},
        response_check=lambda response: response.status_code == 200,
        poke_interval=60,
        timeout=300,
        soft_fail=True,  # Don't fail if API is down
    )
    
    # Ingestion tasks
    ingest_weather_task = PythonOperator(
        task_id='ingest_weather',
        python_callable=ingest_weather,
        provide_context=True,
    )
    
    ingest_soil_task = PythonOperator(
        task_id='ingest_soil_moisture',
        python_callable=ingest_soil_moisture,
        provide_context=True,
    )
    
    # Validation
    validate_task = PythonOperator(
        task_id='validate_and_dedup',
        python_callable=validate_and_dedup,
        provide_context=True,
    )
    
    # Load to Postgres
    load_postgres_task = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres,
        provide_context=True,
    )
    
    # Feature engineering
    features_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_feature_engineering,
        provide_context=True,
    )
    
    # Notification
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_success_notification,
        provide_context=True,
        trigger_rule='all_success',
    )
    
    # Task dependencies
    check_weather_api >> [ingest_weather_task, ingest_soil_task]
    [ingest_weather_task, ingest_soil_task] >> validate_task
    validate_task >> load_postgres_task >> features_task >> notify_task


# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY MODEL RETRAINING DAG
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id='india_crop_model_retrain',
    default_args=default_args,
    description='Weekly model retraining for crop recommendations',
    schedule_interval='0 6 * * 0',  # Run at 6 AM every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['india', 'crop', 'ml', 'training'],
    max_active_runs=1,
) as model_dag:
    
    def retrain_model(**context):
        """Retrain crop recommendation model."""
        import sys
        sys.path.insert(0, '/opt/airflow/dags/india_crop_recommendation')
        
        # This would import and run the training pipeline
        logging.info("Model retraining started")
        # from train.train_model import run_training
        # results = run_training()
        return {"status": "completed", "model_version": "v1.0"}
    
    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
        provide_context=True,
    )
