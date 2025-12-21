"""Scheduler for ETL jobs and model retraining."""
import logging
import time
from datetime import datetime, timedelta
from threading import Thread
from typing import Callable, Dict, Optional
import schedule

from .config import SCHEDULER, STATES
from .ingest import DataIngester, GEO_COORDS
from .etl import ETLPipeline
from .features import engineer_all_features
from .ml_pipeline import ModelPipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# JOB DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def job_weather_ingest():
    """Ingest weather data for all configured locations."""
    log.info("Starting weather ingestion job...")
    ingester = DataIngester()
    
    success = 0
    for (state, district), coords in GEO_COORDS.items():
        if state in STATES:
            try:
                df = ingester.ingest_weather(state, district)
                if not df.empty:
                    success += 1
            except Exception as e:
                log.error(f"Weather ingest failed for {state}/{district}: {e}")
    
    log.info(f"Weather ingestion complete: {success} locations updated")


def job_etl_pipeline():
    """Run full ETL pipeline."""
    log.info("Starting ETL pipeline job...")
    try:
        pipeline = ETLPipeline()
        result = pipeline.run()
        log.info(f"ETL complete: {len(result)} rows processed")
    except Exception as e:
        log.error(f"ETL pipeline failed: {e}")


def job_feature_engineering():
    """Update features for curated data."""
    log.info("Starting feature engineering job...")
    try:
        from .etl import load_curated, save_curated
        
        df = load_curated("irrigation_dataset")
        if df.empty:
            log.warning("No curated data found")
            return
        
        df = engineer_all_features(df, crop="wheat")
        save_curated(df, "irrigation_features")
        log.info(f"Feature engineering complete: {len(df)} rows")
    except Exception as e:
        log.error(f"Feature engineering failed: {e}")


def job_model_retrain():
    """Retrain ML models on latest data."""
    log.info("Starting model retraining job...")
    try:
        from .etl import load_curated
        
        df = load_curated("irrigation_features")
        if df.empty:
            log.warning("No feature data found")
            return
        
        pipeline = ModelPipeline(crop="wheat")
        pipeline.train_all(df)
        pipeline.save_model("best")
        pipeline.compare_models()
        
        log.info("Model retraining complete")
    except Exception as e:
        log.error(f"Model retraining failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class JobScheduler:
    """Manages scheduled jobs for the irrigation system."""
    
    def __init__(self):
        self.running = False
        self.thread: Optional[Thread] = None
        self.jobs: Dict[str, Callable] = {}
    
    def register_job(self, name: str, func: Callable, interval_hours: int):
        """Register a job to run at specified interval."""
        self.jobs[name] = func
        schedule.every(interval_hours).hours.do(func)
        log.info(f"Registered job '{name}' to run every {interval_hours}h")
    
    def register_daily_job(self, name: str, func: Callable, time_str: str = "02:00"):
        """Register a job to run daily at specified time."""
        self.jobs[name] = func
        schedule.every().day.at(time_str).do(func)
        log.info(f"Registered job '{name}' to run daily at {time_str}")
    
    def setup_default_jobs(self):
        """Set up default job schedule."""
        # Weather ingestion every 6 hours
        self.register_job("weather_ingest", job_weather_ingest, 
                         SCHEDULER.weather_interval_hours)
        
        # ETL pipeline daily at 3 AM
        self.register_daily_job("etl_pipeline", job_etl_pipeline, "03:00")
        
        # Feature engineering daily at 4 AM
        self.register_daily_job("feature_engineering", job_feature_engineering, "04:00")
        
        # Model retraining monthly (approximated as every 30 days)
        if SCHEDULER.model_retrain_days <= 30:
            self.register_daily_job("model_retrain", job_model_retrain, "05:00")
    
    def run_job_now(self, name: str):
        """Run a specific job immediately."""
        if name not in self.jobs:
            raise ValueError(f"Unknown job: {name}")
        
        log.info(f"Running job '{name}' now...")
        self.jobs[name]()
    
    def start(self, blocking: bool = True):
        """Start the scheduler."""
        self.running = True
        log.info("Starting scheduler...")
        
        if blocking:
            self._run_loop()
        else:
            self.thread = Thread(target=self._run_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        log.info("Scheduler stopped")
    
    def _run_loop(self):
        """Main scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_next_runs(self) -> Dict[str, str]:
        """Get next run time for each job."""
        next_runs = {}
        for job in schedule.get_jobs():
            next_run = job.next_run
            if next_run:
                # Find job name from function
                for name, func in self.jobs.items():
                    if job.job_func == func:
                        next_runs[name] = next_run.isoformat()
                        break
        return next_runs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler():
    """Run the scheduler from command line."""
    scheduler = JobScheduler()
    scheduler.setup_default_jobs()
    
    print("Smart Irrigation Scheduler")
    print("=" * 40)
    print("Registered jobs:")
    for name in scheduler.jobs:
        print(f"  - {name}")
    print("\nPress Ctrl+C to stop")
    
    try:
        scheduler.start(blocking=True)
    except KeyboardInterrupt:
        scheduler.stop()
        print("\nScheduler stopped")


if __name__ == "__main__":
    run_scheduler()
