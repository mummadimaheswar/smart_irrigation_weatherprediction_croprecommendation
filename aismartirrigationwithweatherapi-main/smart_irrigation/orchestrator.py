"""
Smart Irrigation Orchestrator

Main coordinator module that ties together all system components to implement
the complete decision pipeline described in the methodology pseudocode.

This module implements the algorithm:
1. Load SensorData, WeatherData, CropParameters, KnowledgeBase
2. Normalize and fuse into EnvironmentalState
3. Identify DecisionTriggers
4. For each trigger: evaluate, compute consistency, determine action
5. Apply safety constraints
6. Generate crop guidance and confidence scores
7. Output Decision Report

Usage:
    from smart_irrigation.orchestrator import SmartIrrigationOrchestrator
    
    orchestrator = SmartIrrigationOrchestrator()
    report = orchestrator.run_decision_cycle(
        lat=20.5, lon=78.9,
        crop_type="wheat",
        days_after_sowing=60
    )
    print(report.to_text())
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .config import CROP_PARAMS, STATES, DISTRICTS
from .environmental_state import (
    SensorData, WeatherData, CropContext,
    EnvironmentalState, EnvironmentalFusionEngine
)
from .decision_arbitration import (
    DecisionArbitrator, SafetyConstraints,
    ArbitrationResult, IrrigationAction, ActuationController
)
from .decision_report import (
    DecisionReport, DecisionReportGenerator, CropGuidance
)
from .weather import fetch_openweather, get_forecast_summary

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Raw sensor reading from field device."""
    sensor_id: str
    soil_moisture_pct: float  # 0-100%
    soil_temp_c: float
    ambient_temp_c: float
    ambient_humidity_pct: float
    timestamp: datetime
    battery_pct: float = 100.0
    signal_strength: float = 1.0
    
    def to_sensor_data(self) -> SensorData:
        """Convert to normalized SensorData."""
        # Calculate reliability based on battery and signal
        reliability = min(1.0, (self.battery_pct / 100) * (self.signal_strength))
        
        return SensorData(
            soil_moisture_vwc=self.soil_moisture_pct / 100,  # Convert % to fraction
            soil_temperature_c=self.soil_temp_c,
            ambient_temperature_c=self.ambient_temp_c,
            ambient_humidity_pct=self.ambient_humidity_pct,
            timestamp=self.timestamp,
            sensor_id=self.sensor_id,
            reliability_score=reliability
        )


class SensorDataLoader:
    """
    Loads and aggregates sensor data from multiple sources.
    
    Can read from:
    - Direct sensor readings
    - CSV files (historical soil moisture data)
    - Simulated/synthetic data for testing
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir
    
    def load_latest_reading(
        self,
        sensor_id: str = "default",
        simulate: bool = True
    ) -> SensorReading:
        """
        Load the latest sensor reading.
        
        If simulate=True, generates realistic synthetic data.
        """
        if simulate:
            return self._generate_synthetic_reading(sensor_id)
        
        # TODO: Implement actual sensor interface
        raise NotImplementedError("Real sensor interface not implemented")
    
    def _generate_synthetic_reading(self, sensor_id: str) -> SensorReading:
        """Generate realistic synthetic sensor data for testing."""
        import numpy as np
        
        # Time-based variation
        hour = datetime.utcnow().hour
        day_factor = np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Base values with realistic variation
        base_soil_moisture = 25 + np.random.normal(0, 5)
        base_temp = 28 + 8 * day_factor + np.random.normal(0, 2)
        base_humidity = 65 - 15 * day_factor + np.random.normal(0, 5)
        
        return SensorReading(
            sensor_id=sensor_id,
            soil_moisture_pct=np.clip(base_soil_moisture, 10, 50),
            soil_temp_c=np.clip(base_temp - 5, 15, 45),
            ambient_temp_c=np.clip(base_temp, 15, 50),
            ambient_humidity_pct=np.clip(base_humidity, 30, 95),
            timestamp=datetime.utcnow(),
            battery_pct=np.random.uniform(70, 100),
            signal_strength=np.random.uniform(0.7, 1.0)
        )
    
    def load_from_csv(self, state: str, date: datetime) -> Optional[SensorReading]:
        """Load historical sensor data from state CSV files."""
        if self.data_dir is None:
            return None
        
        # Try to find matching CSV file
        csv_pattern = f"sm_{state.title().replace(' ', '')}*.csv"
        csv_files = list(self.data_dir.glob(csv_pattern))
        
        if not csv_files:
            return None
        
        import pandas as pd
        try:
            df = pd.read_csv(csv_files[0])
            # Extract reading for date
            # Implementation depends on CSV format
            return None
        except Exception as e:
            log.warning(f"Failed to load CSV: {e}")
            return None


class WeatherDataLoader:
    """
    Loads weather data from APIs or generates synthetic forecasts.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def load_forecast(
        self,
        lat: float,
        lon: float
    ) -> WeatherData:
        """
        Load weather forecast for given coordinates.
        
        Uses OpenWeather API if key available, otherwise synthetic data.
        """
        # Fetch from API (handles synthetic fallback internally)
        df = fetch_openweather(lat, lon, self.api_key)
        
        if df is None or df.empty:
            return self._generate_synthetic_forecast(lat, lon)
        
        # Aggregate forecast data
        summary = get_forecast_summary(df, hours=48)
        
        # Get 24h and 48h rain totals
        rain_24h = df.head(24)['precipitation'].sum() if len(df) >= 24 else df['precipitation'].sum()
        rain_48h = df.head(48)['precipitation'].sum() if len(df) >= 48 else df['precipitation'].sum()
        
        # Get temperature range
        temp_max = df.head(24)['temp'].max() if len(df) >= 24 else df['temp'].max()
        temp_min = df.head(24)['temp'].min() if len(df) >= 24 else df['temp'].min()
        
        # Estimate rain probability
        hours_with_rain = (df['precipitation'] > 0.1).sum()
        rain_prob = min(90, hours_with_rain * 5)
        
        return WeatherData(
            forecast_rain_24h_mm=rain_24h,
            forecast_rain_48h_mm=rain_48h,
            forecast_temp_max_c=temp_max,
            forecast_temp_min_c=temp_min,
            forecast_humidity_pct=summary.get('avg_humidity_pct', 60),
            forecast_wind_speed_ms=summary.get('avg_wind_speed_ms', 3),
            forecast_cloud_cover_pct=50.0,
            rain_probability_pct=rain_prob,
            weather_description=df.iloc[0]['weather_desc'] if 'weather_desc' in df.columns else 'unknown',
            forecast_timestamp=datetime.utcnow(),
            source="openweather",
            confidence_score=0.85
        )
    
    def _generate_synthetic_forecast(self, lat: float, lon: float) -> WeatherData:
        """Generate synthetic weather forecast."""
        import numpy as np
        np.random.seed(int(lat * 100 + lon * 10) % 1000)
        
        # Latitude-based temperature
        base_temp = 30 - abs(lat - 23) * 0.5
        
        return WeatherData(
            forecast_rain_24h_mm=np.random.exponential(3),
            forecast_rain_48h_mm=np.random.exponential(5),
            forecast_temp_max_c=base_temp + np.random.uniform(3, 8),
            forecast_temp_min_c=base_temp - np.random.uniform(5, 10),
            forecast_humidity_pct=np.random.uniform(40, 80),
            forecast_wind_speed_ms=np.random.uniform(1, 6),
            forecast_cloud_cover_pct=np.random.uniform(10, 70),
            rain_probability_pct=np.random.uniform(10, 50),
            weather_description="partly cloudy",
            forecast_timestamp=datetime.utcnow(),
            source="synthetic",
            confidence_score=0.6
        )


class KnowledgeBase:
    """
    Knowledge base for RAG-enhanced decision support.
    
    Contains:
    - Crop parameters and requirements
    - Regional agro-climatic information
    - Agricultural guidelines and expert advisories
    """
    
    def __init__(self):
        self.crop_params = CROP_PARAMS
        self.regional_knowledge: Dict[str, Dict] = {}
        self.advisories: List[str] = []
    
    def get_crop_context(
        self,
        crop_type: str,
        days_after_sowing: int
    ) -> CropContext:
        """Get crop-specific context for decision making."""
        return CropContext.from_config(
            crop_type=crop_type,
            days_after_sowing=days_after_sowing,
            crop_params=self.crop_params
        )
    
    def get_regional_advisory(
        self,
        state: str,
        month: int
    ) -> str:
        """Get region and season-specific advisory."""
        # Determine season
        if 6 <= month <= 10:
            season = "Kharif"
        elif month >= 11 or month <= 2:
            season = "Rabi"
        else:
            season = "Zaid"
        
        advisories = {
            ("MAHARASHTRA", "Kharif"): "Monsoon active. Watch for waterlogging in cotton and soybean fields.",
            ("MAHARASHTRA", "Rabi"): "Dry season. Plan irrigation for wheat and chickpea.",
            ("PUNJAB", "Kharif"): "Rice transplanting season. Maintain flooded conditions.",
            ("PUNJAB", "Rabi"): "Wheat growing season. Critical irrigation at crown root stage.",
            ("KARNATAKA", "Kharif"): "Southwest monsoon. Ragi and maize planting.",
            ("UTTAR PRADESH", "Rabi"): "Major wheat belt. Fog may reduce ET.",
        }
        
        return advisories.get(
            (state.upper(), season),
            f"{season} season in {state}. Monitor local weather advisories."
        )
    
    def retrieve_context(
        self,
        state: EnvironmentalState,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for RAG-enhanced guidance.
        
        This simulates RAG by returning structured knowledge.
        In production, this would query a vector database.
        """
        context = {
            "crop_info": {
                "type": state.crop_type,
                "stage": state.growth_stage,
                "critical_thresholds": {
                    "vwc_critical": state.vwc_critical,
                    "vwc_optimal": state.vwc_optimal
                }
            },
            "environmental_factors": {
                "moisture_status": state.moisture_status,
                "water_deficit": state.water_deficit_mm,
                "et_demand": state.etc_mm_day
            },
            "recommendations": []
        }
        
        # Add situation-specific recommendations
        if state.current_vwc < state.vwc_critical:
            context["recommendations"].append("Immediate irrigation required to prevent yield loss")
        elif state.forecast_rain_24h_mm > 10:
            context["recommendations"].append("Heavy rain expected - defer irrigation and monitor drainage")
        elif state.is_critical_window:
            context["recommendations"].append("Critical growth stage - maintain optimal moisture levels")
        
        return context


class SmartIrrigationOrchestrator:
    """
    Main orchestrator for the Smart Irrigation System.
    
    Coordinates all components to implement the complete decision pipeline:
    
    1. Load inputs (sensor data, weather, crop params, knowledge base)
    2. Normalize and fuse into EnvironmentalState
    3. Identify decision triggers
    4. Evaluate consistency and determine action
    5. Apply safety constraints
    6. Generate crop guidance and confidence scores
    7. Output Decision Report
    
    This class is the primary interface for running the irrigation decision system.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: Optional[Path] = None,
        constraints: Optional[SafetyConstraints] = None
    ):
        # Initialize components
        self.sensor_loader = SensorDataLoader(data_dir)
        self.weather_loader = WeatherDataLoader(api_key)
        self.knowledge_base = KnowledgeBase()
        self.fusion_engine = EnvironmentalFusionEngine(CROP_PARAMS)
        self.arbitrator = DecisionArbitrator(constraints)
        self.actuator = ActuationController()
        self.report_generator = DecisionReportGenerator()
        
        # State tracking
        self.last_decision: Optional[ArbitrationResult] = None
        self.decision_history: List[Dict] = []
    
    def run_decision_cycle(
        self,
        lat: float,
        lon: float,
        crop_type: str = "wheat",
        days_after_sowing: int = 60,
        sensor_reading: Optional[SensorReading] = None,
        location_info: Optional[Dict] = None,
        simulate_sensors: bool = True
    ) -> DecisionReport:
        """
        Execute a complete irrigation decision cycle.
        
        This implements the full pseudocode algorithm from the methodology.
        
        Args:
            lat: Latitude of field location
            lon: Longitude of field location
            crop_type: Type of crop (wheat, rice, maize, etc.)
            days_after_sowing: Days since crop was planted
            sensor_reading: Optional direct sensor reading
            location_info: Optional location metadata
            simulate_sensors: If True, generate synthetic sensor data
        
        Returns:
            Complete DecisionReport with irrigation decision, guidance, and confidence
        """
        log.info(f"Starting decision cycle for {crop_type} at ({lat}, {lon})")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Load Inputs
        # ─────────────────────────────────────────────────────────────────────
        
        # Load sensor data
        if sensor_reading is None:
            sensor_reading = self.sensor_loader.load_latest_reading(
                sensor_id=f"sensor_{lat:.2f}_{lon:.2f}",
                simulate=simulate_sensors
            )
        sensor_data = sensor_reading.to_sensor_data()
        log.debug(f"Sensor data: moisture={sensor_data.soil_moisture_vwc:.0%}")
        
        # Load weather forecast
        weather_data = self.weather_loader.load_forecast(lat, lon)
        log.debug(f"Weather: rain={weather_data.forecast_rain_24h_mm:.1f}mm, temp={weather_data.forecast_temp_max_c:.1f}C")
        
        # Load knowledge context
        knowledge_context = self.knowledge_base.retrieve_context
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Normalize and Fuse into EnvironmentalState
        # ─────────────────────────────────────────────────────────────────────
        
        environmental_state = self.fusion_engine.fuse(
            sensor_data=sensor_data,
            weather_data=weather_data,
            crop_type=crop_type,
            days_after_sowing=days_after_sowing,
            latitude=lat
        )
        log.info(f"Environmental state: moisture={environmental_state.moisture_status}, "
                 f"triggers={len(environmental_state.triggers)}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3-5: Evaluate Triggers, Compute Consistency, Determine Action
        # ─────────────────────────────────────────────────────────────────────
        
        arbitration_result = self.arbitrator.arbitrate(environmental_state)
        log.info(f"Decision: {arbitration_result.action.value} "
                 f"(confidence={arbitration_result.confidence_score:.0%})")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Generate Control Signal
        # ─────────────────────────────────────────────────────────────────────
        
        control_signal = self.actuator.generate_control_signal(arbitration_result)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 7: Generate Decision Report
        # ─────────────────────────────────────────────────────────────────────
        
        location = location_info or {
            "lat": lat,
            "lon": lon,
            "state": "Unknown",
            "district": "Unknown"
        }
        
        report = self.report_generator.generate(
            state=environmental_state,
            result=arbitration_result,
            control_signal=control_signal,
            location=location
        )
        
        # Track decision
        self.last_decision = arbitration_result
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": arbitration_result.action.value,
            "confidence": arbitration_result.confidence_score,
            "moisture": environmental_state.current_vwc
        })
        
        return report
    
    def run_for_location(
        self,
        state: str,
        district: Optional[str] = None,
        crop_type: str = "wheat",
        days_after_sowing: int = 60
    ) -> DecisionReport:
        """
        Run decision cycle for a named location (state/district).
        
        Looks up coordinates from config.
        """
        from .config import DISTRICTS
        
        # Get coordinates (use state centroid)
        state_coords = {
            "MAHARASHTRA": (19.75, 75.71),
            "PUNJAB": (31.15, 75.34),
            "KARNATAKA": (15.32, 75.71),
            "ASSAM": (26.20, 92.94),
            "UTTAR PRADESH": (26.85, 80.95),
        }
        
        coords = state_coords.get(state.upper(), (20.0, 78.0))
        
        return self.run_decision_cycle(
            lat=coords[0],
            lon=coords[1],
            crop_type=crop_type,
            days_after_sowing=days_after_sowing,
            location_info={
                "state": state,
                "district": district or "Unknown",
                "lat": coords[0],
                "lon": coords[1]
            }
        )
    
    def save_decision(self, report: DecisionReport, output_path: Path):
        """Save decision report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report.to_json())
        log.info(f"Saved decision to {output_path}")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of recent decisions."""
        if not self.decision_history:
            return {"status": "No decisions yet"}
        
        recent = self.decision_history[-10:]
        actions = [d["action"] for d in recent]
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_count": len(recent),
            "execute_count": actions.count("execute"),
            "defer_count": actions.count("defer"),
            "skip_count": actions.count("skip"),
            "avg_confidence": sum(d["confidence"] for d in recent) / len(recent),
            "last_decision": self.decision_history[-1] if self.decision_history else None
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line interface for running irrigation decisions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Irrigation Decision System"
    )
    parser.add_argument("--lat", type=float, default=20.5, help="Latitude")
    parser.add_argument("--lon", type=float, default=78.9, help="Longitude")
    parser.add_argument("--crop", type=str, default="wheat", help="Crop type")
    parser.add_argument("--das", type=int, default=60, help="Days after sowing")
    parser.add_argument("--state", type=str, help="Indian state name")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = SmartIrrigationOrchestrator()
    
    # Run decision
    if args.state:
        report = orchestrator.run_for_location(
            state=args.state,
            crop_type=args.crop,
            days_after_sowing=args.das
        )
    else:
        report = orchestrator.run_decision_cycle(
            lat=args.lat,
            lon=args.lon,
            crop_type=args.crop,
            days_after_sowing=args.das
        )
    
    # Output
    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())
    
    # Save if requested
    if args.output:
        orchestrator.save_decision(report, Path(args.output))


if __name__ == "__main__":
    main()
