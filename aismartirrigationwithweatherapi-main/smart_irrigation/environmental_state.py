"""
Environmental State Fusion Module

Fuses heterogeneous environmental inputs (sensor data, weather forecasts, crop parameters)
into a unified EnvironmentalState representation for decision arbitration.

This module implements the data normalization and fusion layer described in the methodology:
- Normalize SensorData and WeatherData
- Fuse into unified EnvironmentalState
- Identify DecisionTriggers
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class TriggerType(Enum):
    """Types of decision triggers that initiate irrigation evaluation."""
    LOW_SOIL_MOISTURE = "low_soil_moisture"
    CRITICAL_MOISTURE = "critical_moisture"
    HIGH_ET_DEMAND = "high_et_demand"
    RAINFALL_EXPECTED = "rainfall_expected"
    CROP_STRESS = "crop_stress"
    SCHEDULED_CHECK = "scheduled_check"
    TEMPERATURE_EXTREME = "temperature_extreme"
    GROWTH_STAGE_CRITICAL = "growth_stage_critical"


@dataclass
class SensorData:
    """Normalized sensor readings from field devices."""
    soil_moisture_vwc: float  # Volumetric water content (0-1)
    soil_temperature_c: float
    ambient_temperature_c: float
    ambient_humidity_pct: float
    timestamp: datetime
    sensor_id: Optional[str] = None
    reliability_score: float = 1.0  # 0-1, based on sensor health/calibration
    
    def validate(self) -> bool:
        """Check if sensor values are within physical bounds."""
        return (
            0 <= self.soil_moisture_vwc <= 1 and
            -10 <= self.soil_temperature_c <= 60 and
            -50 <= self.ambient_temperature_c <= 60 and
            0 <= self.ambient_humidity_pct <= 100
        )


@dataclass
class WeatherData:
    """Normalized weather forecast data."""
    forecast_rain_24h_mm: float
    forecast_rain_48h_mm: float
    forecast_temp_max_c: float
    forecast_temp_min_c: float
    forecast_humidity_pct: float
    forecast_wind_speed_ms: float
    forecast_cloud_cover_pct: float
    rain_probability_pct: float
    weather_description: str
    forecast_timestamp: datetime
    source: str = "openweather"
    confidence_score: float = 0.8  # API reliability
    
    @property
    def is_rain_expected(self) -> bool:
        return self.forecast_rain_24h_mm >= 5.0 or self.rain_probability_pct >= 60


@dataclass
class CropContext:
    """Crop-specific parameters and growth stage context."""
    crop_type: str
    growth_stage: str  # initial, development, mid, late
    days_after_sowing: int
    kc: float  # Crop coefficient
    vwc_critical: float  # Critical moisture threshold
    vwc_optimal: float  # Optimal moisture threshold
    root_depth_cm: float
    is_critical_window: bool = False  # Flowering, grain fill, etc.
    
    @classmethod
    def from_config(cls, crop_type: str, days_after_sowing: int, crop_params: Dict) -> "CropContext":
        """Create CropContext from crop parameters config."""
        params = crop_params.get(crop_type.lower(), crop_params.get("wheat", {}))
        growing_days = params.get("growing_days", 120)
        
        # Determine growth stage
        stage_pct = days_after_sowing / growing_days
        if stage_pct < 0.15:
            stage = "initial"
            kc_idx = 0
        elif stage_pct < 0.40:
            stage = "development"
            kc_idx = 1
        elif stage_pct < 0.75:
            stage = "mid"
            kc_idx = 2
        else:
            stage = "late"
            kc_idx = 3
        
        kc_stages = params.get("kc_stages", [0.5, 0.9, 1.1, 0.7])
        
        # Critical windows: mid-stage for most crops
        is_critical = 0.35 <= stage_pct <= 0.65
        
        return cls(
            crop_type=crop_type,
            growth_stage=stage,
            days_after_sowing=days_after_sowing,
            kc=kc_stages[kc_idx],
            vwc_critical=params.get("vwc_critical", 0.18),
            vwc_optimal=params.get("vwc_optimal", 0.28),
            root_depth_cm=params.get("root_depth_cm", 100),
            is_critical_window=is_critical
        )


@dataclass
class DecisionTrigger:
    """Represents a condition that triggers irrigation evaluation."""
    trigger_type: TriggerType
    severity: float  # 0-1, higher = more urgent
    description: str
    field_value: float
    threshold_value: float


@dataclass
class EnvironmentalState:
    """
    Unified representation of environmental conditions.
    
    Fuses sensor data, weather forecasts, and crop context into a single
    state object for decision arbitration.
    """
    # Core measurements
    current_vwc: float
    predicted_vwc_24h: float
    soil_temp_c: float
    ambient_temp_c: float
    humidity_pct: float
    
    # Weather expectations
    forecast_rain_24h_mm: float
    forecast_rain_48h_mm: float
    rain_probability_pct: float
    et0_mm_day: float
    etc_mm_day: float
    water_deficit_mm: float
    
    # Crop context
    crop_type: str
    growth_stage: str
    vwc_critical: float
    vwc_optimal: float
    is_critical_window: bool
    
    # Quality indicators
    sensor_reliability: float
    forecast_confidence: float
    data_staleness_minutes: int
    
    # Triggered conditions
    triggers: List[DecisionTrigger] = field(default_factory=list)
    
    # Timestamps
    timestamp: datetime = field(default_factory=utc_now)
    
    @property
    def overall_data_quality(self) -> float:
        """Combined data quality score (0-1)."""
        staleness_penalty = min(1.0, self.data_staleness_minutes / 120)  # 2 hour max
        return (
            0.4 * self.sensor_reliability +
            0.4 * self.forecast_confidence +
            0.2 * (1 - staleness_penalty)
        )
    
    @property
    def moisture_status(self) -> str:
        """Human-readable moisture status."""
        if self.current_vwc < self.vwc_critical:
            return "critical"
        elif self.current_vwc < self.vwc_optimal:
            return "marginal"
        elif self.current_vwc < self.vwc_optimal * 1.2:
            return "adequate"
        else:
            return "saturated"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_vwc": round(self.current_vwc, 4),
            "predicted_vwc_24h": round(self.predicted_vwc_24h, 4),
            "soil_temp_c": round(self.soil_temp_c, 1),
            "ambient_temp_c": round(self.ambient_temp_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "forecast_rain_24h_mm": round(self.forecast_rain_24h_mm, 1),
            "forecast_rain_48h_mm": round(self.forecast_rain_48h_mm, 1),
            "rain_probability_pct": round(self.rain_probability_pct, 1),
            "et0_mm_day": round(self.et0_mm_day, 2),
            "etc_mm_day": round(self.etc_mm_day, 2),
            "water_deficit_mm": round(self.water_deficit_mm, 2),
            "crop_type": self.crop_type,
            "growth_stage": self.growth_stage,
            "vwc_critical": self.vwc_critical,
            "vwc_optimal": self.vwc_optimal,
            "is_critical_window": self.is_critical_window,
            "moisture_status": self.moisture_status,
            "data_quality": round(self.overall_data_quality, 2),
            "triggers": [
                {
                    "type": t.trigger_type.value,
                    "severity": round(t.severity, 2),
                    "description": t.description
                }
                for t in self.triggers
            ],
            "timestamp": self.timestamp.isoformat()
        }


class EnvironmentalFusionEngine:
    """
    Fuses heterogeneous environmental inputs into unified EnvironmentalState.
    
    Implements data normalization, quality assessment, and trigger identification
    as described in the system methodology.
    """
    
    def __init__(self, crop_params: Optional[Dict] = None):
        self.crop_params = crop_params or {}
    
    def compute_et0_hargreaves(
        self,
        temp_mean: float,
        temp_max: float,
        temp_min: float,
        latitude: float = 20.0
    ) -> float:
        """
        Compute reference ET using Hargreaves equation.
        
        Returns ET0 in mm/day.
        """
        # Approximate extraterrestrial radiation based on latitude and day of year
        day_of_year = utc_now().timetuple().tm_yday
        lat_rad = latitude * np.pi / 180
        
        # Solar declination
        delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
        # Sunset hour angle
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        
        # Relative distance Earth-Sun
        dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
        # Extraterrestrial radiation (MJ/m2/day)
        ra = (24 * 60 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )
        
        # Hargreaves equation
        temp_range = max(temp_max - temp_min, 0.1)
        et0 = 0.0023 * (temp_mean + 17.8) * np.sqrt(temp_range) * ra / 2.45
        
        return max(0, et0)
    
    def predict_vwc_24h(
        self,
        current_vwc: float,
        etc_mm: float,
        forecast_rain_mm: float,
        soil_type: str = "loam"
    ) -> float:
        """
        Simple soil moisture prediction for next 24 hours.
        
        Uses water balance approach: VWC_next = VWC_current + (Rain - ETc) / root_zone
        """
        # Approximate root zone depth in mm
        root_zone_mm = 300  # 30cm typical
        
        # Effective rainfall (accounting for runoff/interception)
        effective_rain = forecast_rain_mm * 0.8
        
        # Water balance
        delta_mm = effective_rain - etc_mm
        delta_vwc = delta_mm / root_zone_mm
        
        predicted = current_vwc + delta_vwc
        
        # Bounds check
        return np.clip(predicted, 0.05, 0.55)
    
    def identify_triggers(
        self,
        current_vwc: float,
        predicted_vwc: float,
        etc_mm: float,
        forecast_rain_mm: float,
        ambient_temp: float,
        crop_context: CropContext
    ) -> List[DecisionTrigger]:
        """
        Identify conditions that trigger irrigation evaluation.
        
        Returns list of active triggers with severity ratings.
        """
        triggers = []
        
        # Critical moisture trigger
        if current_vwc < crop_context.vwc_critical:
            severity = min(1.0, (crop_context.vwc_critical - current_vwc) / 0.1)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.CRITICAL_MOISTURE,
                severity=severity,
                description=f"Soil moisture {current_vwc:.0%} below critical {crop_context.vwc_critical:.0%}",
                field_value=current_vwc,
                threshold_value=crop_context.vwc_critical
            ))
        
        # Low moisture trigger
        elif current_vwc < crop_context.vwc_optimal:
            severity = min(1.0, (crop_context.vwc_optimal - current_vwc) / 0.1)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.LOW_SOIL_MOISTURE,
                severity=severity * 0.6,
                description=f"Soil moisture {current_vwc:.0%} below optimal {crop_context.vwc_optimal:.0%}",
                field_value=current_vwc,
                threshold_value=crop_context.vwc_optimal
            ))
        
        # Predicted moisture drop
        if predicted_vwc < crop_context.vwc_critical:
            severity = min(1.0, (crop_context.vwc_critical - predicted_vwc) / 0.1)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.CROP_STRESS,
                severity=severity * 0.8,
                description=f"Predicted moisture {predicted_vwc:.0%} will drop below critical",
                field_value=predicted_vwc,
                threshold_value=crop_context.vwc_critical
            ))
        
        # High ET demand
        if etc_mm > 5.0:
            severity = min(1.0, (etc_mm - 5.0) / 3.0)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.HIGH_ET_DEMAND,
                severity=severity * 0.5,
                description=f"High evapotranspiration demand: {etc_mm:.1f} mm/day",
                field_value=etc_mm,
                threshold_value=5.0
            ))
        
        # Rainfall expected
        if forecast_rain_mm >= 5.0:
            severity = min(1.0, forecast_rain_mm / 20.0)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.RAINFALL_EXPECTED,
                severity=severity,
                description=f"Rainfall forecast: {forecast_rain_mm:.1f} mm in 24h",
                field_value=forecast_rain_mm,
                threshold_value=5.0
            ))
        
        # Critical growth window
        if crop_context.is_critical_window and current_vwc < crop_context.vwc_optimal * 1.1:
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.GROWTH_STAGE_CRITICAL,
                severity=0.7,
                description=f"Critical growth stage ({crop_context.growth_stage}) requires optimal moisture",
                field_value=current_vwc,
                threshold_value=crop_context.vwc_optimal
            ))
        
        # Temperature extreme
        if ambient_temp > 40:
            severity = min(1.0, (ambient_temp - 40) / 10)
            triggers.append(DecisionTrigger(
                trigger_type=TriggerType.TEMPERATURE_EXTREME,
                severity=severity * 0.6,
                description=f"High temperature stress: {ambient_temp:.1f}°C",
                field_value=ambient_temp,
                threshold_value=40.0
            ))
        
        return sorted(triggers, key=lambda t: t.severity, reverse=True)
    
    def fuse(
        self,
        sensor_data: SensorData,
        weather_data: WeatherData,
        crop_type: str = "wheat",
        days_after_sowing: int = 60,
        latitude: float = 20.0
    ) -> EnvironmentalState:
        """
        Fuse sensor data and weather data into unified EnvironmentalState.
        
        This is the main fusion method implementing the methodology's
        "Fuse SensorData and WeatherData into EnvironmentalState" step.
        """
        # Build crop context
        crop_context = CropContext.from_config(
            crop_type=crop_type,
            days_after_sowing=days_after_sowing,
            crop_params=self.crop_params
        )
        
        # Compute ET0 (reference evapotranspiration)
        et0 = self.compute_et0_hargreaves(
            temp_mean=(weather_data.forecast_temp_max_c + weather_data.forecast_temp_min_c) / 2,
            temp_max=weather_data.forecast_temp_max_c,
            temp_min=weather_data.forecast_temp_min_c,
            latitude=latitude
        )
        
        # Compute ETc (crop evapotranspiration)
        etc = et0 * crop_context.kc
        
        # Water deficit
        water_deficit = etc - weather_data.forecast_rain_24h_mm
        
        # Predict 24h soil moisture
        predicted_vwc = self.predict_vwc_24h(
            current_vwc=sensor_data.soil_moisture_vwc,
            etc_mm=etc,
            forecast_rain_mm=weather_data.forecast_rain_24h_mm
        )
        
        # Calculate data staleness (handle both naive and aware datetimes)
        now = utc_now()
        sensor_ts = sensor_data.timestamp
        if sensor_ts.tzinfo is None:
            sensor_ts = sensor_ts.replace(tzinfo=timezone.utc)
        staleness = (now - sensor_ts).total_seconds() / 60
        
        # Identify triggers
        triggers = self.identify_triggers(
            current_vwc=sensor_data.soil_moisture_vwc,
            predicted_vwc=predicted_vwc,
            etc_mm=etc,
            forecast_rain_mm=weather_data.forecast_rain_24h_mm,
            ambient_temp=sensor_data.ambient_temperature_c,
            crop_context=crop_context
        )
        
        return EnvironmentalState(
            current_vwc=sensor_data.soil_moisture_vwc,
            predicted_vwc_24h=predicted_vwc,
            soil_temp_c=sensor_data.soil_temperature_c,
            ambient_temp_c=sensor_data.ambient_temperature_c,
            humidity_pct=sensor_data.ambient_humidity_pct,
            forecast_rain_24h_mm=weather_data.forecast_rain_24h_mm,
            forecast_rain_48h_mm=weather_data.forecast_rain_48h_mm,
            rain_probability_pct=weather_data.rain_probability_pct,
            et0_mm_day=et0,
            etc_mm_day=etc,
            water_deficit_mm=water_deficit,
            crop_type=crop_type,
            growth_stage=crop_context.growth_stage,
            vwc_critical=crop_context.vwc_critical,
            vwc_optimal=crop_context.vwc_optimal,
            is_critical_window=crop_context.is_critical_window,
            sensor_reliability=sensor_data.reliability_score,
            forecast_confidence=weather_data.confidence_score,
            data_staleness_minutes=int(staleness),
            triggers=triggers,
            timestamp=utc_now()
        )
