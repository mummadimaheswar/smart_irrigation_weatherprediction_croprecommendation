"""
Decision Report Generator

Generates structured, explainable decision reports that include:
- Irrigation Control Decision (EXECUTE/DEFER/OVERRIDE/SKIP)
- Crop Guidance Report (water demand explanation, environmental justification)
- Decision Confidence Score

This module implements the output generation described in the methodology.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json

from .environmental_state import EnvironmentalState
from .decision_arbitration import ArbitrationResult, IrrigationAction


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class CropGuidance:
    """
    Context-aware crop guidance explaining the decision rationale.
    
    Includes:
    - Water demand explanation
    - Environmental justification
    - Recommended action rationale
    """
    water_demand_explanation: str
    environmental_justification: str
    action_rationale: str
    crop_health_tips: List[str]
    risk_factors: List[str]
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "water_demand_explanation": self.water_demand_explanation,
            "environmental_justification": self.environmental_justification,
            "action_rationale": self.action_rationale,
            "crop_health_tips": self.crop_health_tips,
            "risk_factors": self.risk_factors,
            "recommended_actions": self.recommended_actions
        }


@dataclass
class DecisionReport:
    """
    Complete structured decision report.
    
    This is the primary output format as specified in the methodology,
    containing all information for decision transparency and user guidance.
    """
    # Header
    report_id: str
    timestamp: datetime
    location: Dict[str, Any]
    
    # Primary Decision
    irrigation_decision: str  # EXECUTE, DEFER, OVERRIDE, SKIP
    decision_confidence: float
    
    # Environmental Summary
    environmental_summary: Dict[str, Any]
    
    # Crop Guidance
    crop_guidance: CropGuidance
    
    # Technical Details
    arbitration_details: Dict[str, Any]
    control_signal: Dict[str, Any]
    
    # Metadata
    data_quality_indicators: Dict[str, float]
    next_evaluation: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "irrigation_decision": self.irrigation_decision,
            "decision_confidence": round(self.decision_confidence, 3),
            "environmental_summary": self.environmental_summary,
            "crop_guidance": self.crop_guidance.to_dict(),
            "arbitration_details": self.arbitration_details,
            "control_signal": self.control_signal,
            "data_quality_indicators": {k: round(v, 3) for k, v in self.data_quality_indicators.items()},
            "next_evaluation": self.next_evaluation.isoformat()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_text(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 60,
            "SMART IRRIGATION DECISION REPORT",
            "=" * 60,
            f"Report ID: {self.report_id}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "-" * 40,
            "DECISION SUMMARY",
            "-" * 40,
            f"Action: {self.irrigation_decision}",
            f"Confidence: {self.decision_confidence:.0%}",
            "",
        ]
        
        # Environmental conditions
        lines.extend([
            "-" * 40,
            "ENVIRONMENTAL CONDITIONS",
            "-" * 40,
            f"Soil Moisture: {self.environmental_summary.get('current_vwc', 0):.0%} ({self.environmental_summary.get('moisture_status', 'unknown')})",
            f"Predicted (24h): {self.environmental_summary.get('predicted_vwc_24h', 0):.0%}",
            f"Crop Water Demand: {self.environmental_summary.get('etc_mm_day', 0):.1f} mm/day",
            f"Forecast Rain (24h): {self.environmental_summary.get('forecast_rain_24h_mm', 0):.1f} mm",
            f"Water Deficit: {self.environmental_summary.get('water_deficit_mm', 0):.1f} mm",
            "",
        ])
        
        # Crop guidance
        lines.extend([
            "-" * 40,
            "CROP GUIDANCE",
            "-" * 40,
            f"Water Demand: {self.crop_guidance.water_demand_explanation}",
            f"Environment: {self.crop_guidance.environmental_justification}",
            f"Rationale: {self.crop_guidance.action_rationale}",
            "",
            "Recommendations:",
        ])
        for rec in self.crop_guidance.recommended_actions:
            lines.append(f"  • {rec}")
        
        if self.crop_guidance.risk_factors:
            lines.extend(["", "Risk Factors:"])
            for risk in self.crop_guidance.risk_factors:
                lines.append(f"  ⚠ {risk}")
        
        # Control signal
        if self.irrigation_decision == "EXECUTE":
            lines.extend([
                "",
                "-" * 40,
                "IRRIGATION PRESCRIPTION",
                "-" * 40,
                f"Water Amount: {self.control_signal.get('target_volume_mm', 0):.1f} mm",
                f"Duration: {self.control_signal.get('duration_minutes', 0):.0f} minutes",
            ])
        
        # Next evaluation
        lines.extend([
            "",
            "-" * 40,
            f"Next Evaluation: {self.next_evaluation.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class CropGuidanceGenerator:
    """
    Generates context-aware crop guidance based on environmental state and decision.
    
    This component creates human-readable explanations of:
    - Why irrigation is or isn't needed
    - Current crop water demand
    - Environmental conditions affecting the crop
    - Actionable recommendations
    """
    
    # Crop-specific hints
    CROP_HINTS = {
        "rice": {
            "water_needs": "Rice requires saturated to flooded conditions during most growth stages.",
            "stress_signs": "Leaf rolling and yellowing indicate water stress.",
            "tips": ["Maintain 5-10cm standing water during tillering", "Check bunds for leakage"]
        },
        "wheat": {
            "water_needs": "Wheat has moderate water needs, critical at crown root initiation and grain filling.",
            "stress_signs": "Wilting and reduced tillering indicate moisture deficit.",
            "tips": ["Avoid waterlogging", "Critical irrigation at 21, 45, 65, 85 days after sowing"]
        },
        "maize": {
            "water_needs": "Maize is highly sensitive to water stress, especially at tasseling and silking.",
            "stress_signs": "Rolled leaves and delayed silking indicate stress.",
            "tips": ["Ensure adequate moisture 2 weeks before and after tasseling", "Avoid stress during pollination"]
        },
        "cotton": {
            "water_needs": "Cotton needs moderate water, with peak demand during boll development.",
            "stress_signs": "Premature leaf shedding and boll dropping.",
            "tips": ["Avoid excess water during vegetative phase", "Critical irrigation during flowering"]
        },
        "sugarcane": {
            "water_needs": "Sugarcane has high water requirements throughout its long growing season.",
            "stress_signs": "Reduced cane elongation and leaf drying.",
            "tips": ["Maintain consistent moisture", "Reduce irrigation before harvest for sugar concentration"]
        },
        "soybean": {
            "water_needs": "Soybean needs adequate moisture during flowering and pod filling.",
            "stress_signs": "Flower abortion and poor pod set.",
            "tips": ["Critical irrigation at R3-R5 stages", "Avoid waterlogging"]
        }
    }
    
    DEFAULT_HINTS = {
        "water_needs": "Maintain soil moisture between wilting point and field capacity.",
        "stress_signs": "Watch for wilting, leaf discoloration, and reduced growth.",
        "tips": ["Monitor soil moisture regularly", "Adjust irrigation based on weather forecasts"]
    }
    
    def generate(
        self,
        state: EnvironmentalState,
        result: ArbitrationResult
    ) -> CropGuidance:
        """Generate comprehensive crop guidance."""
        crop_info = self.CROP_HINTS.get(state.crop_type.lower(), self.DEFAULT_HINTS)
        
        # Water demand explanation
        if state.etc_mm_day > 6:
            demand_level = "very high"
        elif state.etc_mm_day > 4:
            demand_level = "high"
        elif state.etc_mm_day > 2.5:
            demand_level = "moderate"
        else:
            demand_level = "low"
        
        water_demand = (
            f"{state.crop_type.capitalize()} at {state.growth_stage} stage has {demand_level} water demand "
            f"({state.etc_mm_day:.1f} mm/day). {crop_info['water_needs']}"
        )
        
        # Environmental justification
        env_factors = []
        if state.ambient_temp_c > 35:
            env_factors.append(f"High temperature ({state.ambient_temp_c:.1f}°C) increases evapotranspiration")
        if state.humidity_pct < 40:
            env_factors.append(f"Low humidity ({state.humidity_pct:.0f}%) accelerates soil drying")
        if state.forecast_rain_24h_mm > 5:
            env_factors.append(f"Rain expected ({state.forecast_rain_24h_mm:.1f}mm in 24h)")
        if state.water_deficit_mm > 5:
            env_factors.append(f"Water deficit of {state.water_deficit_mm:.1f}mm accumulating")
        
        if not env_factors:
            env_factors.append("Environmental conditions are favorable for crop growth")
        
        environmental_justification = ". ".join(env_factors) + "."
        
        # Action rationale
        if result.action == IrrigationAction.EXECUTE:
            action_rationale = (
                f"Irrigation recommended due to: {result.primary_reason}. "
                f"Apply {result.recommended_water_mm:.1f}mm over {result.recommended_duration_minutes:.0f} minutes."
            )
        elif result.action == IrrigationAction.DEFER:
            action_rationale = (
                f"Irrigation deferred: {result.primary_reason}. "
                f"Re-evaluate in {result.next_evaluation_hours:.0f} hours."
            )
        elif result.action == IrrigationAction.OVERRIDE:
            action_rationale = (
                f"Decision overridden: {result.primary_reason}. "
                f"Manual intervention may be required."
            )
        else:
            action_rationale = (
                f"No irrigation needed: {result.primary_reason}. "
                f"Soil moisture is adequate for current crop requirements."
            )
        
        # Crop health tips
        tips = list(crop_info.get("tips", []))
        if state.is_critical_window:
            tips.insert(0, f"⚠ Critical growth stage - prioritize adequate moisture")
        
        # Risk factors
        risks = []
        if state.current_vwc < state.vwc_critical * 1.2:
            risks.append(f"Soil moisture approaching critical level ({state.vwc_critical:.0%})")
        if state.ambient_temp_c > 40:
            risks.append("Heat stress risk - consider irrigation for cooling")
        if state.forecast_rain_24h_mm > 30:
            risks.append("Heavy rain expected - watch for waterlogging")
        if state.humidity_pct > 90 and state.ambient_temp_c > 25:
            risks.append("High humidity - monitor for fungal diseases")
        
        # Recommended actions
        actions = []
        if result.action == IrrigationAction.EXECUTE:
            actions.append(f"Apply {result.recommended_water_mm:.1f}mm irrigation")
            actions.append("Irrigate during cooler hours (early morning or evening)")
        elif result.action == IrrigationAction.DEFER:
            actions.append(f"Wait {result.next_evaluation_hours:.0f} hours before re-evaluation")
            if state.forecast_rain_24h_mm > 5:
                actions.append("Monitor actual rainfall and adjust plans")
        
        actions.append("Continue monitoring soil moisture sensors")
        if state.is_critical_window:
            actions.append("Inspect crop for stress symptoms")
        
        return CropGuidance(
            water_demand_explanation=water_demand,
            environmental_justification=environmental_justification,
            action_rationale=action_rationale,
            crop_health_tips=tips,
            risk_factors=risks,
            recommended_actions=actions
        )


class DecisionReportGenerator:
    """
    Generates comprehensive decision reports.
    
    Combines environmental state, arbitration results, and crop guidance
    into a complete, structured report suitable for:
    - User display
    - Logging and audit trails
    - API responses
    - Integration with farm management systems
    """
    
    def __init__(self):
        self.guidance_generator = CropGuidanceGenerator()
        self.report_counter = 0
    
    def generate_report_id(self) -> str:
        """Generate unique report ID."""
        self.report_counter += 1
        timestamp = utc_now().strftime("%Y%m%d%H%M%S")
        return f"IRR-{timestamp}-{self.report_counter:04d}"
    
    def generate(
        self,
        state: EnvironmentalState,
        result: ArbitrationResult,
        control_signal: Dict[str, Any],
        location: Optional[Dict[str, Any]] = None
    ) -> DecisionReport:
        """
        Generate complete decision report.
        
        This is the main report generation method, producing the structured
        output described in the methodology.
        """
        # Generate crop guidance
        guidance = self.guidance_generator.generate(state, result)
        
        # Calculate next evaluation time
        next_eval = utc_now() + timedelta(hours=result.next_evaluation_hours)
        
        # Build report
        return DecisionReport(
            report_id=self.generate_report_id(),
            timestamp=utc_now(),
            location=location or {"state": "Unknown", "district": "Unknown"},
            irrigation_decision=result.action.value.upper(),
            decision_confidence=result.confidence_score,
            environmental_summary=state.to_dict(),
            crop_guidance=guidance,
            arbitration_details=result.to_dict(),
            control_signal=control_signal,
            data_quality_indicators={
                "sensor_reliability": state.sensor_reliability,
                "forecast_confidence": state.forecast_confidence,
                "overall_data_quality": state.overall_data_quality,
                "consistency_score": result.consistency_metrics.overall_consistency
            },
            next_evaluation=next_eval
        )


def generate_quick_report(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    forecast_rain: float,
    crop_type: str = "wheat",
    days_after_sowing: int = 60
) -> str:
    """
    Quick report generation for simple use cases.
    
    Convenience function that creates minimal inputs and generates a text report.
    """
    from .environmental_state import SensorData, WeatherData, EnvironmentalFusionEngine
    from .decision_arbitration import DecisionArbitrator, ActuationController
    
    # Create sensor data
    sensor = SensorData(
        soil_moisture_vwc=soil_moisture,
        soil_temperature_c=temperature - 5,
        ambient_temperature_c=temperature,
        ambient_humidity_pct=humidity,
        timestamp=utc_now()
    )
    
    # Create weather data
    weather = WeatherData(
        forecast_rain_24h_mm=forecast_rain,
        forecast_rain_48h_mm=forecast_rain * 1.5,
        forecast_temp_max_c=temperature + 5,
        forecast_temp_min_c=temperature - 8,
        forecast_humidity_pct=humidity,
        forecast_wind_speed_ms=3.0,
        forecast_cloud_cover_pct=50.0,
        rain_probability_pct=min(90, forecast_rain * 10),
        weather_description="partly cloudy",
        forecast_timestamp=utc_now()
    )
    
    # Fuse data
    fusion = EnvironmentalFusionEngine()
    state = fusion.fuse(sensor, weather, crop_type, days_after_sowing)
    
    # Arbitrate
    arbitrator = DecisionArbitrator()
    result = arbitrator.arbitrate(state)
    
    # Generate control signal
    controller = ActuationController()
    signal = controller.generate_control_signal(result)
    
    # Generate report
    generator = DecisionReportGenerator()
    report = generator.generate(state, result, signal)
    
    return report.to_text()
