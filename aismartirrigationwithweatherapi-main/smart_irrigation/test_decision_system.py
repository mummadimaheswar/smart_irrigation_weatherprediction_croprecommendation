"""
Test Script for Smart Irrigation Decision System

Demonstrates the complete decision pipeline:
1. Environmental State Fusion
2. Decision Arbitration (EXECUTE/DEFER/OVERRIDE)
3. Confidence Scoring
4. Decision Report Generation

Run with: python -m smart_irrigation.test_decision_system
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_irrigation.environmental_state import (
    SensorData, WeatherData, EnvironmentalFusionEngine
)
from smart_irrigation.decision_arbitration import (
    DecisionArbitrator, SafetyConstraints, IrrigationAction
)
from smart_irrigation.decision_report import (
    DecisionReportGenerator, generate_quick_report
)
from smart_irrigation.orchestrator import SmartIrrigationOrchestrator
from smart_irrigation.config import CROP_PARAMS


def test_environmental_fusion():
    """Test the environmental state fusion module."""
    print("\n" + "=" * 60)
    print("TEST 1: Environmental State Fusion")
    print("=" * 60)
    
    # Create sensor data
    sensor = SensorData(
        soil_moisture_vwc=0.22,  # 22% VWC
        soil_temperature_c=25.0,
        ambient_temperature_c=32.0,
        ambient_humidity_pct=65.0,
        timestamp=datetime.utcnow(),
        reliability_score=0.9
    )
    
    # Create weather data
    weather = WeatherData(
        forecast_rain_24h_mm=2.5,
        forecast_rain_48h_mm=8.0,
        forecast_temp_max_c=35.0,
        forecast_temp_min_c=24.0,
        forecast_humidity_pct=60.0,
        forecast_wind_speed_ms=3.5,
        forecast_cloud_cover_pct=40.0,
        rain_probability_pct=25.0,
        weather_description="partly cloudy",
        forecast_timestamp=datetime.utcnow(),
        confidence_score=0.85
    )
    
    # Fuse data
    fusion = EnvironmentalFusionEngine(CROP_PARAMS)
    state = fusion.fuse(
        sensor_data=sensor,
        weather_data=weather,
        crop_type="wheat",
        days_after_sowing=60,
        latitude=20.0
    )
    
    print(f"\n✓ Sensor Data:")
    print(f"  Soil Moisture: {sensor.soil_moisture_vwc:.0%}")
    print(f"  Temperature: {sensor.ambient_temperature_c:.1f}°C")
    
    print(f"\n✓ Weather Forecast:")
    print(f"  Rain (24h): {weather.forecast_rain_24h_mm:.1f} mm")
    print(f"  Temp Range: {weather.forecast_temp_min_c:.1f} - {weather.forecast_temp_max_c:.1f}°C")
    
    print(f"\n✓ Fused Environmental State:")
    print(f"  Current VWC: {state.current_vwc:.0%}")
    print(f"  Predicted VWC (24h): {state.predicted_vwc_24h:.0%}")
    print(f"  ET0: {state.et0_mm_day:.2f} mm/day")
    print(f"  ETc: {state.etc_mm_day:.2f} mm/day")
    print(f"  Water Deficit: {state.water_deficit_mm:.2f} mm")
    print(f"  Moisture Status: {state.moisture_status}")
    print(f"  Data Quality: {state.overall_data_quality:.0%}")
    
    print(f"\n✓ Decision Triggers ({len(state.triggers)}):")
    for trigger in state.triggers:
        print(f"  - {trigger.trigger_type.value}: {trigger.description} (severity: {trigger.severity:.2f})")
    
    return state


def test_decision_arbitration(state=None):
    """Test the decision arbitration module."""
    print("\n" + "=" * 60)
    print("TEST 2: Decision Arbitration")
    print("=" * 60)
    
    if state is None:
        # Create a test state if not provided
        state = test_environmental_fusion()
    
    # Create arbitrator with safety constraints
    constraints = SafetyConstraints(
        min_irrigation_interval_hours=12.0,
        max_daily_water_mm=50.0,
        equipment_available=True,
        water_quota_remaining_mm=100.0
    )
    
    arbitrator = DecisionArbitrator(constraints)
    
    # Run arbitration
    result = arbitrator.arbitrate(state)
    
    print(f"\n✓ Consistency Metrics:")
    metrics = result.consistency_metrics
    print(f"  Field-Forecast Alignment: {metrics.field_forecast_alignment:.0%}")
    print(f"  Temporal Consistency: {metrics.temporal_consistency:.0%}")
    print(f"  Crop Demand Match: {metrics.crop_demand_match:.0%}")
    print(f"  Trigger Coherence: {metrics.trigger_coherence:.0%}")
    print(f"  Overall Consistency: {metrics.overall_consistency:.0%}")
    
    print(f"\n✓ Decision Result:")
    print(f"  Action: {result.action.value.upper()}")
    print(f"  Confidence: {result.confidence_score:.0%}")
    print(f"  Primary Reason: {result.primary_reason}")
    
    if result.secondary_reasons:
        print(f"  Secondary Reasons:")
        for reason in result.secondary_reasons:
            print(f"    - {reason}")
    
    if result.action == IrrigationAction.EXECUTE:
        print(f"\n✓ Irrigation Prescription:")
        print(f"  Water Amount: {result.recommended_water_mm:.1f} mm")
        print(f"  Duration: {result.recommended_duration_minutes:.0f} minutes")
    
    print(f"\n  Next Evaluation: {result.next_evaluation_hours:.0f} hours")
    
    return result


def test_full_orchestrator():
    """Test the complete orchestrator pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Orchestrator Pipeline")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = SmartIrrigationOrchestrator()
    
    # Run decision cycle for Maharashtra wheat field
    print("\n✓ Running decision cycle for Maharashtra wheat field...")
    
    report = orchestrator.run_for_location(
        state="MAHARASHTRA",
        district="PUNE",
        crop_type="wheat",
        days_after_sowing=60
    )
    
    print("\n" + "-" * 40)
    print("DECISION REPORT")
    print("-" * 40)
    print(report.to_text())
    
    return report


def test_multiple_scenarios():
    """Test different irrigation scenarios."""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Critical Moisture - Must Irrigate",
            "soil_moisture": 0.12,
            "temperature": 35,
            "humidity": 40,
            "forecast_rain": 0,
            "crop": "wheat"
        },
        {
            "name": "Rain Expected - Defer",
            "soil_moisture": 0.25,
            "temperature": 28,
            "humidity": 75,
            "forecast_rain": 15,
            "crop": "rice"
        },
        {
            "name": "Adequate Moisture - Skip",
            "soil_moisture": 0.32,
            "temperature": 26,
            "humidity": 60,
            "forecast_rain": 2,
            "crop": "maize"
        },
        {
            "name": "High ET Demand - Execute",
            "soil_moisture": 0.20,
            "temperature": 42,
            "humidity": 25,
            "forecast_rain": 0,
            "crop": "cotton"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        
        # Create sensor data
        sensor = SensorData(
            soil_moisture_vwc=scenario["soil_moisture"],
            soil_temperature_c=scenario["temperature"] - 5,
            ambient_temperature_c=scenario["temperature"],
            ambient_humidity_pct=scenario["humidity"],
            timestamp=datetime.utcnow(),
            reliability_score=0.9
        )
        
        # Create weather data
        weather = WeatherData(
            forecast_rain_24h_mm=scenario["forecast_rain"],
            forecast_rain_48h_mm=scenario["forecast_rain"] * 1.5,
            forecast_temp_max_c=scenario["temperature"] + 5,
            forecast_temp_min_c=scenario["temperature"] - 8,
            forecast_humidity_pct=scenario["humidity"],
            forecast_wind_speed_ms=3.0,
            forecast_cloud_cover_pct=50.0,
            rain_probability_pct=min(90, scenario["forecast_rain"] * 5),
            weather_description="variable",
            forecast_timestamp=datetime.utcnow(),
            confidence_score=0.8
        )
        
        # Fuse and arbitrate
        fusion = EnvironmentalFusionEngine(CROP_PARAMS)
        state = fusion.fuse(sensor, weather, scenario["crop"], 60)
        
        arbitrator = DecisionArbitrator()
        result = arbitrator.arbitrate(state)
        
        print(f"  Moisture: {scenario['soil_moisture']:.0%} | Rain: {scenario['forecast_rain']}mm")
        print(f"  → Decision: {result.action.value.upper()} (confidence: {result.confidence_score:.0%})")
        print(f"  → Reason: {result.primary_reason}")


def test_quick_report():
    """Test the quick report generation function."""
    print("\n" + "=" * 60)
    print("TEST 5: Quick Report Generation")
    print("=" * 60)
    
    report_text = generate_quick_report(
        soil_moisture=0.18,
        temperature=34,
        humidity=45,
        forecast_rain=3,
        crop_type="cotton",
        days_after_sowing=90
    )
    
    print(report_text)


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# SMART IRRIGATION DECISION SYSTEM - TEST SUITE")
    print("#" * 60)
    
    try:
        # Test 1: Environmental Fusion
        state = test_environmental_fusion()
        print("\n✅ Environmental Fusion Test PASSED")
        
        # Test 2: Decision Arbitration
        result = test_decision_arbitration(state)
        print("\n✅ Decision Arbitration Test PASSED")
        
        # Test 3: Full Orchestrator
        report = test_full_orchestrator()
        print("\n✅ Full Orchestrator Test PASSED")
        
        # Test 4: Multiple Scenarios
        test_multiple_scenarios()
        print("\n✅ Multiple Scenarios Test PASSED")
        
        # Test 5: Quick Report
        test_quick_report()
        print("\n✅ Quick Report Test PASSED")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
