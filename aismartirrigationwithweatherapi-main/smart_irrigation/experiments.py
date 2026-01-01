"""
Experiment Suite for Smart Irrigation Decision System

Generates quantitative results for evaluating:
1. Decision accuracy across scenarios
2. Water savings potential
3. Response time analysis
4. Consistency scoring validation
5. Multi-region/multi-crop performance

Run with: python -m smart_irrigation.experiments
"""

import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .environmental_state import (
    SensorData, WeatherData, EnvironmentalFusionEngine, TriggerType
)
from .decision_arbitration import (
    DecisionArbitrator, SafetyConstraints, IrrigationAction
)
from .decision_report import DecisionReportGenerator
from .orchestrator import SmartIrrigationOrchestrator
from .config import CROP_PARAMS, STATES, DISTRICTS

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""
    n_iterations: int = 100
    crops: List[str] = field(default_factory=lambda: ["wheat", "rice", "maize", "cotton", "sugarcane", "soybean"])
    states: List[str] = field(default_factory=lambda: ["MAHARASHTRA", "PUNJAB", "KARNATAKA", "UTTAR PRADESH", "ASSAM"])
    moisture_range: Tuple[float, float] = (0.10, 0.45)
    temperature_range: Tuple[float, float] = (15.0, 45.0)
    rain_range: Tuple[float, float] = (0.0, 50.0)
    random_seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    scenario_id: str
    crop_type: str
    state: str
    soil_moisture: float
    temperature: float
    humidity: float
    forecast_rain: float
    decision: str
    confidence: float
    consistency_score: float
    water_recommended_mm: float
    execution_time_ms: float
    triggers_count: int
    is_critical: bool
    moisture_status: str


@dataclass
class AggregatedResults:
    """Aggregated experiment results for analysis."""
    total_runs: int
    execute_count: int
    defer_count: int
    skip_count: int
    override_count: int
    avg_confidence: float
    avg_consistency: float
    avg_execution_time_ms: float
    water_savings_potential_pct: float
    decision_distribution: Dict[str, float]
    confidence_by_decision: Dict[str, float]
    results_by_crop: Dict[str, Dict]
    results_by_state: Dict[str, Dict]


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class GroundTruthGenerator:
    """
    Generates expected decisions based on agronomic rules.
    Used for validating system decision accuracy.
    """
    
    def __init__(self):
        self.crop_params = CROP_PARAMS
    
    def get_expected_decision(
        self,
        soil_moisture: float,
        forecast_rain: float,
        crop_type: str,
        temperature: float
    ) -> str:
        """
        Determine expected decision based on established agronomic rules.
        """
        params = self.crop_params.get(crop_type, self.crop_params.get("wheat", {}))
        vwc_critical = params.get("vwc_critical", 0.18)
        vwc_optimal = params.get("vwc_optimal", 0.28)
        
        # Rule 1: Critical moisture - always irrigate
        if soil_moisture < vwc_critical:
            return "execute"
        
        # Rule 2: Significant rain expected and moisture adequate
        if forecast_rain >= 10 and soil_moisture >= vwc_critical * 1.2:
            return "defer"
        
        # Rule 3: Adequate moisture
        if soil_moisture >= vwc_optimal:
            return "skip"
        
        # Rule 4: Marginal moisture with light rain expected
        if forecast_rain >= 5 and soil_moisture >= vwc_critical * 1.1:
            return "defer"
        
        # Rule 5: Marginal moisture, no rain - irrigate
        if soil_moisture < vwc_optimal:
            return "execute"
        
        return "skip"


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Runs systematic experiments to evaluate system performance.
    """
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.fusion_engine = EnvironmentalFusionEngine(CROP_PARAMS)
        self.arbitrator = DecisionArbitrator()
        self.ground_truth = GroundTruthGenerator()
        self.results: List[ExperimentResult] = []
        
        np.random.seed(self.config.random_seed)
    
    def generate_scenario(
        self,
        scenario_id: int,
        crop: str,
        state: str
    ) -> Tuple[SensorData, WeatherData]:
        """Generate a random but realistic scenario."""
        # Random environmental conditions
        soil_moisture = np.random.uniform(*self.config.moisture_range)
        temperature = np.random.uniform(*self.config.temperature_range)
        humidity = np.random.uniform(30, 90)
        
        # Rain follows exponential distribution (more common to have low/no rain)
        rain_prob = np.random.random()
        if rain_prob < 0.6:
            forecast_rain = 0.0
        elif rain_prob < 0.85:
            forecast_rain = np.random.exponential(5)
        else:
            forecast_rain = np.random.exponential(15)
        
        forecast_rain = min(forecast_rain, self.config.rain_range[1])
        
        sensor = SensorData(
            soil_moisture_vwc=soil_moisture,
            soil_temperature_c=temperature - 5,
            ambient_temperature_c=temperature,
            ambient_humidity_pct=humidity,
            timestamp=datetime.utcnow(),
            reliability_score=np.random.uniform(0.7, 1.0)
        )
        
        weather = WeatherData(
            forecast_rain_24h_mm=forecast_rain,
            forecast_rain_48h_mm=forecast_rain * 1.5,
            forecast_temp_max_c=temperature + np.random.uniform(3, 8),
            forecast_temp_min_c=temperature - np.random.uniform(5, 10),
            forecast_humidity_pct=humidity,
            forecast_wind_speed_ms=np.random.uniform(1, 8),
            forecast_cloud_cover_pct=np.random.uniform(10, 80),
            rain_probability_pct=min(95, forecast_rain * 5),
            weather_description="variable",
            forecast_timestamp=datetime.utcnow(),
            confidence_score=np.random.uniform(0.6, 0.95)
        )
        
        return sensor, weather, soil_moisture, temperature, humidity, forecast_rain
    
    def run_single_experiment(
        self,
        scenario_id: int,
        crop: str,
        state: str,
        days_after_sowing: int = None
    ) -> ExperimentResult:
        """Run a single experiment scenario."""
        if days_after_sowing is None:
            days_after_sowing = np.random.randint(30, 100)
        
        # Generate scenario
        sensor, weather, moisture, temp, humidity, rain = self.generate_scenario(
            scenario_id, crop, state
        )
        
        # Measure execution time
        start_time = time.perf_counter()
        
        # Fuse environmental state
        state_obj = self.fusion_engine.fuse(
            sensor_data=sensor,
            weather_data=weather,
            crop_type=crop,
            days_after_sowing=days_after_sowing,
            latitude=20.0
        )
        
        # Run arbitration
        result = self.arbitrator.arbitrate(state_obj)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return ExperimentResult(
            scenario_id=f"{state}_{crop}_{scenario_id:04d}",
            crop_type=crop,
            state=state,
            soil_moisture=moisture,
            temperature=temp,
            humidity=humidity,
            forecast_rain=rain,
            decision=result.action.value,
            confidence=result.confidence_score,
            consistency_score=result.consistency_metrics.overall_consistency,
            water_recommended_mm=result.recommended_water_mm,
            execution_time_ms=execution_time_ms,
            triggers_count=len(state_obj.triggers),
            is_critical=moisture < state_obj.vwc_critical,
            moisture_status=state_obj.moisture_status
        )
    
    def run_experiments(self, verbose: bool = True) -> List[ExperimentResult]:
        """Run all experiments across crops and states."""
        self.results = []
        total = self.config.n_iterations * len(self.config.crops) * len(self.config.states)
        
        if verbose:
            print(f"\nRunning {total} experiments...")
            print(f"  Crops: {self.config.crops}")
            print(f"  States: {self.config.states}")
            print(f"  Iterations per combination: {self.config.n_iterations}")
        
        scenario_id = 0
        for crop in self.config.crops:
            for state in self.config.states:
                for i in range(self.config.n_iterations):
                    result = self.run_single_experiment(scenario_id, crop, state)
                    self.results.append(result)
                    scenario_id += 1
        
        if verbose:
            print(f"  Completed {len(self.results)} experiments")
        
        return self.results
    
    def compute_accuracy(self) -> Dict[str, float]:
        """Compute decision accuracy against ground truth."""
        correct = 0
        total = len(self.results)
        
        for r in self.results:
            expected = self.ground_truth.get_expected_decision(
                r.soil_moisture, r.forecast_rain, r.crop_type, r.temperature
            )
            if r.decision == expected:
                correct += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }
    
    def aggregate_results(self) -> AggregatedResults:
        """Aggregate experiment results for analysis."""
        if not self.results:
            raise ValueError("No results to aggregate. Run experiments first.")
        
        df = pd.DataFrame([vars(r) for r in self.results])
        
        # Decision counts
        decision_counts = df['decision'].value_counts()
        total = len(df)
        
        # Compute water savings (comparing to threshold-only baseline)
        # Baseline: Always irrigate when moisture < optimal, using fixed 30mm
        baseline_water = 0
        smart_water = 0
        for _, row in df.iterrows():
            params = CROP_PARAMS.get(row['crop_type'], CROP_PARAMS.get('wheat', {}))
            vwc_optimal = params.get('vwc_optimal', 0.28)
            # Baseline: irrigate if below optimal with fixed amount
            if row['soil_moisture'] < vwc_optimal:
                baseline_water += 30.0
            # Smart system: variable amount only when executing
            if row['decision'] == 'execute':
                smart_water += row['water_recommended_mm']
        
        water_savings = (baseline_water - smart_water) / baseline_water * 100 if baseline_water > 0 else 0
        
        # Results by crop
        results_by_crop = {}
        for crop in self.config.crops:
            crop_df = df[df['crop_type'] == crop]
            results_by_crop[crop] = {
                "count": len(crop_df),
                "execute_pct": (crop_df['decision'] == 'execute').mean() * 100,
                "defer_pct": (crop_df['decision'] == 'defer').mean() * 100,
                "skip_pct": (crop_df['decision'] == 'skip').mean() * 100,
                "avg_confidence": crop_df['confidence'].mean(),
                "avg_water_mm": crop_df[crop_df['decision'] == 'execute']['water_recommended_mm'].mean()
            }
        
        # Results by state
        results_by_state = {}
        for state in self.config.states:
            state_df = df[df['state'] == state]
            results_by_state[state] = {
                "count": len(state_df),
                "execute_pct": (state_df['decision'] == 'execute').mean() * 100,
                "defer_pct": (state_df['decision'] == 'defer').mean() * 100,
                "skip_pct": (state_df['decision'] == 'skip').mean() * 100,
                "avg_confidence": state_df['confidence'].mean()
            }
        
        # Confidence by decision type
        confidence_by_decision = {}
        for decision in ['execute', 'defer', 'skip', 'override']:
            dec_df = df[df['decision'] == decision]
            confidence_by_decision[decision] = dec_df['confidence'].mean() if len(dec_df) > 0 else 0
        
        return AggregatedResults(
            total_runs=total,
            execute_count=decision_counts.get('execute', 0),
            defer_count=decision_counts.get('defer', 0),
            skip_count=decision_counts.get('skip', 0),
            override_count=decision_counts.get('override', 0),
            avg_confidence=df['confidence'].mean(),
            avg_consistency=df['consistency_score'].mean(),
            avg_execution_time_ms=df['execution_time_ms'].mean(),
            water_savings_potential_pct=water_savings,
            decision_distribution={
                k: v/total*100 for k, v in decision_counts.items()
            },
            confidence_by_decision=confidence_by_decision,
            results_by_crop=results_by_crop,
            results_by_state=results_by_state
        )


# ─────────────────────────────────────────────────────────────────────────────
# SPECIFIC EXPERIMENT PROTOCOLS
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1_decision_distribution(n_samples: int = 500) -> Dict:
    """
    Experiment 1: Decision Distribution Analysis
    
    Evaluates the distribution of EXECUTE/DEFER/SKIP decisions
    across varied environmental conditions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Decision Distribution Analysis")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    agg = runner.aggregate_results()
    accuracy = runner.compute_accuracy()
    
    print(f"\n📊 Results ({agg.total_runs} scenarios):")
    print(f"   Decision Distribution:")
    print(f"     EXECUTE: {agg.execute_count:4d} ({agg.decision_distribution.get('execute', 0):.1f}%)")
    print(f"     DEFER:   {agg.defer_count:4d} ({agg.decision_distribution.get('defer', 0):.1f}%)")
    print(f"     SKIP:    {agg.skip_count:4d} ({agg.decision_distribution.get('skip', 0):.1f}%)")
    print(f"     OVERRIDE:{agg.override_count:4d} ({agg.decision_distribution.get('override', 0):.1f}%)")
    print(f"\n   Decision Accuracy: {accuracy['accuracy']*100:.1f}%")
    print(f"   Average Confidence: {agg.avg_confidence*100:.1f}%")
    print(f"   Average Consistency: {agg.avg_consistency*100:.1f}%")
    
    return {
        "experiment": "Decision Distribution",
        "total_samples": agg.total_runs,
        "decision_distribution": agg.decision_distribution,
        "accuracy": accuracy['accuracy'],
        "avg_confidence": agg.avg_confidence,
        "avg_consistency": agg.avg_consistency
    }


def experiment_2_water_savings(n_samples: int = 500) -> Dict:
    """
    Experiment 2: Water Savings Potential
    
    Compares system water recommendations against fixed-schedule baseline.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Water Savings Analysis")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    agg = runner.aggregate_results()
    df = pd.DataFrame([vars(r) for r in runner.results])
    
    # Baseline: Threshold-based, irrigate when moisture < optimal with fixed 30mm
    baseline_water = 0
    smart_water = 0
    
    for _, row in df.iterrows():
        params = CROP_PARAMS.get(row['crop_type'], CROP_PARAMS.get('wheat', {}))
        vwc_optimal = params.get('vwc_optimal', 0.28)
        # Baseline: irrigate whenever below optimal
        if row['soil_moisture'] < vwc_optimal:
            baseline_water += 30.0
        
        # Smart system: defers when rain expected, applies variable amount
        if row['decision'] == 'execute':
            smart_water += row['water_recommended_mm']
    
    savings_pct = (baseline_water - smart_water) / baseline_water * 100 if baseline_water > 0 else 0
    
    print(f"\n💧 Water Usage Comparison:")
    print(f"   Baseline (threshold-based): {baseline_water:.1f} mm total")
    print(f"   Smart System:               {smart_water:.1f} mm total")
    print(f"   Water Savings:              {savings_pct:.1f}%")
    
    # Savings by crop
    print(f"\n   Savings by Crop Type:")
    for crop in config.crops:
        crop_df = df[df['crop_type'] == crop]
        params = CROP_PARAMS.get(crop, CROP_PARAMS.get('wheat', {}))
        vwc_optimal = params.get('vwc_optimal', 0.28)
        crop_baseline = sum(30.0 for _, r in crop_df.iterrows() if r['soil_moisture'] < vwc_optimal)
        crop_smart = crop_df[crop_df['decision'] == 'execute']['water_recommended_mm'].sum()
        crop_savings = (crop_baseline - crop_smart) / crop_baseline * 100 if crop_baseline > 0 else 0
        print(f"     {crop:12s}: {crop_savings:5.1f}% savings")
    
    return {
        "experiment": "Water Savings",
        "baseline_water_mm": baseline_water,
        "smart_water_mm": smart_water,
        "savings_percent": savings_pct,
        "savings_by_crop": agg.results_by_crop
    }


def experiment_3_response_time(n_samples: int = 1000) -> Dict:
    """
    Experiment 3: Response Time Analysis
    
    Measures decision latency for embedded/real-time deployment.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Response Time Analysis")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    df = pd.DataFrame([vars(r) for r in runner.results])
    
    times = df['execution_time_ms']
    
    print(f"\n⏱️  Execution Time Statistics:")
    print(f"   Mean:   {times.mean():.3f} ms")
    print(f"   Median: {times.median():.3f} ms")
    print(f"   Std:    {times.std():.3f} ms")
    print(f"   Min:    {times.min():.3f} ms")
    print(f"   Max:    {times.max():.3f} ms")
    print(f"   P95:    {times.quantile(0.95):.3f} ms")
    print(f"   P99:    {times.quantile(0.99):.3f} ms")
    
    # Real-time feasibility
    rt_threshold = 100  # 100ms for real-time
    rt_compliant = (times < rt_threshold).mean() * 100
    print(f"\n   Real-time Compliance (<{rt_threshold}ms): {rt_compliant:.1f}%")
    
    return {
        "experiment": "Response Time",
        "mean_ms": times.mean(),
        "median_ms": times.median(),
        "std_ms": times.std(),
        "p95_ms": times.quantile(0.95),
        "p99_ms": times.quantile(0.99),
        "realtime_compliant_pct": rt_compliant
    }


def experiment_4_consistency_validation(n_samples: int = 500) -> Dict:
    """
    Experiment 4: Consistency Score Validation
    
    Validates that consistency scores correctly reflect decision confidence.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Consistency Score Validation")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    df = pd.DataFrame([vars(r) for r in runner.results])
    
    # Correlation between consistency and confidence
    correlation = df['consistency_score'].corr(df['confidence'])
    
    print(f"\n📈 Consistency Analysis:")
    print(f"   Consistency-Confidence Correlation: {correlation:.3f}")
    
    # Consistency by decision type
    print(f"\n   Average Consistency by Decision:")
    for decision in ['execute', 'defer', 'skip']:
        dec_df = df[df['decision'] == decision]
        if len(dec_df) > 0:
            print(f"     {decision.upper():8s}: {dec_df['consistency_score'].mean()*100:.1f}%")
    
    # High vs Low consistency decisions
    high_consistency = df[df['consistency_score'] >= 0.7]
    low_consistency = df[df['consistency_score'] < 0.5]
    
    print(f"\n   Decision Distribution by Consistency Level:")
    print(f"     High (≥70%): {len(high_consistency)} decisions")
    print(f"       - Execute: {(high_consistency['decision']=='execute').sum()}")
    print(f"       - Defer:   {(high_consistency['decision']=='defer').sum()}")
    print(f"       - Skip:    {(high_consistency['decision']=='skip').sum()}")
    print(f"     Low (<50%):  {len(low_consistency)} decisions")
    print(f"       - Defer rate: {(low_consistency['decision']=='defer').mean()*100:.1f}%")
    
    return {
        "experiment": "Consistency Validation",
        "consistency_confidence_correlation": correlation,
        "high_consistency_count": len(high_consistency),
        "low_consistency_count": len(low_consistency),
        "low_consistency_defer_rate": (low_consistency['decision']=='defer').mean()
    }


def experiment_5_crop_specific_performance(n_samples: int = 600) -> Dict:
    """
    Experiment 5: Crop-Specific Performance
    
    Evaluates decision quality across different crop types.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Crop-Specific Performance")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    agg = runner.aggregate_results()
    
    print(f"\n🌾 Performance by Crop Type:")
    print(f"   {'Crop':12s} | {'Execute%':8s} | {'Defer%':8s} | {'Skip%':8s} | {'Confidence':10s} | {'Avg Water':10s}")
    print(f"   {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    for crop, stats in agg.results_by_crop.items():
        avg_water = stats.get('avg_water_mm', 0)
        if pd.isna(avg_water):
            avg_water = 0
        print(f"   {crop:12s} | {stats['execute_pct']:7.1f}% | {stats['defer_pct']:7.1f}% | "
              f"{stats['skip_pct']:7.1f}% | {stats['avg_confidence']*100:9.1f}% | {avg_water:8.1f} mm")
    
    return {
        "experiment": "Crop-Specific Performance",
        "results_by_crop": agg.results_by_crop
    }


def experiment_6_regional_analysis(n_samples: int = 500) -> Dict:
    """
    Experiment 6: Regional (State-wise) Analysis
    
    Evaluates decision patterns across Indian states.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Regional (State-wise) Analysis")
    print("=" * 70)
    
    config = ExperimentConfig(n_iterations=n_samples // 30)
    runner = ExperimentRunner(config)
    runner.run_experiments(verbose=True)
    
    agg = runner.aggregate_results()
    
    print(f"\n📍 Performance by State:")
    print(f"   {'State':16s} | {'Execute%':8s} | {'Defer%':8s} | {'Skip%':8s} | {'Confidence':10s}")
    print(f"   {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    
    for state, stats in agg.results_by_state.items():
        print(f"   {state:16s} | {stats['execute_pct']:7.1f}% | {stats['defer_pct']:7.1f}% | "
              f"{stats['skip_pct']:7.1f}% | {stats['avg_confidence']*100:9.1f}%")
    
    return {
        "experiment": "Regional Analysis",
        "results_by_state": agg.results_by_state
    }


def experiment_7_edge_cases() -> Dict:
    """
    Experiment 7: Edge Case Handling
    
    Tests system behavior under extreme/edge conditions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Edge Case Analysis")
    print("=" * 70)
    
    edge_cases = [
        {"name": "Critical Drought", "moisture": 0.08, "rain": 0, "temp": 42},
        {"name": "Flood Risk", "moisture": 0.50, "rain": 80, "temp": 28},
        {"name": "Frost Risk", "moisture": 0.25, "rain": 0, "temp": 1},
        {"name": "Sensor Degraded", "moisture": 0.22, "rain": 5, "temp": 30, "reliability": 0.3},
        {"name": "Conflicting Signals", "moisture": 0.15, "rain": 25, "temp": 35},
        {"name": "Optimal Conditions", "moisture": 0.30, "rain": 0, "temp": 28},
    ]
    
    fusion = EnvironmentalFusionEngine(CROP_PARAMS)
    arbitrator = DecisionArbitrator()
    
    results = []
    
    print(f"\n🔬 Edge Case Results:")
    print(f"   {'Scenario':20s} | {'Decision':10s} | {'Confidence':10s} | {'Reason'}")
    print(f"   {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*40}")
    
    for case in edge_cases:
        reliability = case.get('reliability', 0.9)
        
        sensor = SensorData(
            soil_moisture_vwc=case['moisture'],
            soil_temperature_c=case['temp'] - 5,
            ambient_temperature_c=case['temp'],
            ambient_humidity_pct=60,
            timestamp=datetime.utcnow(),
            reliability_score=reliability
        )
        
        weather = WeatherData(
            forecast_rain_24h_mm=case['rain'],
            forecast_rain_48h_mm=case['rain'] * 1.5,
            forecast_temp_max_c=case['temp'] + 5,
            forecast_temp_min_c=case['temp'] - 8,
            forecast_humidity_pct=60,
            forecast_wind_speed_ms=3,
            forecast_cloud_cover_pct=50,
            rain_probability_pct=min(90, case['rain'] * 3),
            weather_description="test",
            forecast_timestamp=datetime.utcnow(),
            confidence_score=0.8
        )
        
        state = fusion.fuse(sensor, weather, "wheat", 60)
        result = arbitrator.arbitrate(state)
        
        print(f"   {case['name']:20s} | {result.action.value.upper():10s} | "
              f"{result.confidence_score*100:9.1f}% | {result.primary_reason[:40]}")
        
        results.append({
            "scenario": case['name'],
            "decision": result.action.value,
            "confidence": result.confidence_score,
            "reason": result.primary_reason
        })
    
    return {
        "experiment": "Edge Cases",
        "results": results
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_table(all_results: List[Dict]) -> str:
    """Generate a formatted summary table for publication."""
    
    lines = [
        "",
        "=" * 80,
        "EXPERIMENT RESULTS SUMMARY",
        "=" * 80,
        "",
        "Table 1: System Performance Metrics",
        "-" * 80,
        f"{'Metric':<40} | {'Value':>15} | {'Unit':>10}",
        "-" * 80,
    ]
    
    # Extract key metrics
    for result in all_results:
        exp = result.get('experiment', '')
        
        if exp == 'Decision Distribution':
            lines.append(f"{'Decision Accuracy':<40} | {result['accuracy']*100:>14.1f}% | {'%':>10}")
            lines.append(f"{'Average Confidence Score':<40} | {result['avg_confidence']*100:>14.1f}% | {'%':>10}")
            lines.append(f"{'Average Consistency Score':<40} | {result['avg_consistency']*100:>14.1f}% | {'%':>10}")
        
        elif exp == 'Water Savings':
            lines.append(f"{'Water Savings vs Baseline':<40} | {result['savings_percent']:>14.1f}% | {'%':>10}")
        
        elif exp == 'Response Time':
            lines.append(f"{'Mean Response Time':<40} | {result['mean_ms']:>14.3f} | {'ms':>10}")
            lines.append(f"{'P95 Response Time':<40} | {result['p95_ms']:>14.3f} | {'ms':>10}")
            lines.append(f"{'Real-time Compliance Rate':<40} | {result['realtime_compliant_pct']:>14.1f}% | {'%':>10}")
        
        elif exp == 'Consistency Validation':
            lines.append(f"{'Consistency-Confidence Correlation':<40} | {result['consistency_confidence_correlation']:>14.3f} | {'r':>10}")
    
    lines.extend([
        "-" * 80,
        "",
        "Table 2: Decision Distribution",
        "-" * 80,
    ])
    
    for result in all_results:
        if result.get('experiment') == 'Decision Distribution':
            dist = result['decision_distribution']
            lines.append(f"{'EXECUTE':<20} | {dist.get('execute', 0):>10.1f}%")
            lines.append(f"{'DEFER':<20} | {dist.get('defer', 0):>10.1f}%")
            lines.append(f"{'SKIP':<20} | {dist.get('skip', 0):>10.1f}%")
    
    lines.extend([
        "-" * 80,
        "",
    ])
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(samples_per_experiment: int = 300) -> Dict:
    """Run all experiments and generate comprehensive results."""
    
    print("\n" + "#" * 80)
    print("#" + " " * 25 + "EXPERIMENT SUITE" + " " * 25 + "#")
    print("#" + " " * 15 + "Smart Irrigation Decision System" + " " * 15 + "#")
    print("#" * 80)
    
    all_results = []
    
    # Run all experiments
    all_results.append(experiment_1_decision_distribution(samples_per_experiment))
    all_results.append(experiment_2_water_savings(samples_per_experiment))
    all_results.append(experiment_3_response_time(samples_per_experiment))
    all_results.append(experiment_4_consistency_validation(samples_per_experiment))
    all_results.append(experiment_5_crop_specific_performance(samples_per_experiment))
    all_results.append(experiment_6_regional_analysis(samples_per_experiment))
    all_results.append(experiment_7_edge_cases())
    
    # Generate summary
    summary = generate_summary_table(all_results)
    print(summary)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    
    return {
        "experiments": all_results,
        "summary": summary,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    results = run_all_experiments(samples_per_experiment=300)
    
    # Save results to JSON
    output_path = Path(__file__).parent.parent / "experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
