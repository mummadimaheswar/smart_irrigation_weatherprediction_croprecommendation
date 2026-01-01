"""
Supervisory Decision Arbitration Layer

Implements agentic irrigation decision-making by dynamically prioritizing,
deferring, or overriding irrigation actions based on environmental consistency.

This module implements the core decision arbitration logic described in the methodology:
- Evaluate Consistency between field conditions and forecast-informed expectations
- Set IrrigationDecision = EXECUTE | DEFER | OVERRIDE
- Apply SafetyConstraints
- Compute DecisionConfidenceScore
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging

from .environmental_state import EnvironmentalState, TriggerType, DecisionTrigger


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

log = logging.getLogger(__name__)


class IrrigationAction(Enum):
    """Possible irrigation control decisions."""
    EXECUTE = "execute"   # Proceed with irrigation
    DEFER = "defer"       # Postpone due to expected conditions
    OVERRIDE = "override" # Safety/constraint-based override
    SKIP = "skip"         # No action needed


class DeferReason(Enum):
    """Reasons for deferring irrigation."""
    RAIN_EXPECTED = "rain_expected"
    MOISTURE_ADEQUATE = "moisture_adequate"
    FORECAST_UNCERTAINTY = "forecast_uncertainty"
    RECENT_IRRIGATION = "recent_irrigation"
    LOW_CONFIDENCE_DATA = "low_confidence_data"


class OverrideReason(Enum):
    """Reasons for overriding normal decision logic."""
    CRITICAL_MOISTURE = "critical_moisture"
    EQUIPMENT_CONSTRAINT = "equipment_constraint"
    WATER_QUOTA_EXCEEDED = "water_quota_exceeded"
    FROST_RISK = "frost_risk"
    SENSOR_FAILURE = "sensor_failure"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SafetyConstraints:
    """Safety and operational constraints that may override decisions."""
    min_irrigation_interval_hours: float = 12.0
    max_daily_water_mm: float = 50.0
    frost_risk_temp_c: float = 2.0
    equipment_available: bool = True
    water_quota_remaining_mm: float = 1000.0
    last_irrigation_timestamp: Optional[datetime] = None
    manual_override_active: bool = False
    manual_override_action: Optional[IrrigationAction] = None


@dataclass
class ConsistencyMetrics:
    """Metrics measuring alignment between field conditions and forecasts."""
    field_forecast_alignment: float  # 0-1, how well sensor data aligns with forecast
    temporal_consistency: float      # 0-1, consistency of recent readings
    crop_demand_match: float         # 0-1, how well conditions match crop needs
    trigger_coherence: float         # 0-1, agreement between different triggers
    overall_consistency: float       # 0-1, weighted combination
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "field_forecast_alignment": round(self.field_forecast_alignment, 3),
            "temporal_consistency": round(self.temporal_consistency, 3),
            "crop_demand_match": round(self.crop_demand_match, 3),
            "trigger_coherence": round(self.trigger_coherence, 3),
            "overall_consistency": round(self.overall_consistency, 3)
        }


@dataclass
class ArbitrationResult:
    """
    Complete result from the decision arbitration process.
    
    Includes the decision, confidence, reasoning, and supporting metrics.
    """
    action: IrrigationAction
    confidence_score: float  # 0-1
    primary_reason: str
    secondary_reasons: List[str]
    consistency_metrics: ConsistencyMetrics
    triggered_conditions: List[DecisionTrigger]
    defer_reason: Optional[DeferReason] = None
    override_reason: Optional[OverrideReason] = None
    recommended_water_mm: float = 0.0
    recommended_duration_minutes: float = 0.0
    next_evaluation_hours: float = 6.0
    timestamp: datetime = field(default_factory=utc_now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence_score": round(self.confidence_score, 3),
            "primary_reason": self.primary_reason,
            "secondary_reasons": self.secondary_reasons,
            "consistency_metrics": self.consistency_metrics.to_dict(),
            "triggered_conditions": [
                {"type": t.trigger_type.value, "severity": round(t.severity, 2), "description": t.description}
                for t in self.triggered_conditions
            ],
            "defer_reason": self.defer_reason.value if self.defer_reason else None,
            "override_reason": self.override_reason.value if self.override_reason else None,
            "recommended_water_mm": round(self.recommended_water_mm, 1),
            "recommended_duration_minutes": round(self.recommended_duration_minutes, 1),
            "next_evaluation_hours": self.next_evaluation_hours,
            "timestamp": self.timestamp.isoformat()
        }


class DecisionArbitrator:
    """
    Supervisory Decision Arbitration Layer.
    
    Implements agentic decision-making by:
    1. Evaluating consistency between field conditions and forecast expectations
    2. Computing decision confidence based on data quality and alignment
    3. Dynamically choosing EXECUTE, DEFER, or OVERRIDE actions
    4. Applying safety constraints
    5. Generating explainable decision rationale
    
    This avoids static rule enforcement and reactive control limitations
    by considering the full environmental context.
    """
    
    # Thresholds for decision logic
    CONSISTENCY_THRESHOLD = 0.5      # Below this, defer due to uncertainty
    CONFIDENCE_THRESHOLD = 0.4       # Minimum confidence to act
    RAIN_DEFER_THRESHOLD_MM = 5.0    # Rain amount to trigger deferral
    RAIN_PROB_DEFER_THRESHOLD = 0.6  # Rain probability to trigger deferral
    CRITICAL_SEVERITY = 0.8          # Trigger severity requiring immediate action
    
    def __init__(self, constraints: Optional[SafetyConstraints] = None):
        self.constraints = constraints or SafetyConstraints()
    
    def compute_consistency(self, state: EnvironmentalState) -> ConsistencyMetrics:
        """
        Evaluate consistency between field conditions and forecast-informed expectations.
        
        This implements the methodology's "Evaluate Consistency" step.
        """
        # Field-forecast alignment: Do sensor readings match what forecast would predict?
        # Check if current moisture level is consistent with recent weather
        if state.forecast_rain_24h_mm > 10 and state.current_vwc < 0.20:
            # Rain expected but soil very dry - possible sensor issue or very dry conditions
            field_forecast_alignment = 0.5
        elif state.forecast_rain_24h_mm < 1 and state.water_deficit_mm > 5:
            # No rain expected and high deficit - conditions are consistent
            field_forecast_alignment = 0.9
        else:
            # Normal conditions
            field_forecast_alignment = 0.75
        
        # Temporal consistency: Based on data staleness and reliability
        staleness_factor = max(0, 1 - state.data_staleness_minutes / 120)
        temporal_consistency = (state.sensor_reliability + staleness_factor) / 2
        
        # Crop demand match: How well do conditions match crop requirements?
        vwc_ratio = state.current_vwc / state.vwc_optimal
        if 0.8 <= vwc_ratio <= 1.2:
            crop_demand_match = 0.9
        elif 0.6 <= vwc_ratio <= 1.4:
            crop_demand_match = 0.7
        else:
            crop_demand_match = 0.4
        
        # Trigger coherence: Do different triggers agree?
        if not state.triggers:
            trigger_coherence = 0.8  # No conflicting triggers
        else:
            # Check if triggers point in same direction
            irrigation_triggers = [t for t in state.triggers 
                                   if t.trigger_type in (TriggerType.CRITICAL_MOISTURE, 
                                                         TriggerType.LOW_SOIL_MOISTURE,
                                                         TriggerType.HIGH_ET_DEMAND,
                                                         TriggerType.CROP_STRESS)]
            defer_triggers = [t for t in state.triggers 
                              if t.trigger_type == TriggerType.RAINFALL_EXPECTED]
            
            if irrigation_triggers and defer_triggers:
                # Conflicting triggers
                trigger_coherence = 0.4
            elif irrigation_triggers or defer_triggers:
                trigger_coherence = 0.85
            else:
                trigger_coherence = 0.7
        
        # Overall consistency: Weighted combination
        overall = (
            0.30 * field_forecast_alignment +
            0.25 * temporal_consistency +
            0.25 * crop_demand_match +
            0.20 * trigger_coherence
        )
        
        return ConsistencyMetrics(
            field_forecast_alignment=field_forecast_alignment,
            temporal_consistency=temporal_consistency,
            crop_demand_match=crop_demand_match,
            trigger_coherence=trigger_coherence,
            overall_consistency=overall
        )
    
    def compute_confidence(
        self,
        state: EnvironmentalState,
        consistency: ConsistencyMetrics
    ) -> float:
        """
        Compute decision confidence score.
        
        Based on:
        - Sensor reliability
        - Weather forecast alignment
        - Contextual knowledge relevance
        - Consistency metrics
        """
        # Weight components
        sensor_weight = 0.30
        forecast_weight = 0.25
        consistency_weight = 0.30
        data_quality_weight = 0.15
        
        confidence = (
            sensor_weight * state.sensor_reliability +
            forecast_weight * state.forecast_confidence +
            consistency_weight * consistency.overall_consistency +
            data_quality_weight * state.overall_data_quality
        )
        
        # Penalty for stale data
        if state.data_staleness_minutes > 60:
            confidence *= 0.9
        if state.data_staleness_minutes > 120:
            confidence *= 0.8
        
        return min(1.0, max(0.0, confidence))
    
    def check_safety_constraints(
        self,
        state: EnvironmentalState,
        proposed_action: IrrigationAction
    ) -> Tuple[bool, Optional[OverrideReason], str]:
        """
        Check if safety constraints require overriding the proposed action.
        
        Returns: (should_override, reason, explanation)
        """
        # Manual override check
        if self.constraints.manual_override_active:
            return (
                True,
                OverrideReason.MANUAL_OVERRIDE,
                "Manual override is active"
            )
        
        # Critical moisture override - must irrigate regardless of other factors
        if state.current_vwc < state.vwc_critical * 0.8:
            if proposed_action in (IrrigationAction.DEFER, IrrigationAction.SKIP):
                return (
                    True,
                    OverrideReason.CRITICAL_MOISTURE,
                    f"Critical moisture level ({state.current_vwc:.0%}) requires immediate irrigation"
                )
        
        # Equipment availability
        if not self.constraints.equipment_available:
            if proposed_action == IrrigationAction.EXECUTE:
                return (
                    True,
                    OverrideReason.EQUIPMENT_CONSTRAINT,
                    "Irrigation equipment not available"
                )
        
        # Water quota
        if self.constraints.water_quota_remaining_mm <= 0:
            if proposed_action == IrrigationAction.EXECUTE:
                return (
                    True,
                    OverrideReason.WATER_QUOTA_EXCEEDED,
                    "Daily water quota exhausted"
                )
        
        # Frost risk
        if state.ambient_temp_c < self.constraints.frost_risk_temp_c:
            if proposed_action == IrrigationAction.EXECUTE:
                return (
                    True,
                    OverrideReason.FROST_RISK,
                    f"Frost risk at {state.ambient_temp_c:.1f}°C"
                )
        
        # Minimum interval between irrigations
        if self.constraints.last_irrigation_timestamp:
            now = utc_now()
            last_ts = self.constraints.last_irrigation_timestamp
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            hours_since = (now - last_ts).total_seconds() / 3600
            if hours_since < self.constraints.min_irrigation_interval_hours:
                if proposed_action == IrrigationAction.EXECUTE:
                    return (
                        True,
                        OverrideReason.EQUIPMENT_CONSTRAINT,
                        f"Minimum interval not met ({hours_since:.1f}h < {self.constraints.min_irrigation_interval_hours}h)"
                    )
        
        # Sensor reliability too low
        if state.sensor_reliability < 0.3:
            return (
                True,
                OverrideReason.SENSOR_FAILURE,
                f"Sensor reliability too low ({state.sensor_reliability:.0%})"
            )
        
        return (False, None, "")
    
    def calculate_irrigation_amount(self, state: EnvironmentalState) -> Tuple[float, float]:
        """
        Calculate recommended irrigation amount and duration.
        
        Returns: (water_mm, duration_minutes)
        """
        # Target: Bring moisture to optimal level
        target_vwc = state.vwc_optimal
        current_vwc = state.current_vwc
        
        # Calculate deficit
        root_zone_mm = 300  # Approximate root zone depth
        vwc_deficit = max(0, target_vwc - current_vwc)
        water_needed_mm = vwc_deficit * root_zone_mm
        
        # Add buffer for ETc
        water_needed_mm += state.etc_mm_day * 0.5
        
        # Subtract expected rainfall
        effective_rain = state.forecast_rain_24h_mm * 0.7
        water_needed_mm = max(0, water_needed_mm - effective_rain)
        
        # Cap at maximum daily water
        water_needed_mm = min(water_needed_mm, self.constraints.max_daily_water_mm)
        
        # Calculate duration (assuming 5mm/hr application rate)
        application_rate = 5.0  # mm/hr
        duration_minutes = (water_needed_mm / application_rate) * 60
        
        return (water_needed_mm, duration_minutes)
    
    def arbitrate(self, state: EnvironmentalState) -> ArbitrationResult:
        """
        Main decision arbitration method.
        
        Implements the full decision logic:
        1. Compute consistency metrics
        2. Evaluate triggers
        3. Determine preliminary action
        4. Apply safety constraints
        5. Compute confidence
        6. Generate result with reasoning
        """
        # Step 1: Compute consistency
        consistency = self.compute_consistency(state)
        
        # Step 2: Initialize decision components
        action = IrrigationAction.SKIP
        primary_reason = ""
        secondary_reasons = []
        defer_reason = None
        override_reason = None
        
        # Step 3: Evaluate triggers and determine preliminary action
        critical_triggers = [t for t in state.triggers if t.severity >= self.CRITICAL_SEVERITY]
        irrigation_triggers = [t for t in state.triggers 
                               if t.trigger_type in (TriggerType.CRITICAL_MOISTURE,
                                                     TriggerType.LOW_SOIL_MOISTURE,
                                                     TriggerType.HIGH_ET_DEMAND,
                                                     TriggerType.CROP_STRESS,
                                                     TriggerType.GROWTH_STAGE_CRITICAL)]
        rain_triggers = [t for t in state.triggers 
                         if t.trigger_type == TriggerType.RAINFALL_EXPECTED]
        
        # Critical condition - must act
        if critical_triggers:
            action = IrrigationAction.EXECUTE
            primary_reason = critical_triggers[0].description
            secondary_reasons = [t.description for t in critical_triggers[1:3]]
        
        # Rain expected - consider deferral
        elif rain_triggers and (
            state.forecast_rain_24h_mm >= self.RAIN_DEFER_THRESHOLD_MM or
            state.rain_probability_pct >= self.RAIN_PROB_DEFER_THRESHOLD * 100
        ):
            # Check if moisture can wait for rain
            if state.current_vwc >= state.vwc_critical * 1.1:
                action = IrrigationAction.DEFER
                defer_reason = DeferReason.RAIN_EXPECTED
                primary_reason = f"Deferring irrigation - {state.forecast_rain_24h_mm:.1f}mm rain expected"
                secondary_reasons = [rain_triggers[0].description]
            else:
                # Too dry to wait
                action = IrrigationAction.EXECUTE
                primary_reason = f"Cannot defer - moisture {state.current_vwc:.0%} too low despite rain forecast"
        
        # Low consistency - defer with caution
        elif consistency.overall_consistency < self.CONSISTENCY_THRESHOLD:
            action = IrrigationAction.DEFER
            defer_reason = DeferReason.FORECAST_UNCERTAINTY
            primary_reason = f"Data consistency low ({consistency.overall_consistency:.0%}) - deferring for better data"
            secondary_reasons = ["Re-evaluate when data quality improves"]
        
        # Irrigation triggers present
        elif irrigation_triggers:
            action = IrrigationAction.EXECUTE
            primary_reason = irrigation_triggers[0].description
            secondary_reasons = [t.description for t in irrigation_triggers[1:3]]
        
        # No action needed
        else:
            action = IrrigationAction.SKIP
            primary_reason = f"Moisture adequate ({state.current_vwc:.0%})"
            secondary_reasons = [f"Status: {state.moisture_status}"]
        
        # Step 4: Apply safety constraints
        should_override, constraint_reason, constraint_msg = self.check_safety_constraints(state, action)
        
        if should_override:
            if constraint_reason in (OverrideReason.CRITICAL_MOISTURE,):
                action = IrrigationAction.EXECUTE
            else:
                action = IrrigationAction.OVERRIDE
            override_reason = constraint_reason
            primary_reason = constraint_msg
        
        # Step 5: Compute confidence
        confidence = self.compute_confidence(state, consistency)
        
        # Step 6: Calculate irrigation amount if executing
        water_mm, duration_min = (0.0, 0.0)
        if action == IrrigationAction.EXECUTE:
            water_mm, duration_min = self.calculate_irrigation_amount(state)
        
        # Determine next evaluation time
        if action == IrrigationAction.EXECUTE:
            next_eval = 12.0  # Re-evaluate after irrigation settles
        elif action == IrrigationAction.DEFER:
            next_eval = 4.0   # Re-evaluate soon
        else:
            next_eval = 6.0   # Standard interval
        
        return ArbitrationResult(
            action=action,
            confidence_score=confidence,
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
            consistency_metrics=consistency,
            triggered_conditions=state.triggers,
            defer_reason=defer_reason,
            override_reason=override_reason,
            recommended_water_mm=water_mm,
            recommended_duration_minutes=duration_min,
            next_evaluation_hours=next_eval,
            timestamp=utc_now()
        )


class ActuationController:
    """
    Actuation Control Layer.
    
    Translates arbitration decisions into control signals for
    irrigation flow-control elements.
    """
    
    def __init__(self, valve_id: str = "main_valve"):
        self.valve_id = valve_id
        self.is_active = False
        self.current_flow_rate = 0.0
    
    def generate_control_signal(
        self,
        result: ArbitrationResult
    ) -> Dict[str, Any]:
        """
        Generate actuation control signal from arbitration result.
        
        Returns control signal dictionary suitable for hardware interface.
        """
        if result.action == IrrigationAction.EXECUTE:
            return {
                "valve_id": self.valve_id,
                "command": "OPEN",
                "duration_minutes": result.recommended_duration_minutes,
                "target_volume_mm": result.recommended_water_mm,
                "timestamp": utc_now().isoformat(),
                "confidence": result.confidence_score
            }
        elif result.action in (IrrigationAction.DEFER, IrrigationAction.OVERRIDE):
            return {
                "valve_id": self.valve_id,
                "command": "HOLD",
                "reason": result.primary_reason,
                "next_evaluation": result.next_evaluation_hours,
                "timestamp": utc_now().isoformat()
            }
        else:  # SKIP
            return {
                "valve_id": self.valve_id,
                "command": "MAINTAIN",
                "timestamp": utc_now().isoformat()
            }
