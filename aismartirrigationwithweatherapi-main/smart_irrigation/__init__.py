"""
Smart Irrigation System - AI-Powered Decision Engine
Version: 2.1.0

A coordinated decision architecture for smart irrigation that arbitrates
between field-derived environmental states and forecast-informed 
meteorological conditions to generate context-aware irrigation actions.

Features:
- Environmental State Fusion (sensor + weather data)
- Supervisory Decision Arbitration (EXECUTE/DEFER/OVERRIDE)
- Confidence-based decision making
- Explainable crop guidance generation
- Modular, scalable architecture
"""

__version__ = "2.1.0"
__author__ = "Smart Irrigation Team"

# Core modules
from . import data
from . import weather
from . import et
from . import models
from . import decision
from . import advisory

# ML Pipeline modules
from . import config
from . import ingest
from . import etl
from . import features
from . import ml_pipeline
from . import api
from . import scheduler

# New Decision Architecture modules
from . import environmental_state
from . import decision_arbitration
from . import decision_report
from . import orchestrator
from . import experiments

# Convenience imports for main classes
from .environmental_state import (
    SensorData, WeatherData, EnvironmentalState, 
    EnvironmentalFusionEngine, TriggerType
)
from .decision_arbitration import (
    DecisionArbitrator, IrrigationAction, 
    SafetyConstraints, ArbitrationResult
)
from .decision_report import (
    DecisionReport, DecisionReportGenerator, 
    CropGuidance, generate_quick_report
)
from .orchestrator import SmartIrrigationOrchestrator

__all__ = [
    # Core
    'data', 'weather', 'et', 'models', 'decision', 'advisory',
    # Pipeline
    'config', 'ingest', 'etl', 'features', 'ml_pipeline', 'api', 'scheduler',
    # New Architecture
    'environmental_state', 'decision_arbitration', 'decision_report', 'orchestrator',
    # Key Classes
    'SensorData', 'WeatherData', 'EnvironmentalState', 'EnvironmentalFusionEngine',
    'DecisionArbitrator', 'IrrigationAction', 'SafetyConstraints', 'ArbitrationResult',
    'DecisionReport', 'DecisionReportGenerator', 'CropGuidance',
    'SmartIrrigationOrchestrator', 'generate_quick_report', 'TriggerType'
]