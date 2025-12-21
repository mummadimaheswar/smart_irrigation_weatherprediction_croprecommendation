"""
Smart Irrigation System - AI-Powered Decision Engine
Version: 2.0.0
"""

__version__ = "2.0.0"
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

__all__ = [
    # Core
    'data', 'weather', 'et', 'models', 'decision', 'advisory',
    # Pipeline
    'config', 'ingest', 'etl', 'features', 'ml_pipeline', 'api', 'scheduler'
]