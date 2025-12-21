"""
Evapotranspiration (ET) calculation module.
Implements FAO-56 Penman-Monteith with Hargreaves fallback.
"""

import numpy as np
import pandas as pd
from typing import Optional

def compute_et0_fao56(temp_c: float, temp_min_c: float, temp_max_c: float,
                     humidity_pct: float, wind_speed_ms: float, 
                     lat_deg: float = 0, day_of_year: int = 1,
                     elevation_m: float = 0) -> float:
    """
    Compute reference evapotranspiration (ET₀) using FAO-56 Penman-Monteith.
    
    Returns:
        ET₀ in mm/day
    """
    P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.665e-3 * P
    
    e_tmax = 0.6108 * np.exp(17.27 * temp_max_c / (temp_max_c + 237.3))
    e_tmin = 0.6108 * np.exp(17.27 * temp_min_c / (temp_min_c + 237.3))
    es = (e_tmax + e_tmin) / 2
    ea = es * (humidity_pct / 100)
    
    delta = 4098 * (0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))) / ((temp_c + 237.3) ** 2)
    
    Ra = _extraterrestrial_radiation(lat_deg, day_of_year)
    solar_radiation = 0.16 * np.sqrt(temp_max_c - temp_min_c) * Ra
    Rn = solar_radiation * 0.77
    G = 0
    
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (temp_c + 273)) * wind_speed_ms * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_speed_ms)
    
    et0 = numerator / denominator
    return max(0, et0)

def compute_et0_hargreaves(temp_c: float, temp_min_c: float, temp_max_c: float,
                          lat_deg: float, day_of_year: int) -> float:
    """Compute ET₀ using Hargreaves equation (fallback when limited data available)."""
    Ra = _extraterrestrial_radiation(lat_deg, day_of_year)
    et0 = 0.0023 * (temp_c + 17.8) * np.sqrt(temp_max_c - temp_min_c) * Ra / 2.45
    return max(0, et0)

def _extraterrestrial_radiation(lat_deg: float, day_of_year: int) -> float:
    """Calculate extraterrestrial radiation (Ra) in MJ/m²/day."""
    lat_rad = np.pi * lat_deg / 180
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    
    Gsc = 0.0820
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(delta) + 
        np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )
    return Ra

def compute_et0(weather_row: pd.Series, lat_deg: float = 28.6, 
               elevation_m: float = 200, method: str = 'auto') -> float:
    """Compute ET₀ with automatic method selection based on available data."""
    temp_c = weather_row.get('temp', weather_row.get('air_temp', 25))
    humidity_pct = weather_row.get('humidity', 60)
    wind_speed_ms = weather_row.get('wind_speed', 2)
    temp_min_c = weather_row.get('temp_min', temp_c - 5)
    temp_max_c = weather_row.get('temp_max', temp_c + 5)
    
    if hasattr(weather_row, 'name') and isinstance(weather_row.name, pd.Timestamp):
        day_of_year = weather_row.name.timetuple().tm_yday
    else:
        day_of_year = 180
    
    if method == 'auto':
        method = 'fao56' if humidity_pct > 0 and wind_speed_ms > 0 else 'hargreaves'
    
    if method == 'fao56':
        return compute_et0_fao56(temp_c, temp_min_c, temp_max_c, humidity_pct, 
                                wind_speed_ms, lat_deg, day_of_year, elevation_m)
    else:
        return compute_et0_hargreaves(temp_c, temp_min_c, temp_max_c, lat_deg, day_of_year)

def compute_etc(et0: float, kc: float) -> float:
    """Compute crop evapotranspiration (ETc) from reference ET₀ and crop coefficient."""
    return et0 * kc

DEFAULT_KC_VALUES = {
    'rice': 1.05,
    'wheat': 0.95,
    'maize': 0.85,
    'cotton': 0.90,
    'soybean': 0.90,
    'tomato': 1.00,
    'potato': 0.95,
    'default': 0.90
}

def get_kc(crop_type: str, growth_stage: str = 'mid') -> float:
    """Get crop coefficient for given crop and growth stage."""
    stage_factors = {
        'initial': 0.5,
        'development': 0.8,
        'mid': 1.0,
        'late': 0.7
    }
    
    base_kc = DEFAULT_KC_VALUES.get(crop_type.lower(), DEFAULT_KC_VALUES['default'])
    return base_kc * stage_factors.get(growth_stage, 1.0)
