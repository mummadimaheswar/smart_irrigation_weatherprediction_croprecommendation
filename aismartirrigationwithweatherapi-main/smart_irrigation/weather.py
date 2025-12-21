"""
Weather data module - Fetches OpenWeather One Call 3.0 forecasts.
Includes synthetic weather generator for testing without API key.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta


OPENWEATHER_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

def fetch_openweather(lat: float, lon: float, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch weather forecast from OpenWeather One Call 3.0 API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeather API key (if None, uses env var or generates synthetic)
    
    Returns:
        DataFrame with columns: timestamp, temp, humidity, wind_speed, pressure, 
                                clouds, precipitation, weather_desc
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key or api_key == "":
        print("Warning: no OpenWeather API key found. Generating synthetic forecast.")
        return generate_synthetic_forecast(lat, lon)
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric',
            'exclude': 'minutely'
        }
        
        response = requests.get(OPENWEATHER_ONECALL_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = []
        if 'hourly' in data:
            for hour in data['hourly'][:48]:
                hourly_data.append({
                    'timestamp': datetime.utcfromtimestamp(hour['dt']),
                    'temp': hour['temp'],
                    'humidity': hour['humidity'],
                    'wind_speed': hour['wind_speed'],
                    'pressure': hour['pressure'],
                    'clouds': hour['clouds'],
                    'precipitation': hour.get('rain', {}).get('1h', 0) + hour.get('snow', {}).get('1h', 0),
                    'weather_desc': hour['weather'][0]['description'] if hour.get('weather') else 'clear'
                })
        
        df_hourly = pd.DataFrame(hourly_data)
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'], utc=True)
        df_hourly = df_hourly.set_index('timestamp')
        
        return df_hourly
    
    except requests.exceptions.RequestException as e:
        print(f"Warning: OpenWeather API error: {e}. Generating synthetic forecast.")
        return generate_synthetic_forecast(lat, lon)


def generate_synthetic_forecast(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    """Generate synthetic weather forecast for testing without API key."""
    np.random.seed(int(lat * 100) % 1000)
    
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    timestamps = [now + timedelta(hours=i) for i in range(hours)]
    
    base_temp = 25 - abs(lat) * 0.3
    day_of_year = now.timetuple().tm_yday
    seasonal_adjustment = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    data = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        temp = base_temp + seasonal_adjustment + 8 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 1.5)
        humidity = 70 - (temp - base_temp) * 2 + np.random.normal(0, 5)
        humidity = np.clip(humidity, 30, 95)
        wind_speed = np.abs(np.random.normal(3, 2))
        pressure = 1013 + np.random.normal(0, 8)
        clouds = np.clip(np.random.beta(2, 5) * 100, 0, 100)
        
        precipitation = 0
        if np.random.random() < 0.1:
            precipitation = np.random.exponential(2.5)
        
        weather_desc = 'clear sky' if clouds < 20 else ('partly cloudy' if clouds < 60 else 'overcast')
        if precipitation > 0:
            weather_desc = 'light rain' if precipitation < 2 else 'moderate rain'
        
        data.append({
            'timestamp': ts,
            'temp': round(temp, 2),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 2),
            'pressure': round(pressure, 1),
            'clouds': round(clouds, 1),
            'precipitation': round(precipitation, 2),
            'weather_desc': weather_desc
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    return df


def get_forecast_summary(weather_df: pd.DataFrame, hours: int = 48) -> Dict[str, float]:
    """Generate summary statistics for forecast period."""
    df = weather_df.head(hours) if len(weather_df) >= hours else weather_df
    
    return {
        'total_precipitation_mm': df['precipitation'].sum(),
        'avg_temp_c': df['temp'].mean(),
        'max_temp_c': df['temp'].max(),
        'min_temp_c': df['temp'].min(),
        'avg_humidity_pct': df['humidity'].mean(),
        'avg_wind_speed_ms': df['wind_speed'].mean(),
        'hours_with_rain': (df['precipitation'] > 0.1).sum()
    }