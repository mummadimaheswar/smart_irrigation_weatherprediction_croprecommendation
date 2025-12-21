"""FastAPI service for irrigation recommendations."""
import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd

from .config import API, CROP_PARAMS, STATES, DISTRICTS
from .decision import decide_irrigation, compute_crop_etc
from .weather import fetch_openweather, get_forecast_summary
from .et import compute_et0
from .ingest import GEO_COORDS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Irrigation API",
    description="AI-powered irrigation recommendations for Indian agriculture",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LocationInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    state: Optional[str] = None
    district: Optional[str] = None


class SensorInput(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=1, description="Soil moisture (0-1)")
    soil_type: str = Field("loam", description="Soil type")


class CropInput(BaseModel):
    crop_type: str = Field("wheat", description="Crop type")
    growth_stage: Optional[str] = Field(None, description="Growth stage")
    days_after_sowing: Optional[int] = Field(None, ge=0)


class IrrigationRequest(BaseModel):
    location: LocationInput
    sensor: SensorInput
    crop: CropInput


class IrrigationResponse(BaseModel):
    decision: str
    reason: str
    advisory: str
    confidence: float
    details: dict
    timestamp: str


class WeatherResponse(BaseModel):
    location: dict
    forecast_24h: dict
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/v1/states")
async def list_states():
    """List supported states."""
    return {"states": STATES}


@app.get("/api/v1/districts/{state}")
async def list_districts(state: str):
    """List districts for a state."""
    state = state.upper()
    if state not in DISTRICTS:
        raise HTTPException(404, f"State '{state}' not found")
    return {"state": state, "districts": DISTRICTS[state]}


@app.get("/api/v1/crops")
async def list_crops():
    """List supported crops with parameters."""
    return {
        "crops": list(CROP_PARAMS.keys()),
        "parameters": {
            k: {
                "growing_days": v["growing_days"],
                "sowing_months": v["sowing_months"],
                "vwc_critical": v["vwc_critical"],
                "vwc_optimal": v["vwc_optimal"]
            }
            for k, v in CROP_PARAMS.items()
        }
    }


@app.get("/api/v1/weather")
async def get_weather(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    api_key: Optional[str] = None
):
    """Get weather forecast for location."""
    weather_df = fetch_openweather(lat, lon, api_key)
    
    if weather_df.empty:
        raise HTTPException(503, "Weather service unavailable")
    
    summary = get_forecast_summary(weather_df, hours=24)
    
    return {
        "location": {"lat": lat, "lon": lon},
        "forecast_24h": summary,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/api/v1/recommend", response_model=IrrigationResponse)
async def get_recommendation(request: IrrigationRequest):
    """Get irrigation recommendation."""
    loc = request.location
    sensor = request.sensor
    crop = request.crop
    
    # Fetch weather
    weather_df = fetch_openweather(loc.lat, loc.lon)
    if weather_df.empty:
        raise HTTPException(503, "Weather service unavailable")
    
    forecast = get_forecast_summary(weather_df, hours=24)
    
    # Compute ET
    et0 = compute_et0(weather_df.iloc[0], lat_deg=loc.lat)
    etc = compute_crop_etc(et0, crop.crop_type)
    
    # Predict moisture (simple decay)
    rain = forecast["total_precipitation_mm"]
    pred_vwc = max(sensor.soil_moisture - max(min((etc - rain) / 100, 0.05), 0.01), 0)
    
    # Get decision
    decision, reason, details = decide_irrigation(
        current_vwc=sensor.soil_moisture,
        predicted_vwc_24h=pred_vwc,
        forecast_rain_24h_mm=rain,
        et0_mm_day=et0,
        crop_type=crop.crop_type
    )
    
    # Generate advisory
    from .advisory import generate_advisory
    advisory = generate_advisory(decision, reason, details, crop.crop_type)
    
    # Confidence based on data quality
    confidence = 0.85 if rain < 50 else 0.7
    
    return {
        "decision": decision,
        "reason": reason,
        "advisory": advisory,
        "confidence": confidence,
        "details": {
            **details,
            "rain_24h_mm": rain,
            "temp_avg_c": forecast["avg_temp_c"],
            "location": {"lat": loc.lat, "lon": loc.lon}
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/v1/recommend/quick")
async def quick_recommendation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    soil_moisture: float = Query(..., ge=0, le=1),
    crop: str = Query("wheat")
):
    """Quick recommendation with minimal input."""
    request = IrrigationRequest(
        location=LocationInput(lat=lat, lon=lon),
        sensor=SensorInput(soil_moisture=soil_moisture),
        crop=CropInput(crop_type=crop)
    )
    return await get_recommendation(request)


@app.get("/api/v1/locations")
async def list_locations():
    """List pre-configured locations."""
    locations = []
    for (state, district), (lat, lon) in GEO_COORDS.items():
        locations.append({
            "state": state,
            "district": district,
            "lat": lat,
            "lon": lon
        })
    return {"locations": locations}


# ─────────────────────────────────────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────────────────────────────────────

def run_server():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "smart_irrigation.api:app",
        host=API.host,
        port=API.port,
        workers=API.workers,
        reload=True
    )


if __name__ == "__main__":
    run_server()
