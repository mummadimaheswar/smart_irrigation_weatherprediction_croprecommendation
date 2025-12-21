"""
FastAPI Application for Crop Recommendation
India Crop Recommendation System

PROMPT 8: REST API with:
- POST /recommend - crop recommendations
- GET /status - health check
- Pydantic models for request/response
"""
import os
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_VERSION = "1.0.0"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Import models (with fallback)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train.train_model import RuleBasedCropRecommender, CropRecommenderML
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    log.warning("ML models not available, using embedded rule-based")

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = API_VERSION
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models_loaded: bool = False


class CropRecommendation(BaseModel):
    crop: str
    confidence: float = Field(ge=0, le=1)
    season: Optional[str] = None
    notes: Optional[str] = None


class RecommendRequest(BaseModel):
    state: str = Field(..., description="Indian state name", examples=["Maharashtra"])
    district: Optional[str] = Field(default=None, description="District name", examples=["Pune"])
    planting_date: Optional[str] = Field(default=None, description="Target planting date (YYYY-MM-DD)")
    soil_moisture_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Soil moisture percentage")
    temperature_c: Optional[float] = Field(default=None, description="Current/expected temperature")
    rainfall_mm: Optional[float] = Field(default=None, ge=0, description="Recent/expected rainfall")
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Humidity percentage")
    budget_inr: Optional[float] = Field(default=None, ge=0, description="Budget in INR")
    land_size_ha: Optional[float] = Field(default=None, ge=0, description="Land size in hectares")
    irrigation_available: bool = Field(default=True, description="Irrigation availability")
    sensor_readings: Optional[List[float]] = Field(default=None, description="20 sensor readings")
    num_sensors: Optional[int] = Field(default=None, description="Number of sensors")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "state": "Maharashtra",
                "district": "Pune",
                "planting_date": "2024-06-15",
                "soil_moisture_pct": 35.0,
                "temperature_c": 28.5,
                "rainfall_mm": 50.0,
                "irrigation_available": True
            }]
        }
    }


class RecommendResponse(BaseModel):
    request_id: str
    timestamp: datetime
    location: Dict[str, str]
    recommendations: List[CropRecommendation]
    weather_summary: Optional[Dict[str, Any]] = None
    model_version: str = "rule_based_v1"
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "request_id": "abc123",
                "timestamp": "2024-01-15T10:30:00",
                "location": {"state": "Maharashtra", "district": "Pune"},
                "recommendations": [
                    {"crop": "cotton", "confidence": 0.85, "season": "kharif", "notes": "Ideal conditions"},
                    {"crop": "soybean", "confidence": 0.75, "season": "kharif", "notes": "Good alternative"},
                    {"crop": "groundnut", "confidence": 0.65, "season": "kharif", "notes": "Consider if irrigation limited"}
                ],
                "model_version": "rule_based_v1"
            }]
        }
    }


class StateInfo(BaseModel):
    name: str
    code: str
    lat: float
    lon: float


class WeatherResponse(BaseModel):
    state: str
    date_str: str
    temp_min_c: float
    temp_max_c: float
    temp_mean_c: float
    precip_mm: float
    humidity_pct: float
    source: str


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="India Crop Recommendation API",
    description="""
    AI-powered crop recommendation system for Indian agriculture.
    
    ## Features
    - Get personalized crop recommendations based on location, soil, and weather
    - Supports all Indian states
    - Multiple model options (rule-based and ML)
    
    ## Usage
    1. Call `/recommend` with your location and conditions
    2. Receive top 3 crop recommendations with confidence scores
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

STATES = {
    "Andhra Pradesh": {"code": "AP", "lat": 15.9129, "lon": 79.7400},
    "Assam": {"code": "AS", "lat": 26.2006, "lon": 92.9376},
    "Bihar": {"code": "BR", "lat": 25.0961, "lon": 85.3131},
    "Gujarat": {"code": "GJ", "lat": 22.2587, "lon": 71.1924},
    "Haryana": {"code": "HR", "lat": 29.0588, "lon": 76.0856},
    "Karnataka": {"code": "KA", "lat": 15.3173, "lon": 75.7139},
    "Kerala": {"code": "KL", "lat": 10.8505, "lon": 76.2711},
    "Madhya Pradesh": {"code": "MP", "lat": 22.9734, "lon": 78.6569},
    "Maharashtra": {"code": "MH", "lat": 19.7515, "lon": 75.7139},
    "Odisha": {"code": "OR", "lat": 20.9517, "lon": 85.0985},
    "Punjab": {"code": "PB", "lat": 31.1471, "lon": 75.3412},
    "Rajasthan": {"code": "RJ", "lat": 27.0238, "lon": 74.2179},
    "Tamil Nadu": {"code": "TN", "lat": 11.1271, "lon": 78.6569},
    "Telangana": {"code": "TS", "lat": 18.1124, "lon": 79.0193},
    "Uttar Pradesh": {"code": "UP", "lat": 26.8467, "lon": 80.9462},
    "West Bengal": {"code": "WB", "lat": 22.9868, "lon": 87.8550},
}

# Crop requirements for rule-based recommendations
CROP_RULES = {
    "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "precip_min": 100, "season": "kharif"},
    "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "precip_min": 40, "season": "rabi"},
    "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "precip_min": 50, "season": "kharif"},
    "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "precip_min": 60, "season": "kharif"},
    "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "precip_min": 150, "season": "perennial"},
    "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "precip_min": 50, "season": "kharif"},
    "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "precip_min": 60, "season": "kharif"},
    "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "precip_min": 25, "season": "rabi"},
    "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "precip_min": 30, "season": "rabi"},
    "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "precip_min": 40, "season": "rabi"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_season(month: int) -> str:
    """Determine season from month."""
    if 6 <= month <= 10:
        return "kharif"
    elif month >= 10 or month <= 3:
        return "rabi"
    else:
        return "zaid"


def score_crop(
    crop: str,
    soil_moisture: float,
    temp: float,
    precip: float,
    month: int
) -> float:
    """Score crop suitability."""
    if crop not in CROP_RULES:
        return 0.0
    
    rules = CROP_RULES[crop]
    score = 0.0
    
    # Soil moisture (0-30)
    if rules["sm_min"] <= soil_moisture <= rules["sm_max"]:
        mid = (rules["sm_min"] + rules["sm_max"]) / 2
        dist = abs(soil_moisture - mid) / (rules["sm_max"] - rules["sm_min"])
        score += 30 * (1 - dist)
    
    # Temperature (0-30)
    if rules["temp_min"] <= temp <= rules["temp_max"]:
        mid = (rules["temp_min"] + rules["temp_max"]) / 2
        dist = abs(temp - mid) / (rules["temp_max"] - rules["temp_min"])
        score += 30 * (1 - dist)
    
    # Precipitation (0-20)
    if precip >= rules["precip_min"]:
        score += 20
    else:
        score += 20 * (precip / rules["precip_min"])
    
    # Season (0-20)
    if rules["season"] == get_season(month) or rules["season"] == "perennial":
        score += 20
    
    return score


def get_recommendations(
    soil_moisture: float,
    temp: float,
    precip: float,
    month: int,
    n: int = 3
) -> List[CropRecommendation]:
    """Get top N crop recommendations."""
    scores = {
        crop: score_crop(crop, soil_moisture, temp, precip, month)
        for crop in CROP_RULES
    }
    
    total = sum(scores.values())
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    recommendations = []
    for crop, score in sorted_crops:
        confidence = score / 100.0 if total > 0 else 0
        rules = CROP_RULES[crop]
        
        notes = []
        if soil_moisture < rules["sm_min"]:
            notes.append("Consider irrigation")
        if temp > rules["temp_max"]:
            notes.append("High temperature risk")
        if precip < rules["precip_min"]:
            notes.append("May need supplemental irrigation")
        
        recommendations.append(CropRecommendation(
            crop=crop,
            confidence=round(confidence, 3),
            season=rules["season"],
            notes="; ".join(notes) if notes else "Good conditions"
        ))
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """API root - redirects to docs."""
    return {"message": "India Crop Recommendation API", "docs": "/docs"}


@app.get("/status", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    models_loaded = MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib"))
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        models_loaded=models_loaded
    )


@app.get("/states", response_model=List[StateInfo], tags=["Reference"])
async def list_states():
    """List all supported Indian states."""
    return [
        StateInfo(name=name, code=info["code"], lat=info["lat"], lon=info["lon"])
        for name, info in STATES.items()
    ]


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend_crops(request: RecommendRequest):
    """
    Get crop recommendations based on location and conditions.
    
    Accepts state, district, soil moisture, temperature, and rainfall.
    Returns top 3 crop recommendations with confidence scores.
    """
    import uuid
    
    # Validate state
    if request.state not in STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown state: {request.state}. Use /states to list valid states."
        )
    
    # Get date info
    target_date = request.date or date.today()
    month = target_date.month
    
    # Use provided values or defaults
    soil_moisture = request.soil_moisture_pct or 35.0
    temp = request.temperature_c or 25.0
    precip = request.rainfall_mm or 50.0
    
    # Adjust defaults by season
    if get_season(month) == "kharif":
        precip = request.rainfall_mm or 100.0
        temp = request.temperature_c or 28.0
    elif get_season(month) == "rabi":
        precip = request.rainfall_mm or 30.0
        temp = request.temperature_c or 20.0
    
    # Get recommendations
    recommendations = get_recommendations(soil_moisture, temp, precip, month, n=3)
    
    return RecommendResponse(
        request_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow(),
        location={
            "state": request.state,
            "district": request.district or "N/A"
        },
        recommendations=recommendations,
        weather_summary={
            "soil_moisture_pct": soil_moisture,
            "temperature_c": temp,
            "rainfall_mm": precip,
            "season": get_season(month)
        },
        model_version="rule_based_v1"
    )


@app.get("/recommend/quick", response_model=RecommendResponse, tags=["Recommendations"])
async def quick_recommend(
    state: str = Query(..., description="State name"),
    month: int = Query(None, ge=1, le=12, description="Month (1-12)"),
    soil_moisture: float = Query(35.0, ge=0, le=100, description="Soil moisture %")
):
    """Quick recommendation endpoint with minimal parameters."""
    import uuid
    
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    
    month = month or datetime.now().month
    
    # Season-based defaults
    if get_season(month) == "kharif":
        temp, precip = 28.0, 100.0
    else:
        temp, precip = 20.0, 30.0
    
    recommendations = get_recommendations(soil_moisture, temp, precip, month, n=3)
    
    return RecommendResponse(
        request_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow(),
        location={"state": state, "district": "N/A"},
        recommendations=recommendations,
        model_version="rule_based_v1"
    )


@app.get("/weather/{state}", response_model=WeatherResponse, tags=["Weather"])
async def get_weather(state: str):
    """Get current weather for a state (simulated)."""
    import random
    
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    
    # Simulated weather
    month = datetime.now().month
    base_temp = 25 - (STATES[state]["lat"] - 20) * 0.5
    
    if 6 <= month <= 9:  # Monsoon
        temp_mean = base_temp + random.uniform(2, 5)
        precip = random.uniform(50, 200)
        humidity = random.uniform(70, 95)
    else:
        temp_mean = base_temp + random.uniform(-5, 5)
        precip = random.uniform(0, 30)
        humidity = random.uniform(40, 70)
    
    return WeatherResponse(
        state=state,
        date_str=str(date.today()),
        temp_min_c=round(temp_mean - 5, 1),
        temp_max_c=round(temp_mean + 5, 1),
        temp_mean_c=round(temp_mean, 1),
        precip_mm=round(precip, 1),
        humidity_pct=round(humidity, 1),
        source="simulated"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROK CHATBOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

from .grok_chat import get_chatbot, GrokChatBot


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", examples=["What crops should I grow in Maharashtra?"])
    state: Optional[str] = Field(default=None, description="Current state for context")
    district: Optional[str] = Field(default=None, description="Current district")
    soil_moisture_pct: Optional[float] = Field(default=None, description="Current soil moisture")
    temperature_c: Optional[float] = Field(default=None, description="Current temperature")
    rainfall_mm: Optional[float] = Field(default=None, description="Recent rainfall")
    sensor_readings: Optional[List[float]] = Field(default=None, description="Sensor readings array")
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Current month")
    api_key: Optional[str] = Field(default=None, description="Grok API key (optional, overrides env var)")


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_grok(request: ChatRequest):
    """
    Chat with Grok AI for agricultural advice.
    
    Send natural language questions about:
    - Crop recommendations
    - Soil and water management
    - Weather-based farming decisions
    - Regional agricultural practices
    
    Optionally provide context (location, soil moisture, etc.) for more relevant advice.
    You can pass your Grok API key in the request (api_key field) or set it via GROK_API_KEY env var.
    """
    # Get chatbot with optional API key override
    chatbot = get_chatbot(api_key=request.api_key)
    
    # Build context from request
    context = {}
    if request.state:
        context["state"] = request.state
    if request.district:
        context["district"] = request.district
    if request.soil_moisture_pct is not None:
        context["soil_moisture_pct"] = request.soil_moisture_pct
    if request.temperature_c is not None:
        context["temperature_c"] = request.temperature_c
    if request.rainfall_mm is not None:
        context["rainfall_mm"] = request.rainfall_mm
    if request.sensor_readings:
        context["sensor_readings"] = request.sensor_readings
    if request.month is not None:
        context["month"] = request.month
    else:
        context["month"] = datetime.now().month
    
    result = await chatbot.chat_async(request.message, context if context else None)
    
    return ChatResponse(
        success=result["success"],
        response=result.get("response"),
        error=result.get("error")
    )


@app.post("/chat/clear", tags=["Chatbot"])
async def clear_chat_history():
    """Clear the chat conversation history."""
    chatbot = get_chatbot()
    chatbot.clear_history()
    return {"success": True, "message": "Chat history cleared"}


@app.get("/chat/history", tags=["Chatbot"])
async def get_chat_history():
    """Get the current chat conversation history."""
    chatbot = get_chatbot()
    return {"history": chatbot.get_history()}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the API server."""
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
