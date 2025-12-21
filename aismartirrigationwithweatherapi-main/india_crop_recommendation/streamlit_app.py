"""
India Crop Recommendation System - Streamlit Frontend
Easy-to-use web interface with Grok AI chatbot integration

Run with: streamlit run streamlit_app.py
"""
import os
import sys
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🌾 India Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .crop-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .sensor-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
    }
    
    /* Chat UI - Dark Theme */
    .chat-container {
        background: #1a1a2e;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .chat-user {
        background: #0066cc;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-ai {
        background: #16213e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border-left: 3px solid #0066cc;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-ai strong, .chat-ai b {
        color: #4da6ff;
    }
    .chat-ai ul, .chat-ai ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .chat-ai li {
        margin: 0.3rem 0;
    }
    .chat-ai code {
        background: #0d1b2a;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #4da6ff;
    }
    .chat-wrapper {
        background: #0d1b2a;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #1e3a5f;
    }
    .chat-header {
        background: #0066cc;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════════════════

INDIAN_STATES = [
    "Andhra Pradesh", "Gujarat", "Himachal Pradesh", "Maharashtra",
    "Punjab", "Rajasthan", "Tamil Nadu", "Telangana",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

SAMPLE_DATA = {
    "Maharashtra": [21.43, 11.54, 12.98, 19.20, 15.28, 19.04, 10.16, 9.01, 16.17, 18.23,
                   22.15, 14.67, 17.89, 20.45, 13.22, 18.90, 15.67, 21.34, 16.78, 19.56],
    "Gujarat": [18.5, 15.2, 22.1, 19.8, 16.4, 20.3, 17.9, 14.6, 21.7, 18.2,
               15.8, 19.4, 16.9, 20.8, 17.3, 15.9, 21.2, 18.7, 16.1, 19.9],
    "Punjab": [25.3, 28.1, 24.7, 26.9, 23.5, 27.4, 25.8, 29.2, 24.1, 26.3,
              28.7, 25.0, 27.1, 24.9, 26.5, 28.3, 25.6, 27.8, 24.4, 26.8],
    "Rajasthan": [12.5, 10.2, 14.1, 11.8, 13.4, 9.3, 15.9, 11.6, 12.7, 10.2,
                13.8, 11.4, 14.9, 10.8, 12.3, 15.9, 11.2, 13.7, 10.1, 14.9],
    "Tamil Nadu": [19.5, 22.2, 18.1, 21.8, 17.4, 23.3, 19.9, 20.6, 18.7, 22.2,
                 17.8, 21.4, 18.9, 22.8, 19.3, 20.9, 18.2, 21.7, 19.1, 20.9],
    "Andhra Pradesh": [20.5, 18.2, 22.1, 19.8, 21.4, 17.3, 23.9, 18.6, 20.7, 19.2,
                      22.8, 17.4, 21.9, 18.8, 20.3, 22.9, 18.2, 21.7, 19.1, 20.9],
    "Telangana": [18.5, 21.2, 17.1, 20.8, 16.4, 22.3, 18.9, 19.6, 17.7, 21.2,
                16.8, 20.4, 17.9, 21.8, 18.3, 19.9, 17.2, 20.7, 18.1, 19.9],
    "Uttar Pradesh": [22.5, 25.2, 21.1, 24.8, 20.4, 26.3, 22.9, 23.6, 21.7, 25.2,
                     20.8, 24.4, 21.9, 25.8, 22.3, 23.9, 21.2, 24.7, 22.1, 23.9],
    "West Bengal": [28.5, 31.2, 27.1, 30.8, 26.4, 32.3, 28.9, 29.6, 27.7, 31.2,
                   26.8, 30.4, 27.9, 31.8, 28.3, 29.9, 27.2, 30.7, 28.1, 29.9],
    "Himachal Pradesh": [15.5, 18.2, 14.1, 17.8, 13.4, 19.3, 15.9, 16.6, 14.7, 18.2,
                        13.8, 17.4, 14.9, 18.8, 15.3, 16.9, 14.2, 17.7, 15.1, 16.9],
    "Uttarakhand": [16.5, 19.2, 15.1, 18.8, 14.4, 20.3, 16.9, 17.6, 15.7, 19.2,
                  14.8, 18.4, 15.9, 19.8, 16.3, 17.9, 15.2, 18.7, 16.1, 17.9]
}

CROP_RULES = {
    "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "season": "kharif", "emoji": "🌾"},
    "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "season": "rabi", "emoji": "🌾"},
    "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "season": "kharif", "emoji": "🌽"},
    "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "season": "kharif", "emoji": "☁️"},
    "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "season": "perennial", "emoji": "🎋"},
    "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "season": "kharif", "emoji": "🥜"},
    "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "season": "kharif", "emoji": "🫘"},
    "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "season": "rabi", "emoji": "🌻"},
    "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "season": "rabi", "emoji": "🫘"},
    "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "season": "rabi", "emoji": "🥔"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'sensor_values' not in st.session_state:
    st.session_state.sensor_values = [0.0] * 20

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'grok_api_key' not in st.session_state:
    st.session_state.grok_api_key = os.getenv("GROK_API_KEY", "")

if 'weather_api_key' not in st.session_state:
    st.session_state.weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_season(month: int) -> str:
    """Determine agricultural season from month."""
    if 6 <= month <= 10:
        return "kharif"
    elif month >= 10 or month <= 3:
        return "rabi"
    return "zaid"


def calculate_crop_scores(sm: float, temp: float, month: int) -> List[Dict]:
    """Calculate crop suitability scores."""
    season = get_season(month)
    scores = []
    
    for crop, rules in CROP_RULES.items():
        score = 0
        
        # Soil moisture score (40%)
        if rules["sm_min"] <= sm <= rules["sm_max"]:
            optimal_sm = (rules["sm_min"] + rules["sm_max"]) / 2
            sm_score = 1 - abs(sm - optimal_sm) / (rules["sm_max"] - rules["sm_min"])
            score += sm_score * 0.4
        
        # Temperature score (30%)
        if rules["temp_min"] <= temp <= rules["temp_max"]:
            optimal_temp = (rules["temp_min"] + rules["temp_max"]) / 2
            temp_score = 1 - abs(temp - optimal_temp) / (rules["temp_max"] - rules["temp_min"])
            score += temp_score * 0.3
        
        # Season match (30%)
        if rules["season"] == season or rules["season"] == "perennial":
            score += 0.3
        elif rules["season"] in ["kharif", "rabi"] and season == "zaid":
            score += 0.1
        
        if score > 0.3:
            scores.append({
                "crop": crop.title(),
                "confidence": min(score, 1.0),
                "season": rules["season"],
                "emoji": rules["emoji"],
                "notes": f"Optimal moisture: {rules['sm_min']}-{rules['sm_max']}%"
            })
    
    return sorted(scores, key=lambda x: x["confidence"], reverse=True)[:5]


def fetch_weather(city: str, api_key: str) -> Optional[Dict]:
    """Fetch weather data from OpenWeatherMap."""
    if not api_key:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "city": data["name"]
            }
    except Exception as e:
        st.warning(f"Weather fetch error: {e}")
    return None


def call_grok_api(message: str, context: Dict, api_key: str) -> Optional[str]:
    """Call Groq Cloud API for chat responses."""
    if not api_key:
        return "⚠️ No API key provided. Please enter your Groq API key in the sidebar."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert Indian agricultural advisor. Help farmers with:
- Crop recommendations based on location, season, soil moisture
- Soil & water management advice
- Weather-based farming decisions
Be concise, practical, and use simple language."""
        
        context_str = ""
        if context:
            context_str = f"\n\nCurrent context: {json.dumps(context)}"
        
        # Groq Cloud models
        models_to_try = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"]
        
        last_error = None
        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt + context_str},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                return "❌ **Invalid API Key**: Please check your Groq API key is correct."
            elif response.status_code == 404 or response.status_code == 400:
                # Model not found, try next
                last_error = f"Model {model_name} not available"
                continue
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
        
        return f"❌ **API Error**: {last_error}"
        
    except requests.exceptions.Timeout:
        return "❌ **Timeout**: The API request took too long. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Cannot reach the Groq API. Check your internet connection."
    except Exception as e:
        return f"❌ **Error**: {str(e)}"


def get_offline_response(message: str, context: Dict) -> str:
    """Generate offline response when API is not available."""
    msg = message.lower()
    state = context.get("state", "your region")
    moisture = context.get("soil_moisture")
    month = context.get("month", datetime.now().month)
    season = get_season(month)
    
    if any(word in msg for word in ["recommend", "crop", "grow", "plant"]):
        if season == "kharif":
            crops = ["Rice", "Cotton", "Maize", "Soybean", "Groundnut"]
        elif season == "rabi":
            crops = ["Wheat", "Mustard", "Chickpea", "Potato", "Barley"]
        else:
            crops = ["Watermelon", "Muskmelon", "Cucumber", "Vegetables"]
        
        response = f"**{season.title()} Season Recommendations for {state}:**\n\n"
        for i, crop in enumerate(crops, 1):
            response += f"{i}. {crop}\n"
        
        if moisture:
            response += f"\n💧 With soil moisture at {moisture:.1f}%, "
            if moisture < 25:
                response += "consider drought-resistant crops."
            elif moisture > 50:
                response += "water-loving crops would thrive."
            else:
                response += "most crops should do well."
        
        return response
    
    if any(word in msg for word in ["weather", "rain", "monsoon"]):
        return f"""🌧️ **Weather Guidance for {state}:**

• Current season: **{season.title()}**
• If expecting heavy rain, ensure proper drainage
• Light rain is ideal for sowing operations
• Monitor forecasts for timely irrigation"""
    
    if any(word in msg for word in ["irrigation", "water"]):
        return f"""💧 **Irrigation Tips:**

• Morning irrigation (6-9 AM) reduces evaporation
• Drip irrigation saves 30-50% water
• Mulching helps retain soil moisture
• Consider rainwater harvesting"""
    
    return """I'm your AI farming assistant! I can help with:

• **"What crops should I grow?"** - Get recommendations
• **"Irrigation tips"** - Water management advice  
• **"Weather guidance"** - Seasonal advice

*Note: Enter your Grok API key in the sidebar for AI-powered responses.*"""

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - API KEYS & SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔑 API Keys")
    
    # Groq API Key Section
    st.markdown("##### 🤖 Groq Cloud API")
    st.caption("Get FREE key: [console.groq.com](https://console.groq.com)")
    grok_key = st.text_input(
        "Groq API Key",
        value=st.session_state.grok_api_key,
        type="password",
        placeholder="gsk_xxxxxxxxxxxx",
        help="Get your FREE key from console.groq.com"
    )
    if grok_key != st.session_state.grok_api_key:
        st.session_state.grok_api_key = grok_key
    
    if st.button("🧪 Test Groq", use_container_width=True):
        if grok_key:
            if not grok_key.startswith("gsk_"):
                st.error("❌ Keys start with 'gsk_'")
            else:
                with st.spinner("Testing..."):
                    result = call_grok_api("Say hello", {}, grok_key)
                    if result and "❌" not in result:
                        st.success("✅ Connected!")
                    else:
                        st.error(result)
        else:
            st.warning("Enter key first")
    
    st.markdown("---")
    
    # Weather API Key Section
    st.markdown("##### 🌤️ OpenWeatherMap API")
    st.caption("Get key: [openweathermap.org](https://openweathermap.org/api)")
    weather_key = st.text_input(
        "Weather API Key",
        value=st.session_state.weather_api_key,
        type="password",
        placeholder="Enter API key",
        help="Get your key from openweathermap.org"
    )
    if weather_key != st.session_state.weather_api_key:
        st.session_state.weather_api_key = weather_key
    
    if st.button("🧪 Test Weather", use_container_width=True):
        if weather_key:
            with st.spinner("Testing..."):
                result = fetch_weather("Delhi", weather_key)
                if result:
                    st.success(f"✅ {result['temp']}°C")
                else:
                    st.error("❌ Invalid key")
        else:
            st.warning("Enter key first")
    
    st.markdown("---")
    
    # Quick Stats
    st.header("📊 Sensor Stats")
    sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
    if sensor_vals:
        avg = sum(sensor_vals) / len(sensor_vals)
        st.metric("Average Moisture", f"{avg:.1f}%")
        col1, col2 = st.columns(2)
        col1.metric("Min", f"{min(sensor_vals):.1f}%")
        col2.metric("Max", f"{max(sensor_vals):.1f}%")
        st.metric("Active Sensors", f"{len(sensor_vals)}/20")
    else:
        st.info("Enter sensor readings")
    
    st.markdown("---")
    st.markdown("**India Crop Recommendation v2.0**")
    st.markdown("🌾 Streamlit Edition")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🌾 India Crop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered crop recommendations with 20 sensor inputs & Grok chatbot</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "🤖 AI Chatbot", "📊 Data Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Location & Weather")
        
        state = st.selectbox("State", INDIAN_STATES, index=3)  # Maharashtra default
        district = st.text_input("District (optional)", placeholder="e.g., Pune")
        
        # Weather fetch
        weather_col1, weather_col2 = st.columns(2)
        with weather_col1:
            temperature = st.number_input("Temperature (°C)", value=28.0, min_value=0.0, max_value=50.0)
        with weather_col2:
            if st.button("🌤️ Auto-fetch Weather"):
                if st.session_state.weather_api_key:
                    city = district if district else state
                    weather = fetch_weather(city, st.session_state.weather_api_key)
                    if weather:
                        st.success(f"{weather['city']}: {weather['temp']}°C, {weather['description']}")
                        temperature = weather['temp']
                else:
                    st.warning("Add Weather API key in sidebar")
        
        rainfall = st.number_input("Rainfall (mm)", value=50.0, min_value=0.0, max_value=500.0)
        humidity = st.number_input("Humidity (%)", value=65.0, min_value=0.0, max_value=100.0)
        irrigation = st.checkbox("Irrigation Available", value=True)
        planting_date = st.date_input("Planting Date", value=date.today())
    
    with col2:
        st.subheader("💧 20 Sensor Readings")
        st.caption("Soil moisture % at 15cm depth")
        
        # Quick actions
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("📊 Load Sample", use_container_width=True):
                if state in SAMPLE_DATA:
                    st.session_state.sensor_values = SAMPLE_DATA[state].copy()
                    st.rerun()
        with action_col2:
            if st.button("🎲 Random", use_container_width=True):
                import random
                st.session_state.sensor_values = [round(random.uniform(10, 40), 2) for _ in range(20)]
                st.rerun()
        with action_col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.sensor_values = [0.0] * 20
                st.rerun()
        
        # Sensor input grid (4 columns x 5 rows)
        for row in range(5):
            cols = st.columns(4)
            for col_idx in range(4):
                sensor_idx = row * 4 + col_idx
                with cols[col_idx]:
                    st.session_state.sensor_values[sensor_idx] = st.number_input(
                        f"S{sensor_idx + 1}",
                        value=st.session_state.sensor_values[sensor_idx],
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key=f"sensor_{sensor_idx}"
                    )
    
    st.markdown("---")
    
    # Get Recommendations Button
    if st.button("🌾 Get Crop Recommendations", type="primary", use_container_width=True):
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        
        if len(sensor_vals) < 5:
            st.error("Please enter at least 5 sensor readings")
        else:
            avg_moisture = sum(sensor_vals) / len(sensor_vals)
            month = planting_date.month
            
            with st.spinner("Analyzing conditions..."):
                recommendations = calculate_crop_scores(avg_moisture, temperature, month)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} suitable crops!")
                
                # Display recommendations
                for i, rec in enumerate(recommendations):
                    confidence_pct = int(rec["confidence"] * 100)
                    
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.markdown(f"### {rec['emoji']}")
                    with col_b:
                        st.markdown(f"**{rec['crop']}** - {rec['season'].title()} season")
                        st.caption(rec['notes'])
                    with col_c:
                        st.metric("Confidence", f"{confidence_pct}%")
                    
                    st.progress(rec["confidence"])
                    st.markdown("---")
                
                # Summary
                st.info(f"""
                **Summary:**
                - State: {state} | District: {district or 'N/A'}
                - Season: {get_season(month).title()}
                - Avg Soil Moisture: {avg_moisture:.1f}%
                - Temperature: {temperature}°C
                - Active Sensors: {len(sensor_vals)}/20
                """)
            else:
                st.warning("No suitable crops found for current conditions")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    # Dark themed chat wrapper
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 AI Farming Assistant | Powered by Groq</div>', unsafe_allow_html=True)
    
    # Chat container with dark background
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if not st.session_state.chat_history:
            st.markdown('''
            <div class="chat-ai">
                👋 <strong>Hello! I'm your AI farming assistant.</strong><br><br>
                I can help you with:<br>
                • <strong>Crop recommendations</strong> for your region<br>
                • <strong>Soil & water management</strong> advice<br>
                • <strong>Weather-based</strong> farming decisions<br><br>
                Just ask me anything about Indian agriculture!
            </div>
            ''', unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    # Convert markdown to HTML for better display
                    content = msg["content"].replace('\n', '<br>').replace('**', '<strong>').replace('*', '<em>')
                    content = content.replace('<strong><strong>', '<strong>').replace('</strong></strong>', '</strong>')
                    st.markdown(f'<div class="chat-ai">{content}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🌾 Recommend Crops"):
            st.session_state.pending_message = "What crops should I grow based on my current conditions?"
    with quick_col2:
        if st.button("💧 Soil Analysis"):
            st.session_state.pending_message = "Analyze my soil moisture readings and give advice"
    with quick_col3:
        if st.button("🚿 Irrigation Tips"):
            st.session_state.pending_message = "Give me irrigation tips for my region"
    with quick_col4:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Check for pending message
    if 'pending_message' in st.session_state:
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    else:
        user_input = None
    
    # Chat input
    user_message = st.chat_input("Ask about crops, soil, weather...")
    
    if user_message or user_input:
        message = user_message or user_input
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": message})
        
        # Build context
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        context = {
            "state": state if 'state' in dir() else "Maharashtra",
            "month": datetime.now().month,
        }
        if sensor_vals:
            context["soil_moisture"] = sum(sensor_vals) / len(sensor_vals)
            context["sensor_count"] = len(sensor_vals)
        
        # Get response
        with st.spinner("🤖 Thinking..."):
            if st.session_state.grok_api_key:
                response = call_grok_api(message, context, st.session_state.grok_api_key)
                # Response is always a string now, check for error markers
            else:
                response = get_offline_response(message, context)
        
        # Add AI response
        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("📊 Sensor Data Analysis")
    
    sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
    
    if sensor_vals:
        # Visualization
        df = pd.DataFrame({
            "Sensor": [f"S{i+1}" for i, v in enumerate(st.session_state.sensor_values) if v > 0],
            "Moisture (%)": sensor_vals
        })
        
        st.bar_chart(df.set_index("Sensor"))
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{sum(sensor_vals)/len(sensor_vals):.2f}%")
        with col2:
            st.metric("Minimum", f"{min(sensor_vals):.2f}%")
        with col3:
            st.metric("Maximum", f"{max(sensor_vals):.2f}%")
        with col4:
            variance = sum((x - sum(sensor_vals)/len(sensor_vals))**2 for x in sensor_vals) / len(sensor_vals)
            st.metric("Std Dev", f"{variance**0.5:.2f}")
        
        # Data table
        st.markdown("### Raw Data")
        st.dataframe(df, use_container_width=True)
        
        # Moisture interpretation
        avg = sum(sensor_vals) / len(sensor_vals)
        st.markdown("### Interpretation")
        if avg < 20:
            st.error("⚠️ **Low Moisture**: Consider immediate irrigation. Drought-resistant crops recommended.")
        elif avg < 35:
            st.warning("💧 **Moderate Moisture**: Good for most crops. Monitor regularly.")
        elif avg < 55:
            st.success("✅ **Optimal Moisture**: Excellent conditions for crop growth.")
        else:
            st.info("🌊 **High Moisture**: Good for water-loving crops. Ensure proper drainage.")
    else:
        st.info("📝 Enter sensor readings in the Recommendations tab to see analysis")
        
        # Show sample data preview
        st.markdown("### Available Sample Data")
        sample_df = pd.DataFrame(SAMPLE_DATA).T
        sample_df.columns = [f"S{i+1}" for i in range(20)]
        sample_df["Average"] = sample_df.mean(axis=1)
        st.dataframe(sample_df[["Average"]], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 India Crop Recommendation System v2.0 | Streamlit Edition</p>
    <p>Built for Indian Agriculture 🇮🇳 | 20 Sensor Inputs | Grok AI Chatbot</p>
</div>
""", unsafe_allow_html=True)
