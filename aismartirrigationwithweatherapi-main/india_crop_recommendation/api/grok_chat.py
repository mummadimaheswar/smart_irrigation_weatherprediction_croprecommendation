"""
Grok AI Chatbot Integration for Crop Recommendations
Uses xAI's Grok API for natural language conversations

Usage:
    from api.grok_chat import GrokChatBot
    bot = GrokChatBot(api_key="your_key")
    response = bot.chat("What crops should I grow in Maharashtra during monsoon?")
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

# Grok API endpoint (xAI)
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# System prompt for agricultural assistant
SYSTEM_PROMPT = """You are an expert Indian agricultural advisor AI assistant. Your role is to help farmers with:

1. **Crop Recommendations**: Suggest suitable crops based on:
   - Location (state, district)
   - Season (Kharif: June-Oct, Rabi: Oct-March, Zaid: March-June)
   - Soil moisture levels (%)
   - Temperature and rainfall
   - Available irrigation

2. **Soil & Water Management**: Advise on irrigation schedules, soil health, moisture conservation.

3. **Weather Guidance**: Interpret weather data for farming decisions.

4. **Regional Knowledge**: You know Indian states, their agro-climatic zones, and traditional crops:
   - Maharashtra: Cotton, Soybean, Sugarcane, Jowar
   - Punjab: Wheat, Rice, Cotton
   - Gujarat: Groundnut, Cotton, Castor
   - Tamil Nadu: Rice, Sugarcane, Banana
   - Rajasthan: Bajra, Mustard, Pulses

**Response Style**:
- Be concise and practical
- Use simple language farmers can understand
- Give specific, actionable advice
- Include confidence levels when recommending crops
- Mention risks and precautions
- Use bullet points for clarity

**Available Data Context** (when provided):
- Soil moisture sensor readings (20 sensors at 15cm depth)
- State-wise historical soil moisture from CSV datasets
- Weather forecasts

Always be helpful, accurate, and supportive of Indian farmers."""


class GrokChatBot:
    """Grok AI chatbot for agricultural advice."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        self.model = "grok-beta"  # or "grok-2-latest"
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 messages for context
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_messages(self, user_message: str, context: Optional[Dict] = None) -> List[Dict]:
        """Build message list with system prompt, history, and user message."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add context if provided (sensor data, location, etc.)
        if context:
            context_str = self._format_context(context)
            messages.append({
                "role": "system", 
                "content": f"Current farming context:\n{context_str}"
            })
        
        # Add conversation history
        messages.extend(self.conversation_history[-self.max_history:])
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _format_context(self, context: Dict) -> str:
        """Format context data for the AI."""
        lines = []
        
        if "state" in context:
            lines.append(f"📍 Location: {context['state']}, {context.get('district', 'N/A')}")
        
        if "soil_moisture_pct" in context:
            lines.append(f"💧 Soil Moisture: {context['soil_moisture_pct']:.1f}%")
        
        if "sensor_readings" in context and context["sensor_readings"]:
            readings = context["sensor_readings"]
            avg = sum(readings) / len(readings)
            lines.append(f"📡 Sensors ({len(readings)} readings): avg={avg:.1f}%, min={min(readings):.1f}%, max={max(readings):.1f}%")
        
        if "temperature_c" in context:
            lines.append(f"🌡️ Temperature: {context['temperature_c']}°C")
        
        if "rainfall_mm" in context:
            lines.append(f"🌧️ Rainfall: {context['rainfall_mm']}mm")
        
        if "month" in context:
            month = context["month"]
            season = "Kharif" if 6 <= month <= 10 else "Rabi" if month >= 10 or month <= 3 else "Zaid"
            lines.append(f"📅 Month: {month} (Season: {season})")
        
        return "\n".join(lines) if lines else "No specific context provided."
    
    async def chat_async(
        self, 
        message: str, 
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Send a message and get a response (async)."""
        if not self.api_key:
            return {
                "success": False,
                "error": "Grok API key not configured. Set GROK_API_KEY or XAI_API_KEY environment variable.",
                "response": None
            }
        
        messages = self._build_messages(message, context)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    GROK_API_URL,
                    headers=self._get_headers(),
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            assistant_message = data["choices"][0]["message"]["content"]
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return {
                "success": True,
                "response": assistant_message,
                "model": data.get("model", self.model),
                "usage": data.get("usage", {})
            }
            
        except httpx.HTTPStatusError as e:
            log.error(f"Grok API error: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"API error: {e.response.status_code}",
                "response": None
            }
        except Exception as e:
            log.error(f"Grok chat error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def chat_sync(
        self, 
        message: str, 
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Send a message and get a response (sync)."""
        if not self.api_key:
            return {
                "success": False,
                "error": "Grok API key not configured. Set GROK_API_KEY or XAI_API_KEY environment variable.",
                "response": None
            }
        
        messages = self._build_messages(message, context)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    GROK_API_URL,
                    headers=self._get_headers(),
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            assistant_message = data["choices"][0]["message"]["content"]
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return {
                "success": True,
                "response": assistant_message,
                "model": data.get("model", self.model),
                "usage": data.get("usage", {})
            }
            
        except Exception as e:
            log.error(f"Grok chat error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def set_api_key(self, api_key: str):
        """Update the API key dynamically."""
        self.api_key = api_key


# Singleton instance
_chatbot: Optional[GrokChatBot] = None

def get_chatbot(api_key: Optional[str] = None) -> GrokChatBot:
    """
    Get or create the chatbot instance.
    
    Args:
        api_key: Optional API key to use. If provided, updates the chatbot's key.
                If not provided, uses the existing key or environment variable.
    """
    global _chatbot
    if _chatbot is None:
        _chatbot = GrokChatBot(api_key=api_key)
    elif api_key:
        # Update API key if provided (from web UI)
        _chatbot.set_api_key(api_key)
    return _chatbot
