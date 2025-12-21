"""Generate concise agronomic advisories."""

HINTS = {
    "rice": "Rice needs saturated soil; check bunds.",
    "wheat": "Wheat prefers moderate moisture.",
    "maize": "Maize is stress-sensitive at tasseling.",
}


def generate_advisory(decision: str, reason: str, details: dict, crop: str = "wheat") -> str:
    vwc = details.get("current_vwc", 0) or 0
    rain = details.get("forecast_rain_24h_mm", 0) or 0
    etc = details.get("etc_mm_day", 0) or 0

    parts = [HINTS.get(crop.lower(), "Monitor crop health.")]

    if vwc < 0.18:
        parts.append("Soil critically dry.")
    elif vwc < 0.24:
        parts.append("Soil moisture marginal.")

    if rain >= 10:
        parts.append(f"Heavy rain expected ({rain:.0f} mm).")
    elif rain >= 5:
        parts.append(f"Light rain forecast ({rain:.0f} mm).")

    if etc >= 5:
        parts.append(f"High ETc ({etc:.1f} mm/d).")

    return " ".join(parts)