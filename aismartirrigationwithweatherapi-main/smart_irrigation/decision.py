"""Irrigation decision engine with crop-specific thresholds."""
from .et import compute_etc

# Crop parameters: (kc, vwc_min, vwc_critical, et_threshold_mm)
CROPS = {
    "rice":  (1.05, 0.30, 0.25, 5.0),
    "wheat": (0.95, 0.24, 0.20, 4.0),
    "maize": (0.90, 0.22, 0.18, 3.5),
}
DEFAULT = (0.90, 0.23, 0.18, 4.0)
RAIN_THRESHOLD = 5.0


def _params(crop: str):
    return CROPS.get(crop.lower(), DEFAULT)


def compute_crop_etc(et0: float, crop: str) -> float:
    return compute_etc(et0, _params(crop)[0])


def decide_irrigation(
    *, current_vwc: float, predicted_vwc_24h: float,
    forecast_rain_24h_mm: float, et0_mm_day: float,
    crop_type: str = "wheat", **_
) -> tuple[str, str, dict]:
    kc, vwc_min, vwc_crit, et_thresh = _params(crop_type)
    etc = compute_etc(et0_mm_day, kc)
    deficit = etc - forecast_rain_24h_mm

    d = {
        "current_vwc": round(current_vwc, 4),
        "predicted_vwc_24h": round(predicted_vwc_24h, 4),
        "forecast_rain_24h_mm": round(forecast_rain_24h_mm, 2),
        "et0_mm_day": round(et0_mm_day, 2),
        "etc_mm_day": round(etc, 2),
        "water_deficit_mm": round(deficit, 2),
    }

    # Critical moisture
    if current_vwc < vwc_crit:
        return "Irrigate", f"Moisture {current_vwc:.0%} < critical {vwc_crit:.0%}.", d

    # Rain expected
    if forecast_rain_24h_mm >= RAIN_THRESHOLD and predicted_vwc_24h >= vwc_min:
        return "Skip", f"Rain {forecast_rain_24h_mm:.1f} mm forecast; defer irrigation.", d

    # Predicted drop
    if predicted_vwc_24h < vwc_min:
        return "Irrigate", f"Predicted moisture {predicted_vwc_24h:.0%} < min {vwc_min:.0%}.", d

    # High ET deficit
    if deficit > et_thresh:
        return "Irrigate", f"Water deficit {deficit:.1f} mm > threshold {et_thresh:.1f} mm.", d

    return "Skip", f"Moisture adequate ({current_vwc:.0%}).", d