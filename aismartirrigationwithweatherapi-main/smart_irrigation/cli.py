"""CLI for Smart Irrigation decision engine."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import advisory, data, decision, weather
from .et import compute_et0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smart Irrigation")
    p.add_argument("--data", required=True, help="Sensor CSV path")
    p.add_argument("--lat", type=float, default=28.6139)
    p.add_argument("--lon", type=float, default=77.2090)
    p.add_argument("--crop", default="wheat")
    p.add_argument("--state")
    p.add_argument("--district")
    p.add_argument("--output", default="last_decision.json")
    p.add_argument("--api-key")
    args = p.parse_args(argv)

    path = Path(args.data).expanduser().resolve()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = data.load_and_clean(str(path), state=args.state, district=args.district)
    if df.empty:
        sys.exit("No records after filtering.")

    row = df.sort_values("date").iloc[-1]
    vwc = float(row["soil_moisture"])

    wdf = weather.fetch_openweather(args.lat, args.lon, args.api_key)
    if wdf.empty:
        sys.exit("Weather forecast unavailable.")

    summary = weather.get_forecast_summary(wdf, hours=24)
    rain = summary["total_precipitation_mm"]
    et0 = compute_et0(wdf.iloc[0], lat_deg=args.lat)
    etc = decision.compute_crop_etc(et0, args.crop)

    # Simple 24h moisture prediction based on water balance
    pred_vwc = max(vwc - min(max((etc - rain) / 100, 0.01), 0.05), 0.0)

    label, reason, details = decision.decide_irrigation(
        current_vwc=vwc,
        predicted_vwc_24h=pred_vwc,
        forecast_rain_24h_mm=rain,
        et0_mm_day=et0,
        crop_type=args.crop,
    )
    adv = advisory.generate_advisory(label, reason, details, args.crop)

    print(f"Soil moisture: {vwc:.2%} | Rain 24h: {rain:.1f} mm | ET0: {et0:.2f} mm/d")
    print(f"Decision: {label}")
    print(f"Reason: {reason}")
    print(f"Advisory: {adv}")

    out = Path(args.output).expanduser().resolve()
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "decision": label,
        "reason": reason,
        "soil_moisture": vwc,
        "predicted_24h": pred_vwc,
        "rain_24h_mm": rain,
        "et0_mm_day": et0,
        "etc_mm_day": etc,
        "advisory": adv,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())