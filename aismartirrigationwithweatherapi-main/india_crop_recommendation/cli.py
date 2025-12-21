"""
India Crop Recommendation System - Unified CLI
Main entry point for all commands

Usage:
    python -m india_crop_recommendation.cli <command> [options]
    
Commands:
    ingest      Load and process data from CSV/API sources
    train       Train ML models
    serve       Start the FastAPI server
    recommend   Get crop recommendations (CLI mode)
    stats       Show data statistics
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("india_crop")

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: ingest
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_ingest(args):
    """Ingest data from CSV or API sources."""
    from ingest.csv_loader import load_all_states, load_state, export_to_parquet, list_available_states
    
    if args.list:
        states = list_available_states()
        print(f"📁 Available CSV data for {len(states)} states:")
        for s in states:
            print(f"  • {s}")
        return
    
    if args.export:
        log.info(f"Exporting to Parquet: {args.export}")
        export_to_parquet(args.export, args.year)
        print(f"✅ Exported to {args.export}")
        return
    
    if args.state:
        df = load_state(args.state, args.year)
        print(f"\n📊 {args.state} ({len(df)} rows)")
    else:
        df = load_all_states(args.year)
        print(f"\n📊 All States ({len(df)} rows)")
    
    print(df.head(args.rows).to_string(index=False))
    print(f"\n📈 Soil Moisture: mean={df['soil_moisture_pct'].mean():.1f}%, "
          f"range=[{df['soil_moisture_pct'].min():.1f}, {df['soil_moisture_pct'].max():.1f}]%")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: train
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train crop recommendation models."""
    from train.train_model import RuleBasedCropRecommender, CropRecommenderML, generate_synthetic_training_data
    
    print("🚀 Starting model training...")
    
    # Generate or load training data
    if args.data:
        import pandas as pd
        df = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        print(f"📂 Loaded {len(df)} training samples from {args.data}")
    else:
        print("📝 Generating synthetic training data...")
        df = generate_synthetic_training_data(n_samples=args.samples)
    
    # Train rule-based
    if args.model in ("rule", "all"):
        print("\n📐 Rule-based model ready (no training needed)")
        rb = RuleBasedCropRecommender()
        print(f"   Crops: {len(rb.CROP_RULES)}")
    
    # Train ML model
    if args.model in ("ml", "all"):
        print("\n🤖 Training ML model...")
        ml = CropRecommenderML()
        metrics = ml.train(df)
        ml.save(Path(args.output) / "lightgbm_model.joblib")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"   F1 Score: {metrics.get('f1_weighted', 0):.2%}")
        print(f"✅ Model saved to {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: serve
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_serve(args):
    """Start FastAPI server."""
    import uvicorn
    
    print(f"🚀 Starting API server on {args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   UI:   Open ui/index.html in browser")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: recommend
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_recommend(args):
    """Get crop recommendations from CLI."""
    from train.train_model import RuleBasedCropRecommender
    
    model = RuleBasedCropRecommender()
    
    # Parse sensor readings if provided
    sensor_avg = args.moisture
    if args.sensors:
        readings = [float(x) for x in args.sensors.split(",")]
        sensor_avg = sum(readings) / len(readings)
        print(f"📡 {len(readings)} sensor readings, avg: {sensor_avg:.1f}%")
    
    recommendations = model.recommend(
        soil_moisture_pct=sensor_avg,
        temp_mean_c=args.temp,
        precip_mm=args.rainfall,
        month=args.month
    )
    
    print(f"\n🌾 Crop Recommendations for {args.state or 'your location'}:")
    print(f"   Conditions: soil={sensor_avg:.0f}%, temp={args.temp}°C, rain={args.rainfall}mm")
    print("-" * 50)
    
    for i, rec in enumerate(recommendations[:args.top], 1):
        conf = rec["confidence"] * 100
        emoji = {"rice": "🌾", "wheat": "🌾", "maize": "🌽", "cotton": "🧵", 
                "sugarcane": "🎋", "groundnut": "🥜", "potato": "🥔"}.get(rec["crop"], "🌱")
        print(f"   {i}. {emoji} {rec['crop'].title():12} {conf:5.1f}% | {rec.get('notes', '')}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: stats
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_stats(args):
    """Show data statistics."""
    from ingest.csv_loader import load_all_states, get_district_stats
    
    if args.state:
        stats = get_district_stats(args.state)
        print(f"\n📊 {args.state} District Statistics ({len(stats)} districts):")
        print(stats.to_string(index=False))
    else:
        df = load_all_states()
        print(f"\n📊 Overall Statistics ({len(df)} records)")
        
        by_state = df.groupby("state").agg({
            "soil_moisture_pct": ["count", "mean", "std", "min", "max"],
            "district": "nunique"
        }).round(2)
        by_state.columns = ["records", "mean%", "std%", "min%", "max%", "districts"]
        print(by_state.to_string())
        
        print(f"\n📅 Date range: {df['date'].min()} to {df['date'].max()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="india_crop",
        description="🌾 India Crop Recommendation System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m india_crop_recommendation.cli ingest --list
  python -m india_crop_recommendation.cli ingest -s Maharashtra
  python -m india_crop_recommendation.cli train --model all
  python -m india_crop_recommendation.cli serve --port 8000
  python -m india_crop_recommendation.cli recommend -m 25 -t 28 -r 50
  python -m india_crop_recommendation.cli stats -s Gujarat
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Load data from CSV/API")
    p_ingest.add_argument("-s", "--state", help="State name")
    p_ingest.add_argument("-y", "--year", type=int, help="Year filter")
    p_ingest.add_argument("-l", "--list", action="store_true", help="List available states")
    p_ingest.add_argument("-e", "--export", help="Export to Parquet file")
    p_ingest.add_argument("-n", "--rows", type=int, default=10, help="Rows to display")
    p_ingest.set_defaults(func=cmd_ingest)
    
    # train
    p_train = subparsers.add_parser("train", help="Train ML models")
    p_train.add_argument("-d", "--data", help="Training data file (CSV/Parquet)")
    p_train.add_argument("-m", "--model", choices=["rule", "ml", "all"], default="all")
    p_train.add_argument("-s", "--samples", type=int, default=5000, help="Synthetic samples")
    p_train.add_argument("-o", "--output", default="./models", help="Output directory")
    p_train.set_defaults(func=cmd_train)
    
    # serve
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("-H", "--host", default="0.0.0.0", help="Host")
    p_serve.add_argument("-p", "--port", type=int, default=8000, help="Port")
    p_serve.add_argument("-r", "--reload", action="store_true", help="Auto-reload")
    p_serve.set_defaults(func=cmd_serve)
    
    # recommend
    p_rec = subparsers.add_parser("recommend", help="Get crop recommendations")
    p_rec.add_argument("-s", "--state", help="State name")
    p_rec.add_argument("-m", "--moisture", type=float, default=30, help="Soil moisture %")
    p_rec.add_argument("-t", "--temp", type=float, default=28, help="Temperature °C")
    p_rec.add_argument("-r", "--rainfall", type=float, default=50, help="Rainfall mm")
    p_rec.add_argument("--month", type=int, default=6, help="Month (1-12)")
    p_rec.add_argument("--sensors", help="Comma-separated sensor readings")
    p_rec.add_argument("--top", type=int, default=5, help="Top N recommendations")
    p_rec.set_defaults(func=cmd_recommend)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show data statistics")
    p_stats.add_argument("-s", "--state", help="State name")
    p_stats.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
