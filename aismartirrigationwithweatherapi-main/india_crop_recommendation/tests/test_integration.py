"""
Integration Tests for India Crop Recommendation System
Tests end-to-end flows: data loading → model → API

Run with: pytest tests/test_integration.py -v
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCSVLoader:
    """Test CSV data loading functionality."""
    
    def test_list_available_states(self):
        from ingest.csv_loader import list_available_states
        states = list_available_states()
        assert len(states) > 0, "Should find at least one state"
        assert isinstance(states, list)
    
    def test_load_single_state(self):
        from ingest.csv_loader import load_state
        df = load_state("Maharashtra")
        assert len(df) > 0, "Should load data for Maharashtra"
        assert "soil_moisture_pct" in df.columns
        assert "date" in df.columns
        assert "district" in df.columns
    
    def test_load_all_states(self):
        from ingest.csv_loader import load_all_states
        df = load_all_states()
        assert len(df) > 0, "Should load data from all CSVs"
        assert df["state"].nunique() > 1, "Should have multiple states"
    
    def test_column_standardization(self):
        from ingest.csv_loader import load_state
        df = load_state("Gujarat")
        expected_cols = ["date", "state", "district", "soil_moisture_pct", "year", "month"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_date_parsing(self):
        from ingest.csv_loader import load_state
        import pandas as pd
        df = load_state("Punjab")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestRuleBasedModel:
    """Test rule-based crop recommender."""
    
    def test_recommend_returns_results(self):
        from train.train_model import RuleBasedCropRecommender
        model = RuleBasedCropRecommender()
        recs = model.recommend(soil_moisture_pct=35, temp_mean_c=28, precip_mm=50, month=7)
        assert len(recs) > 0, "Should return recommendations"
    
    def test_recommend_format(self):
        from train.train_model import RuleBasedCropRecommender
        model = RuleBasedCropRecommender()
        recs = model.recommend(soil_moisture_pct=25, temp_mean_c=22, precip_mm=30, month=11)
        
        for rec in recs:
            assert "crop" in rec
            assert "confidence" in rec
            assert 0 <= rec["confidence"] <= 1
    
    def test_kharif_season_crops(self):
        from train.train_model import RuleBasedCropRecommender
        model = RuleBasedCropRecommender()
        # Monsoon conditions
        recs = model.recommend(soil_moisture_pct=50, temp_mean_c=30, precip_mm=150, month=7)
        crops = [r["crop"] for r in recs[:3]]
        # Should favor kharif crops
        assert any(c in crops for c in ["rice", "maize", "cotton", "soybean"])
    
    def test_rabi_season_crops(self):
        from train.train_model import RuleBasedCropRecommender
        model = RuleBasedCropRecommender()
        # Winter conditions
        recs = model.recommend(soil_moisture_pct=30, temp_mean_c=18, precip_mm=20, month=11)
        crops = [r["crop"] for r in recs[:3]]
        # Should favor rabi crops
        assert any(c in crops for c in ["wheat", "mustard", "chickpea", "potato"])


class TestDataPreparation:
    """Test training data preparation."""
    
    def test_synthetic_data_generation(self):
        from train.prepare_data import generate_fully_synthetic
        df = generate_fully_synthetic(100)
        assert len(df) == 100
        assert "crop_name" in df.columns
        assert "soil_moisture_pct" in df.columns
        assert "temp_mean_c" in df.columns
    
    def test_crop_label_assignment(self):
        from train.prepare_data import assign_crop_labels
        import pandas as pd
        
        df = pd.DataFrame({
            "soil_moisture_pct": [50, 25, 35],
            "temp_mean_c": [30, 18, 28],
            "month": [7, 11, 6]
        })
        df = assign_crop_labels(df)
        assert "crop_name" in df.columns
        assert df["crop_name"].notna().all()


class TestAPIModels:
    """Test API Pydantic models."""
    
    def test_request_model(self):
        from api.main import RecommendRequest
        req = RecommendRequest(
            state="Maharashtra",
            soil_moisture_pct=35.0,
            temperature_c=28.0
        )
        assert req.state == "Maharashtra"
        assert req.soil_moisture_pct == 35.0
    
    def test_request_validation(self):
        from api.main import RecommendRequest
        from pydantic import ValidationError
        
        # soil_moisture_pct must be 0-100
        with pytest.raises(ValidationError):
            RecommendRequest(state="Test", soil_moisture_pct=150)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_csv_to_recommendation(self):
        """Load CSV → Extract features → Get recommendation."""
        from ingest.csv_loader import load_state
        from train.train_model import RuleBasedCropRecommender
        
        # Load real data
        df = load_state("Maharashtra")
        assert len(df) > 0
        
        # Get sample row
        sample = df.iloc[0]
        
        # Get recommendation
        model = RuleBasedCropRecommender()
        recs = model.recommend(
            soil_moisture_pct=sample["soil_moisture_pct"],
            month=sample["month"]
        )
        assert len(recs) > 0
    
    def test_data_stats(self):
        """Verify data statistics are reasonable."""
        from ingest.csv_loader import load_all_states
        
        df = load_all_states()
        
        # Check value ranges
        assert df["soil_moisture_pct"].min() >= 0
        assert df["soil_moisture_pct"].max() <= 100
        assert df["month"].min() >= 1
        assert df["month"].max() <= 12


# Quick sanity check when run directly
if __name__ == "__main__":
    print("Running quick sanity checks...")
    
    # Test CSV loading
    from ingest.csv_loader import list_available_states, load_state
    states = list_available_states()
    print(f"✓ Found {len(states)} states with CSV data")
    
    df = load_state("Maharashtra")
    print(f"✓ Loaded {len(df)} rows for Maharashtra")
    
    # Test model
    from train.train_model import RuleBasedCropRecommender
    model = RuleBasedCropRecommender()
    recs = model.recommend(soil_moisture_pct=30, temp_mean_c=25, month=7)
    print(f"✓ Got {len(recs)} recommendations")
    print(f"  Top crop: {recs[0]['crop']} ({recs[0]['confidence']*100:.0f}%)")
    
    print("\n✅ All sanity checks passed!")
