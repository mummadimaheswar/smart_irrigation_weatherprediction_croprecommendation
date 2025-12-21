"""
ML Training Pipeline for Crop Recommendation
India Crop Recommendation System

PROMPT 7: Two model approaches:
1. Rule-based baseline
2. ML model (LightGBM / RandomForest)

With:
- Cross-validation (time-split by year)
- Metrics logging
- Feature importance
- Hyperparameter grid
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    log.warning("LightGBM not installed, using sklearn alternatives")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CROPS = [
    "rice", "wheat", "maize", "cotton", "sugarcane",
    "groundnut", "soybean", "mustard", "chickpea", "potato"
]

FEATURE_COLS = [
    "soil_moisture_pct", "temp_mean_c", "temp_min_c", "temp_max_c",
    "precip_mm", "humidity_pct", "wind_speed_ms",
    "ndvi", "month", "lat", "lon",
    "precip_7d_sum", "temp_7d_mean", "gdd_base10", "water_deficit_mm"
]

TARGET_COL = "crop_name"

RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedCropRecommender:
    """
    Rule-based baseline model for crop recommendation.
    
    Pros:
    - Interpretable: clear decision rules
    - No training required
    - Works with limited data
    - Encodes domain knowledge
    
    Cons:
    - Rigid rules may not capture complex patterns
    - Requires expert knowledge to define rules
    - Cannot learn from data
    - May miss regional variations
    """
    
    # Crop requirements (soil_moisture_min, soil_moisture_max, temp_min, temp_max, precip_min)
    CROP_RULES = {
        "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "precip_min": 100, "season": "kharif"},
        "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "precip_min": 40, "season": "rabi"},
        "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "precip_min": 50, "season": "kharif"},
        "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "precip_min": 60, "season": "kharif"},
        "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "precip_min": 150, "season": "kharif"},
        "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "precip_min": 50, "season": "kharif"},
        "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "precip_min": 60, "season": "kharif"},
        "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "precip_min": 25, "season": "rabi"},
        "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "precip_min": 30, "season": "rabi"},
        "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "precip_min": 40, "season": "rabi"},
    }
    
    def __init__(self):
        self.name = "rule_based"
    
    def _get_season(self, month: int) -> str:
        """Determine season from month."""
        if 6 <= month <= 10:
            return "kharif"
        elif 10 <= month <= 3 or month <= 3:
            return "rabi"
        else:
            return "zaid"
    
    def _score_crop(
        self,
        crop: str,
        soil_moisture: float,
        temp_mean: float,
        precip: float,
        month: int
    ) -> float:
        """Score how suitable a crop is for given conditions."""
        if crop not in self.CROP_RULES:
            return 0.0
        
        rules = self.CROP_RULES[crop]
        score = 0.0
        
        # Soil moisture match (0-30 points)
        if rules["sm_min"] <= soil_moisture <= rules["sm_max"]:
            # Best score at midpoint
            midpoint = (rules["sm_min"] + rules["sm_max"]) / 2
            distance = abs(soil_moisture - midpoint) / (rules["sm_max"] - rules["sm_min"])
            score += 30 * (1 - distance)
        
        # Temperature match (0-30 points)
        if rules["temp_min"] <= temp_mean <= rules["temp_max"]:
            midpoint = (rules["temp_min"] + rules["temp_max"]) / 2
            distance = abs(temp_mean - midpoint) / (rules["temp_max"] - rules["temp_min"])
            score += 30 * (1 - distance)
        
        # Precipitation match (0-20 points)
        if precip >= rules["precip_min"]:
            score += 20
        else:
            score += 20 * (precip / rules["precip_min"])
        
        # Season match (0-20 points)
        current_season = self._get_season(month)
        if rules["season"] == current_season:
            score += 20
        
        return score
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict best crop for each sample."""
        predictions = []
        
        for _, row in X.iterrows():
            sm = row.get("soil_moisture_pct", 30)
            temp = row.get("temp_mean_c", 25)
            precip = row.get("precip_mm", 50)
            month = row.get("month", 6)
            
            # Score all crops
            scores = {
                crop: self._score_crop(crop, sm, temp, precip, month)
                for crop in self.CROP_RULES
            }
            
            # Best crop
            best_crop = max(scores, key=scores.get)
            predictions.append(best_crop)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each crop."""
        n_samples = len(X)
        n_crops = len(self.CROP_RULES)
        crops = list(self.CROP_RULES.keys())
        
        probas = np.zeros((n_samples, n_crops))
        
        for i, (_, row) in enumerate(X.iterrows()):
            sm = row.get("soil_moisture_pct", 30)
            temp = row.get("temp_mean_c", 25)
            precip = row.get("precip_mm", 50)
            month = row.get("month", 6)
            
            scores = [
                self._score_crop(crop, sm, temp, precip, month)
                for crop in crops
            ]
            
            # Normalize to probabilities
            total = sum(scores)
            if total > 0:
                probas[i] = [s / total for s in scores]
            else:
                probas[i] = [1 / n_crops] * n_crops
        
        return probas
    
    def get_top_n_recommendations(
        self,
        X: pd.DataFrame,
        n: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """Get top N crop recommendations with confidence scores."""
        probas = self.predict_proba(X)
        crops = list(self.CROP_RULES.keys())
        
        recommendations = []
        for proba in probas:
            sorted_idx = np.argsort(proba)[::-1][:n]
            recs = [(crops[i], float(proba[i])) for i in sorted_idx]
            recommendations.append(recs)
        
        return recommendations

    def recommend(
        self,
        soil_moisture_pct: float = 30,
        temp_mean_c: float = 25,
        precip_mm: float = 50,
        month: int = 6,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to get recommendations from raw values.
        
        Returns list of dicts with crop, confidence, season, notes.
        """
        X = pd.DataFrame([{
            "soil_moisture_pct": soil_moisture_pct,
            "temp_mean_c": temp_mean_c,
            "precip_mm": precip_mm,
            "month": month
        }])
        
        probas = self.predict_proba(X)[0]
        crops = list(self.CROP_RULES.keys())
        
        results = []
        for i, crop in enumerate(crops):
            rules = self.CROP_RULES[crop]
            notes = []
            if soil_moisture_pct < rules["sm_min"]:
                notes.append("Low soil moisture")
            if soil_moisture_pct > rules["sm_max"]:
                notes.append("High soil moisture")
            if temp_mean_c < rules["temp_min"]:
                notes.append("Temperature too low")
            if temp_mean_c > rules["temp_max"]:
                notes.append("Temperature too high")
            if precip_mm < rules["precip_min"]:
                notes.append("May need irrigation")
            
            results.append({
                "crop": crop,
                "confidence": float(probas[i]),
                "season": rules["season"],
                "notes": "; ".join(notes) if notes else "Good conditions"
            })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CropRecommenderML:
    """
    ML-based crop recommendation model.
    
    Supports: LightGBM, RandomForest, GradientBoosting
    
    Pros:
    - Learns complex patterns from data
    - Adapts to regional variations
    - Can improve with more data
    - Feature importance for interpretability
    
    Cons:
    - Requires labeled training data
    - May overfit on small datasets
    - Less interpretable than rules
    - Computationally more expensive
    """
    
    HYPERPARAMETER_GRIDS = {
        "lightgbm": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 70],
            "min_child_samples": [20, 50],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_samples_split": [2, 5],
        }
    }
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.classes_ = None
        self.feature_importance_ = None
        
    def _create_model(self, **kwargs):
        """Create model instance."""
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                **kwargs
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **kwargs
            )
        else:  # gradient_boosting
            return GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                **kwargs
            )
    
    def _prepare_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Prepare features for training/prediction."""
        # Select feature columns
        available_cols = [c for c in FEATURE_COLS if c in X.columns]
        self.feature_cols = available_cols
        
        X_features = X[available_cols].copy()
        
        # Fill missing values
        X_features = X_features.fillna(X_features.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_features)
        else:
            X_scaled = self.scaler.transform(X_features)
        
        return X_scaled
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        use_time_split: bool = True
    ) -> Dict[str, float]:
        """
        Train the model with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series (crop names)
            cv_folds: Number of CV folds
            use_time_split: Use time-based split (by year)
        
        Returns:
            Cross-validation metrics
        """
        # Prepare features
        X_scaled = self._prepare_features(X, fit=True)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Create model
        self.model = self._create_model()
        
        # Cross-validation
        if use_time_split and "year" in X.columns:
            # Time-based split
            years = X["year"].unique()
            years.sort()
            
            cv_scores = []
            for i in range(1, len(years)):
                train_years = years[:i]
                val_year = years[i]
                
                train_mask = X["year"].isin(train_years)
                val_mask = X["year"] == val_year
                
                if train_mask.sum() < 10 or val_mask.sum() < 10:
                    continue
                
                X_train = X_scaled[train_mask]
                y_train = y_encoded[train_mask]
                X_val = X_scaled[val_mask]
                y_val = y_encoded[val_mask]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                
                cv_scores.append(accuracy_score(y_val, y_pred))
            
            cv_mean = np.mean(cv_scores) if cv_scores else 0
            cv_std = np.std(cv_scores) if cv_scores else 0
        else:
            # Standard K-fold
            cv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        # Fit on full data
        self.model.fit(X_scaled, y_encoded)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                self.feature_cols,
                self.model.feature_importances_
            ))
        
        return {
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "n_samples": len(X),
            "n_features": len(self.feature_cols),
            "n_classes": len(self.classes_)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crop labels."""
        X_scaled = self._prepare_features(X, fit=False)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self._prepare_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def get_top_n_recommendations(
        self,
        X: pd.DataFrame,
        n: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """Get top N recommendations with confidence."""
        probas = self.predict_proba(X)
        
        recommendations = []
        for proba in probas:
            sorted_idx = np.argsort(proba)[::-1][:n]
            recs = [(self.classes_[i], float(proba[i])) for i in sorted_idx]
            recommendations.append(recs)
        
        return recommendations
    
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 3
    ) -> Dict[str, Any]:
        """Tune hyperparameters using grid search."""
        X_scaled = self._prepare_features(X, fit=True)
        y_encoded = self.label_encoder.fit_transform(y)
        
        param_grid = self.HYPERPARAMETER_GRIDS.get(self.model_type, {})
        
        # Use smaller grid for efficiency
        small_grid = {k: v[:2] for k, v in param_grid.items()}
        
        base_model = self._create_model()
        grid_search = GridSearchCV(
            base_model,
            small_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        self.model = grid_search.best_estimator_
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": pd.DataFrame(grid_search.cv_results_).to_dict()
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "classes_": self.classes_,
            "feature_importance_": self.feature_importance_,
            "model_type": self.model_type
        }
        joblib.dump(model_data, path)
        log.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "CropRecommenderML":
        """Load model from disk."""
        model_data = joblib.load(path)
        
        instance = cls(model_type=model_data["model_type"])
        instance.model = model_data["model"]
        instance.label_encoder = model_data["label_encoder"]
        instance.scaler = model_data["scaler"]
        instance.feature_cols = model_data["feature_cols"]
        instance.classes_ = model_data["classes_"]
        instance.feature_importance_ = model_data["feature_importance_"]
        
        return instance


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive classification evaluation.
    
    Metrics:
    - Accuracy
    - Precision/Recall/F1 (macro & weighted)
    - Top-N accuracy (if probabilities provided)
    - Confusion matrix
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-N accuracy
    if y_proba is not None:
        for n in [2, 3, 5]:
            if y_proba.shape[1] >= n:
                # Need to encode true labels if they're strings
                if isinstance(y_true[0], str) and classes is not None:
                    le = LabelEncoder()
                    le.classes_ = np.array(classes)
                    y_true_encoded = le.transform(y_true)
                else:
                    y_true_encoded = y_true
                
                metrics[f"top_{n}_accuracy"] = top_k_accuracy_score(
                    y_true_encoded, y_proba, k=n
                )
    
    # Classification report
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def evaluate_yield_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluate yield prediction (regression)."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def load_training_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training data."""
    df = pd.read_parquet(data_path)
    
    # Filter to rows with crop labels
    df = df[df[TARGET_COL].notna()]
    
    # Add derived features if missing
    if "month" not in df.columns and "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.month
    
    if "temp_mean_c" not in df.columns:
        if "t_mean_c" in df.columns:
            df["temp_mean_c"] = df["t_mean_c"]
        elif "temp_min_c" in df.columns and "temp_max_c" in df.columns:
            df["temp_mean_c"] = (df["temp_min_c"] + df["temp_max_c"]) / 2
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    return X, y


def generate_synthetic_training_data(n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic training data for testing."""
    np.random.seed(RANDOM_STATE)
    
    crops = list(RuleBasedCropRecommender.CROP_RULES.keys())
    
    records = []
    for crop in crops:
        rules = RuleBasedCropRecommender.CROP_RULES[crop]
        n_per_crop = n_samples // len(crops)
        
        for _ in range(n_per_crop):
            # Generate features centered on crop's ideal conditions
            sm = np.random.normal((rules["sm_min"] + rules["sm_max"]) / 2, 10)
            temp = np.random.normal((rules["temp_min"] + rules["temp_max"]) / 2, 5)
            precip = np.random.normal(rules["precip_min"] * 1.5, rules["precip_min"] * 0.5)
            
            # Season-appropriate month
            if rules["season"] == "kharif":
                month = np.random.choice([6, 7, 8, 9])
            else:
                month = np.random.choice([10, 11, 12, 1, 2])
            
            records.append({
                "soil_moisture_pct": np.clip(sm, 0, 100),
                "temp_mean_c": np.clip(temp, 0, 50),
                "temp_min_c": np.clip(temp - 5, 0, 45),
                "temp_max_c": np.clip(temp + 5, 5, 55),
                "precip_mm": max(0, precip),
                "humidity_pct": np.random.uniform(40, 90),
                "wind_speed_ms": np.random.uniform(1, 8),
                "ndvi": np.random.uniform(0.3, 0.8),
                "month": month,
                "lat": np.random.uniform(8, 35),
                "lon": np.random.uniform(68, 97),
                "year": np.random.choice([2018, 2019, 2020, 2021, 2022]),
                "crop_name": crop
            })
    
    df = pd.DataFrame(records)
    X = df.drop(columns=["crop_name"])
    y = df["crop_name"]
    
    return X, y


def run_training(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_types: List[str] = ["rule_based", "random_forest", "lightgbm"]
) -> Dict[str, Any]:
    """
    Run full training pipeline.
    
    Args:
        data_path: Path to training data (Parquet)
        output_dir: Directory to save models and metrics
        model_types: List of model types to train
    
    Returns:
        Training results with metrics for each model
    """
    output_dir = output_dir or str(MODELS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    if data_path and Path(data_path).exists():
        log.info(f"Loading training data from {data_path}")
        X, y = load_training_data(data_path)
    else:
        log.info("Generating synthetic training data")
        X, y = generate_synthetic_training_data(10000)
    
    log.info(f"Training data: {len(X)} samples, {len(y.unique())} classes")
    
    # Train-test split by year
    if "year" in X.columns:
        test_year = X["year"].max()
        train_mask = X["year"] < test_year
        test_mask = X["year"] == test_year
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            range(len(X)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        train_mask = pd.Series([i in train_idx for i in range(len(X))])
        test_mask = ~train_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    for model_type in model_types:
        log.info(f"\n{'='*50}")
        log.info(f"Training: {model_type}")
        log.info(f"{'='*50}")
        
        if model_type == "rule_based":
            model = RuleBasedCropRecommender()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            cv_metrics = {"cv_accuracy_mean": None, "cv_accuracy_std": None}
        else:
            if model_type == "lightgbm" and not HAS_LIGHTGBM:
                log.warning("LightGBM not available, skipping")
                continue
            
            model = CropRecommenderML(model_type=model_type)
            cv_metrics = model.fit(X_train, y_train, cv_folds=5)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        
        # Evaluate
        eval_metrics = evaluate_classification(
            y_test.values, y_pred, y_proba,
            classes=model.classes_ if hasattr(model, 'classes_') else CROPS
        )
        
        # Combine metrics
        results[model_type] = {
            **cv_metrics,
            "test_accuracy": eval_metrics["accuracy"],
            "test_f1_macro": eval_metrics["f1_macro"],
            "top_3_accuracy": eval_metrics.get("top_3_accuracy", None),
            "feature_importance": model.feature_importance_ if hasattr(model, 'feature_importance_') else None
        }
        
        log.info(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")
        log.info(f"Test F1 (macro): {eval_metrics['f1_macro']:.4f}")
        if "top_3_accuracy" in eval_metrics:
            log.info(f"Top-3 Accuracy: {eval_metrics['top_3_accuracy']:.4f}")
        
        # Save model
        if model_type != "rule_based":
            model_path = Path(output_dir) / f"{model_type}_model.joblib"
            model.save(str(model_path))
    
    # Save metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for model_type, metrics in results.items():
            json_results[model_type] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if v is not None
            }
        json.dump(json_results, f, indent=2, default=str)
    
    log.info(f"\nMetrics saved to {metrics_path}")
    
    # Compare models
    log.info("\n" + "="*50)
    log.info("MODEL COMPARISON")
    log.info("="*50)
    comparison_df = pd.DataFrame([
        {
            "Model": model_type,
            "CV Accuracy": results[model_type].get("cv_accuracy_mean", "-"),
            "Test Accuracy": results[model_type]["test_accuracy"],
            "F1 Macro": results[model_type]["test_f1_macro"],
            "Top-3 Acc": results[model_type].get("top_3_accuracy", "-")
        }
        for model_type in results
    ])
    log.info("\n" + comparison_df.to_string(index=False))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train crop recommendation models")
    parser.add_argument("--data", type=str, help="Path to training data (Parquet)")
    parser.add_argument("--output", type=str, default="./models", help="Output directory")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["rule_based", "random_forest"],
                       help="Model types to train")
    
    args = parser.parse_args()
    
    results = run_training(
        data_path=args.data,
        output_dir=args.output,
        model_types=args.models
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
