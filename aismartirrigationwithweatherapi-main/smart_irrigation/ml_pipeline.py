"""ML model pipeline: rules baseline, decision tree, ensemble models."""
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from .config import MODEL, MODELS_DIR, METRICS, CROP_PARAMS
from .features import get_feature_columns

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedModel:
    """Interpretable rule-based irrigation decision model."""
    
    def __init__(self, crop: str = "wheat"):
        self.crop = crop
        self.params = CROP_PARAMS.get(crop, CROP_PARAMS["wheat"])
        self.thresholds = {
            "vwc_critical": self.params["vwc_critical"],
            "vwc_optimal": self.params["vwc_optimal"],
            "rain_block_mm": 5.0,
            "et_high_mm": 4.0,
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Apply rules to predict irrigation need."""
        predictions = []
        
        for _, row in X.iterrows():
            pred = self._apply_rules(row)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _apply_rules(self, row: pd.Series) -> int:
        """Apply decision rules. Returns 1=irrigate, 0=skip."""
        sm = row.get("soil_moisture", row.get("sm_mean_3d", 0.25))
        rain_7d = row.get("rain_7d", row.get("rain_sum", 0))
        et = row.get("et0_hargreaves", row.get("et0_proxy", 3))
        deficit = row.get("cum_deficit_7d", 0)
        critical = row.get("critical_window", 0)
        
        # Rule 1: Critical moisture
        if sm < self.thresholds["vwc_critical"]:
            return 1
        
        # Rule 2: Rain blocking
        if rain_7d > self.thresholds["rain_block_mm"] * 2:
            return 0
        
        # Rule 3: Critical growth stage + stress
        if critical and sm < self.thresholds["vwc_optimal"]:
            return 1
        
        # Rule 4: High ET demand
        if et > self.thresholds["et_high_mm"] and deficit < -10:
            return 1
        
        # Rule 5: Moderate stress
        if sm < self.thresholds["vwc_optimal"] * 0.9 and deficit < -15:
            return 1
        
        return 0
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return pseudo-probabilities based on confidence."""
        preds = self.predict(X)
        # Simple confidence: further from threshold = higher confidence
        proba = np.zeros((len(X), 2))
        
        for i, (_, row) in enumerate(X.iterrows()):
            sm = row.get("soil_moisture", 0.25)
            if preds[i] == 1:
                conf = min(1.0, (self.thresholds["vwc_optimal"] - sm) / 0.1 + 0.5)
            else:
                conf = min(1.0, (sm - self.thresholds["vwc_optimal"]) / 0.1 + 0.5)
            proba[i, preds[i]] = conf
            proba[i, 1 - preds[i]] = 1 - conf
        
        return proba
    
    def explain(self, row: pd.Series) -> str:
        """Explain prediction for a single row."""
        pred = self._apply_rules(row)
        sm = row.get("soil_moisture", 0.25)
        rain = row.get("rain_7d", 0)
        
        if pred == 1:
            if sm < self.thresholds["vwc_critical"]:
                return f"IRRIGATE: Soil moisture ({sm:.0%}) below critical ({self.thresholds['vwc_critical']:.0%})"
            return f"IRRIGATE: Water deficit detected (moisture: {sm:.0%})"
        else:
            if rain > 10:
                return f"SKIP: Recent rain ({rain:.0f}mm) sufficient"
            return f"SKIP: Moisture adequate ({sm:.0%})"


# ─────────────────────────────────────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    model: Any
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None


def train_decision_tree(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    max_depth: int = 8
) -> ModelResult:
    """Train interpretable decision tree."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=MODEL.random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    importance = dict(zip(X_train.columns, model.feature_importances_))
    
    return ModelResult("decision_tree", model, metrics, importance)


def train_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    n_estimators: int = 100
) -> ModelResult:
    """Train random forest ensemble."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=MODEL.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    importance = dict(zip(X_train.columns, model.feature_importances_))
    
    return ModelResult("random_forest", model, metrics, importance)


def train_gradient_boosting(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    n_estimators: int = 100
) -> ModelResult:
    """Train gradient boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=MODEL.random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    importance = dict(zip(X_train.columns, model.feature_importances_))
    
    return ModelResult("gradient_boosting", model, metrics, importance)


def train_xgboost(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    n_estimators: int = 100
) -> ModelResult:
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("XGBoost not available, using GradientBoosting")
        return train_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=MODEL.random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    importance = dict(zip(X_train.columns, model.feature_importances_))
    
    return ModelResult("xgboost", model, metrics, importance)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def cross_validate_model(
    model, X: pd.DataFrame, y: np.ndarray, cv: int = 5
) -> Dict[str, float]:
    """Perform stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=MODEL.random_state)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")
    
    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "cv_scores": scores.tolist()
    }


def print_report(result: ModelResult, X_test: pd.DataFrame, y_test: np.ndarray):
    """Print detailed evaluation report."""
    print(f"\n{'='*60}")
    print(f"Model: {result.name}")
    print(f"{'='*60}")
    
    print("\nMetrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")
    
    if result.feature_importance:
        print("\nTop 10 Features:")
        sorted_imp = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10]
        for feat, imp in sorted_imp:
            print(f"  {feat}: {imp:.4f}")
    
    y_pred = result.model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Skip", "Irrigate"]))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class ModelPipeline:
    """End-to-end model training pipeline."""
    
    def __init__(self, crop: str = "wheat"):
        self.crop = crop
        self.results: Dict[str, ModelResult] = {}
        self.best_model: Optional[ModelResult] = None
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "irrigation_needed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare train/test split."""
        
        # Create target if not exists
        if target_col not in df.columns:
            # Derive from soil moisture
            if "soil_moisture" in df.columns:
                vwc_crit = CROP_PARAMS.get(self.crop, {}).get("vwc_critical", 0.20)
                df[target_col] = (df["soil_moisture"] < vwc_crit + 0.05).astype(int)
            else:
                raise ValueError(f"No target column '{target_col}' and can't derive from data")
        
        # Get feature columns
        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if c != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL.test_size, 
            random_state=MODEL.random_state, stratify=y
        )
        
        log.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
        log.info(f"Target distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_all(self, df: pd.DataFrame, target_col: str = "irrigation_needed"):
        """Train all model types."""
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Rule-based baseline
        log.info("Training rule-based baseline...")
        rule_model = RuleBasedModel(self.crop)
        y_pred_rules = rule_model.predict(X_test)
        rule_metrics = compute_metrics(y_test, y_pred_rules)
        self.results["rules"] = ModelResult("rules", rule_model, rule_metrics)
        
        # Decision Tree
        log.info("Training decision tree...")
        self.results["decision_tree"] = train_decision_tree(X_train, y_train, X_test, y_test)
        
        # Random Forest
        log.info("Training random forest...")
        self.results["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Gradient Boosting / XGBoost
        log.info("Training gradient boosting...")
        if MODEL.ensemble == "xgboost":
            self.results["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test)
        else:
            self.results["gradient_boosting"] = train_gradient_boosting(
                X_train, y_train, X_test, y_test
            )
        
        # Find best model
        primary_metric = METRICS.primary
        best_name = max(self.results, key=lambda k: self.results[k].metrics.get(primary_metric, 0))
        self.best_model = self.results[best_name]
        
        log.info(f"Best model: {best_name} ({primary_metric}: {self.best_model.metrics[primary_metric]:.4f})")
        
        return self.results
    
    def save_model(self, name: str = "best"):
        """Save model to disk."""
        if name == "best" and self.best_model:
            result = self.best_model
        elif name in self.results:
            result = self.results[name]
        else:
            raise ValueError(f"Model '{name}' not found")
        
        path = MODELS_DIR / f"{result.name}_{self.crop}.pkl"
        with open(path, "wb") as f:
            pickle.dump(result, f)
        
        log.info(f"Saved model to {path}")
        return path
    
    def load_model(self, path: str) -> ModelResult:
        """Load model from disk."""
        with open(path, "rb") as f:
            result = pickle.load(f)
        log.info(f"Loaded model: {result.name}")
        return result
    
    def compare_models(self):
        """Print comparison of all trained models."""
        print(f"\n{'Model Comparison':=^60}")
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        
        for name, result in sorted(self.results.items(), 
                                   key=lambda x: -x[1].metrics.get("f1_weighted", 0)):
            m = result.metrics
            print(f"{name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        print("-" * 60)
        if self.best_model:
            print(f"Best: {self.best_model.name}")
