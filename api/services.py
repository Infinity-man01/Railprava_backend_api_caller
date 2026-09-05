import os
import xgboost as xgb
import pandas as pd
from typing import Dict, Any
from .models import AssetFeatures

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model_artifacts', 'model_artifacts')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'failure_classifier.json')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'risk_regressor.json')

# Load models globally
classifier = xgb.XGBClassifier()
classifier.load_model(CLASSIFIER_PATH)

regressor = xgb.XGBRegressor()
regressor.load_model(REGRESSOR_PATH)

# Feature list required by the model in exact order
EXPECTED_FEATURES = [
    'age_years', 'last_inspection_days_ago', 'overdue_ratio', 'traffic_density_trains_per_day', 
    'max_speed_kmph', 'load_tonnage_daily', 'weather_exposure_index', 'temperature_extremity_index', 
    'gradient_curvature_index', 'condition_rating', 'corrosion_index', 'historical_failures_last_2yrs', 
    'avg_repair_time_hours', 'distance_from_depot_km', 'redundancy_available', 
    'asset_type__Bridge', 'asset_type__Level_Crossing', 'asset_type__OHE', 'asset_type__Points_and_Crossing', 
    'asset_type__Signal', 'asset_type__Track', 
    'section_type__Branch_Line', 'section_type__Main_Line', 'section_type__Siding', 'section_type__Yard', 
    'zone__Central', 'zone__Eastern', 'zone__NorthEastern', 'zone__Northern', 'zone__Southern', 'zone__Western'
]

def preprocess_features(data: AssetFeatures) -> pd.DataFrame:
    """
    Convert Pydantic model into a DataFrame with one-hot encoded categorical variables.
    Matches the EXPECTED_FEATURES order.
    """
    # Initialize dictionary with zeros for all expected features
    feature_dict = {feat: 0.0 for feat in EXPECTED_FEATURES}
    
    # Fill in continuous features
    for field in data.model_fields.keys():
        if field in feature_dict:
            feature_dict[field] = getattr(data, field)
            
    # Set one-hot encoded categorical features to 1
    asset_col = f"asset_type__{data.asset_type}"
    if asset_col in feature_dict:
        feature_dict[asset_col] = 1.0
        
    section_col = f"section_type__{data.section_type}"
    if section_col in feature_dict:
        feature_dict[section_col] = 1.0
        
    zone_col = f"zone__{data.zone}"
    if zone_col in feature_dict:
        feature_dict[zone_col] = 1.0
        
    df = pd.DataFrame([feature_dict])
    # Ensure correct column order
    return df[EXPECTED_FEATURES]

def predict_failure_probability(df: pd.DataFrame) -> float:
    # predict_proba returns array of shape (n_samples, n_classes)
    proba = classifier.predict_proba(df)[0][1] # Probability of positive class (failure)
    return float(proba)

def predict_risk_score(df: pd.DataFrame) -> float:
    risk = regressor.predict(df)[0]
    # Assuming risk is normalized or can be bounded
    return float(max(0.0, risk))

def generate_recommendation(priority: float, fail_prob: float, risk: float) -> str:
    """Generate a human readable recommendation based on scores."""
    if priority >= 80:
        return "CRITICAL: Immediate maintenance required. High risk of failure and severe impact."
    elif priority >= 50:
        return "HIGH: Schedule block at the earliest availability. Asset showing signs of degradation."
    elif priority >= 30:
        return "MEDIUM: Monitor closely. Plan for routine maintenance in the upcoming schedule."
    else:
        return "LOW: Asset is in good condition. Standard periodic inspection applies."

def calculate_priority_pipeline(data: AssetFeatures) -> Dict[str, Any]:
    df = preprocess_features(data)
    
    fail_prob = predict_failure_probability(df)
    risk_score = predict_risk_score(df)
    
    # Simple weighted formula for Priority Score out of 100
    # Weights can be adjusted based on domain knowledge.
    # Using 60% failure probability and 40% risk score
    # Assuming risk score is naturally between 0 and 1, if not, it should be normalized.
    # From metrics, RMSE is ~0.06 on risk, so risk is likely between 0 and 1.
    priority_score = (fail_prob * 60.0) + (risk_score * 40.0)
    # Ensure it's capped at 100
    priority_score = min(100.0, max(0.0, priority_score))
    
    recommendation = generate_recommendation(priority_score, fail_prob, risk_score)
    
    return {
        "risk_score": risk_score,
        "failure_probability": fail_prob,
        "priority_score": priority_score,
        "recommendation": recommendation
    }
