import os
import xgboost as xgb
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .models import AssetFeatures, ActionPlan, PriorityResponse

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

# Feature list required by the model in exact order (31 features)
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

# Strictly the 19 live model input features
INPUT_FEATURE_KEYS = [
    'asset_id', 'asset_type', 'section_type', 'zone', 'age_years',
    'last_inspection_days_ago', 'overdue_ratio', 'traffic_density_trains_per_day',
    'max_speed_kmph', 'load_tonnage_daily', 'weather_exposure_index',
    'temperature_extremity_index', 'gradient_curvature_index', 'condition_rating',
    'corrosion_index', 'historical_failures_last_2yrs', 'avg_repair_time_hours',
    'distance_from_depot_km', 'redundancy_available'
]

def db_record_to_asset_features(record: Dict[str, Any]) -> AssetFeatures:
    """
    Safely extract strictly the 19 live model input features from a database record.
    Explicitly filters out ground-truth targets (risk_score, failure_within_30_days)
    and metadata (created_at) to guarantee ZERO target leakage.
    """
    features = {k: record[k] for k in INPUT_FEATURE_KEYS if k in record}
    # Ensure boolean redundancy_available is converted to float (1.0 or 0.0) for model input
    if "redundancy_available" in features:
        features["redundancy_available"] = 1.0 if features["redundancy_available"] else 0.0
    return AssetFeatures(**features)

def fetch_live_asset_features(asset_id: str) -> Optional[AssetFeatures]:
    """Fetch an asset from PostgreSQL by ID and return validated AssetFeatures (19 inputs)."""
    from .database import get_asset_by_id
    record = get_asset_by_id(asset_id)
    if not record:
        return None
    return db_record_to_asset_features(record)

def fetch_live_assets_batch_features(asset_ids: List[str]) -> List[AssetFeatures]:
    """Fetch multiple assets from PostgreSQL by IDs and return validated AssetFeatures list."""
    from .database import get_assets_batch
    records = get_assets_batch(asset_ids)
    return [db_record_to_asset_features(r) for r in records]

def preprocess_features(data: AssetFeatures) -> pd.DataFrame:
    """
    Convert Pydantic model into a DataFrame with one-hot encoded categorical variables.
    """
    return preprocess_batch_features([data])

def preprocess_batch_features(data_list: List[AssetFeatures]) -> pd.DataFrame:
    """
    Convert a list of Pydantic models into a DataFrame with 31 encoded features.
    """
    dict_list = []
    for data in data_list:
        feature_dict = {feat: 0.0 for feat in EXPECTED_FEATURES}
        for field in data.model_fields.keys():
            if field in feature_dict:
                feature_dict[field] = getattr(data, field)
        
        # Categorical one-hot encoding
        asset_col = f"asset_type__{data.asset_type}"
        if asset_col in feature_dict: feature_dict[asset_col] = 1.0
            
        section_col = f"section_type__{data.section_type}"
        if section_col in feature_dict: feature_dict[section_col] = 1.0
            
        zone_col = f"zone__{data.zone}"
        if zone_col in feature_dict: feature_dict[zone_col] = 1.0
            
        dict_list.append(feature_dict)
        
    df = pd.DataFrame(dict_list)
    return df[EXPECTED_FEATURES]

def predict_failure_probability_batch(df: pd.DataFrame) -> List[float]:
    probas = classifier.predict_proba(df)[:, 1]
    return [float(p) for p in probas]

def predict_risk_score_batch(df: pd.DataFrame) -> List[float]:
    risks = regressor.predict(df)
    return [float(max(0.0, r)) for r in risks]

def generate_action_plan(priority: float, asset_type: str) -> ActionPlan:
    """Generate structured AI recommendations."""
    if priority >= 80:
        urgency = "Critical"
        if asset_type == "Track":
            action = "Immediate Ultrasonic Flaw Detection and Track Replacement."
            team = "Emergency Track Crew"
        elif asset_type == "Bridge":
            action = "Urgent Structural Integrity Check and Load Restriction."
            team = "Bridge Engineering Unit"
        else:
            action = "Immediate Replacement or Overhaul."
            team = "Specialized Maintenance Unit"
    elif priority >= 50:
        urgency = "High"
        action = "Schedule maintenance block in the upcoming week for detailed inspection."
        team = "Routine Maintenance Crew"
    elif priority >= 30:
        urgency = "Medium"
        action = "Monitor condition. Add to routine monthly check."
        team = "Local Inspection Team"
    else:
        urgency = "Low"
        action = "No immediate action required. Standard periodic inspection."
        team = "Standard Patrol"
        
    return ActionPlan(
        urgency_level=urgency,
        recommended_action=action,
        suggested_team=team
    )

def get_top_risk_factors(data: AssetFeatures) -> List[str]:
    """Heuristic logic to explain the AI model's decision."""
    factors = []
    if data.overdue_ratio > 1.0:
        factors.append(f"Severely overdue for inspection (Ratio: {data.overdue_ratio})")
    elif data.last_inspection_days_ago > 180:
        factors.append("No inspection in the last 6 months")
        
    if data.condition_rating >= 4.0:
        factors.append("Poor visual condition rating")
        
    if data.historical_failures_last_2yrs >= 3:
        factors.append("High history of recent failures")
        
    if data.age_years > 30 and data.asset_type in ["Bridge", "Track"]:
        factors.append("Asset has exceeded standard lifecycle age")
        
    if data.corrosion_index > 0.7:
        factors.append("Critical corrosion levels detected")
        
    if data.weather_exposure_index > 0.8:
        factors.append("High environmental stress exposure")
        
    if not factors:
        factors.append("No severe anomalies detected.")
        
    return factors

def calculate_priority_pipeline(data: AssetFeatures) -> PriorityResponse:
    results = batch_calculate_priority_pipeline([data])
    return results[0]

def batch_calculate_priority_pipeline(data_list: List[AssetFeatures]) -> List[PriorityResponse]:
    if not data_list:
        return []
        
    df = preprocess_batch_features(data_list)
    
    fail_probs = predict_failure_probability_batch(df)
    risk_scores = predict_risk_score_batch(df)
    
    responses = []
    for i, data in enumerate(data_list):
        f_prob = fail_probs[i]
        r_score = risk_scores[i]
        
        # Formula: 60% failure, 40% risk
        p_score = (f_prob * 60.0) + (r_score * 40.0)
        p_score = min(100.0, max(0.0, p_score))
        
        action_plan = generate_action_plan(p_score, data.asset_type)
        risk_factors = get_top_risk_factors(data)
        
        resp = PriorityResponse(
            asset_id=data.asset_id,
            risk_score=r_score,
            failure_probability=f_prob,
            priority_score=p_score,
            action_plan=action_plan,
            top_risk_factors=risk_factors
        )
        responses.append(resp)
        
    return responses
