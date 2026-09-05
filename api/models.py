from pydantic import BaseModel, Field
from typing import List, Optional

class AssetFeatures(BaseModel):
    # Categorical Features
    asset_id: str = Field(default="unknown", description="Unique identifier for the asset")
    asset_type: str = Field(..., description="Type of the asset (e.g., Track, Bridge, Signal)")
    section_type: str = Field(..., description="Section type (e.g., Main_Line, Branch_Line)")
    zone: str = Field(..., description="Railway zone (e.g., Northern, Southern)")

    # Continuous Features
    age_years: float
    last_inspection_days_ago: float
    overdue_ratio: float
    traffic_density_trains_per_day: float
    max_speed_kmph: float
    load_tonnage_daily: float
    weather_exposure_index: float
    temperature_extremity_index: float
    gradient_curvature_index: float
    condition_rating: float
    corrosion_index: float
    historical_failures_last_2yrs: float
    avg_repair_time_hours: float
    distance_from_depot_km: float
    redundancy_available: float

class ActionPlan(BaseModel):
    urgency_level: str = Field(..., description="Critical, High, Medium, or Low")
    recommended_action: str = Field(..., description="Specific maintenance action")
    suggested_team: str = Field(..., description="Recommended team to handle the task")

class PriorityResponse(BaseModel):
    asset_id: str
    risk_score: float = Field(..., description="Predicted risk score from regressor")
    failure_probability: float = Field(..., description="Predicted failure probability from classifier")
    priority_score: float = Field(..., description="Calculated priority score (0-100)")
    action_plan: ActionPlan = Field(..., description="Structured AI recommendation")
    top_risk_factors: List[str] = Field(default_factory=list, description="Heuristic explainability insights")

class BatchAssetFeatures(BaseModel):
    assets: List[AssetFeatures]

class BatchPriorityResponse(BaseModel):
    results: List[PriorityResponse]

class RiskResponse(BaseModel):
    risk_score: float

class FailureResponse(BaseModel):
    failure_probability: float
