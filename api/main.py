from fastapi.responses import FileResponse
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    AssetFeatures, PriorityResponse, RiskResponse, FailureResponse, 
    BatchAssetFeatures, BatchPriorityResponse, BatchLiveAssetRequest
)
from .services import (
    calculate_priority_pipeline, 
    batch_calculate_priority_pipeline,
    predict_risk_score_batch, 
    predict_failure_probability_batch,
    preprocess_batch_features,
    fetch_live_asset_features,
    fetch_live_assets_batch_features
)
from .database import (
    log_prediction, 
    log_batch_predictions, 
    get_history,
    check_db_health,
    get_asset_by_id,
    get_assets_batch,
    list_assets
)

app = FastAPI(
    title="AI / ML Priority Engine V2",
    description="Hackathon Winning Backend API for predicting asset risk, failure probability and generating maintenance priority scores with XAI and Batch Processing.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Priority Engine API V2"}

@app.get("/api/v1/database/health")
def database_health():
    """Check live PostgreSQL (Supabase) database health."""
    health = check_db_health()
    if health.get("status") != "connected":
        raise HTTPException(status_code=503, detail=health)
    return health

@app.get("/api/v1/assets")
def get_assets_list(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    asset_type: Optional[str] = None,
    zone: Optional[str] = None
):
    """Retrieve paginated assets live from PostgreSQL."""
    try:
        assets = list_assets(limit=limit, offset=offset, asset_type=asset_type, zone=zone)
        return {"count": len(assets), "limit": limit, "offset": offset, "assets": assets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/assets/{asset_id}")
def get_single_asset(asset_id: str):
    """Fetch an asset live from PostgreSQL by its asset_id."""
    try:
        asset = get_asset_by_id(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Asset with ID '{asset_id}' not found in database.")
        return asset
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# LIVE ML INFERENCE PIPELINE (PostgreSQL -> 19 Features -> XGBoost)
# -------------------------------------------------------------------

@app.get("/api/v1/predict/live/{asset_id}", response_model=PriorityResponse)
def get_live_asset_priority(asset_id: str):
    """
    Fetch an asset LIVE from PostgreSQL, extract strictly the 19 input features,
    run live XGBoost risk and failure models, and return priority score + action plan.
    Guarantees ZERO target leakage (risk_score and failure_within_30_days are excluded).
    """
    try:
        features = fetch_live_asset_features(asset_id)
        if not features:
            raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found in database.")
        
        result = calculate_priority_pipeline(features)
        
        # Log to telemetry DB
        log_prediction(
            asset_id=result.asset_id,
            asset_type=features.asset_type,
            priority_score=result.priority_score,
            urgency_level=result.action_plan.urgency_level,
            risk_factors=result.top_risk_factors,
            input_features=features.model_dump()
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/live/batch", response_model=BatchPriorityResponse)
def get_live_assets_priority_batch(payload: BatchLiveAssetRequest):
    """
    Fetch multiple assets LIVE from PostgreSQL, run vectorized XGBoost inference,
    and return priority scores for all requested assets.
    """
    try:
        features_list = fetch_live_assets_batch_features(payload.asset_ids)
        if not features_list:
            raise HTTPException(status_code=404, detail="None of the specified assets were found in the database.")
        
        results = batch_calculate_priority_pipeline(features_list)
        log_batch_predictions(results, features_list)
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# EXISTING DIRECT PAYLOAD ENDPOINTS (100% PRESERVED)
# -------------------------------------------------------------------

@app.post("/api/v1/predict/risk", response_model=RiskResponse)
def get_risk(features: AssetFeatures):
    """Predict continuous risk score for an asset."""
    try:
        df = preprocess_batch_features([features])
        risk = predict_risk_score_batch(df)[0]
        return {"risk_score": risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/failure", response_model=FailureResponse)
def get_failure_probability(features: AssetFeatures):
    """Predict probability of failure for an asset."""
    try:
        df = preprocess_batch_features([features])
        prob = predict_failure_probability_batch(df)[0]
        return {"failure_probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/priority", response_model=PriorityResponse)
def get_priority(features: AssetFeatures):
    """
    Calculate the final priority score combining risk and failure probability,
    get AI recommendations, and heuristic insights.
    """
    try:
        result = calculate_priority_pipeline(features)
        
        # Log to telemetry DB
        log_prediction(
            asset_id=result.asset_id,
            asset_type=features.asset_type,
            priority_score=result.priority_score,
            urgency_level=result.action_plan.urgency_level,
            risk_factors=result.top_risk_factors,
            input_features=features.model_dump()
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/priority/batch", response_model=BatchPriorityResponse)
def get_priority_batch(payload: BatchAssetFeatures):
    """
    Process hundreds of assets at once (Vectorized processing).
    Critical for Optimization Engine.
    """
    try:
        results = batch_calculate_priority_pipeline(payload.assets)
        
        # Log batch to DB
        log_batch_predictions(results, payload.assets)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
def get_prediction_history(limit: int = 50):
    """
    Fetch the telemetry history of predictions.
    """
    try:
        return {"history": get_history(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the interactive Railprava web dashboard."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(base_dir, "frontend_tester.html"))
