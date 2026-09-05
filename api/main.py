from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import AssetFeatures, PriorityResponse, RiskResponse, FailureResponse, BatchAssetFeatures, BatchPriorityResponse
from .services import (
    calculate_priority_pipeline, 
    batch_calculate_priority_pipeline,
    predict_risk_score_batch, 
    predict_failure_probability_batch,
    preprocess_batch_features
)
from .database import log_prediction, log_batch_predictions, get_history

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
