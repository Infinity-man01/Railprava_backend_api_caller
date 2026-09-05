from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import AssetFeatures, PriorityResponse, RiskResponse, FailureResponse
from .services import (
    calculate_priority_pipeline, 
    predict_risk_score, 
    predict_failure_probability,
    preprocess_features
)

app = FastAPI(
    title="AI / ML Priority Engine",
    description="Backend API for predicting asset risk, failure probability and generating maintenance priority scores.",
    version="1.0.0"
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
    return {"status": "healthy", "service": "Priority Engine API"}

@app.post("/api/v1/predict/risk", response_model=RiskResponse)
def get_risk(features: AssetFeatures):
    """Predict continuous risk score for an asset."""
    try:
        df = preprocess_features(features)
        risk = predict_risk_score(df)
        return {"risk_score": risk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/failure", response_model=FailureResponse)
def get_failure_probability(features: AssetFeatures):
    """Predict probability of failure for an asset."""
    try:
        df = preprocess_features(features)
        prob = predict_failure_probability(df)
        return {"failure_probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/priority", response_model=PriorityResponse)
def get_priority(features: AssetFeatures):
    """
    Calculate the final priority score combining risk and failure probability,
    and get AI recommendations.
    """
    try:
        result = calculate_priority_pipeline(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
