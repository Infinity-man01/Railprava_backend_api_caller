# AI / ML Priority Engine

This repository contains **Module 3: AI / ML Priority Engine** for the **Automatic Railway Block Planning System (SIH 2026)**.

## 🏗️ System Architecture & Workflow

This microservice acts as the critical decision-making brain in the 6-module architecture:

1. **Member 2 (Data Integration & DB):** Collects raw asset data from TMS, SMMS, TDMS, etc.
2. **👉 Member 3 (AI Priority Engine) [This Repository]:** Ingests asset data, predicts risk/failure probabilities using ML (XGBoost), and generates a unified Priority Score and Action Plan.
3. **Member 4 (Block Optimization Engine):** Consumes the batch priority scores from this API to intelligently schedule maintenance blocks and detect corridor conflicts.
4. **Member 5 & 6 (Frontend & Backend):** Displays these recommendations and priority scores on the user dashboard.

### Core Deliverables Achieved
* **Priority Scoring & Risk Prediction:** Employs XGBoost classifiers and regressors to evaluate asset criticality, age, weather exposure, and historical failures.
* **AI Recommendation Engine:** Generates specific, actionable insights (e.g., "Urgent Structural Integrity Check") based on heuristic risk factors.
* **Batch Processing API:** Fully vectorized endpoints designed to process hundreds of assets simultaneously, enabling the Block Optimization Engine to run its constraint-solving algorithms efficiently.

---

## 🚀 Tech Stack

* **Framework:** FastAPI (Python 3.11)
* **Machine Learning:** XGBoost, Scikit-learn, Pandas
* **Database (Telemetry):** SQLite (`predictions.db` for logging AI decisions)
* **Deployment:** Docker, Render (`render.yaml` configured for automated deployment)

---

## 📡 API Endpoints

The API is deployed and accessible at: `https://priority-engine-api.onrender.com`

### 1. Health Check
`GET /api/v1/health`
Validates that the API is running and serving requests.

### 2. Single Asset Priority
`POST /api/v1/priority`
Evaluates a single asset and returns its failure probability, risk score, and an AI-generated Action Plan.

### 3. Batch Asset Processing (For Optimization Engine)
`POST /api/v1/priority/batch`
**Critical for Member 4.** Accepts a payload of multiple assets and efficiently processes them in bulk, returning priority scores for all assets at once.

### 4. Telemetry History
`GET /api/v1/history?limit=50`
Returns a history of recent AI predictions and recommendations from the SQLite database.

---

## 💻 Running Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Test the Frontend UI:**
   Open `frontend_tester.html` in your browser to interactively test the live or local API.

---

## ☁️ Deployment

This service is configured for zero-downtime automated deployment on **Render** using Docker.
* **Configuration:** `render.yaml`
* **Environment:** `python:3.11-slim` container
* Pushing to the `main` branch automatically triggers a deployment (assuming Render auto-deploy is enabled).
