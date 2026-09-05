import sqlite3
import os
import json
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

def init_db():
    """Initialize the SQLite database for telemetry logging."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            asset_id TEXT,
            asset_type TEXT,
            priority_score REAL,
            urgency_level TEXT,
            risk_factors TEXT,
            input_features TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(asset_id: str, asset_type: str, priority_score: float, urgency_level: str, risk_factors: List[str], input_features: Dict[str, Any]):
    """Log a single prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (asset_id, asset_type, priority_score, urgency_level, risk_factors, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        asset_id,
        asset_type,
        priority_score,
        urgency_level,
        json.dumps(risk_factors),
        json.dumps(input_features)
    ))
    conn.commit()
    conn.close()

def log_batch_predictions(results: List[Any], inputs: List[Any]):
    """Log a batch of predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    data = []
    for i, res in enumerate(results):
        inp = inputs[i]
        data.append((
            res.asset_id,
            inp.asset_type,
            res.priority_score,
            res.action_plan.urgency_level,
            json.dumps(res.top_risk_factors),
            inp.model_dump_json()
        ))
        
    cursor.executemany('''
        INSERT INTO predictions (asset_id, asset_type, priority_score, urgency_level, risk_factors, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve prediction history."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "asset_id": r["asset_id"],
            "asset_type": r["asset_type"],
            "priority_score": r["priority_score"],
            "urgency_level": r["urgency_level"],
            "risk_factors": json.loads(r["risk_factors"]),
            "input_features": json.loads(r["input_features"])
        })
    return history

# Initialize on import
init_db()
