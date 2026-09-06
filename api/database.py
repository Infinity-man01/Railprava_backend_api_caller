import sqlite3
import os
import json
import time
import re
import urllib.parse
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

# -------------------------------------------------------------
# 1. PostgreSQL (Supabase) Connection Layer
# -------------------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, '.env'))
DATABASE_URL = os.getenv("DATABASE_URL")

def sanitize_db_url(url: str) -> str:
    """Safely URL-encode passwords containing special characters (like '@' or '!') in PostgreSQL connection strings."""
    if not url:
        return url
    url = url.strip()
    pattern = r'^(postgres(?:ql)?:\/\/)([^:]+):(.*)@([^@]+)$'
    m = re.match(pattern, url)
    if m:
        scheme, user, password, rest = m.groups()
        password = urllib.parse.unquote(password.strip())
        encoded_password = urllib.parse.quote(password, safe='')
        return f"{scheme}{user.strip()}:{encoded_password}@{rest.strip()}".strip()
    return url.strip()

_pg_pool: Optional[pool.ThreadedConnectionPool] = None

def get_pg_pool() -> pool.ThreadedConnectionPool:
    """Initialize or return the global ThreadedConnectionPool."""
    global _pg_pool
    if _pg_pool is None or _pg_pool.closed:
        url = (os.getenv("DATABASE_URL") or DATABASE_URL or "").strip()
        if not url:
            raise ValueError("DATABASE_URL environment variable is not configured.")
        _pg_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=sanitize_db_url(url)
        )
    return _pg_pool

@contextmanager
def get_db_connection():
    """Context manager for acquiring and releasing pooled PostgreSQL connections."""
    p = get_pg_pool()
    conn = p.getconn()
    try:
        yield conn
    finally:
        p.putconn(conn)

def check_db_health() -> Dict[str, Any]:
    """Check live database connectivity and latency."""
    try:
        t0 = time.time()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
                cur.execute("SELECT COUNT(*) FROM railway_assets;")
                asset_count = cur.fetchone()[0]
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "status": "connected",
            "latency_ms": latency_ms,
            "asset_count": asset_count,
            "engine": "PostgreSQL (Supabase)"
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

def get_asset_by_id(asset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an asset by its asset_id from PostgreSQL."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    asset_id, asset_type, section_type, zone, age_years,
                    last_inspection_days_ago, overdue_ratio, traffic_density_trains_per_day,
                    max_speed_kmph, load_tonnage_daily, weather_exposure_index,
                    temperature_extremity_index, gradient_curvature_index, condition_rating,
                    corrosion_index, historical_failures_last_2yrs, avg_repair_time_hours,
                    redundancy_available, distance_from_depot_km, risk_score,
                    failure_within_30_days, created_at
                FROM railway_assets
                WHERE asset_id = %s;
            """, (asset_id,))
            row = cur.fetchone()
            return dict(row) if row else None

def get_assets_batch(asset_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple assets by IDs in a single query."""
    if not asset_ids:
        return []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    asset_id, asset_type, section_type, zone, age_years,
                    last_inspection_days_ago, overdue_ratio, traffic_density_trains_per_day,
                    max_speed_kmph, load_tonnage_daily, weather_exposure_index,
                    temperature_extremity_index, gradient_curvature_index, condition_rating,
                    corrosion_index, historical_failures_last_2yrs, avg_repair_time_hours,
                    redundancy_available, distance_from_depot_km, risk_score,
                    failure_within_30_days, created_at
                FROM railway_assets
                WHERE asset_id = ANY(%s);
            """, (asset_ids,))
            return [dict(r) for r in cur.fetchall()]

def list_assets(
    limit: int = 50, 
    offset: int = 0, 
    asset_type: Optional[str] = None, 
    zone: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List assets from PostgreSQL with optional filtering."""
    query = """
        SELECT 
            asset_id, asset_type, section_type, zone, age_years,
            last_inspection_days_ago, overdue_ratio, traffic_density_trains_per_day,
            max_speed_kmph, load_tonnage_daily, weather_exposure_index,
            temperature_extremity_index, gradient_curvature_index, condition_rating,
            corrosion_index, historical_failures_last_2yrs, avg_repair_time_hours,
            redundancy_available, distance_from_depot_km, risk_score,
            failure_within_30_days, created_at
        FROM railway_assets
        WHERE 1=1
    """
    params = []
    if asset_type:
        query += " AND asset_type = %s"
        params.append(asset_type)
    if zone:
        query += " AND zone = %s"
        params.append(zone)
    query += " ORDER BY asset_id ASC LIMIT %s OFFSET %s;"
    params.extend([limit, offset])
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            return [dict(r) for r in cur.fetchall()]

# -------------------------------------------------------------
# 2. SQLite Telemetry Logging (Preserved for prediction telemetry)
# -------------------------------------------------------------
def init_db():
    """Initialize the SQLite database for telemetry logging."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    conn.commit()
    conn.close()

def log_prediction(asset_id: str, asset_type: str, priority_score: float, urgency_level: str, risk_factors: List[str], input_features: Dict[str, Any]):
    """Log a single prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (asset_id, asset_type, priority_score, urgency_level, risk_factors, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
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
        
    cursor.executemany("""
        INSERT INTO predictions (asset_id, asset_type, priority_score, urgency_level, risk_factors, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
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

# Initialize SQLite on import
init_db()
