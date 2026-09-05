import requests
import json

print("--- Testing Batch Endpoint ---")
batch_url = "http://127.0.0.1:8000/api/v1/priority/batch"
batch_payload = {
    "assets": [
        {
            "asset_id": "TRK-001",
            "asset_type": "Track",
            "section_type": "Main_Line",
            "zone": "Northern",
            "age_years": 1.0,
            "last_inspection_days_ago": 5.0,
            "overdue_ratio": 0.0,
            "traffic_density_trains_per_day": 20.0,
            "max_speed_kmph": 100.0,
            "load_tonnage_daily": 10000.0,
            "weather_exposure_index": 0.2,
            "temperature_extremity_index": 0.1,
            "gradient_curvature_index": 0.1,
            "condition_rating": 1.0,
            "corrosion_index": 0.1,
            "historical_failures_last_2yrs": 0.0,
            "avg_repair_time_hours": 1.0,
            "distance_from_depot_km": 10.0,
            "redundancy_available": 1.0
        },
        {
            "asset_id": "BRG-099",
            "asset_type": "Bridge",
            "section_type": "Main_Line",
            "zone": "Northern",
            "age_years": 80.0,
            "last_inspection_days_ago": 400.0,
            "overdue_ratio": 2.5,
            "traffic_density_trains_per_day": 150.0,
            "max_speed_kmph": 130.0,
            "load_tonnage_daily": 80000.0,
            "weather_exposure_index": 0.9,
            "temperature_extremity_index": 0.8,
            "gradient_curvature_index": 0.6,
            "condition_rating": 5.0,
            "corrosion_index": 0.9,
            "historical_failures_last_2yrs": 5.0,
            "avg_repair_time_hours": 24.0,
            "distance_from_depot_km": 150.0,
            "redundancy_available": 0.0
        }
    ]
}

r_batch = requests.post(batch_url, json=batch_payload)
print("Batch Response Code:", r_batch.status_code)
print("Batch Response:")
print(json.dumps(r_batch.json(), indent=2))

print("\n--- Testing History Endpoint ---")
history_url = "http://127.0.0.1:8000/api/v1/history"
r_hist = requests.get(history_url)
print("History Response Code:", r_hist.status_code)
print("History Response:")
print(json.dumps(r_hist.json(), indent=2))
