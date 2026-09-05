import requests
import json

url = "https://priority-engine-api.onrender.com/api/v1/priority"

print("--- Testing Model Behavior ---\n")

# Scenario 1: Perfect Condition
payload_good = {
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
}

r1 = requests.post(url, json=payload_good)
print("Scenario 1 (New Track, Good Condition):")
print(json.dumps(r1.json(), indent=2))
print()

# Scenario 2: Terrible Condition
payload_bad = {
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

r2 = requests.post(url, json=payload_bad)
print("Scenario 2 (Old Bridge, Terrible Condition, Overdue):")
print(json.dumps(r2.json(), indent=2))
