import requests

url = "http://127.0.0.1:8000/api/v1/priority"
payload = {
  "asset_type": "Track",
  "section_type": "Main_Line",
  "zone": "Northern",
  "age_years": 15.5,
  "last_inspection_days_ago": 120.0,
  "overdue_ratio": 0.2,
  "traffic_density_trains_per_day": 85.0,
  "max_speed_kmph": 130.0,
  "load_tonnage_daily": 50000.0,
  "weather_exposure_index": 0.8,
  "temperature_extremity_index": 0.6,
  "gradient_curvature_index": 0.3,
  "condition_rating": 4.5,
  "corrosion_index": 0.4,
  "historical_failures_last_2yrs": 2.0,
  "avg_repair_time_hours": 6.5,
  "distance_from_depot_km": 45.0,
  "redundancy_available": 1.0
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
