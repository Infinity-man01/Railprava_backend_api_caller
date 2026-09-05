import pickle
import json

encoder_path = r'c:\Users\panda\OneDrive\Desktop\RailPrava\Railway_priority_engine\model_artifacts\model_artifacts\encoders.pkl'
classifier_path = r'c:\Users\panda\OneDrive\Desktop\RailPrava\Railway_priority_engine\model_artifacts\model_artifacts\failure_classifier.json'

with open(encoder_path, 'rb') as f:
    encoders = pickle.load(f)
print("Encoders found:", list(encoders.keys()))

with open(classifier_path, 'r') as f:
    classifier = json.load(f)
    print("Features expected in classifier:")
    try:
        # XGBoost models saved as json typically store feature names in 'learner' -> 'feature_names'
        print(classifier.get('learner', {}).get('feature_names', []))
    except Exception as e:
        print("Could not extract feature names", e)
