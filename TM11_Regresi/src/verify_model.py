import joblib
import pandas as pd
import json

# 1. Load Model & Metadata
print("--- LOADING MODEL & METADATA ---")
model = joblib.load("models/best_model.joblib")
with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Best Model Loaded: {metadata['best_model_name']}")

# 2. Prepare Sample Input
# Using the same columns as defined in metadata
input_data = pd.DataFrame([{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
}])

# 3. Predict
print("\n--- RUNNING PREDICTION ---")
prediction = model.predict(input_data)
print(f"Input Data:\n{input_data.iloc[0]}")
print(f"\nPredicted House Value: ${prediction[0]:,.2f}")

# 4. Verify with X_test samples
print("\n--- VERIFYING WITH X_TEST SAMPLES ---")
X_test = pd.read_csv("hasil_orang1_final/X_test.csv", index_col=0)
samples = X_test.head(3)
preds = model.predict(samples)
print(f"Predictions for first 3 rows of X_test: {preds}")
