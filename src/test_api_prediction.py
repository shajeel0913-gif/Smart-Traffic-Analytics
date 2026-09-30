import numpy as np
import requests

# Load one real test sequence
X_test = np.load("data/processed/X_test.npy")

sequence = X_test[0].tolist()

payload = {
    "sequence": sequence
}

print("=" * 60)
print("TESTING SMART TRAFFIC ANALYTICS API")
print("=" * 60)

print("\nInput sequence shape:")
print(np.asarray(sequence).shape)

response = requests.post(
    "http://127.0.0.1:8001/predict",
    json=payload
)

print("\nHTTP status:", response.status_code)

print("\nAPI response:")

data = response.json()

print("Model:", data.get("model"))
print("Sequence length:", data.get("sequence_length"))
print("Prediction horizon:", data.get("prediction_horizon"))
print("Number of sensors:", data.get("number_of_sensors"))

predictions = data.get("predictions", {})

print("\nNumber of predictions:", len(predictions))

print("\nFirst 10 predictions:")

for sensor, value in list(predictions.items())[:10]:
    print(f"Sensor {sensor}: {value:.2f}")

print("\n" + "=" * 60)
print("API PREDICTION TEST COMPLETE")
print("=" * 60)
