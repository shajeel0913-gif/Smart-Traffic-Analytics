import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("LSTM ACTUAL VS PREDICTED")
print("=" * 60)

# Load data
y_true = np.load("results/lstm_actual.npy")
y_pred = np.load("results/lstm_predictions.npy")

print("\nActual shape:", y_true.shape)
print("Prediction shape:", y_pred.shape)

# Select one sensor
sensor = 0

# Number of points to display
n_points = 200

actual = y_true[:n_points, sensor]
predicted = y_pred[:n_points, sensor]

# Create output directory
os.makedirs("results/figures", exist_ok=True)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    actual,
    label="Actual"
)

plt.plot(
    predicted,
    label="Predicted"
)

plt.xlabel("Time Step")
plt.ylabel("Normalized Traffic Speed")

plt.title(
    f"LSTM Actual vs Predicted Traffic Speed - Sensor {sensor}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

output_path = "results/figures/lstm_actual_vs_predicted.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(output_path)

print("\n" + "=" * 60)
print("PREDICTION PLOT COMPLETE")
print("=" * 60)
