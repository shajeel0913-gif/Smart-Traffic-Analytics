import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("BI-LSTM ACTUAL VS PREDICTED")
print("=" * 60)

actual = np.load("results/bilstm_actual.npy")
predicted = np.load("results/bilstm_predictions.npy")

print(f"\nActual shape: {actual.shape}")
print(f"Prediction shape: {predicted.shape}")

# Select one sensor for visualization
sensor = 0

plt.figure(figsize=(14, 6))

plt.plot(
    actual[:, sensor],
    label="Actual"
)

plt.plot(
    predicted[:, sensor],
    label="Bi-LSTM Predicted"
)

plt.xlabel("Time Step")
plt.ylabel("Normalized Traffic Speed")
plt.title(
    f"Bi-LSTM Actual vs Predicted Traffic Speed - Sensor {sensor}"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/bilstm_actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("results/figures/bilstm_actual_vs_predicted.png")

print("\n" + "=" * 60)
print("PREDICTION PLOT COMPLETE")
print("=" * 60)
