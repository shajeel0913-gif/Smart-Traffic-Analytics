import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("METR-LA BI-LSTM MODEL EVALUATION")
print("=" * 60)

# Load actual and predicted values
actual = np.load("results/bilstm_actual.npy")
predicted = np.load("results/bilstm_predictions.npy")

print("\nData shapes:")
print(f"Actual    : {actual.shape}")
print(f"Predicted : {predicted.shape}")

# Flatten all sensor predictions
actual_flat = actual.flatten()
predicted_flat = predicted.flatten()

# Calculate metrics
mae = mean_absolute_error(actual_flat, predicted_flat)
mse = mean_squared_error(actual_flat, predicted_flat)
rmse = np.sqrt(mse)
r2 = r2_score(actual_flat, predicted_flat)

print("\n" + "=" * 60)
print("BI-LSTM PERFORMANCE METRICS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
