import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("METR-LA LSTM MODEL EVALUATION")
print("=" * 60)

# Load actual and predicted values
y_true = np.load("results/lstm_actual.npy")
y_pred = np.load("results/lstm_predictions.npy")

print("\nData shapes:")
print("Actual    :", y_true.shape)
print("Predicted :", y_pred.shape)

# Flatten all samples and sensors
y_true_flat = y_true.reshape(-1)
y_pred_flat = y_pred.reshape(-1)

# Calculate metrics
mae = mean_absolute_error(y_true_flat, y_pred_flat)
mse = mean_squared_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_flat, y_pred_flat)

print("\n" + "=" * 60)
print("LSTM PERFORMANCE METRICS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
