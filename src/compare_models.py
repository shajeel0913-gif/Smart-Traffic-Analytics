import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("LSTM vs BI-LSTM MODEL COMPARISON")
print("=" * 60)

# ============================================================
# Load LSTM results
# ============================================================

lstm_actual = np.load("results/lstm_actual.npy")
lstm_pred = np.load("results/lstm_predictions.npy")

# ============================================================
# Load Bi-LSTM results
# ============================================================

bilstm_actual = np.load("results/bilstm_actual.npy")
bilstm_pred = np.load("results/bilstm_predictions.npy")

# Flatten arrays
lstm_actual_flat = lstm_actual.flatten()
lstm_pred_flat = lstm_pred.flatten()

bilstm_actual_flat = bilstm_actual.flatten()
bilstm_pred_flat = bilstm_pred.flatten()

# ============================================================
# Calculate LSTM metrics
# ============================================================

lstm_mae = mean_absolute_error(
    lstm_actual_flat,
    lstm_pred_flat
)

lstm_mse = mean_squared_error(
    lstm_actual_flat,
    lstm_pred_flat
)

lstm_rmse = np.sqrt(lstm_mse)

lstm_r2 = r2_score(
    lstm_actual_flat,
    lstm_pred_flat
)

# ============================================================
# Calculate Bi-LSTM metrics
# ============================================================

bilstm_mae = mean_absolute_error(
    bilstm_actual_flat,
    bilstm_pred_flat
)

bilstm_mse = mean_squared_error(
    bilstm_actual_flat,
    bilstm_pred_flat
)

bilstm_rmse = np.sqrt(bilstm_mse)

bilstm_r2 = r2_score(
    bilstm_actual_flat,
    bilstm_pred_flat
)

# ============================================================
# Create comparison table
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "LSTM",
        "Bi-LSTM"
    ],

    "MAE": [
        lstm_mae,
        bilstm_mae
    ],

    "MSE": [
        lstm_mse,
        bilstm_mse
    ],

    "RMSE": [
        lstm_rmse,
        bilstm_rmse
    ],

    "R2": [
        lstm_r2,
        bilstm_r2
    ]
})

print("\n")
print(comparison.to_string(index=False))

# ============================================================
# Save comparison table
# ============================================================

comparison.to_csv(
    "results/tables/model_comparison.csv",
    index=False
)

print("\nSaved:")
print("results/tables/model_comparison.csv")

# ============================================================
# Create comparison plots
# ============================================================

metrics = ["MAE", "MSE", "RMSE", "R2"]

for metric in metrics:

    plt.figure(figsize=(8, 6))

    plt.bar(
        comparison["Model"],
        comparison[metric]
    )

    plt.ylabel(metric)
    plt.title(f"LSTM vs Bi-LSTM - {metric}")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"results/figures/"
        f"lstm_vs_bilstm_{metric.lower()}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

# ============================================================
# Final conclusion
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    f"\nLSTM    → "
    f"MAE={lstm_mae:.6f}, "
    f"RMSE={lstm_rmse:.6f}, "
    f"R²={lstm_r2:.6f}"
)

print(
    f"Bi-LSTM → "
    f"MAE={bilstm_mae:.6f}, "
    f"RMSE={bilstm_rmse:.6f}, "
    f"R²={bilstm_r2:.6f}"
)

if bilstm_r2 > lstm_r2 and bilstm_rmse < lstm_rmse:
    print("\nOVERALL WINNER: Bi-LSTM")
else:
    print("\nOVERALL WINNER: LSTM")

print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)
