import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("BI-LSTM TRAINING HISTORY PLOTS")
print("=" * 60)

# Load training history
history = np.load(
    "results/bilstm_history.npy",
    allow_pickle=True
).item()

epochs = range(1, len(history["loss"]) + 1)

# ---------------------------------------------------------
# Loss plot
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Bi-LSTM Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/bilstm_training_loss.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# MAE plot
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["mae"],
    label="Training MAE"
)

plt.plot(
    epochs,
    history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Bi-LSTM Training and Validation MAE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/bilstm_training_mae.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("results/figures/bilstm_training_loss.png")
print("results/figures/bilstm_training_mae.png")

print("\n" + "=" * 60)
print("PLOTTING COMPLETE")
print("=" * 60)
