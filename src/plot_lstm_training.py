import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("LSTM TRAINING HISTORY PLOT")
print("=" * 60)

# Load training history
history = np.load(
    "results/lstm_history.npy",
    allow_pickle=True
).item()

print("\nAvailable history keys:")
print(history.keys())

# Extract values
train_loss = history["loss"]
val_loss = history["val_loss"]

train_mae = history["mae"]
val_mae = history["val_mae"]

epochs = range(1, len(train_loss) + 1)

# Make sure output directory exists
os.makedirs("results/figures", exist_ok=True)

# ---------------------------------------------------------
# LOSS GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = "results/figures/lstm_training_loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(loss_path)

# ---------------------------------------------------------
# MAE GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_mae,
    label="Training MAE"
)

plt.plot(
    epochs,
    val_mae,
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("LSTM Training and Validation MAE")

plt.legend()
plt.grid(True)

plt.tight_layout()

mae_path = "results/figures/lstm_training_mae.png"

plt.savefig(
    mae_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(mae_path)

print("\n" + "=" * 60)
print("PLOTTING COMPLETE")
print("=" * 60)
