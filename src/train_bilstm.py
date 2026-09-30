import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
import os

print("=" * 60)
print("METR-LA BI-LSTM TRAINING")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading sequence data...")

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ============================================================
# 2. CHECK DATA
# ============================================================

print("\n[2] Checking data...")

datasets = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test
}

for name, data in datasets.items():
    print(
        f"{name}: "
        f"NaN={np.isnan(data).sum()}, "
        f"Inf={np.isinf(data).sum()}"
    )

# ============================================================
# 3. BUILD BI-LSTM MODEL
# ============================================================

print("\n[3] Building Bi-LSTM model...")

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1], X_train.shape[2])),

    layers.Bidirectional(
        layers.LSTM(128)
    ),

    layers.Dropout(0.2),

    layers.Dense(207)
])

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

print("\nModel summary:")
model.summary()

# ============================================================
# 4. CALLBACKS
# ============================================================

print("\n[4] Preparing callbacks...")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/bilstm_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# 5. TRAIN
# ============================================================

print("\n[5] Starting Bi-LSTM training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,
    batch_size=32,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

# ============================================================
# 6. SAVE FINAL MODEL
# ============================================================

print("\n[6] Saving final model...")

model.save(
    "models/bilstm_final.keras"
)

# ============================================================
# 7. SAVE HISTORY
# ============================================================

print("\n[7] Saving training history...")

np.save(
    "results/bilstm_history.npy",
    history.history
)

# ============================================================
# 8. EVALUATE
# ============================================================

print("\n[8] Evaluating Bi-LSTM on test data...")

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Results:")
print(f"  Test Loss (MSE): {test_loss:.6f}")
print(f"  Test MAE:        {test_mae:.6f}")

# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\n[9] Generating test predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

print(
    "Prediction shape:",
    predictions.shape
)

# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

np.save(
    "results/bilstm_predictions.npy",
    predictions
)

np.save(
    "results/bilstm_actual.npy",
    y_test
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BI-LSTM TRAINING COMPLETE")
print("=" * 60)

print(
    "Total epochs trained:",
    len(history.history["loss"])
)

best_epoch = (
    np.argmin(history.history["val_loss"]) + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Test loss: "
    f"{test_loss:.6f}"
)

print(
    f"Test MAE: "
    f"{test_mae:.6f}"
)

print("\nSaved files:")

print(" - models/bilstm_best.keras")
print(" - models/bilstm_final.keras")
print(" - results/bilstm_history.npy")
print(" - results/bilstm_predictions.npy")
print(" - results/bilstm_actual.npy")

print("\n" + "=" * 60)
