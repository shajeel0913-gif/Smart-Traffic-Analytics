import os

# Reduce unnecessary TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50

SEQUENCE_LENGTH = 12
N_SENSORS = 207


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("METR-LA LSTM TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1] Loading sequence data...")

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val.npy"
)

y_val = np.load(
    f"{DATA_DIR}/y_val.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# DATA VALIDATION
# ============================================================

print("\n[2] Checking data...")

for name, data in [
    ("X_train", X_train),
    ("y_train", y_train),
    ("X_val", X_val),
    ("y_val", y_val),
    ("X_test", X_test),
    ("y_test", y_test)
]:

    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()

    print(
        f"{name}: "
        f"NaN={nan_count}, "
        f"Inf={inf_count}"
    )

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\n[3] Building LSTM model...")

model = Sequential([
    
    LSTM(
        128,
        input_shape=(
            SEQUENCE_LENGTH,
            N_SENSORS
        ),
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        207,
        activation="linear"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


print("\nModel summary:\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

print("\n[4] Preparing callbacks...")

checkpoint = ModelCheckpoint(
    f"{MODEL_DIR}/lstm_best.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    mode="min",
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n[5] Starting training...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n[6] Saving final model...")

model.save(
    f"{MODEL_DIR}/lstm_final.keras"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

print("\n[7] Saving training history...")

np.save(
    f"{RESULTS_DIR}/lstm_history.npy",
    history.history,
    allow_pickle=True
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n[8] Evaluating model on test data...")

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest Results:")
print(
    f"  Test Loss (MSE): {test_loss:.6f}"
)

print(
    f"  Test MAE:        {test_mae:.6f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n[9] Generating test predictions...")

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(
    "Prediction shape:",
    predictions.shape
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    f"{RESULTS_DIR}/lstm_predictions.npy",
    predictions
)

np.save(
    f"{RESULTS_DIR}/lstm_actual.npy",
    y_test
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

best_epoch = np.argmin(
    history.history["val_loss"]
) + 1

best_val_loss = min(
    history.history["val_loss"]
)

print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETE")
print("=" * 60)

print(
    f"Total epochs trained: "
    f"{len(history.history['loss'])}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
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

print(" - models/lstm_best.keras")
print(" - models/lstm_final.keras")
print(" - results/lstm_history.npy")
print(" - results/lstm_predictions.npy")
print(" - results/lstm_actual.npy")

print("\n" + "=" * 60)
