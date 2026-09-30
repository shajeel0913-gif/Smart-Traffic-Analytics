import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")

# 12 × 5 minutes = 60 minutes of history
SEQUENCE_LENGTH = 12


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("METR-LA SEQUENCE CREATION")
print("=" * 60)

print("\n[1] Loading processed data...")

train = np.load(PROCESSED_DIR / "train.npy")
val = np.load(PROCESSED_DIR / "val.npy")
test = np.load(PROCESSED_DIR / "test.npy")

train_valid = np.load(
    PROCESSED_DIR / "train_valid.npy"
)

val_valid = np.load(
    PROCESSED_DIR / "val_valid.npy"
)

test_valid = np.load(
    PROCESSED_DIR / "test_valid.npy"
)

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

print("Train validity mask:", train_valid.shape)
print("Validation validity mask:", val_valid.shape)
print("Test validity mask:", test_valid.shape)


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(data, valid_mask, name):
    """
    Create supervised learning sequences.

    Input:
        Previous 12 time steps
        Each time step contains 207 sensors.

    Output:
        Next time step
        Contains traffic speed for all 207 sensors.

    A sequence is created only when:
        1. All 12 historical timestamps are valid.
        2. The target timestamp is valid.
        3. No NaN or Inf exists in the sequence.
    """

    print(f"\nCreating {name} sequences...")

    X = []
    y = []

    total_possible = len(data) - SEQUENCE_LENGTH

    skipped_invalid = 0
    skipped_nan = 0

    for i in range(total_possible):

        # ----------------------------------------------------
        # Historical window
        # ----------------------------------------------------

        start = i
        end = i + SEQUENCE_LENGTH

        # 12 historical timestamps
        history_valid = valid_mask[start:end]

        # Target timestamp
        target_valid = valid_mask[end]

        # ----------------------------------------------------
        # Check validity mask
        # ----------------------------------------------------

        if not history_valid.all() or not target_valid:
            skipped_invalid += 1
            continue

        # ----------------------------------------------------
        # Extract sequence and target
        # ----------------------------------------------------

        sequence = data[start:end]
        target = data[end]

        # ----------------------------------------------------
        # Final NaN / Inf safety check
        # ----------------------------------------------------

        if not np.isfinite(sequence).all():
            skipped_nan += 1
            continue

        if not np.isfinite(target).all():
            skipped_nan += 1
            continue

        X.append(sequence)
        y.append(target)

    # Convert lists to NumPy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    print(f"\n{name} sequence statistics:")
    print("  Possible sequences:", total_possible)
    print("  Created sequences:", len(X))
    print("  Skipped invalid:", skipped_invalid)
    print("  Skipped NaN/Inf:", skipped_nan)

    if len(X) == 0:
        raise ValueError(
            f"No valid sequences were created for {name}."
        )

    print("  X shape:", X.shape)
    print("  y shape:", y.shape)

    return X, y


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

print("\n[2] Creating training sequences...")

X_train, y_train = create_sequences(
    train,
    train_valid,
    "TRAIN"
)


# ============================================================
# CREATE VALIDATION SEQUENCES
# ============================================================

print("\n[3] Creating validation sequences...")

X_val, y_val = create_sequences(
    val,
    val_valid,
    "VALIDATION"
)


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

print("\n[4] Creating test sequences...")

X_test, y_test = create_sequences(
    test,
    test_valid,
    "TEST"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n[5] Final sequence validation...")

datasets = [
    ("X_train", X_train),
    ("y_train", y_train),
    ("X_val", X_val),
    ("y_val", y_val),
    ("X_test", X_test),
    ("y_test", y_test),
]

for name, data in datasets:

    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    print(
        f"{name}: "
        f"shape={data.shape}, "
        f"NaN={nan_count}, "
        f"Inf={inf_count}"
    )

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# ============================================================
# SAVE SEQUENCES
# ============================================================

print("\n[6] Saving sequences...")

np.save(
    PROCESSED_DIR / "X_train.npy",
    X_train
)

np.save(
    PROCESSED_DIR / "y_train.npy",
    y_train
)

np.save(
    PROCESSED_DIR / "X_val.npy",
    X_val
)

np.save(
    PROCESSED_DIR / "y_val.npy",
    y_val
)

np.save(
    PROCESSED_DIR / "X_test.npy",
    X_test
)

np.save(
    PROCESSED_DIR / "y_test.npy",
    y_test
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE CREATION COMPLETE")
print("=" * 60)

print("\nSaved files:")

print(" - X_train.npy")
print(" - y_train.npy")
print(" - X_val.npy")
print(" - y_val.npy")
print(" - X_test.npy")
print(" - y_test.npy")

print("\nFinal shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\n" + "=" * 60)
