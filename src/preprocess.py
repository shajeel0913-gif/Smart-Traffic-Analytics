import pandas as pd
import numpy as np
import tables
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/metr-la.h5"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# METR-LA has one observation every 5 minutes.
# 12 × 5 minutes = 1 hour.
MAX_INTERPOLATION_GAP = 12


# ============================================================
# LOAD METR-LA DATASET
# ============================================================

def load_metr_la(path):
    """
    Load the METR-LA HDF5 dataset using PyTables.

    Returns
    -------
    pandas.DataFrame
        Rows    = timestamps
        Columns = traffic sensors
    """

    with tables.open_file(path, mode="r") as f:

        values = f.root.df.block0_values.read()
        sensor_ids = f.root.df.axis0.read()
        timestamps = f.root.df.axis1.read()

    # Convert sensor IDs from bytes to strings
    sensor_ids = [
        x.decode("utf-8")
        if isinstance(x, bytes)
        else str(x)
        for x in sensor_ids
    ]

    # METR-LA timestamps are Unix timestamps in nanoseconds
    timestamps = pd.to_datetime(
        timestamps,
        unit="ns"
    )

    df = pd.DataFrame(
        values,
        index=timestamps,
        columns=sensor_ids
    )

    return df


# ============================================================
# CLEAN ONE DATA SPLIT
# ============================================================

def clean_split(df, name):
    """
    Clean one chronological data split.

    Steps:
    1. Convert zero traffic speeds to NaN.
    2. Interpolate only short gaps.
    3. Keep long gaps as NaN.
    4. Create a timestamp-level validity mask.

    A timestamp is fully valid only when all sensors
    contain valid values.
    """

    print(f"\nCleaning {name} split...")

    # --------------------------------------------------------
    # STEP 1: Convert zero values to missing
    # --------------------------------------------------------

    df = df.replace(0, np.nan)

    missing_before = int(
        df.isna().sum().sum()
    )

    print(
        "Missing after zero conversion:",
        missing_before
    )

    print(
        "Missing percentage:",
        f"{missing_before / df.size * 100:.2f}%"
    )

    # --------------------------------------------------------
    # STEP 2: Interpolate ONLY short gaps
    # --------------------------------------------------------

    df = df.interpolate(
        method="time",
        limit=MAX_INTERPOLATION_GAP,
        limit_direction="both"
    )

    missing_after = int(
        df.isna().sum().sum()
    )

    print(
        "Missing after short-gap interpolation:",
        missing_after
    )

    print(
        "Remaining missing percentage:",
        f"{missing_after / df.size * 100:.2f}%"
    )

    # --------------------------------------------------------
    # STEP 3: Create timestamp-level validity mask
    # --------------------------------------------------------

    valid = ~df.isna().any(axis=1)

    valid_count = int(valid.sum())

    print(
        f"Fully valid timestamps: "
        f"{valid_count} / {len(valid)}"
    )

    return df, valid


# ============================================================
# FIT SCALER
# ============================================================

def fit_scaler(train_df, train_valid):
    """
    Fit MinMaxScaler using ONLY fully valid training rows.

    Each sensor gets its own scaling range.
    """

    print(
        "\nPreparing training data for scaling..."
    )

    train_valid_values = train_df.loc[
        train_valid
    ].to_numpy(
        dtype=np.float64
    )

    print(
        "Valid training rows:",
        train_valid_values.shape[0]
    )

    print(
        "Training features:",
        train_valid_values.shape[1]
    )

    if train_valid_values.shape[0] == 0:
        raise ValueError(
            "No fully valid training rows available."
        )

    scaler = MinMaxScaler()

    scaler.fit(
        train_valid_values
    )

    return scaler


# ============================================================
# SCALE DATA WHILE PRESERVING NaN
# ============================================================

def scale_with_nan(df, scaler):
    """
    Scale the data sensor-by-sensor while preserving NaN.

    The scaler was fitted using 207 sensor features.
    Therefore, transformation is performed on the
    complete 207-column matrix.

    NaN positions are restored after scaling.
    """

    values = df.to_numpy(
        dtype=np.float64
    )

    # Save locations of missing values
    nan_mask = np.isnan(values)

    # Create temporary copy
    temp = values.copy()

    # Replace NaN temporarily with the training
    # minimum for the corresponding sensor.
    #
    # This allows sklearn's MinMaxScaler to process
    # the complete matrix.
    for sensor_idx in range(
        values.shape[1]
    ):

        sensor_nan = nan_mask[:, sensor_idx]

        if sensor_nan.any():

            temp[
                sensor_nan,
                sensor_idx
            ] = scaler.data_min_[sensor_idx]

    # Scale all 207 sensors
    scaled = scaler.transform(
        temp
    )

    # Restore original NaN positions
    scaled[nan_mask] = np.nan

    return scaled


# ============================================================
# VERIFY DATA
# ============================================================

def verify_data(name, data):
    """
    Verify NaN, Inf and numerical range.
    """

    nan_count = int(
        np.isnan(data).sum()
    )

    inf_count = int(
        np.isinf(data).sum()
    )

    finite_values = data[
        np.isfinite(data)
    ]

    if finite_values.size == 0:
        raise ValueError(
            f"{name} contains no finite values."
        )

    print(
        f"{name}: "
        f"NaN={nan_count}, "
        f"Inf={inf_count}, "
        f"Min={finite_values.min():.4f}, "
        f"Max={finite_values.max():.4f}"
    )

    if inf_count > 0:
        raise ValueError(
            f"{name} contains Inf values."
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("METR-LA PREPROCESSING")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

df = load_metr_la(
    DATA_PATH
)

print(
    "Original shape:",
    df.shape
)

print(
    "Number of timestamps:",
    len(df)
)

print(
    "Number of sensors:",
    len(df.columns)
)

print(
    "Start:",
    df.index.min()
)

print(
    "End:",
    df.index.max()
)


# ============================================================
# 2. CHRONOLOGICAL SPLIT
# ============================================================

print(
    "\n[2] Creating chronological split..."
)

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (TRAIN_RATIO + VAL_RATIO)
)

train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()

print(
    "Train shape:",
    train_df.shape
)

print(
    "Validation shape:",
    val_df.shape
)

print(
    "Test shape:",
    test_df.shape
)


# ============================================================
# 3. CLEAN EACH SPLIT
# ============================================================

print(
    "\n[3] Cleaning splits..."
)

train_df, train_valid = clean_split(
    train_df,
    "train"
)

val_df, val_valid = clean_split(
    val_df,
    "validation"
)

test_df, test_valid = clean_split(
    test_df,
    "test"
)


# ============================================================
# 4. FIT SCALER ONLY ON TRAINING DATA
# ============================================================

print(
    "\n[4] Fitting scaler..."
)

scaler = fit_scaler(
    train_df,
    train_valid
)

print(
    "Scaler fitted using training data only."
)


# ============================================================
# 5. SCALE DATA
# ============================================================

print(
    "\n[5] Scaling data..."
)

train_scaled = scale_with_nan(
    train_df,
    scaler
)

val_scaled = scale_with_nan(
    val_df,
    scaler
)

test_scaled = scale_with_nan(
    test_df,
    scaler
)


# ============================================================
# 6. VERIFY PROCESSED DATA
# ============================================================

print(
    "\n[6] Final verification..."
)

verify_data(
    "train",
    train_scaled
)

verify_data(
    "validation",
    val_scaled
)

verify_data(
    "test",
    test_scaled
)


# ============================================================
# 7. SAVE SCALER
# ============================================================

print(
    "\n[7] Saving scaler..."
)

scaler_path = (
    PROCESSED_DIR /
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

print(
    "Scaler saved:",
    scaler_path
)


# ============================================================
# 8. SAVE VALIDITY MASKS
# ============================================================

print(
    "\n[8] Saving validity masks..."
)

np.save(
    PROCESSED_DIR /
    "train_valid.npy",
    train_valid.to_numpy()
)

np.save(
    PROCESSED_DIR /
    "val_valid.npy",
    val_valid.to_numpy()
)

np.save(
    PROCESSED_DIR /
    "test_valid.npy",
    test_valid.to_numpy()
)

print(
    "Validity masks saved."
)


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

print(
    "\n[9] Saving processed datasets..."
)

np.save(
    PROCESSED_DIR /
    "train.npy",
    train_scaled
)

np.save(
    PROCESSED_DIR /
    "val.npy",
    val_scaled
)

np.save(
    PROCESSED_DIR /
    "test.npy",
    test_scaled
)


# ============================================================
# 10. SAVE TIMESTAMPS
# ============================================================

np.save(
    PROCESSED_DIR /
    "train_timestamps.npy",
    train_df.index.to_numpy()
)

np.save(
    PROCESSED_DIR /
    "val_timestamps.npy",
    val_df.index.to_numpy()
)

np.save(
    PROCESSED_DIR /
    "test_timestamps.npy",
    test_df.index.to_numpy()
)


# ============================================================
# 11. SAVE SENSOR IDS
# ============================================================

np.save(
    PROCESSED_DIR /
    "sensor_ids.npy",
    np.array(
        df.columns
    )
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PREPROCESSING COMPLETE"
)

print(
    "=" * 60
)

print(
    "Train:",
    train_scaled.shape
)

print(
    "Validation:",
    val_scaled.shape
)

print(
    "Test:",
    test_scaled.shape
)

print(
    "\nFully valid timestamps:"
)

print(
    "Train:",
    int(train_valid.sum()),
    "/",
    len(train_valid)
)

print(
    "Validation:",
    int(val_valid.sum()),
    "/",
    len(val_valid)
)

print(
    "Test:",
    int(test_valid.sum()),
    "/",
    len(test_valid)
)

print(
    "\nProcessed files:"
)

print(" - train.npy")
print(" - val.npy")
print(" - test.npy")
print(" - scaler.pkl")
print(" - sensor_ids.npy")
print(" - train_timestamps.npy")
print(" - val_timestamps.npy")
print(" - test_timestamps.npy")
print(" - train_valid.npy")
print(" - val_valid.npy")
print(" - test_valid.npy")

print(
    "\nNext step:"
)

print(
    "Create valid LSTM sequences while excluding"
)

print(
    "windows containing NaN values."
)

print(
    "=" * 60
)
