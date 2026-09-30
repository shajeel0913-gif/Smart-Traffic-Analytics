import pandas as pd
import tables
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/metr-la.h5"
FIGURE_DIR = Path("results/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD METR-LA DATASET
# ============================================================

def load_metr_la(path):
    """Load METR-LA HDF5 data using PyTables."""

    with tables.open_file(path, mode="r") as f:
        values = f.root.df.block0_values.read()
        sensor_ids = f.root.df.axis0.read()
        timestamps = f.root.df.axis1.read()

    # Convert sensor IDs from bytes to strings
    sensor_ids = [
        sensor.decode("utf-8") if isinstance(sensor, bytes) else str(sensor)
        for sensor in sensor_ids
    ]

    # Convert Unix nanosecond timestamps
    timestamps = pd.to_datetime(timestamps, unit="ns")

    df = pd.DataFrame(
        values,
        index=timestamps,
        columns=sensor_ids
    )

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("METR-LA TRAFFIC DATASET - EDA")
print("=" * 60)

df = load_metr_la(DATA_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n===== DATASET INFORMATION =====")
print("Shape:", df.shape)
print("Number of timestamps:", len(df.index))
print("Number of sensors:", len(df.columns))

print("\n===== TIME RANGE =====")
print("Start:", df.index.min())
print("End:", df.index.max())

print("\n===== SAMPLING INTERVAL =====")
print(df.index.to_series().diff().value_counts().head())


# ============================================================
# SENSOR INFORMATION
# ============================================================

print("\n===== SENSOR INFORMATION =====")
print("First 10 sensors:")
print(df.columns[:10].tolist())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n===== MISSING VALUES =====")
total_missing = df.isna().sum().sum()
print("Total missing values:", total_missing)

if total_missing == 0:
    print("No missing values found.")


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n===== BASIC STATISTICS =====")
print(df.describe().T.head(10))


# ============================================================
# OVERALL TRAFFIC STATISTICS
# ============================================================

print("\n===== OVERALL TRAFFIC SPEED =====")

all_values = df.values.flatten()

print("Minimum speed:", all_values.min())
print("Maximum speed:", all_values.max())
print("Mean speed:", all_values.mean())
print("Median speed:", pd.Series(all_values).median())


# ============================================================
# ZERO VALUES
# ============================================================

zero_count = (df == 0).sum().sum()
total_values = df.size
zero_percentage = (zero_count / total_values) * 100

print("\n===== ZERO VALUES =====")
print("Zero-valued observations:", zero_count)
print("Zero percentage:", round(zero_percentage, 2), "%")


# ============================================================
# DAILY AVERAGE TRAFFIC SPEED
# ============================================================

daily_average = df.mean(axis=1).resample("D").mean()

print("\n===== DAILY AVERAGE =====")
print(daily_average.head())


# ============================================================
# HOURLY AVERAGE TRAFFIC SPEED
# ============================================================

hourly_average = df.mean(axis=1).resample("h").mean()

print("\n===== HOURLY AVERAGE =====")
print(hourly_average.head())


# ============================================================
# PLOT 1 — OVERALL TRAFFIC SPEED
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df.mean(axis=1)
)

plt.title("METR-LA Overall Average Traffic Speed")
plt.xlabel("Time")
plt.ylabel("Average Speed")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "overall_traffic_speed.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — DAILY AVERAGE
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_average.index,
    daily_average.values
)

plt.title("METR-LA Daily Average Traffic Speed")
plt.xlabel("Date")
plt.ylabel("Average Speed")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "daily_average_speed.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 — SAMPLE SENSOR
# ============================================================

sample_sensor = df.columns[0]

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df[sample_sensor]
)

plt.title(
    f"Traffic Speed - Sensor {sample_sensor}"
)

plt.xlabel("Time")
plt.ylabel("Speed")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "sample_sensor_speed.png",
    dpi=300
)

plt.close()

# ============================================================
# ZERO-VALUE ANALYSIS
# ============================================================

print("\n===== ZERO-VALUE ANALYSIS =====")

# Zero count for each sensor
zero_count_per_sensor = (df == 0).sum()

zero_percentage_per_sensor = (
    zero_count_per_sensor / len(df)
) * 100

zero_summary = pd.DataFrame({
    "zero_count": zero_count_per_sensor,
    "zero_percentage": zero_percentage_per_sensor
})

zero_summary = zero_summary.sort_values(
    "zero_percentage",
    ascending=False
)

print("\nTop 10 sensors with highest zero percentage:")
print(zero_summary.head(10))


# ============================================================
# ZERO PERCENTAGE STATISTICS
# ============================================================

print("\n===== ZERO PERCENTAGE STATISTICS =====")

print(
    zero_summary["zero_percentage"].describe()
)


# ============================================================
# HOURLY ZERO VALUES
# ============================================================

hourly_zero_percentage = (
    (df == 0)
    .resample("h")
    .mean()
    .mean(axis=1)
    * 100
)

print("\n===== HOURLY ZERO PERCENTAGE =====")
print(hourly_zero_percentage.head(10))


# ============================================================
# ZERO-VALUE PLOT
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    hourly_zero_percentage.index,
    hourly_zero_percentage.values
)

plt.title("Hourly Percentage of Zero Traffic-Speed Values")
plt.xlabel("Time")
plt.ylabel("Zero Values (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "zero_value_analysis.png",
    dpi=300
)

plt.close()

# ============================================================
# CONSECUTIVE ZERO ANALYSIS
# ============================================================

print("\n===== CONSECUTIVE ZERO ANALYSIS =====")


def longest_zero_streak(series):
    """Return the longest consecutive sequence of zero values."""

    is_zero = series.eq(0)

    groups = is_zero.ne(is_zero.shift()).cumsum()

    streaks = (
        is_zero
        .groupby(groups)
        .sum()
    )

    return int(streaks.max())


zero_streaks = {}

for sensor in df.columns:
    zero_streaks[sensor] = longest_zero_streak(df[sensor])


zero_streak_summary = (
    pd.Series(zero_streaks)
    .sort_values(ascending=False)
)


print("\nTop 10 sensors by longest consecutive zero streak:")

print(zero_streak_summary.head(10))


print("\nLongest zero streak in 5-minute intervals:")
print(zero_streak_summary.head(10))

print("\nLongest zero streak in hours:")
print(
    (zero_streak_summary.head(10) * 5 / 60)
)

# ============================================================
# FINISHED
# ============================================================

print("\n===== EDA COMPLETE =====")
print("Figures saved to:", FIGURE_DIR)
