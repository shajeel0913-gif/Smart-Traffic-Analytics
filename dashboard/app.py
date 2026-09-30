import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_DIR / "results" / "figures"
TABLES_DIR = PROJECT_DIR / "results" / "tables"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Analytics",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Smart Traffic Analytics")

st.markdown(
    """
    ### METR-LA Traffic Speed Prediction using LSTM and Bi-LSTM

    This dashboard presents the preprocessing results, model
    performance comparison, and traffic speed predictions.
    """
)

st.divider()


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison_file = TABLES_DIR / "model_comparison.csv"

if comparison_file.exists():

    comparison = pd.read_csv(comparison_file)

else:

    st.error("model_comparison.csv not found.")
    st.stop()


# ============================================================
# LOAD SENSOR IDs
# ============================================================

sensor_file = PROCESSED_DIR / "sensor_ids.npy"

if sensor_file.exists():

    sensor_ids = np.load(
        sensor_file,
        allow_pickle=True
    )

else:

    sensor_ids = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Comparison",
        "Training Results",
        "Predictions",
        "Dataset Analysis"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sensors",
            "207"
        )

    with col2:
        st.metric(
            "History",
            "12 × 5 min"
        )

    with col3:
        st.metric(
            "Prediction",
            "Next 5 min"
        )

    with col4:
        st.metric(
            "Best Model",
            "Bi-LSTM"
        )

    st.divider()

    st.subheader("Project Pipeline")

    st.markdown(
        """
        **METR-LA Dataset**

        ↓

        **Data Preprocessing**

        ↓

        **Missing Value Handling**

        ↓

        **Min-Max Scaling**

        ↓

        **12 Time-Step Sequences**

        ↓

        **LSTM / Bi-LSTM**

        ↓

        **Traffic Speed Prediction**

        ↓

        **FastAPI REST API**

        ↓

        **Streamlit Dashboard**
        """
    )

    st.divider()

    st.subheader("Final Model")

    st.success(
        "Bi-LSTM is selected as the overall winner because "
        "it achieves the highest R² and the lowest MSE/RMSE."
    )

    st.info(
        "The model predicts traffic speed for all 207 sensors "
        "using the previous 12 observations (1 hour of history)."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("📊 LSTM vs Bi-LSTM")

    st.subheader("Performance Metrics")

    display_df = comparison.copy()

    for column in [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]:
        display_df[column] = display_df[column].round(6)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("MAE Comparison")

        image_path = (
            FIGURES_DIR /
            "lstm_vs_bilstm_mae.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    with col2:

        st.subheader("MSE Comparison")

        image_path = (
            FIGURES_DIR /
            "lstm_vs_bilstm_mse.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("RMSE Comparison")

        image_path = (
            FIGURES_DIR /
            "lstm_vs_bilstm_rmse.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    with col2:

        st.subheader("R² Comparison")

        image_path = (
            FIGURES_DIR /
            "lstm_vs_bilstm_r2.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    st.divider()

    st.subheader("🏆 Winner")

    st.success(
        "Bi-LSTM — R² = 0.832548"
    )


# ============================================================
# TRAINING RESULTS
# ============================================================

elif page == "Training Results":

    st.header("📈 Model Training Results")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("LSTM Training Loss")

        image_path = (
            FIGURES_DIR /
            "lstm_training_loss.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    with col2:

        st.subheader("Bi-LSTM Training Loss")

        image_path = (
            FIGURES_DIR /
            "bilstm_training_loss.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("LSTM Training MAE")

        image_path = (
            FIGURES_DIR /
            "lstm_training_mae.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    with col2:

        st.subheader("Bi-LSTM Training MAE")

        image_path = (
            FIGURES_DIR /
            "bilstm_training_mae.png"
        )

        if image_path.exists():
            st.image(
                Image.open(image_path),
                use_container_width=True
            )


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "Predictions":

    st.header("🚗 Traffic Speed Predictions")

    st.markdown(
        """
        The Bi-LSTM model predicts traffic speed at **207 sensors**
        for the next 5-minute time step.
        """
    )

    prediction_file = (
        PROJECT_DIR /
        "results" /
        "bilstm_predictions.npy"
    )

    actual_file = (
        PROJECT_DIR /
        "results" /
        "bilstm_actual.npy"
    )

    if prediction_file.exists():

        predictions = np.load(
            prediction_file
        )

        actual = np.load(
            actual_file
        )

        sensor_index = st.selectbox(
            "Select Sensor",
            range(len(sensor_ids)),
            format_func=lambda x: str(sensor_ids[x])
        )

        predicted_speed = float(
            predictions[0, sensor_index]
        )

        actual_speed = float(
            actual[0, sensor_index]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sensor ID",
                str(sensor_ids[sensor_index])
            )

        with col2:
            st.metric(
                "Predicted Speed",
                f"{predicted_speed:.2f}"
            )

        with col3:
            st.metric(
                "Actual Speed",
                f"{actual_speed:.2f}"
            )

        st.divider()

        st.subheader("Bi-LSTM Actual vs Predicted")

        image_path = (
            FIGURES_DIR /
            "bilstm_actual_vs_predicted.png"
        )

        if image_path.exists():

            st.image(
                Image.open(image_path),
                use_container_width=True
            )

    else:

        st.warning(
            "Prediction file not found."
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.header("📁 METR-LA Dataset Analysis")

    images = [
        (
            "Overall Traffic Speed",
            "overall_traffic_speed.png"
        ),
        (
            "Daily Average Speed",
            "daily_average_speed.png"
        ),
        (
            "Sample Sensor Speed",
            "sample_sensor_speed.png"
        ),
        (
            "Zero Value Analysis",
            "zero_value_analysis.png"
        )
    ]

    for title, filename in images:

        st.subheader(title)

        image_path = FIGURES_DIR / filename

        if image_path.exists():

            st.image(
                Image.open(image_path),
                use_container_width=True
            )

        else:

            st.warning(
                f"{filename} not found."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Traffic Analytics | METR-LA | "
    "LSTM & Bi-LSTM Traffic Speed Prediction"
)
