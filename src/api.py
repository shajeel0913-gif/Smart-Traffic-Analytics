import numpy as np
import joblib
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/bilstm_best.keras"
SCALER_PATH = "data/processed/scaler.pkl"
SENSOR_PATH = "data/processed/sensor_ids.npy"


# ============================================================
# LOAD MODEL AND PREPROCESSING ASSETS
# ============================================================

print("Loading Bi-LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Loading sensor IDs...")

sensor_ids = np.load(
    SENSOR_PATH,
    allow_pickle=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Smart Traffic Analytics API",
    description=(
        "REST API for METR-LA traffic speed prediction "
        "using a Bi-LSTM model."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    sequence: list[list[float]]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Smart Traffic Analytics API",
        "model": "Bi-LSTM",
        "sensors": len(sensor_ids),
        "sequence_length": 12,
        "prediction_horizon": "next 5 minutes"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "number_of_sensors": len(sensor_ids)
    }


# ============================================================
# SENSOR ENDPOINT
# ============================================================

@app.get("/sensors")
def get_sensors():

    return {
        "count": len(sensor_ids),
        "sensors": sensor_ids.tolist()
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # Convert input to NumPy array
        sequence = np.asarray(
            request.sequence,
            dtype=np.float32
        )

        # Validate shape
        if sequence.shape != (12, 207):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid sequence shape. "
                    "Expected (12, 207), "
                    f"received {sequence.shape}."
                )
            )

        # Check NaN
        if np.isnan(sequence).any():

            raise HTTPException(
                status_code=400,
                detail="Input contains NaN values."
            )

        # Check infinity
        if np.isinf(sequence).any():

            raise HTTPException(
                status_code=400,
                detail="Input contains infinite values."
            )

        # Add batch dimension
        X = np.expand_dims(
            sequence,
            axis=0
        )

        # Generate prediction
        prediction_scaled = model.predict(
            X,
            verbose=0
        )[0]

        # Convert back to original scale
        prediction_original = scaler.inverse_transform(
            prediction_scaled.reshape(1, -1)
        )[0]

        # Create sensor → prediction mapping
        predictions = {}

        for sensor, value in zip(
            sensor_ids,
            prediction_original
        ):

            predictions[str(sensor)] = float(value)

        return {

            "model": "Bi-LSTM",

            "sequence_length": 12,

            "prediction_horizon": "next 5 minutes",

            "number_of_sensors": len(sensor_ids),

            "predictions": predictions

        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
