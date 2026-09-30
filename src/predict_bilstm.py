import numpy as np
import joblib
import tensorflow as tf


MODEL_PATH = "models/bilstm_best.keras"
SCALER_PATH = "data/processed/scaler.pkl"


def load_model_and_scaler():
    """
    Load the trained Bi-LSTM model and
    the scaler used during preprocessing.
    """

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


def predict_from_sequence(sequence):
    """
    Generate a traffic-speed prediction.

    Parameters
    ----------
    sequence : numpy array
        Shape must be:
        (12, 207)

    Returns
    -------
    prediction_scaled : numpy array
        Normalized prediction, shape (207,)

    prediction_original : numpy array
        Prediction converted back to
        original traffic-speed scale.
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (12, 207):
        raise ValueError(
            f"Expected input shape (12, 207), "
            f"but received {sequence.shape}"
        )

    # Add batch dimension
    X = np.expand_dims(
        sequence,
        axis=0
    )

    model, scaler = load_model_and_scaler()

    # Generate prediction
    prediction_scaled = model.predict(
        X,
        verbose=0
    )[0]

    # Convert normalized prediction
    # back to original traffic-speed scale
    prediction_original = scaler.inverse_transform(
        prediction_scaled.reshape(1, -1)
    )[0]

    return (
        prediction_scaled,
        prediction_original
    )


if __name__ == "__main__":

    print("=" * 60)
    print("BI-LSTM INFERENCE TEST")
    print("=" * 60)

    # Load one real test sequence
    X_test = np.load(
        "data/processed/X_test.npy"
    )

    print(
        "\nX_test shape:",
        X_test.shape
    )

    # Select first test sequence
    sample = X_test[0]

    print(
        "Sample shape:",
        sample.shape
    )

    # Generate prediction
    scaled_prediction, original_prediction = (
        predict_from_sequence(sample)
    )

    print(
        "\nScaled prediction shape:",
        scaled_prediction.shape
    )

    print(
        "Original prediction shape:",
        original_prediction.shape
    )

    print("\nFirst 10 predicted sensor speeds:")

    for i, value in enumerate(
        original_prediction[:10]
    ):
        print(
            f"Sensor {i}: {value:.2f}"
        )

    print("\n" + "=" * 60)
    print("INFERENCE TEST COMPLETE")
    print("=" * 60)
