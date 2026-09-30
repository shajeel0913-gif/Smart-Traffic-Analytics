# Smart Traffic Analytics

A deep learning-based traffic speed forecasting project using the METR-LA traffic dataset. The project compares LSTM and Bidirectional LSTM (Bi-LSTM) models for short-term traffic speed prediction and provides API and dashboard integration using FastAPI and Streamlit.

## Project Overview

Traffic congestion and changing traffic conditions make short-term traffic forecasting an important problem in intelligent transportation systems.

This project uses historical traffic speed data from the METR-LA dataset to predict future traffic speed. Two deep learning architectures are implemented and compared:

- Long Short-Term Memory (LSTM)
- Bidirectional Long Short-Term Memory (Bi-LSTM)

The project also includes:

- Data preprocessing and exploratory data analysis
- Time-series sequence generation
- Model training and evaluation
- Prediction visualization
- FastAPI prediction API
- Streamlit dashboard

## Dataset

The project uses the METR-LA traffic speed dataset.

The dataset contains traffic measurements from 207 sensors in the Los Angeles metropolitan area.

Traffic observations are recorded at 5-minute intervals.

The project uses historical traffic observations to predict the next traffic speed value.

The raw dataset is not included in this repository because of its size.

## Methodology

The project follows this workflow:

METR-LA Dataset → Data Loading → EDA → Preprocessing → Sequence Creation → Train/Validation/Test Split → LSTM and Bi-LSTM → Evaluation → Predictions → FastAPI and Streamlit

## Models

### LSTM

The LSTM model is used to learn temporal patterns from historical traffic speed observations.

### Bi-LSTM

The Bidirectional LSTM model processes the input sequence in both forward and backward directions to capture temporal relationships within the input sequence.

## Model Performance

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| LSTM | 0.050020 | 0.006393 | 0.079954 | 0.830173 |
| Bi-LSTM | 0.050081 | 0.006303 | 0.079393 | 0.832548 |

Detailed results are available in:

`results/tables/model_comparison.csv`

## Project Structure

```text
Smart_Traffic_Analytics/
├── dashboard/
│   └── app.py
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── models/
│   ├── bilstm_best.keras
│   ├── bilstm_final.keras
│   ├── lstm_best.keras
│   └── lstm_final.keras
├── notebooks/
├── reports/
├── results/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── src/
│   ├── api.py
│   ├── collect_traffic.py
│   ├── compare_models.py
│   ├── create_sequences.py
│   ├── eda.py
│   ├── evaluate_bilstm.py
│   ├── evaluate_lstm.py
│   ├── plot_bilstm_predictions.py
│   ├── plot_bilstm_training.py
│   ├── plot_lstm_predictions.py
│   ├── plot_lstm_training.py
│   ├── predict_bilstm.py
│   ├── preprocess.py
│   ├── test_api_prediction.py
│   ├── train_bilstm.py
│   ├── train_lstm.py
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
