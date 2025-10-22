"""
Bitcoin Price Prediction - Gradio MVP
Main application file integrating all models with Gradio interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# Data loading
def load_bitcoin_data():
    """Load Bitcoin price data from local CSV file"""
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "bitcoin_2020-10-22_2025-10-21.csv"
    )
    df = pd.read_csv(data_path, parse_dates=["Start"])
    df = df.sort_values("Start")  # Ensure chronological order
    df.set_index("Start", inplace=True)

    # Use Close price as the target (equivalent to Closing Price)
    bitcoin_prices = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"})
    return bitcoin_prices


# Global data
BITCOIN_PRICES = load_bitcoin_data()
PRICES = BITCOIN_PRICES["Price"].to_numpy()
TIMESTEPS = BITCOIN_PRICES.index.to_numpy()

# Create multivariate data (price + returns)
BITCOIN_PRICES["Return"] = BITCOIN_PRICES["Price"].pct_change().fillna(0)
MULTIVARIATE_DATA = BITCOIN_PRICES[["Price", "Return"]].to_numpy()


# Utility functions
def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
    """Utility function for plotting time series"""
    plt.figure(figsize=(10, 7))
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    return plt


def create_plotly_time_series(
    timesteps, values, predictions=None, pred_label="Predictions"
):
    """Create interactive plotly chart for time series"""
    fig = make_subplots(rows=1, cols=1)

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=values,
            mode="lines",
            name="Actual Prices",
            line=dict(color="blue", width=2),
        )
    )

    # Plot predictions if provided
    if predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=timesteps[len(timesteps) - len(predictions) :],
                y=predictions,
                mode="lines",
                name=pred_label,
                line=dict(color="red", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title="Bitcoin Price Time Series",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
    )

    return fig


def create_comparison_chart(historical_timesteps, historical_prices, all_predictions):
    """Create comparison chart showing all models' future predictions"""
    fig = go.Figure()

    # Plot historical prices
    fig.add_trace(
        go.Scatter(
            x=historical_timesteps,
            y=historical_prices,
            mode="lines",
            name="Historical Prices",
            line=dict(color="blue", width=2),
        )
    )

    # Define colors for different models
    colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    # Plot each model's predictions
    for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
        if pred_data["future_predictions"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=pred_data["future_timesteps"],
                    y=pred_data["future_predictions"],
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=colors[idx % len(colors)], width=2, dash="dash"),
                )
            )

    fig.update_layout(
        title="Bitcoin Price Prediction - 7 Days Future Comparison (All Models)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )

    return fig


# Windowing functions
def get_labelled_windows(x, horizon=1):
    """Create windows and labels from time series data"""
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size=7, horizon=1):
    """Create sliding windows from time series data"""
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = (
        window_step
        + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    )
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels


def make_train_test_splits(windows, labels, test_split=0.2):
    """Split windows and labels into train/test sets"""
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


# Multivariate windowing functions
def get_labelled_windows_mv(x, window_size=7, horizon=1):
    """Create windows and labels from multivariate time series data"""
    windows = []
    labels = []
    for i in range(len(x) - window_size - horizon + 1):
        window = x[i : i + window_size]
        label = x[i + window_size : i + window_size + horizon, 0]  # predict price only
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)


# Evaluation functions
def mean_absolute_scaled_error(y_true, y_pred):
    """Calculate Mean Absolute Scaled Error"""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    """Evaluate predictions using multiple metrics"""
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_metric = tf.keras.metrics.MeanSquaredError()
    mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()

    mae_metric.update_state(y_true, y_pred)
    mse_metric.update_state(y_true, y_pred)
    mape_metric.update_state(y_true, y_pred)

    rmse = tf.sqrt(mse_metric.result())
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {
        "mae": mae_metric.result().numpy(),
        "mse": mse_metric.result().numpy(),
        "rmse": rmse.numpy(),
        "mape": mape_metric.result().numpy(),
        "mase": mase.numpy(),
    }


# Model prediction function
def make_preds(model, input_data):
    """Make predictions with a trained model"""
    forecast = model.predict(input_data, verbose=0)
    return tf.squeeze(forecast)


def predict_future_recursive(model, last_window, n_future=7, window_size=7):
    """Recursively predict future prices using a trained model"""
    future_preds = []
    current_window = last_window.copy()

    for _ in range(n_future):
        # Predict next value
        pred = model.predict(current_window.reshape(1, -1), verbose=0)
        if len(pred.shape) > 1:
            pred_value = pred[0][0]
        else:
            pred_value = pred[0]

        future_preds.append(float(pred_value))

        # Update window: remove oldest, add prediction
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred_value

    return np.array(future_preds)


def predict_future_multivariate(model, last_window_mv, n_future=7, window_size=7):
    """Recursively predict future prices using a trained multivariate model"""
    future_preds = []
    current_window = last_window_mv.copy()

    for _ in range(n_future):
        # Predict next value
        pred = model.predict(current_window.reshape(1, window_size, 2), verbose=0)
        if len(pred.shape) > 1:
            pred_value = pred[0][0]
        else:
            pred_value = pred[0]

        future_preds.append(float(pred_value))

        # Calculate return for the new prediction
        last_price = current_window[-1, 0]
        new_return = (pred_value - last_price) / last_price if last_price != 0 else 0

        # Update window: remove oldest, add prediction with return
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1] = [pred_value, new_return]

    return np.array(future_preds)


# Model implementations
def model_0_naive_forecast(window_size=7, horizon=1):
    """Model 0: Naive Forecast (Baseline)"""
    # Create naive forecast: predict next value as current value
    split_size = int(0.8 * len(PRICES))
    train_prices = PRICES[:split_size]
    test_prices = PRICES[split_size:]

    # Naive forecast: y_t = y_{t-1}
    naive_forecast = test_prices[:-1]

    # Get corresponding timesteps
    test_timesteps = TIMESTEPS[split_size:]
    pred_timesteps = test_timesteps[1:]

    # Evaluate
    results = evaluate_preds(y_true=test_prices[1:], y_pred=naive_forecast)

    # Future prediction: just repeat the last known price
    last_price = PRICES[-1]
    future_preds = np.array([last_price] * 7)

    # Create future timesteps
    last_date = TIMESTEPS[-1]
    future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

    return {
        "predictions": naive_forecast,
        "actual": test_prices[1:],
        "timesteps": pred_timesteps,
        "metrics": results,
        "model_name": "Naive Forecast",
        "future_predictions": future_preds,
        "future_timesteps": future_timesteps,
    }


def model_5_lstm(window_size=7, horizon=1):
    """Model 5: LSTM Model"""
    try:
        # Prepare windowed data
        full_windows, full_labels = make_windows(
            PRICES, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build model
        tf.random.set_seed(42)
        inputs = tf.keras.layers.Input(shape=(window_size,))
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
        x = tf.keras.layers.LSTM(128, activation="relu")(x)
        output = tf.keras.layers.Dense(horizon)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_lstm")

        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

        # Train model
        model.fit(
            train_windows,
            train_labels,
            batch_size=128,
            epochs=10,
            verbose=0,
            validation_data=(test_windows, test_labels),
        )

        # Make predictions
        predictions = make_preds(model, test_windows)

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(predictions)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=predictions)

        # Future prediction
        last_window = PRICES[-window_size:]
        future_preds = predict_future_recursive(
            model, last_window, n_future=7, window_size=window_size
        )

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"LSTM Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"LSTM Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


def model_6_multivariate(window_size=7, horizon=1):
    """Model 6: Multivariate Dense Model"""
    try:
        # Prepare windowed multivariate data
        full_windows, full_labels = get_labelled_windows_mv(
            MULTIVARIATE_DATA, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build model
        tf.random.set_seed(42)
        inputs = tf.keras.layers.Input(
            shape=(window_size, 2)
        )  # 2 features: price and return
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        output = tf.keras.layers.Dense(horizon)(x)
        model = tf.keras.Model(
            inputs=inputs, outputs=output, name="model_6_multivariate_dense"
        )

        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

        # Train model
        model.fit(
            train_windows,
            train_labels,
            batch_size=128,
            epochs=10,
            verbose=0,
            validation_data=(test_windows, test_labels),
        )

        # Make predictions
        predictions = make_preds(model, test_windows)

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(predictions)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=predictions)

        # Future prediction
        last_window_mv = MULTIVARIATE_DATA[-window_size:]
        future_preds = predict_future_multivariate(
            model, last_window_mv, n_future=7, window_size=window_size
        )

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"Multivariate Dense Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Multivariate Dense Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


# N-BEATS Block for Model 7
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, units, input_size, horizon, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(units, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units, activation="relu")
        self.fc3 = tf.keras.layers.Dense(units, activation="relu")
        self.fc4 = tf.keras.layers.Dense(units, activation="relu")
        self.theta_layer = tf.keras.layers.Dense(
            units, activation="linear"
        )  # Changed to units
        self.backcast_layer = tf.keras.layers.Dense(
            input_size, activation="linear"
        )  # Changed to input_size
        self.forecast_layer = tf.keras.layers.Dense(horizon, activation="linear")

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        theta = self.theta_layer(x)
        backcast = self.backcast_layer(theta)
        forecast = self.forecast_layer(theta)
        return backcast, forecast


def model_7_nbeats(window_size=7, horizon=1):
    """Model 7: N-BEATS Model"""
    try:
        # Prepare windowed data
        full_windows, full_labels = make_windows(
            PRICES, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build N-BEATS model
        tf.random.set_seed(42)
        inputs = tf.keras.layers.Input(shape=(window_size,))
        x = inputs
        for _ in range(3):  # Stack 3 N-BEATS blocks
            backcast, forecast = NBeatsBlock(128, window_size, horizon)(x)
            x = x - backcast  # Residual connection
        outputs = forecast

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model_7_nbeats")
        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

        # Train model
        model.fit(
            train_windows,
            train_labels,
            batch_size=128,
            epochs=10,
            verbose=0,
            validation_data=(test_windows, test_labels),
        )

        # Make predictions
        predictions = make_preds(model, test_windows)

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(predictions)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=predictions)

        # Future prediction
        last_window = PRICES[-window_size:]
        future_preds = predict_future_recursive(
            model, last_window, n_future=7, window_size=window_size
        )

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"N-BEATS Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"N-BEATS Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


def model_8_ensemble(window_size=7, horizon=1):
    """Model 8: Ensemble Model"""
    try:
        # Prepare windowed data
        full_windows, full_labels = make_windows(
            PRICES, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build and train multiple models
        tf.random.set_seed(42)

        # Dense Model
        dense_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128, activation="relu", input_shape=(window_size,)
                ),
                tf.keras.layers.Dense(horizon),
            ]
        )
        dense_model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
        dense_model.fit(
            train_windows, train_labels, epochs=10, verbose=0, batch_size=128
        )

        # Conv1D Model
        conv1d_model = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape((window_size, 1), input_shape=(window_size,)),
                tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(horizon),
            ]
        )
        conv1d_model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
        conv1d_model.fit(
            train_windows, train_labels, epochs=10, verbose=0, batch_size=128
        )

        # LSTM Model
        lstm_model = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=1), input_shape=(window_size,)
                ),
                tf.keras.layers.LSTM(128, activation="relu"),
                tf.keras.layers.Dense(horizon),
            ]
        )
        lstm_model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
        lstm_model.fit(
            train_windows, train_labels, epochs=10, verbose=0, batch_size=128
        )

        # Make predictions with each model
        dense_preds = make_preds(dense_model, test_windows)
        conv1d_preds = make_preds(conv1d_model, test_windows)
        lstm_preds = make_preds(lstm_model, test_windows)

        # Ensemble: average predictions
        ensemble_preds = (dense_preds + conv1d_preds + lstm_preds) / 3

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(ensemble_preds)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=ensemble_preds)

        # Future prediction - ensemble of all three models
        last_window = PRICES[-window_size:]
        dense_future = predict_future_recursive(
            dense_model, last_window, n_future=7, window_size=window_size
        )
        conv1d_future = predict_future_recursive(
            conv1d_model, last_window, n_future=7, window_size=window_size
        )
        lstm_future = predict_future_recursive(
            lstm_model, last_window, n_future=7, window_size=window_size
        )
        future_preds = (dense_future + conv1d_future + lstm_future) / 3

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": ensemble_preds.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"Ensemble Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Ensemble Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


def model_1_dense(window_size=7, horizon=1):
    """Model 1: Dense Model (Window=7, Horizon=1)"""
    try:
        # Prepare windowed data
        full_windows, full_labels = make_windows(
            PRICES, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build model
        tf.random.set_seed(42)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(horizon, activation="linear"),
            ],
            name="model_1_dense",
        )

        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

        # Train model (reduced epochs for demo)
        model.fit(
            x=train_windows,
            y=train_labels,
            epochs=10,  # Reduced for faster demo
            verbose=0,
            batch_size=128,
            validation_data=(test_windows, test_labels),
        )

        # Make predictions
        predictions = make_preds(model, test_windows)

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(predictions)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=predictions)

        # Future prediction
        last_window = PRICES[-window_size:]
        future_preds = predict_future_recursive(
            model, last_window, n_future=7, window_size=window_size
        )

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"Dense Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Dense Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


def model_4_conv1d(window_size=7, horizon=1):
    """Model 4: Conv1D Model"""
    try:
        # Prepare windowed data
        full_windows, full_labels = make_windows(
            PRICES, window_size=window_size, horizon=horizon
        )
        train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
            full_windows, full_labels
        )

        # Build model
        tf.random.set_seed(42)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                tf.keras.layers.Conv1D(
                    filters=128, kernel_size=5, padding="causal", activation="relu"
                ),
                tf.keras.layers.Dense(horizon),
            ],
            name="model_4_conv1D",
        )

        model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

        # Train model
        model.fit(
            train_windows,
            train_labels,
            batch_size=128,
            epochs=10,
            verbose=0,
            validation_data=(test_windows, test_labels),
        )

        # Make predictions
        predictions = make_preds(model, test_windows)

        # Get timesteps for plotting
        split_size = int(len(full_windows) * 0.8)
        test_timesteps = TIMESTEPS[
            window_size + split_size : window_size + split_size + len(predictions)
        ]

        # Evaluate
        results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=predictions)

        # Future prediction
        last_window = PRICES[-window_size:]
        future_preds = predict_future_recursive(
            model, last_window, n_future=7, window_size=window_size
        )

        # Create future timesteps
        last_date = TIMESTEPS[-1]
        future_timesteps = pd.date_range(last_date, periods=8, freq="D")[1:]

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"Conv1D Model (Window={window_size}, Horizon={horizon})",
            "future_predictions": future_preds,
            "future_timesteps": future_timesteps,
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Conv1D Model Error: {str(e)}",
            "future_predictions": None,
            "future_timesteps": None,
        }


# Gradio interface functions
def compare_all_models():
    """Train all models and compare their 7-day future predictions"""
    try:
        print("Training all models...")
        all_results = {}

        # Train all models
        models_to_run = [
            ("Naive Forecast", model_0_naive_forecast),
            ("Dense (W=7, H=1)", lambda: model_1_dense(window_size=7, horizon=1)),
            ("Dense (W=30, H=1)", lambda: model_1_dense(window_size=30, horizon=1)),
            ("Conv1D", model_4_conv1d),
            ("LSTM", model_5_lstm),
            ("Multivariate", model_6_multivariate),
            ("N-BEATS", model_7_nbeats),
            ("Ensemble", model_8_ensemble),
        ]

        for model_name, model_func in models_to_run:
            print(f"Training {model_name}...")
            result = model_func()
            if result["future_predictions"] is not None:
                all_results[model_name] = result

        # Create comparison chart
        fig = create_comparison_chart(
            TIMESTEPS[-60:],  # Last 60 days of historical data
            PRICES[-60:],
            all_results,
        )

        # Create metrics comparison table
        metrics_data = []
        for model_name, result in all_results.items():
            metrics = result["metrics"]
            future_mean = (
                np.mean(result["future_predictions"])
                if result["future_predictions"] is not None
                else 0
            )
            metrics_data.append(
                [
                    model_name,
                    f"{metrics.get('mae', 0):.2f}",
                    f"{metrics.get('rmse', 0):.2f}",
                    f"{metrics.get('mape', 0):.2f}",
                    f"{metrics.get('mase', 0):.2f}",
                    f"${future_mean:.2f}",
                ]
            )

        metrics_df = pd.DataFrame(
            metrics_data,
            columns=[
                "Model",
                "Avg Error ($)",
                "Error Range ($)",
                "Error %",
                "vs Simple Guess",
                "Avg 7-Day Prediction",
            ],
        )

        # Create detailed prediction table
        prediction_table = []
        for model_name, result in all_results.items():
            if result["future_predictions"] is not None:
                preds = result["future_predictions"]
                prediction_table.append(
                    [
                        model_name,
                        f"${preds[0]:,.2f}",
                        f"${preds[1]:,.2f}",
                        f"${preds[2]:,.2f}",
                        f"${preds[3]:,.2f}",
                        f"${preds[4]:,.2f}",
                        f"${preds[5]:,.2f}",
                        f"${preds[6]:,.2f}",
                    ]
                )

        pred_df = pd.DataFrame(
            prediction_table,
            columns=[
                "Model",
                "Tomorrow",
                "Day 2",
                "Day 3",
                "Day 4",
                "Day 5",
                "Day 6",
                "Day 7",
            ],
        )

        # Find best model by MAE and calculate stats
        best_model_idx = metrics_df["Avg Error ($)"].astype(float).idxmin()
        best_model = metrics_df.iloc[best_model_idx]["Model"]
        best_mae = metrics_df.iloc[best_model_idx]["Avg Error ($)"]

        # Calculate prediction spread
        all_day7_preds = [
            result["future_predictions"][-1]
            for result in all_results.values()
            if result["future_predictions"] is not None
        ]
        min_pred = min(all_day7_preds)
        max_pred = max(all_day7_preds)
        avg_pred = np.mean(all_day7_preds)
        spread_percent = ((max_pred - min_pred) / avg_pred) * 100

        current_price = PRICES[-1]
        avg_change = ((avg_pred - current_price) / current_price) * 100
        consensus = (
            "📈 BULLISH (Prices Going Up)"
            if avg_change > 1
            else (
                "📉 BEARISH (Prices Going Down)"
                if avg_change < -1
                else "➡️ STABLE (Little Change)"
            )
        )

        metrics_text = f"""
        ## 📊 All Models Comparison
        
        ### 🎯 Which Model is Most Accurate?
        
        **🏆 Winner: {best_model}**
        - Has the smallest average error: ${best_mae}
        - This model's past predictions were closest to actual prices
        
        ### 📈 Accuracy Scores Explained:
        
        {metrics_df.to_markdown(index=False)}
        
        **How to read this table:**
        - **Avg Error ($)**: Lower = better (how much predictions miss by, on average)
        - **Error Range ($)**: Lower = better (typical range of errors)
        - **Error %**: Lower = better (percentage off from actual price)
        - **vs Simple Guess**: Less than 1.0 = better than just guessing "tomorrow = today"
        
        ---
        
        ### 🔮 What Do Models Predict for the Next 7 Days?
        
        **Current Bitcoin Price: ${current_price:,.2f}**
        
        {pred_df.to_markdown(index=False)}
        
        ### 💡 Consensus Summary:
        
        - **Market Direction**: {consensus}
        - **Day 7 Average Prediction**: ${avg_pred:,.2f} ({avg_change:+.2f}% from current)
        - **Prediction Range**: ${min_pred:,.2f} - ${max_pred:,.2f}
        - **Agreement Level**: {spread_percent:.1f}% spread
          - **Low spread (<5%)** = Models mostly agree → Higher confidence
          - **High spread (>10%)** = Models disagree → More uncertainty
        
        ---
        
        **⚠️ Important Disclaimer:**
        These are AI predictions based on historical patterns. Cryptocurrency markets are highly volatile 
        and unpredictable. Use these predictions as one of many tools, not as financial advice!
        """

        return fig, metrics_text

    except Exception as e:
        return None, f"Error comparing models: {str(e)}"


def run_model(model_name, window_size=7):
    """Run selected model with given parameters"""
    try:
        horizon = 1  # All models use horizon=1

        if model_name == "Model 0: Naive Forecast":
            result = model_0_naive_forecast(window_size=window_size, horizon=horizon)
        elif model_name == "Model 1: Dense":
            result = model_1_dense(window_size=window_size, horizon=horizon)
        elif model_name == "Model 4: Conv1D":
            result = model_4_conv1d(window_size=window_size, horizon=horizon)
        elif model_name == "Model 5: LSTM":
            result = model_5_lstm(window_size=window_size, horizon=horizon)
        elif model_name == "Model 6: Multivariate Dense":
            result = model_6_multivariate(window_size=window_size, horizon=horizon)
        elif model_name == "Model 7: N-BEATS":
            result = model_7_nbeats(window_size=window_size, horizon=horizon)
        elif model_name == "Model 8: Ensemble":
            result = model_8_ensemble(window_size=window_size, horizon=horizon)

        # Create plot showing both test predictions and future predictions
        if result["predictions"] is not None:
            # Create a figure with test predictions and future predictions
            fig = go.Figure()

            # Plot actual test prices
            fig.add_trace(
                go.Scatter(
                    x=result["timesteps"],
                    y=result["actual"],
                    mode="lines",
                    name="Actual Test Prices",
                    line=dict(color="green", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=result["timesteps"],
                    y=result["predictions"],
                    mode="lines",
                    name="Test Predictions",
                    line=dict(color="orange", width=2, dash="dash"),
                )
            )

            # Plot future predictions if available
            if result.get("future_predictions") is not None:
                fig.add_trace(
                    go.Scatter(
                        x=result["future_timesteps"],
                        y=result["future_predictions"],
                        mode="lines+markers",
                        name="7-Day Future Prediction",
                        line=dict(color="red", width=3, dash="dot"),
                        marker=dict(size=8),
                    )
                )

            fig.update_layout(
                title=f"{result['model_name']} - Predictions",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
            )
        else:
            fig = None

        # Format metrics for display
        future_text = ""
        if result.get("future_predictions") is not None:
            future_mean = np.mean(result["future_predictions"])
            future_min = np.min(result["future_predictions"])
            future_max = np.max(result["future_predictions"])
            current_price = PRICES[-1]
            price_change = ((future_mean - current_price) / current_price) * 100
            direction = "📈 UP" if price_change > 0 else "📉 DOWN"

            future_text = f"""
        
        ### 🔮 7-Day Future Prediction:
        - **Average Predicted Price**: ${future_mean:,.2f}
        - **Predicted Range**: ${future_min:,.2f} - ${future_max:,.2f}
        - **Current Price**: ${current_price:,.2f}
        - **Expected Change**: {direction} {abs(price_change):.2f}%
        """

        metrics_text = f"""
        ## {result['model_name']}
        
        ### ✅ How Accurate Was This Model?
        
        These scores show how close the model's predictions (orange line) were to the actual prices (green line):
        
        - **Average Error**: ${result['metrics'].get('mae', 0):,.2f}
          - On average, predictions were off by this amount
          - **Lower is better** - means predictions are closer to reality
        
        - **Typical Error Range**: ${result['metrics'].get('rmse', 0):,.2f}
          - Most predictions fall within this range of the actual price
          - **Lower is better** - means more consistent predictions
        
        - **Error Percentage**: {result['metrics'].get('mape', 0):.2f}%
          - What percent off the predictions typically are
          - **Lower is better** - under 5% is good, under 2% is excellent
        
        - **Accuracy vs Simple Guess**: {result['metrics'].get('mase', 0):.2f}x
          - Compares to just guessing "tomorrow = today"
          - **Less than 1.0 is good** - means this model beats simple guessing
          - **More than 1.0** - simple guessing would be more accurate
        {future_text}
        """

        return fig, metrics_text

    except Exception as e:
        return None, f"Error running model: {str(e)}"


# Create Gradio interface
def create_gradio_interface():
    """Create and launch Gradio interface"""

    # Model options
    model_options = [
        "Model 0: Naive Forecast",
        "Model 1: Dense",
        "Model 4: Conv1D",
        "Model 5: LSTM",
        "Model 6: Multivariate Dense",
        "Model 7: N-BEATS",
        "Model 8: Ensemble",
    ]

    with gr.Blocks(
        title="Bitcoin Price Prediction", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# 🪙 Bitcoin Price Prediction - Deep Learning Models Comparison")
        gr.Markdown(
            "*Compare multiple time series forecasting models with 7-day future predictions*"
        )

        with gr.Tabs():
            # Tab 1: Individual Model Testing
            with gr.Tab("Individual Model"):
                gr.Markdown(
                    """
                ### 📈 How to Use:
                1. **Select a model** from the dropdown below
                2. **Adjust the window size** - how many days of history the model looks at
                3. **Click "Train & Predict"** - the model will learn from historical Bitcoin data
                4. **View results** - see how accurate the model is and what it predicts for the next 7 days
                """
                )

                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    value="Model 0: Naive Forecast",
                    label="Select Model",
                )

                window_size_slider = gr.Slider(
                    minimum=3,
                    maximum=60,
                    value=7,
                    step=1,
                    label="Window Size (days)",
                    info="How many days of historical data the model uses to make predictions. Smaller = faster/simpler, Larger = more context but slower",
                )

                run_btn = gr.Button("🚀 Train & Predict", variant="primary", size="lg")

                plot_output = gr.Plot(label="Bitcoin Price Predictions")

                gr.Markdown(
                    """
                ### 📊 How to Read the Chart:
                
                **🟢 Green Line - What Actually Happened**
                - Shows real Bitcoin prices from the recent past
                - Example: If Bitcoin was $50,000 on Jan 1st, the green line shows $50,000
                
                **🟠 Orange Line - How Well the Model Predicted the Past**
                - Shows what the model predicted for prices we already know
                - This helps us understand if the model is reliable
                - Example: If green shows $50,000 and orange shows $49,500, the model was off by $500
                
                **🔴 Red Line - Future Forecast (Next 7 Days)**
                - This is the actual prediction for tomorrow and the next 6 days
                - These are unknown prices - we don't know yet if they'll be accurate
                - Example: If red shows $51,000 for tomorrow, the model thinks Bitcoin will be worth $51,000
                
                **💡 Why Compare Orange vs Green?**
                The closer the orange and green lines are, the more accurate the model has been in the past, 
                which suggests it might be more reliable for future predictions (red line).
                """
                )

                metrics_output = gr.Markdown(label="Model Accuracy Scores")

                # Event handler for individual model
                run_btn.click(
                    fn=run_model,
                    inputs=[model_dropdown, window_size_slider],
                    outputs=[plot_output, metrics_output],
                )

            # Tab 2: Compare All Models
            with gr.Tab("Compare All Models"):
                gr.Markdown(
                    """
                ### 🏆 Compare All Models Side-by-Side
                
                **What This Does:**
                - Trains 7 different AI models on the same Bitcoin price history
                - Shows each model's prediction for the next 7 days
                - Helps you see which models agree or disagree about the future
                
                **⏱️ Note:** This takes 2-4 minutes to train all 7 models
                """
                )

                compare_btn = gr.Button(
                    "🚀 Train All Models & Compare", variant="primary", size="lg"
                )

                comparison_plot = gr.Plot(
                    label="All Models - 7 Day Future Predictions Comparison"
                )

                gr.Markdown(
                    """
                ### 📊 How to Read the Comparison Chart:
                
                - **Each colored line** = One model's prediction for the next 7 days
                - **Lines close together?** Models agree on the future direction → More confidence
                - **Lines spread apart?** Models disagree → More uncertainty about future prices
                
                **Example:**
                - If 5 out of 7 models predict Bitcoin will be $52,000 in 7 days, that's more reliable
                - If predictions range from $48,000 to $56,000, there's high uncertainty
                """
                )

                comparison_metrics = gr.Markdown(label="Which Model is Most Accurate?")

                # Event handler for comparison
                compare_btn.click(
                    fn=compare_all_models, outputs=[comparison_plot, comparison_metrics]
                )

        gr.Markdown(
            """
        ---
        ### 🤖 About the AI Models:
        
        Each model uses a different approach to learn from Bitcoin's price history:
        
        - **Naive Forecast**: Simplest approach - just assumes tomorrow's price = today's price (baseline to beat)
        - **Dense Models**: Standard neural networks that find patterns in price sequences (like 7-day or 30-day windows)
        - **Conv1D**: Designed to detect trends and patterns in time-based data (similar to how images are analyzed)
        - **LSTM**: Specialized for remembering long-term patterns (can "remember" what happened weeks ago)
        - **Multivariate**: Looks at both price AND rate of change for smarter predictions
        - **N-BEATS**: Advanced architecture specifically designed for forecasting with interpretable results
        - **Ensemble**: Combines Dense, Conv1D, and LSTM - like getting a second (and third) opinion
        
        ---
        
        **📅 Data Source:** Bitcoin daily closing prices from October 22, 2020 to October 21, 2025 (5 years)
        """
        )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
