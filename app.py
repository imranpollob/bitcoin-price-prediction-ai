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


def model_5_lstm():
    """Model 5: LSTM Model (Window=7, Horizon=1)"""
    try:
        window_size, horizon = 7, 1
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
            "model_name": "LSTM Model (Window=7, Horizon=1)",
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


def model_6_multivariate():
    """Model 6: Multivariate Dense Model (Window=7, Horizon=1)"""
    try:
        window_size, horizon = 7, 1
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
            "model_name": "Multivariate Dense Model (Window=7, Horizon=1)",
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


def model_7_nbeats():
    """Model 7: N-BEATS Model (Window=7, Horizon=1)"""
    try:
        window_size, horizon = 7, 1
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
            "model_name": "N-BEATS Model (Window=7, Horizon=1)",
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


def model_8_ensemble():
    """Model 8: Ensemble Model (Window=7, Horizon=1)"""
    try:
        window_size, horizon = 7, 1
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
            "model_name": "Ensemble Model (Window=7, Horizon=1)",
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


def model_2_dense():
    """Model 2: Dense Model (Window=30, Horizon=1)"""
    return model_1_dense(window_size=30, horizon=1)


def model_3_dense():
    """Model 3: Dense Model (Window=30, Horizon=7)"""
    return model_1_dense(window_size=30, horizon=7)


def model_4_conv1d():
    """Model 4: Conv1D Model (Window=7, Horizon=1)"""
    try:
        window_size, horizon = 7, 1
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
            "model_name": "Conv1D Model (Window=7, Horizon=1)",
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
            columns=["Model", "MAE", "RMSE", "MAPE", "MASE", "Avg 7-Day Pred"],
        )

        # Create detailed prediction table
        prediction_table = []
        for model_name, result in all_results.items():
            if result["future_predictions"] is not None:
                preds = result["future_predictions"]
                prediction_table.append(
                    [
                        model_name,
                        f"${preds[0]:.2f}",
                        f"${preds[1]:.2f}",
                        f"${preds[2]:.2f}",
                        f"${preds[3]:.2f}",
                        f"${preds[4]:.2f}",
                        f"${preds[5]:.2f}",
                        f"${preds[6]:.2f}",
                    ]
                )

        pred_df = pd.DataFrame(
            prediction_table,
            columns=[
                "Model",
                "Day 1",
                "Day 2",
                "Day 3",
                "Day 4",
                "Day 5",
                "Day 6",
                "Day 7",
            ],
        )

        # Find best model by MAE
        best_model_idx = metrics_df["MAE"].astype(float).idxmin()
        best_model = metrics_df.iloc[best_model_idx]["Model"]
        best_mae = metrics_df.iloc[best_model_idx]["MAE"]

        metrics_text = f"""
        ## 📊 Model Comparison Results
        
        ### Test Set Performance Metrics:
        
        {metrics_df.to_markdown(index=False)}
        
        ### 7-Day Future Predictions (Starting from {TIMESTEPS[-1].strftime('%Y-%m-%d')}):
        
        {pred_df.to_markdown(index=False)}
        
        ---
        
        **🏆 Best Model:** {best_model} (MAE: {best_mae})
        
        **Note:** 
        - Models are trained on 80% of data and evaluated on 20% test set
        - Lower MAE, RMSE, MAPE, and MASE indicate better performance
        - Future predictions are based on the most recent 7-day window
        - Current Bitcoin price (last known): ${PRICES[-1]:.2f}
        """

        return fig, metrics_text

    except Exception as e:
        return None, f"Error comparing models: {str(e)}"


def run_model(model_name):
    """Run selected model with given parameters"""
    try:
        if model_name == "Model 0: Naive Forecast":
            result = model_0_naive_forecast()
        elif model_name == "Model 1: Dense (Window=7, Horizon=1)":
            result = model_1_dense()
        elif model_name == "Model 2: Dense (Window=30, Horizon=1)":
            result = model_2_dense()
        elif model_name == "Model 3: Dense (Window=30, Horizon=7)":
            result = model_3_dense()
        elif model_name == "Model 4: Conv1D (Window=7, Horizon=1)":
            result = model_4_conv1d()
        elif model_name == "Model 5: LSTM (Window=7, Horizon=1)":
            result = model_5_lstm()
        elif model_name == "Model 6: Multivariate Dense (Window=7, Horizon=1)":
            result = model_6_multivariate()
        elif model_name == "Model 7: N-BEATS (Window=7, Horizon=1)":
            result = model_7_nbeats()
        elif model_name == "Model 8: Ensemble (Window=7, Horizon=1)":
            result = model_8_ensemble()

        # Create plot showing both test predictions and future predictions
        if result["predictions"] is not None:
            # Create a figure with historical data, test predictions, and future predictions
            fig = go.Figure()

            # Plot historical prices (last 90 days)
            historical_timesteps = TIMESTEPS[-90:]
            historical_prices = PRICES[-90:]
            fig.add_trace(
                go.Scatter(
                    x=historical_timesteps,
                    y=historical_prices,
                    mode="lines",
                    name="Historical Prices",
                    line=dict(color="blue", width=2),
                )
            )

            # Plot test predictions
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
            future_text = f"\n        - **7-Day Avg Prediction**: ${future_mean:.2f}"

        metrics_text = f"""
        **{result['model_name']} Results:**

        - **MAE**: {result['metrics'].get('mae', 'N/A'):.4f}
        - **RMSE**: {result['metrics'].get('rmse', 'N/A'):.4f}
        - **MAPE**: {result['metrics'].get('mape', 'N/A'):.4f}
        - **MASE**: {result['metrics'].get('mase', 'N/A'):.4f}{future_text}
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
        "Model 1: Dense (Window=7, Horizon=1)",
        "Model 2: Dense (Window=30, Horizon=1)",
        "Model 3: Dense (Window=30, Horizon=7)",
        "Model 4: Conv1D (Window=7, Horizon=1)",
        "Model 5: LSTM (Window=7, Horizon=1)",
        "Model 6: Multivariate Dense (Window=7, Horizon=1)",
        "Model 7: N-BEATS (Window=7, Horizon=1)",
        "Model 8: Ensemble (Window=7, Horizon=1)",
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
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=model_options,
                            value="Model 0: Naive Forecast",
                            label="Select Model",
                        )
                        run_btn = gr.Button("Run Model", variant="primary", size="lg")
                        gr.Markdown(
                            """
                        **Instructions:**
                        1. Select a model from the dropdown
                        2. Click "Run Model" to train and evaluate
                        3. View predictions and metrics on the right
                        
                        Each model predicts 7 days into the future.
                        """
                        )

                    with gr.Column(scale=2):
                        plot_output = gr.Plot(label="Price Predictions")
                        metrics_output = gr.Markdown(label="Performance Metrics")

                # Event handler for individual model
                run_btn.click(
                    fn=run_model,
                    inputs=[model_dropdown],
                    outputs=[plot_output, metrics_output],
                )

            # Tab 2: Compare All Models
            with gr.Tab("Compare All Models"):
                gr.Markdown(
                    """
                ### 📊 Train and Compare All Models
                
                This will train all available models and compare their 7-day future predictions.
                **Note:** This may take a few minutes as it trains 8 different models.
                """
                )

                compare_btn = gr.Button(
                    "🚀 Train & Compare All Models", variant="primary", size="lg"
                )

                comparison_plot = gr.Plot(
                    label="All Models Comparison - 7 Day Future Predictions"
                )
                comparison_metrics = gr.Markdown(label="Models Performance Comparison")

                # Event handler for comparison
                compare_btn.click(
                    fn=compare_all_models, outputs=[comparison_plot, comparison_metrics]
                )

        gr.Markdown(
            """
        ---
        **About the Models:**
        - **Naive Forecast**: Simple baseline that predicts the last known price
        - **Dense Models**: Fully connected neural networks with different window sizes
        - **Conv1D**: Convolutional neural network for temporal patterns
        - **LSTM**: Recurrent neural network with long short-term memory
        - **Multivariate**: Uses both price and return rate features
        - **N-BEATS**: Neural Basis Expansion Analysis for interpretable forecasting
        - **Ensemble**: Combines Dense, Conv1D, and LSTM predictions
        
        **Data:** Bitcoin prices from 2020-10-22 to 2025-10-21 (5 years)
        """
        )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
