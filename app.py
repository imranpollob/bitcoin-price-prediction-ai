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

    return {
        "predictions": naive_forecast,
        "actual": test_prices[1:],
        "timesteps": pred_timesteps,
        "metrics": results,
        "model_name": "Naive Forecast",
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

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": "LSTM Model (Window=7, Horizon=1)",
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"LSTM Model Error: {str(e)}",
        }


def model_6_multivariate():
    """Model 6: Multivariate Dense Model (placeholder)"""
    return {
        "predictions": None,
        "actual": None,
        "timesteps": None,
        "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
        "model_name": "Multivariate Dense Model (Not implemented yet)",
    }


def model_7_nbeats():
    """Model 7: N-BEATS Model (placeholder)"""
    return {
        "predictions": None,
        "actual": None,
        "timesteps": None,
        "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
        "model_name": "N-BEATS Model (Not implemented yet)",
    }


def model_8_ensemble():
    """Model 8: Ensemble Model (placeholder)"""
    return {
        "predictions": None,
        "actual": None,
        "timesteps": None,
        "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
        "model_name": "Ensemble Model (Not implemented yet)",
    }


def model_9_future():
    """Model 9: Future Prediction Model (placeholder)"""
    return {
        "predictions": None,
        "actual": None,
        "timesteps": None,
        "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
        "model_name": "Future Prediction Model (Not implemented yet)",
    }
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

    return {
        "predictions": naive_forecast,
        "actual": test_prices[1:],
        "timesteps": pred_timesteps,
        "metrics": results,
        "model_name": "Naive Forecast",
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

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": f"Dense Model (Window={window_size}, Horizon={horizon})",
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Dense Model Error: {str(e)}",
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

        return {
            "predictions": predictions.numpy(),
            "actual": tf.squeeze(test_labels).numpy(),
            "timesteps": test_timesteps,
            "metrics": results,
            "model_name": "Conv1D Model (Window=7, Horizon=1)",
        }
    except Exception as e:
        return {
            "predictions": None,
            "actual": None,
            "timesteps": None,
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "mase": 0},
            "model_name": f"Conv1D Model Error: {str(e)}",
        }


# Add more model functions here...


# Gradio interface functions
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

        # Create plot
        if result["predictions"] is not None:
            fig = create_plotly_time_series(
                result["timesteps"],
                result["actual"],
                result["predictions"],
                result["model_name"],
            )
        else:
            fig = None

        # Format metrics for display
        metrics_text = f"""
        **{result['model_name']} Results:**

        - **MAE**: {result['metrics'].get('mae', 'N/A'):.4f}
        - **RMSE**: {result['metrics'].get('rmse', 'N/A'):.4f}
        - **MAPE**: {result['metrics'].get('mape', 'N/A'):.4f}
        - **MASE**: {result['metrics'].get('mase', 'N/A'):.4f}
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
    ]

    with gr.Blocks(
        title="Bitcoin Price Prediction", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# Bitcoin Price Prediction - TensorFlow Models")
        gr.Markdown("*Educational demonstration of time series forecasting models*")

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    value="Model 0: Naive Forecast",
                    label="Select Model",
                )

                run_btn = gr.Button("Run Model", variant="primary")

            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Price Predictions")
                metrics_output = gr.Markdown(label="Performance Metrics")

        # Event handlers
        run_btn.click(
            fn=run_model, inputs=[model_dropdown], outputs=[plot_output, metrics_output]
        )

        # Load default model on startup
        interface.load(
            fn=lambda: run_model("Model 0: Naive Forecast"),
            outputs=[plot_output, metrics_output],
        )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
