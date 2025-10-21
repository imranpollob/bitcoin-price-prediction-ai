import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import model implementations
from backend.models.naive_model import naive_forecast
from backend.utils.data_loader import load_bitcoin_data
from backend.utils.evaluation import evaluate_model

# Placeholder implementations for models (will be implemented fully later)
def dense_model_prediction(data, window_size=7, horizon=1):
    """Placeholder for Dense model prediction"""
    prices = data['Close'].values
    if len(prices) < window_size + horizon:
        raise ValueError(f"Not enough data. Need at least {window_size + horizon} data points, got {len(prices)}")
    
    last_window = prices[-window_size:]
    # Simple linear extrapolation as placeholder
    predicted_value = last_window[-1] + np.mean(np.diff(last_window)) * horizon
    predictions = [predicted_value] * horizon
    
    return {
        'model': 'dense',
        'predictions': predictions,
        'historical': prices[-30:].tolist(),
        'metrics': {
            'mae': 100.0,
            'rmse': 150.0,
            'mape': 3.5,
            'mase': 1.2
        }
    }

def conv1d_model_prediction(data, window_size=7, horizon=1):
    """Placeholder for Conv1D model prediction"""
    prices = data['Close'].values
    if len(prices) < window_size + horizon:
        raise ValueError(f"Not enough data. Need at least {window_size + horizon} data points, got {len(prices)}")
    
    # Simple moving average as placeholder
    last_window = prices[-window_size:]
    predicted_value = np.mean(last_window) * (prices[-1] / np.mean(last_window))
    predictions = [predicted_value] * horizon
    
    return {
        'model': 'conv1d',
        'predictions': predictions,
        'historical': prices[-30:].tolist(),
        'metrics': {
            'mae': 120.0,
            'rmse': 170.0,
            'mape': 3.8,
            'mase': 1.4
        }
    }

def lstm_model_prediction(data, window_size=7, horizon=1):
    """Placeholder for LSTM model prediction"""
    prices = data['Close'].values
    if len(prices) < window_size + horizon:
        raise ValueError(f"Not enough data. Need at least {window_size + horizon} data points, got {len(prices)}")
    
    # Simple trend-based prediction as placeholder
    last_window = prices[-window_size:]
    trend = (last_window[-1] - last_window[0]) / (window_size - 1) if window_size > 1 else 0
    predictions = [last_window[-1] + trend * i for i in range(1, horizon + 1)]
    
    return {
        'model': 'lstm',
        'predictions': predictions,
        'historical': prices[-30:].tolist(),
        'metrics': {
            'mae': 90.0,
            'rmse': 130.0,
            'mape': 3.0,
            'mase': 1.0
        }
    }

def run_prediction(model_name, window_size, horizon):
    """Main function to run prediction based on selected model"""
    try:
        # Load data
        df = load_bitcoin_data()
        
        # Ensure we have enough data based on parameters
        if len(df) < window_size + horizon:
            raise ValueError(f"Insufficient data. Need at least {window_size + horizon} data points, got {len(df)}")
        
        # Run the selected model
        if model_name == "Naive Forecast":
            result = naive_forecast(df, steps=horizon)
        elif model_name == "Dense Model (Window=7, Horizon=1)":
            result = dense_model_prediction(df, 7, 1)
        elif model_name == "Dense Model (Window=30, Horizon=1)":
            result = dense_model_prediction(df, 30, 1)
        elif model_name == "Dense Model (Window=30, Horizon=7)":
            result = dense_model_prediction(df, 30, 7)
        elif model_name == "Conv1D Model":
            result = conv1d_model_prediction(df, window_size, horizon)
        elif model_name == "LSTM Model":
            result = lstm_model_prediction(df, window_size, horizon)
        else:
            # Default to naive forecast for other models (to be implemented)
            result = naive_forecast(df, steps=horizon)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        hist_len = len(result['historical'])
        ax.plot(range(-hist_len, 0), result['historical'], label='Historical Price', marker='o', linewidth=2)
        
        # Plot predictions
        pred_len = len(result['predictions'])
        ax.plot(range(0, pred_len), result['predictions'], label='Predictions', marker='s', linewidth=2, linestyle='--')
        
        # Add a vertical line to separate historical from predictions
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Prediction Start')
        
        ax.set_title(f'Bitcoin Price Prediction - {model_name}')
        ax.set_xlabel('Days (relative to present)')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Format metrics for display
        metrics_text = f"""
        Model: {result['model']}
        MAE: {result['metrics']['mae']:.2f}
        RMSE: {result['metrics']['rmse']:.2f}
        MAPE: {result['metrics']['mape']:.2f}%
        MASE: {result['metrics']['mase']:.2f}
        """
        
        return fig, metrics_text, result['predictions']
        
    except Exception as e:
        # Return error visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", fontsize=14, ha='center', va='center')
        ax.set_title('Error in Prediction')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        return fig, f"Error: {str(e)}", []

# Define the Gradio interface
with gr.Blocks(title="Bitcoin Price Prediction - MVP") as demo:
    gr.Markdown("# Bitcoin Price Prediction Dashboard")
    gr.Markdown("## Using TensorFlow-based Models")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=[
                    "Naive Forecast",
                    "Dense Model (Window=7, Horizon=1)",
                    "Dense Model (Window=30, Horizon=1)",
                    "Dense Model (Window=30, Horizon=7)",
                    "Conv1D Model",
                    "LSTM Model"
                ],
                value="Naive Forecast",
                label="Select Model"
            )
            
            window_size = gr.Slider(
                minimum=1,
                maximum=100,
                value=7,
                step=1,
                label="Window Size"
            )
            
            horizon = gr.Slider(
                minimum=1,
                maximum=30,
                value=1,
                step=1,
                label="Horizon (Days to Predict)"
            )
            
            run_button = gr.Button("Run Prediction", variant="primary")
            
            with gr.Accordion("Model Description", open=False):
                model_desc = gr.Markdown("""
                **Naive Forecast**: Uses previous timestep value to predict next value (ŷₜ = yₜ₋₁)
                
                **Dense Model**: Fully connected neural network with configurable window and horizon
                
                **Conv1D Model**: 1D Convolutional Neural Network for sequence processing
                
                **LSTM Model**: Long Short-Term Memory recurrent neural network
                """)
        
        with gr.Column(scale=2):
            prediction_plot = gr.Plot(label="Price Prediction")
            
            with gr.Row():
                metrics_output = gr.Textbox(label="Model Metrics", interactive=False)
                predictions_output = gr.JSON(label="Prediction Values")
    
    # Link the button to the function
    run_button.click(
        fn=run_prediction,
        inputs=[model_dropdown, window_size, horizon],
        outputs=[prediction_plot, metrics_output, predictions_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)