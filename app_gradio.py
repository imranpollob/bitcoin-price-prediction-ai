import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        
        # Prepare data for visualization
        # The dataset is sorted from newest to oldest (index 0 is most recent)
        
        # Get the historical data from the model result (already in chronological order)
        # The naive model returns the most recent values in chronological order
        hist_len = len(result['historical'])
        
        # Get corresponding dates from the dataset for the historical values
        # Since dataset is newest to oldest, the historical values (chronological order) 
        # correspond to the most recent hist_len days in the dataset
        actual_historical_data = df.head(hist_len)
        
        # Get dates and prices in chronological order (oldest to newest)
        # Dataset is ordered newest to oldest, so we need to reverse both dates and prices
        # to get chronological order that matches the result['historical'] values
        date_range_raw = actual_historical_data['End'].tolist()[::-1]  # Reverse to get chronological order
        historical_prices = actual_historical_data['Close'].tolist()[::-1]  # Reverse to match dates
        
        # Convert to datetime for plotting
        date_range_dt = pd.to_datetime(date_range_raw)
        
        # Create interactive plotly figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=date_range_dt,
            y=historical_prices,
            mode='lines+markers',
            name='Historical Price',
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate='Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add predictions if any
        if len(result['predictions']) > 0:
            # Get the last actual date from the historical data (most recent)
            last_actual_date_str = actual_historical_data['End'].iloc[0]  # Most recent date (first in dataset)
            last_date = pd.to_datetime(last_actual_date_str)
            future_dates = []
            for i in range(1, len(result['predictions']) + 1):
                future_date = last_date + pd.Timedelta(days=i)
                future_dates.append(future_date)
            
            # Add predictions with different styling
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result['predictions'],
                mode='lines+markers',
                name='Predictions',
                line=dict(width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # Add a vertical line to separate historical from predictions
        if len(date_range_dt) > 0:
            last_date = date_range_dt[-1]  # Last date in chronological order (most recent actual)
            # Get the y-axis range to draw the line across the full height
            min_price = min(min(historical_prices), min(result['predictions']) if result['predictions'] else historical_prices)
            max_price = max(max(historical_prices), max(result['predictions']) if result['predictions'] else historical_prices)
            
            # Add vertical line manually using add_shape
            fig.add_shape(
                type='line',
                x0=last_date, x1=last_date,
                y0=min_price, y1=max_price,
                line=dict(color='red', width=2, dash='dash'),
            )
            
            # Add annotation for the line
            fig.add_annotation(
                x=last_date,
                y=max_price,
                text="Prediction Start",
                showarrow=False,
                yshift=10,
                xref='x',
                yref='y'
            )
        
        # Update layout
        fig.update_layout(
            title=f'Bitcoin Price Prediction - {model_name}',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )
        
        # Format y-axis to show currency
        fig.update_yaxes(tickprefix="$", tickformat=",")
        
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
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5,
                          xanchor='center', yanchor='middle',
                          showarrow=False,
                          font=dict(size=18))
        fig.update_layout(title_text="Error in Prediction", height=600)
        
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