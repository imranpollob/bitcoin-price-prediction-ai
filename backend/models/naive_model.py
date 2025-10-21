import pandas as pd
import numpy as np
from backend.utils.evaluation import evaluate_model

def naive_forecast(data, steps=1):
    """
    Naive forecast: uses previous timestep value to predict next value
    Formula: ŷₜ = yₜ₋₁
    """
    # Assuming 'data' is a pandas DataFrame with a 'Close' column
    if isinstance(data, pd.DataFrame):
        prices = data['Close'].values
    else:
        prices = data
    
    if len(prices) < 2:
        raise ValueError("Need at least 2 data points for naive forecast")
    
    if steps < 1:
        raise ValueError("Steps must be at least 1")
    
    # The naive forecast is simply the most recent observed value (first in dataset)
    # Since the dataset is ordered from newest to oldest (index 0 is most recent)
    most_recent_value = prices[0]
    
    # For multi-step prediction, we repeat the most recent value
    predictions = [most_recent_value] * steps
    
    # Calculate metrics using walk-forward validation approach
    # Since we're predicting future values that don't exist yet, we calculate 
    # metrics using historical data with the same naive approach
    if len(prices) >= 3:  # Need at least 3 values to compute meaningful metrics
        # Use walk-forward validation: predict each value using previous value
        # For dataset ordered newest to oldest, use next value (in time) as prediction for current
        y_true = prices[:-1]    # All values except the last (oldest)
        y_pred = prices[1:]     # All values except the first (most recent) - this is our naive predictions
        
        metrics = evaluate_model(y_true, y_pred)
    else:
        # Default metrics if insufficient historical data for evaluation
        metrics = {
            'mae': 0.0,  # Placeholder when actual metrics can't be calculated
            'rmse': 0.0,
            'mape': 0.0,
            'mase': 0.0
        }
    
    # Return all available historical values for visualization (in chronological order)
    # For better visualization, we can return more historical points
    hist_len = min(365, len(prices))  # Return up to 365 days of historical data for visualization
    # Since the dataset is ordered newest to oldest, take the first 'hist_len' values
    # and reverse them to get chronological order (oldest to newest for visualization)
    historical_data = prices[:hist_len][::-1].tolist()
    
    result = {
        'model': 'naive',
        'predictions': predictions,
        'historical': historical_data,  # Return most recent values in chronological order
        'metrics': metrics
    }
    
    return result