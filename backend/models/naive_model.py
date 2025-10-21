import pandas as pd
import numpy as np

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
    
    # The naive forecast is simply the last observed value
    last_value = prices[-1]
    
    # For multi-step prediction, we repeat the last value
    predictions = [last_value] * steps
    
    result = {
        'model': 'naive',
        'predictions': predictions,
        'historical': prices[-30:].tolist(),  # Return last 30 values for visualization
        'metrics': {
            'mae': 0.0,  # Placeholder - in real implementation, calculate on validation set
            'rmse': 0.0,
            'mape': 0.0,
            'mase': 0.0
        }
    }
    
    return result