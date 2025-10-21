import numpy as np

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_mase(y_true, y_pred, y_naive_scale=None):
    """
    Calculate Mean Absolute Scaled Error
    The scaling is typically done by the in-sample one-step naive forecast error
    """
    mae = calculate_mae(y_true, y_pred)
    
    if y_naive_scale is None:
        # Calculate scaling factor using naive forecast (y_t = y_{t-1})
        naive_errors = np.abs(y_true[1:] - y_true[:-1])
        scale = np.mean(naive_errors)
    else:
        scale = y_naive_scale
    
    mase_value = mae / scale
    return mase_value

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mase': mase
    }