import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from backend.utils.evaluation import evaluate_model
from backend.utils.data_loader import prepare_windowed_data, make_train_test_splits
import os

def dense_model_prediction(data, window_size=7, horizon=1):
    """
    Dense Neural Network model for Bitcoin price prediction
    Uses a window of past values to predict the next 'horizon' values
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Extract prices from the data
    if isinstance(data, pd.DataFrame):
        prices = data['Close'].values
    else:
        prices = data
    
    if len(prices) < window_size + horizon:
        raise ValueError(f"Not enough data. Need at least {window_size + horizon} data points, got {len(prices)}")
    
    # Create windowed dataset
    windows, horizons = prepare_windowed_data(data, window_size=window_size, horizon=horizon)
    
    # Split into train and test sets
    train_windows, train_horizons, test_windows, test_horizons = make_train_test_splits(windows, horizons)
    
    # Build the dense model
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(window_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(horizon, activation='linear')
    ], name='dense_model')
    
    # Compile the model
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    
    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_windows,
        train_horizons,
        epochs=100,
        batch_size=32,
        validation_data=(test_windows, test_horizons),
        callbacks=[early_stopping],
        verbose=0  # Set to 1 if you want to see training progress
    )
    
    # Make predictions on test data
    test_predictions = model.predict(test_windows, verbose=0)
    
    # Evaluate the model on test data
    # Flatten predictions and horizons for metric calculation if needed
    if horizon == 1:
        y_true = test_horizons.flatten()
        y_pred = test_predictions.flatten()
    else:
        y_true = test_horizons
        y_pred = test_predictions
    
    # Calculate metrics using the test predictions
    metrics = evaluate_model(y_true, y_pred)
    
    # Prepare final prediction for future values
    # Use the last window to predict future values
    last_window = windows[-1:]  # Get the last window as a 2D array
    future_prediction = model.predict(last_window, verbose=0).flatten()
    
    # Return the required historical data for visualization (last ~90 days in chronological order, approx 3 months)
    # Assuming ~30 days per month, so 3 months would be ~90 days
    hist_len = min(90, len(prices))
    # Since the dataset is ordered newest to oldest, take the most recent values and reverse them
    # to get chronological order (oldest to newest)
    historical_data = prices[:hist_len][::-1].tolist()
    
    result = {
        'model': f'dense_window{window_size}_horizon{horizon}',
        'predictions': future_prediction.tolist(),
        'historical': historical_data,
        'metrics': metrics
    }
    
    return result

def dense_model_1_prediction(data, window_size=7, horizon=1):
    """
    Dense Model (Window=7, Horizon=1) - Model 1
    This is a specific implementation matching the notebook model_1_dense_window7_horizon1
    """
    return dense_model_prediction(data, window_size=7, horizon=1)

def dense_model_2_prediction(data, window_size=30, horizon=1):
    """
    Dense Model (Window=30, Horizon=1) - Model 2
    """
    return dense_model_prediction(data, window_size=30, horizon=1)

def dense_model_3_prediction(data, window_size=30, horizon=7):
    """
    Dense Model (Window=30, Horizon=7) - Model 3
    """
    return dense_model_prediction(data, window_size=30, horizon=7)