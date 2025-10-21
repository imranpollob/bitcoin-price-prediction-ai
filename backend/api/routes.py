from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from backend.models.naive_model import naive_forecast
from backend.utils.data_loader import load_bitcoin_data

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return jsonify({"message": "Bitcoin Price Prediction API", "status": "running"})

@main_bp.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models = [
        {"id": "naive", "name": "Naive Forecast", "description": "Uses previous timestep value to predict next value"},
        {"id": "dense71", "name": "Dense Model (Window=7, Horizon=1)", "description": "Simple dense neural network with 7-day window"},
        {"id": "dense301", "name": "Dense Model (Window=30, Horizon=1)", "description": "Simple dense neural network with 30-day window"},
        {"id": "dense307", "name": "Dense Model (Window=30, Horizon=7)", "description": "Dense model predicting 7 days ahead"},
        {"id": "conv1d", "name": "Conv1D Model", "description": "1D Convolutional Neural Network"},
        {"id": "lstm", "name": "LSTM Model", "description": "Long Short-Term Memory recurrent neural network"},
        {"id": "multivariate", "name": "Multivariate Dense Model", "description": "Uses additional features like block rewards"},
        {"id": "nbeats", "name": "N-BEATS Algorithm", "description": "Neural Basis Expansion Analysis for Interpretable Time Series Forecasting"},
        {"id": "ensemble", "name": "Ensemble Model", "description": "Combines multiple models with different loss functions"},
        {"id": "future", "name": "Future Prediction Model", "description": "Trained on full dataset for real-world deployment"}
    ]
    return jsonify({"models": models})

@main_bp.route('/api/predict', methods=['POST'])
def predict():
    """Generate prediction using selected model"""
    data = request.json
    
    model_id = data.get('model_id', 'naive')
    window_size = data.get('window_size', 7)
    horizon = data.get('horizon', 1)
    
    # Load data
    df = load_bitcoin_data()
    
    # For now, return mock response - in real implementation, this would call the selected model
    if model_id == 'naive':
        result = naive_forecast(df, steps=horizon)
    else:
        # Mock response for other models
        result = {
            'model': model_id,
            'predictions': [35000 + np.random.uniform(-1000, 1000) for _ in range(horizon)],
            'historical': df['Close'].tail(30).tolist(),
            'metrics': {
                'mae': np.random.uniform(500, 1500),
                'rmse': np.random.uniform(700, 2000),
                'mape': np.random.uniform(2, 5)
            }
        }
    
    return jsonify(result)

@main_bp.route('/api/data', methods=['GET'])
def get_data():
    """Get historical Bitcoin data"""
    df = load_bitcoin_data()
    # Return last 100 data points as sample
    data = df.tail(100).to_dict('records')
    return jsonify(data)