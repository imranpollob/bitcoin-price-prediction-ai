# 🪙 Bitcoin Price Prediction with TensorFlow

A comprehensive time series forecasting application that compares multiple deep learning models to predict Bitcoin prices. Features a user-friendly Gradio interface for training models and comparing 7-day future predictions.

## 🎯 Features

- **8 Different Models**: Compare Naive Forecast, Dense Networks, Conv1D, LSTM, Multivariate, N-BEATS, and Ensemble models
- **7-Day Future Predictions**: All models predict Bitcoin prices for the next 7 days
- **Interactive UI**: Gradio-based web interface for easy model training and comparison
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and MASE for model evaluation
- **Visual Comparison**: Side-by-side comparison of all models' predictions

## 📊 Available Models

1. **Naive Forecast** - Baseline model (predicts last known price)
2. **Dense (Window=7, Horizon=1)** - Simple feed-forward neural network
3. **Dense (Window=30, Horizon=1)** - Dense model with larger window
4. **Conv1D** - 1D Convolutional neural network for pattern recognition
5. **LSTM** - Recurrent neural network with long short-term memory
6. **Multivariate** - Uses both price and return rate features
7. **N-BEATS** - Neural Basis Expansion Analysis for interpretable forecasting
8. **Ensemble** - Combines Dense, Conv1D, and LSTM predictions

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Gradio interface
python app.py
```

The application will open in your browser at `http://localhost:7860`

### Testing Models

```bash
# Run test script to verify all models work
python test_models.py
```

## 📖 Usage

### Individual Model Testing

1. Open the "Individual Model" tab
2. Select a model from the dropdown
3. Click "Run Model" to train and evaluate
4. View predictions and metrics (including 7-day future forecast)

### Compare All Models

1. Open the "Compare All Models" tab
2. Click "Train & Compare All Models"
3. Wait for all 8 models to train (takes a few minutes)
4. View comparison chart and detailed metrics table

## 📈 Data

- **Source**: Bitcoin historical price data
- **Period**: 2020-10-22 to 2025-10-21 (5 years)
- **Features**: Close price, daily returns
- **Split**: 80% training, 20% testing

## 🛠️ Technical Details

### Model Architecture

- **Window Size**: 7 or 30 days (depending on model)
- **Horizon**: 1 or 7 days ahead
- **Loss Function**: Mean Absolute Error (MAE)
- **Optimizer**: Adam
- **Epochs**: 10 (for quick demo, can be increased for better performance)

### Metrics

- **MAE**: Mean Absolute Error - Average absolute difference between predicted and actual values
- **RMSE**: Root Mean Squared Error - Square root of average squared differences
- **MAPE**: Mean Absolute Percentage Error - Average percentage error
- **MASE**: Mean Absolute Scaled Error - Scaled error relative to naive forecast

## 📁 Project Structure

```
.
├── app.py                 # Main application with Gradio interface
├── test_models.py         # Model testing script
├── requirements.txt       # Python dependencies
├── data/
│   └── bitcoin_2020-10-22_2025-10-21.csv
└── README.md
```

## 🔧 Customization

### Modify Training Parameters

Edit `app.py` to change:
- Window size (lines ~250+)
- Prediction horizon
- Number of epochs (currently 10 for fast demo)
- Batch size (currently 128)
- Model architecture

### Add New Models

1. Define model function following the pattern:
   ```python
   def model_X_name():
       # Train model
       # Make predictions
       # Return results with future predictions
       return {
           "predictions": test_predictions,
           "actual": test_actuals,
           "timesteps": test_timesteps,
           "metrics": metrics_dict,
           "model_name": "Model Name",
           "future_predictions": future_7_days,
           "future_timesteps": future_dates,
       }
   ```

2. Add to comparison in `compare_all_models()` function
3. Add to dropdown options in `create_gradio_interface()`

## 🎓 Educational Purpose

This project demonstrates:
- **Time Series Forecasting**: Multiple approaches to predicting sequential data
- **Deep Learning Architectures**: Dense, CNN, RNN/LSTM, and ensemble methods
- **Model Comparison**: Systematic evaluation using multiple metrics
- **Financial Data Analysis**: Working with real-world cryptocurrency data
- **TensorFlow/Keras**: Building and training neural networks
- **Gradio Interface**: Creating interactive ML applications

## 📊 Expected Results

Based on testing with the provided data:

| Model          | Typical MAE  | Speed   | Complexity |
| -------------- | ------------ | ------- | ---------- |
| Naive Forecast | ~1,586       | Instant | Very Low   |
| Dense (W=7)    | ~2,607       | Fast    | Low        |
| Conv1D         | ~2,543       | Fast    | Medium     |
| LSTM           | ~2,701       | Medium  | High       |
| Multivariate   | ~2,500-2,700 | Fast    | Medium     |
| N-BEATS        | ~2,600-2,800 | Slow    | Very High  |
| Ensemble       | ~2,400-2,600 | Slow    | High       |

*Note: Actual results may vary based on training run and data splits*

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This application is for educational and research purposes only
2. **No Investment Decisions**: Do not use these predictions for actual trading
3. **Past Performance**: Historical data does not guarantee future results
4. **Market Volatility**: Cryptocurrency markets are highly unpredictable
5. **Model Limitations**: All forecasting models have inherent limitations

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA/GPU errors
```bash
# Set TensorFlow to use CPU only
export CUDA_VISIBLE_DEVICES=""
python app.py
```

**Issue**: Out of memory
- Reduce batch size in model training functions
- Reduce number of epochs
- Use CPU instead of GPU

**Issue**: Missing dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork and experiment
- Add new model architectures
- Improve documentation
- Report bugs or suggest features

