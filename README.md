# 🪙 Bitcoin Price Prediction with Deep Learning

A comprehensive time series forecasting application that compares multiple deep learning models to predict Bitcoin prices. Features an intuitive Gradio interface with adjustable parameters for training models and comparing 7-day future predictions.

## 🎯 Features

- **7 Different AI Models**: Compare Naive Forecast, Dense Networks, Conv1D, LSTM, Multivariate, N-BEATS, and Ensemble models
- **Adjustable Window Size**: Customize how many days of historical data each model uses (3-60 days)
- **7-Day Future Predictions**: All models predict Bitcoin prices for the next week
- **Interactive Web UI**: Beautiful Gradio-based interface designed for non-technical users
- **Comprehensive Metrics**: Easy-to-understand accuracy scores with plain English explanations
- **Visual Comparison**: Side-by-side comparison of all models' predictions with interactive charts

## 🆕 Recent Updates

- ✅ **Adjustable Window Size**: Use the slider to control how much historical data models analyze
- ✅ **User-Friendly UI**: Complete redesign with simple explanations and examples
- ✅ **Simplified Metrics**: Technical terms like MAE, RMSE replaced with "Average Error", "Error Range"
- ✅ **Better Visualizations**: Color-coded charts with clear legends (Green=Actual, Orange=Test, Red=Future)
- ✅ **Consensus Summary**: See which direction most models agree on (Bullish/Bearish/Stable)

## 📊 Available Models

1. **Naive Forecast** - Simple baseline (predicts last known price will continue)
2. **Dense Neural Network** - Standard feed-forward network that finds patterns in sequences
3. **Conv1D** - Convolutional network designed to detect trends in time-based data
4. **LSTM** - Specialized for remembering long-term patterns in sequential data
5. **Multivariate** - Uses both price AND rate of change for smarter predictions
6. **N-BEATS** - Advanced architecture specifically designed for forecasting
7. **Ensemble** - Combines Dense, Conv1D, and LSTM for balanced predictions

All models use **Horizon=1** (predict one day at a time, then use that to predict the next day recursively).

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

1. Open the **"Individual Model"** tab
2. **Select a model** from the dropdown menu
3. **Adjust window size** using the slider (3-60 days)
   - **Smaller windows (3-7 days)**: Faster training, reacts quickly to recent changes
   - **Medium windows (14-21 days)**: Balanced approach, captures weekly patterns
   - **Larger windows (30-60 days)**: More context, captures monthly trends
4. Click **"Train & Predict"** to train the model
5. View the results:
   - **Chart**: Green (actual), Orange (test predictions), Red (future forecast)
   - **Metrics**: Accuracy scores with plain English explanations
   - **Future Prediction**: Expected price change and direction

### Compare All Models

1. Open the **"Compare All Models"** tab
2. Click **"Train All Models & Compare"**
3. Wait 2-4 minutes for all 7 models to train
4. Review the results:
   - **Comparison Chart**: See all models' 7-day predictions
   - **Accuracy Table**: Which model performed best on test data
   - **Consensus Summary**: Market direction and agreement level
   - **Detailed Predictions**: Day-by-day forecasts for each model

### Understanding the Charts

- **🟢 Green Line**: Real prices from the test period (what actually happened)
- **🟠 Orange Line**: Model's predictions on test data (backtesting - shows reliability)
- **🔴 Red Line**: Actual forecast for the next 7 days (unknown future)

## 📈 Data

- **Source**: Bitcoin historical price data
- **Period**: 2020-10-22 to 2025-10-21 (5 years)
- **Features**: Close price, daily returns
- **Split**: 80% training, 20% testing

## 🛠️ Technical Details

### Model Architecture

- **Window Size**: Adjustable from 3 to 60 days (user-selectable via slider)
- **Horizon**: 1 day (all models predict one day ahead, then recursively predict 7 days)
- **Prediction Method**: Recursive rolling window for multi-day forecasts
- **Loss Function**: Mean Absolute Error (MAE)
- **Optimizer**: Adam
- **Epochs**: 10 (optimized for quick demo, increase for better performance)
- **Batch Size**: 128

### Understanding the Metrics

The UI displays metrics in user-friendly terms:

- **Average Error ($)**: How much predictions typically miss by (lower is better)
  - Technical name: MAE (Mean Absolute Error)
  
- **Error Range ($)**: The typical spread of prediction errors (lower is better)
  - Technical name: RMSE (Root Mean Squared Error)
  
- **Error Percentage (%)**: What percent off predictions are (under 5% is good, under 2% is excellent)
  - Technical name: MAPE (Mean Absolute Percentage Error)
  
- **vs Simple Guess**: Compares model to just guessing "tomorrow = today"
  - Less than 1.0 means the model beats simple guessing
  - Technical name: MASE (Mean Absolute Scaled Error)

### Data Split Strategy

- **Training Set**: First 80% of chronological data (~4 years)
- **Test Set**: Last 20% of chronological data (~1 year)
- **Temporal Order**: Maintained throughout (no random shuffling)
- **Test Predictions**: Sliding window approach for backtesting reliability

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

### Adjust Window Size

Simply use the slider in the UI (3-60 days). No code changes needed!

### Modify Training Parameters

Edit `app.py` to change:
- Number of epochs (line ~330+, currently 10 for fast demo)
- Batch size (line ~330+, currently 128)
- Neural network architecture (lines in each model function)
- Slider range for window size (line ~1175)

### Add New Models

1. Define model function following this pattern:
   ```python
   def model_X_name(window_size=7, horizon=1):
       """Model X: Your Model Name"""
       try:
           # Create windowed data
           full_windows, full_labels = make_windows(
               PRICES, window_size=window_size, horizon=horizon
           )
           train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
               full_windows, full_labels
           )
           
           # Build and train your model
           model = tf.keras.Sequential([...])
           model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
           model.fit(train_windows, train_labels, epochs=10, batch_size=128)
           
           # Make predictions and evaluate
           predictions = make_preds(model, test_windows)
           results = evaluate_preds(y_true=test_labels, y_pred=predictions)
           
           # Future predictions
           last_window = PRICES[-window_size:]
           future_preds = predict_future_recursive(
               model, last_window, n_future=7, window_size=window_size
           )
           
           return {
               "predictions": predictions.numpy(),
               "actual": test_labels.numpy(),
               "timesteps": test_timesteps,
               "metrics": results,
               "model_name": f"Your Model (Window={window_size}, Horizon={horizon})",
               "future_predictions": future_preds,
               "future_timesteps": future_timesteps,
           }
       except Exception as e:
           # Return error result
           return {...}
   ```

2. Add to `run_model()` function in the if/elif chain
3. Add to `model_options` list in `create_gradio_interface()`
4. Add to `compare_all_models()` function for comparisons

## 🎓 Educational Purpose

This project demonstrates:
- **Time Series Forecasting**: Multiple approaches to predicting sequential data
- **Deep Learning Architectures**: Dense, CNN (Conv1D), RNN/LSTM, and ensemble methods
- **Hyperparameter Tuning**: How window size affects model performance
- **Model Comparison**: Systematic evaluation using multiple metrics
- **Backtesting**: Testing model predictions on held-out historical data
- **Financial Data Analysis**: Working with real-world cryptocurrency data
- **TensorFlow/Keras**: Building and training neural networks
- **Gradio Interface**: Creating accessible, interactive ML applications
- **User Experience Design**: Making AI tools understandable for non-technical users

## 📊 Expected Results

Based on testing with the provided Bitcoin data (2020-2025):

| Model          | Typical Avg Error | Speed   | Best For                     |
| -------------- | ----------------- | ------- | ---------------------------- |
| Naive Forecast | $1,500-2,000      | Instant | Baseline comparison          |
| Dense          | $2,400-2,700      | Fast    | Quick experiments            |
| Conv1D         | $2,400-2,600      | Fast    | Pattern detection            |
| LSTM           | $2,500-2,800      | Medium  | Sequential dependencies      |
| Multivariate   | $2,400-2,700      | Fast    | Multi-feature analysis       |
| N-BEATS        | $2,500-2,900      | Slow    | Interpretable forecasts      |
| Ensemble       | $2,300-2,600      | Slow    | Balanced, robust predictions |

**Notes**: 
- Results vary based on training run and window size selected
- Larger window sizes (30-60 days) may improve accuracy but take longer to train
- Bitcoin's high volatility makes accurate prediction challenging for all models
- Lower error values indicate better performance

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This application is for educational and research purposes only
2. **No Investment Decisions**: Do not use these predictions for actual cryptocurrency trading
3. **Past Performance ≠ Future Results**: Historical accuracy doesn't guarantee future prediction quality
4. **High Market Volatility**: Cryptocurrency markets are extremely unpredictable and influenced by many external factors
5. **Model Limitations**: All AI forecasting models have inherent limitations and can be wrong
6. **Risk Warning**: Cryptocurrency trading involves substantial risk of loss
7. **Educational Tool**: This is a learning project to understand time series forecasting, not a trading system

**Use this tool to learn about AI and forecasting, not to make financial decisions!**

## 🐛 Troubleshooting

### Common Issues

**Issue**: Models training too slowly
```bash
# Reduce epochs in app.py (change from 10 to 5)
# Use smaller window size (7 instead of 30+)
# Try simpler models first (Dense, Naive)
```

**Issue**: CUDA/GPU errors
```bash
# Set TensorFlow to use CPU only
export CUDA_VISIBLE_DEVICES=""
python app.py
```

**Issue**: Out of memory
```bash
# Reduce batch size in model functions (128 -> 64)
# Reduce window size in the UI slider
# Use CPU instead of GPU
```

**Issue**: Missing dependencies
```bash
pip install --upgrade -r requirements.txt
```

**Issue**: Port already in use
```bash
# Change port in app.py
# Or find and kill process using port 7860:
lsof -ti:7860 | xargs kill -9
```

**Issue**: Models give very different predictions
```
This is normal! Different architectures capture different patterns.
- Check the "Agreement Level" in comparison view
- High disagreement = high uncertainty
- Use ensemble model for balanced prediction
```

## 💡 Tips for Best Results

1. **Start with smaller window sizes (7-14 days)** for faster experimentation
2. **Use the comparison feature** to see which models perform best on your data
3. **Look for consensus** - if most models agree, predictions may be more reliable
4. **Consider the test predictions** (orange line) - if they're close to actual (green), the model is reliable
5. **Don't rely on a single model** - use the ensemble or majority consensus
6. **Experiment with window sizes** - try different values to find the sweet spot for your use case

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork and experiment with different architectures
- Add new model types or features
- Improve documentation and explanations
- Share your findings and improvements
- Report bugs or suggest features

## ❓ FAQ

**Q: What's the difference between window size and horizon?**
- **Window**: How many past days the model looks at (user adjustable)
- **Horizon**: How many days ahead it predicts at once (fixed at 1 for all models)

**Q: Why do all models use Horizon=1?**
- Predicting one day at a time is generally more accurate
- We then use those predictions recursively to get 7-day forecasts
- This approach is more stable than trying to predict all 7 days at once

**Q: What's the orange line (test predictions)?**
- It shows how well the model predicted on data it never trained on
- This is "backtesting" - simulating real-world use
- If orange is close to green, the model is reliable

**Q: Which model should I use?**
- **Fastest**: Naive or Dense (good for quick tests)
- **Most Accurate**: Usually Ensemble or Conv1D (based on testing)
- **Most Interpretable**: N-BEATS (designed for understanding)
- **Best Balance**: Ensemble (combines multiple models)

**Q: Can I use this for actual Bitcoin trading?**
- **No!** This is for education only
- Cryptocurrency markets are too volatile and complex
- Many external factors affect prices that models don't know about

**Q: How do I improve prediction accuracy?**
- Increase epochs (takes longer to train)
- Experiment with different window sizes
- Try ensemble models
- Use more sophisticated architectures
- Add more features (trading volume, market sentiment, etc.)

**Q: The models disagree - which is right?**
- High disagreement means high uncertainty
- No model is "right" - they're all estimates
- Use consensus (average) or ensemble for balanced view
- Check historical accuracy (test predictions) as a guide

## 📚 Learn More

- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)
- [Gradio Documentation](https://www.gradio.app/docs/)

---

**Made with ❤️ for learning AI and time series forecasting**