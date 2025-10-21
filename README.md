# Bitcoin Price Prediction Dashboard

## Project Overview

Interactive web-based dashboard for Bitcoin price prediction using TensorFlow and advanced time series forecasting models. This project demonstrates the application of deep learning for financial forecasting through an intuitive user interface. The dashboard allows users to apply different models to predict Bitcoin prices and visualize the results in real-time.

**Key Warning**: This is not financial advice. Time series forecasting for stock/crypto markets is notoriously difficult and often produces poor results. Use this tool for educational purposes only.

## Features

### Time Series Forecasting Capabilities
* **10 Different Models**: 
  * Model 0: Naive Forecast (Baseline)
  * Model 1-3: Dense Models with various window/horizon configurations
  * Model 4: Conv1D Model
  * Model 5: LSTM Model
  * Model 6: Multivariate Dense Model
  * Model 7: N-BEATS Algorithm
  * Model 8: Ensemble Model
  * Model 9: Future Prediction Model
  * Model 10: Turkey Problem Demonstration
* **Window and Horizon Configuration**: Adjust the number of historical timesteps (window) and future prediction timesteps (horizon)
* **Multiple Evaluation Metrics**: MAE, RMSE, MAPE, and MASE for comprehensive model evaluation
* **Proper Train/Test Splitting**: Sequential splitting appropriate for time series data

### UI Features
* **Interactive Dashboard**: Visual overview of Bitcoin price trends with model predictions
* **Model Selection Interface**: Intuitive dropdown/panel to choose from the 10 different models
* **Parameter Control Panel**: Real-time adjustment of window size, horizon, and other hyperparameters
* **Real-time Chart Visualization**: Interactive plots showing historical data, predictions, and confidence intervals
* **Model Comparison Tool**: Side-by-side comparison of multiple models' performance
* **Data Input Interface**: Ability to upload custom data or select date ranges for analysis
* **Performance Metrics Display**: Real-time visualization of MAE, RMSE, MAPE, MASE metrics
* **Prediction Confidence Indicators**: Visual representation of prediction uncertainty
* **Historical Prediction Backtesting**: Test models on past data and visualize results
* **Export Functionality**: Download predictions, charts, and performance reports
* **Model Training Interface**: Option to retrain models with custom parameters
* **Feature Engineering Panel**: Select additional features like block rewards for multivariate models
* **Alert System**: Configurable notifications when predictions exceed certain thresholds
* **Scenario Analysis Tool**: Compare different prediction scenarios
* **Model Performance Historical Tracking**: Track how models performed over time
* **Custom Time Range Selection**: Select specific date ranges for training/prediction
* **Interactive Data Annotation**: Mark significant events on the timeline
* **Multi-timeframe Visualization**: View predictions on different time scales (daily, weekly, monthly)
* **Risk Assessment Display**: Show potential risks based on model uncertainty

## Key Concepts

### Time Series Fundamentals
* **Horizon**: Number of timesteps to predict into the future
* **Window**: Number of timesteps from the past used to predict the horizon
* **Example**: To predict tomorrow's price (horizon=1) using the previous week's prices (window=7)

### Evaluation Metrics
* **Scale-dependent errors**:
  * **MAE** (Mean Absolute Error): Easy to interpret; forecasts minimizing MAE lead to median forecasts
  * **RMSE** (Root Mean Square Error): Forecasts minimizing RMSE lead to mean forecasts
* **Percentage errors**:
  * **MAPE** (Mean Absolute Percentage Error): Most commonly used percentage error
* **Scaled errors**:
  * **MASE** (Mean Absolute Scaled Error): Compares forecast to naive forecast performance

## Technical Implementation

### Core Components
* **Custom Functions**:
  * `make_windows()`: Creates time series windows for supervised learning
  * `make_train_test_splits()`: Properly splits time series data
  * `evaluate_preds()`: Comprehensive model evaluation with multiple metrics
  * `mean_absolute_scaled_error()`: MASE implementation
  * `make_preds()`: Standardized prediction interface
* **TensorFlow Techniques**:
  * Custom layer creation for N-BEATS algorithm
  * Functional API for complex model architectures
  * tf.data.Dataset for efficient data pipeline
  * Model checkpointing for best model selection
  * Callback implementation for training optimization

### UI Technologies
* React.js for interactive user interface
* D3.js or Chart.js for data visualization
* TensorFlow.js for client-side model inference (optional)
* Flask/FastAPI backend for model serving
* WebSocket connections for real-time updates

## Dataset

* Historical Bitcoin prices from Coindesk (2013-10-01 to 2021-05-18)
* Data includes closing prices and block reward information
* Total of 2,787 samples (relatively small for deep learning)

## Key Insights

### Forecasting in Open Systems
* Markets like Bitcoin are influenced by countless external factors
* Neural networks struggle with the inherent unpredictability
* Even sophisticated models often perform similarly to or worse than naive forecasts

### The Turkey Problem
* Demonstrates how observational data can fail to predict catastrophic events
* Highlights the importance of considering black swan events in forecasting

### Model Performance Reality Check
* Most deep learning models performed comparably to or slightly worse than naive forecasts
* Adding complexity doesn't necessarily improve results
* Single data point changes can devastate model performance

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bitcoin-price-prediction-dashboard.git
   cd bitcoin-price-prediction-dashboard
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   npm install
   ```

3. Prepare the data:
   * Place your Bitcoin price data file in the `data/` directory
   * Ensure the data follows the expected format (date, price, block reward)

4. Start the backend server:
   ```
   python app.py
   ```

5. Start the frontend development server:
   ```
   npm start
   ```

6. Open your browser to `http://localhost:3000` to access the dashboard

## Usage Guide

1. **Select Model**: Choose from the 10 available model types using the model selection panel
2. **Configure Parameters**: Adjust window size, horizon, and other model-specific parameters
3. **Select Date Range**: Choose the historical data range for training and testing
4. **Run Prediction**: Click the "Predict" button to generate forecasts
5. **View Results**: See the interactive chart with historical data and predictions
6. **Compare Models**: Toggle between different models to compare performance metrics
7. **Export Data**: Download predictions and performance metrics in CSV format

## Best Practices

1. **Start Simple**: Always establish a strong baseline with naive forecasts
2. **Proper Evaluation**: Use multiple metrics and visualizations for model comparison
3. **Realistic Expectations**: Understand the limitations of forecasting in open systems like crypto markets
4. **Validation**: Use proper time series splits and out-of-sample testing

## Challenges

* **Data Limitations**: Limited historical data for deep learning approaches
* **Model Overfitting**: Complex models may overfit to training data
* **Evaluation Difficulties**: Metrics can be misleading in volatile environments

## Future Directions

1. **Closed System Applications**: More promising results in predictable domains
2. **Hybrid Approaches**: Combining ML with traditional econometric methods
3. **Uncertainty Quantification**: Better methods for prediction intervals
4. **Real-time Adaptation**: Models that continuously update with new data
5. **Additional Cryptocurrencies**: Extend to other digital assets
6. **Sentiment Analysis Integration**: Incorporate social media and news sentiment
7. **Ensemble Improvements**: Advanced ensemble methods for better performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Based on time series forecasting research and TensorFlow implementation
* Historical Bitcoin data sourced from Coindesk
* Inspired by the challenges of financial forecasting in open systems

---

*This project demonstrates that while deep learning offers powerful tools for time series forecasting, applying them successfully requires:*
* *Deep understanding of the domain*
* *Careful consideration of problem characteristics*
* *Realistic expectations about model capabilities*
* *Robust evaluation methodologies*