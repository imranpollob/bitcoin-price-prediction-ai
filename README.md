# Bitcoin Price Prediction - Gradio MVP

## Project Overview

MVP for Bitcoin price prediction using TensorFlow and advanced time series forecasting models. Built with Gradio for rapid model demonstration and testing. This project focuses on showcasing machine learning expertise through implementation of 10 different models for financial forecasting.

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

### Gradio UI Features
* **Simple Model Selection**: Intuitive dropdown to choose from different models
* **Parameter Controls**: Sliders for window size and horizon configuration
* **Interactive Chart Visualization**: Real-time plots showing historical data and predictions
* **Performance Metrics Display**: Visualization of MAE, RMSE, MAPE, MASE metrics
* **Quick Model Testing**: Rapid iteration and comparison between models
* **Easy Sharing**: Gradio's sharing capabilities for demonstration

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
