# Time Series Forecasting with TensorFlow: Bitcoin Price Prediction

## Project Overview

This project demonstrates time series forecasting using TensorFlow by attempting to predict the price of Bitcoin. It serves as Milestone Project 3 in a deep learning course, focusing on the challenges of forecasting in open systems like cryptocurrency markets.

**Key Warning**: This is not financial advice. Time series forecasting for stock/crypto markets is notoriously difficult and often produces poor results.

## What is a Time Series Problem?

Time series problems deal with data over time, such as:
- Staff members in a company over 10 years
- Sales of computers for the past 5 years
- Electricity usage for the past 50 years

Two main categories of time series problems:
1. **Classification**: Anomaly detection, time series identification
2. **Forecasting**: Stock market prices, demand forecasting

## Project Goals

Build a series of models to predict Bitcoin prices using historical data from October 1, 2013 to May 18, 2021.

## Key Concepts Covered

### Time Series Fundamentals
- **Horizon**: Number of timesteps to predict into the future
- **Window**: Number of timesteps from the past used to predict the horizon
- **Example**: To predict tomorrow's price (horizon=1) using the previous week's prices (window=7)

### Evaluation Metrics
Scale-dependent errors:
- **MAE** (Mean Absolute Error): Easy to interpret; forecasts minimizing MAE lead to median forecasts
- **RMSE** (Root Mean Square Error): Forecasts minimizing RMSE lead to mean forecasts

Percentage errors:
- **MAPE** (Mean Absolute Percentage Error): Most commonly used percentage error

Scaled errors:
- **MASE** (Mean Absolute Scaled Error): Compares forecast to naive forecast performance

### Important Notes on Evaluation
- Lower scores are better for all metrics
- MAE is often a good starting point due to its interpretability
- In open systems like crypto markets, beating a naive forecast is extremely challenging

## Data Preparation Steps

### 1. Loading and Exploring Data
- Historical Bitcoin prices from Coindesk (2013-10-01 to 2021-05-18)
- Data includes closing prices and block reward information
- Total of 2,787 samples (relatively small for deep learning)

### 2. Train/Test Splitting (Correct Method)
Unlike typical ML problems, time series requires sequential splitting:
- Training set: Past data
- Test set: Future data (artificial future for evaluation)

### 3. Windowing Dataset
Convert time series into supervised learning problem:
- Create sliding windows of historical data
- Use past values (window) to predict future values (horizon)

## Modelling Experiments

The project implements 10 different models with increasing complexity:

### Model 0: Naive Forecast (Baseline)
- Uses previous timestep value to predict next value
- Formula: ŷₜ = yₜ₋₁
- Extremely difficult to beat in open systems

### Model 1: Dense Model (Window=7, Horizon=1)
- Simple dense neural network
- Window size of 7 days to predict next day

### Model 2: Dense Model (Window=30, Horizon=1)
- Same architecture with larger window
- Uses previous 30 days to predict next day

### Model 3: Dense Model (Window=30, Horizon=7)
- Predicts 7 days ahead using 30 days of history

### Model 4: Conv1D Model
- 1D Convolutional Neural Network
- Better suited for sequence modeling

### Model 5: LSTM Model
- Long Short-Term Memory recurrent neural network
- Designed for sequential data processing

### Model 6: Multivariate Dense Model
- Incorporates additional features (Bitcoin block reward)
- Uses both price history and block reward size

### Model 7: N-BEATS Algorithm
- Implements the Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
- State-of-the-art algorithm from M4 forecasting competition
- Complex architecture with residual stacking

### Model 8: Ensemble Model
- Combines multiple models with different loss functions
- Uses averaging of predictions from various architectures

### Model 9: Future Prediction Model
- Trains on full dataset to make predictions into actual future
- Demonstrates real-world deployment considerations

### Model 10: Turkey Problem Demonstration
- Illustrates the impact of black swan events
- Shows how a single catastrophic data point affects model performance

## Key Insights and Learnings

### 1. Forecasting in Open Systems is Extremely Difficult
- Markets like Bitcoin are influenced by countless external factors
- Neural networks struggle with the inherent unpredictability
- Even sophisticated models often perform similarly to or worse than naive forecasts

### 2. The Turkey Problem
- Demonstrates how observational data can fail to predict catastrophic events
- A turkey living 1000 good days has no reason to predict day 1001 disaster
- Highlights the importance of considering black swan events in forecasting

### 3. Model Performance Reality Check
- Most deep learning models performed comparably to or slightly worse than naive forecasts
- Adding complexity doesn't necessarily improve results
- Single data point changes can devastate model performance

### 4. Best Practices Learned
- Always start with a simple baseline (naive forecast)
- Proper train/test splitting is crucial for time series
- Feature engineering and data quality often matter more than model complexity
- Evaluation must include multiple metrics and visualization

## Technical Implementation Highlights

### Custom Functions Created
- `make_windows()`: Creates time series windows for supervised learning
- `make_train_test_splits()`: Properly splits time series data
- `evaluate_preds()`: Comprehensive model evaluation with multiple metrics
- `mean_absolute_scaled_error()`: MASE implementation
- `make_preds()`: Standardized prediction interface

### TensorFlow Techniques Used
- Custom layer creation for N-BEATS algorithm
- Functional API for complex model architectures
- tf.data.Dataset for efficient data pipeline
- Model checkpointing for best model selection
- Callback implementation for training optimization

## Challenges Encountered

### 1. Data Limitations
- Only ~2,800 samples available (small for deep learning)
- Irregular patterns in cryptocurrency data
- High volatility makes forecasting inherently difficult

### 2. Model Overfitting Issues
- Complex models tended to overfit training data
- Poor generalization to test data
- Difficulty distinguishing signal from noise

### 3. Evaluation Difficulties
- Metrics can be misleading in volatile environments
- Single metric optimization doesn't guarantee real-world success
- Need for multiple evaluation perspectives

## Lessons for Practitioners

### 1. Start Simple
- Always establish a strong baseline
- Naive models are surprisingly hard to beat
- Complexity should only be added when proven beneficial

### 2. Understand Your Data
- Recognize patterns and irregularities
- Consider domain-specific factors (block rewards, market events)
- Be aware of data limitations and biases

### 3. Embrace Uncertainty
- Quantify prediction intervals
- Acknowledge model limitations
- Consider both aleatoric and epistemic uncertainty

### 4. Validate Thoroughly
- Use proper time series splits
- Multiple evaluation metrics
- Out-of-sample testing
- Stress testing with edge cases

## Conclusion

This project demonstrates that while deep learning offers powerful tools for time series forecasting, applying them successfully requires:
- Deep understanding of the domain
- Careful consideration of problem characteristics
- Realistic expectations about model capabilities
- Robust evaluation methodologies

The key takeaway is that forecasting in open systems like cryptocurrency markets remains an unsolved challenge, and practitioners should approach such problems with appropriate skepticism and thorough validation methodologies.

## Future Directions

1. **Closed System Applications**: More promising results likely in predictable domains
2. **Hybrid Approaches**: Combining ML with traditional econometric methods
3. **Uncertainty Quantification**: Better methods for prediction intervals
4. **Real-time Adaptation**: Models that continuously update with new data

---

*This summary is based on a comprehensive exploration of time series forecasting using TensorFlow, emphasizing practical insights over theoretical perfection.*