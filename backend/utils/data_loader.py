import pandas as pd
import os
import numpy as np

def load_bitcoin_data():
    """
    Load Bitcoin price data from CSV file
    """
    # Define the path to the Bitcoin data file
    # Check in the main project root first
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bitcoin_2020-10-22_2025-10-21.csv')
    
    # If that doesn't work, check the backend directory
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bitcoin_2020-10-22_2025-10-21.csv')
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Ensure we have a 'Close' price column
    if 'Close' not in df.columns:
        # If Close column doesn't exist, try other common names
        if 'close' in df.columns:
            df['Close'] = df['close']
        elif 'Price' in df.columns:
            df['Close'] = df['Price']
        elif df.shape[1] > 0:
            # Use the first numeric column as Close price if no standard name is found
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df['Close'] = df[col]
                    break
    
    # Fill any missing values
    df = df.ffill().bfill()
    
    return df

def prepare_windowed_data(data, window_size=7, horizon=1):
    """
    Convert time series data into windowed format for supervised learning
    """
    prices = data['Close'].values
    windows = []
    horizons = []
    
    for i in range(len(prices) - window_size - horizon + 1):
        window = prices[i:(i + window_size)]
        horizon_data = prices[(i + window_size):(i + window_size + horizon)]
        
        windows.append(window)
        horizons.append(horizon_data)
    
    return np.array(windows), np.array(horizons)

def make_train_test_splits(windows, horizons, test_split=0.2):
    """
    Splits matching pairs of windows and horizons into train and test splits
    This is important for time series as we need to maintain temporal order
    """
    split_size = int(len(windows) * (1 - test_split))  # Default 80% for training
    
    train_windows = windows[:split_size]
    train_horizons = horizons[:split_size]
    test_windows = windows[split_size:]
    test_horizons = horizons[split_size:]
    
    return train_windows, train_horizons, test_windows, test_horizons

def add_features(df):
    """
    Add additional features to the dataframe for more sophisticated models
    """
    df_features = df.copy()
    
    # Add basic technical indicators
    df_features['Price_Change'] = df_features['Close'].pct_change()
    df_features['Price_Diff'] = df_features['Close'].diff()
    
    # Add rolling statistics
    df_features['Rolling_Mean_7'] = df_features['Close'].rolling(window=7).mean()
    df_features['Rolling_Mean_30'] = df_features['Close'].rolling(window=30).mean()
    df_features['Rolling_Std_7'] = df_features['Close'].rolling(window=7).std()
    df_features['Rolling_Std_30'] = df_features['Close'].rolling(window=30).std()
    
    # Add lag features
    df_features['Close_Lag_1'] = df_features['Close'].shift(1)
    df_features['Close_Lag_7'] = df_features['Close'].shift(7)
    
    # Add rolling min/max
    df_features['Rolling_Min_7'] = df_features['Close'].rolling(window=7).min()
    df_features['Rolling_Max_7'] = df_features['Close'].rolling(window=7).max()
    
    # Fill NaN values created by rolling operations
    df_features = df_features.ffill().bfill()
    
    return df_features