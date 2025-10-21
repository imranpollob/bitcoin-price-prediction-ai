import pandas as pd
import os

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