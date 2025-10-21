import pytest
import pandas as pd
import numpy as np
from backend.models.naive_model import naive_forecast

def test_naive_forecast_single_step():
    """Test naive forecast with single step prediction"""
    # Create test data - in the actual dataset, index 0 is the most recent (newest to oldest)
    data = pd.DataFrame({'Close': [110, 107, 103, 105, 100]})  # 110 is most recent
    
    result = naive_forecast(data, steps=1)
    
    # Check that prediction matches most recent value (first in dataset)
    assert result['predictions'] == [110]
    assert result['model'] == 'naive'
    assert len(result['historical']) <= 30  # Should return at most 30 historical values
    
    # Check metrics exist
    assert 'mae' in result['metrics']
    assert 'rmse' in result['metrics']
    assert 'mape' in result['metrics']
    assert 'mase' in result['metrics']

def test_naive_forecast_multi_step():
    """Test naive forecast with multi-step prediction"""
    # Create test data - in the actual dataset, index 0 is the most recent (newest to oldest)
    data = pd.DataFrame({'Close': [110, 107, 103, 105, 100]})  # 110 is most recent
    
    result = naive_forecast(data, steps=3)
    
    # Check that all predictions match most recent value (first in dataset)
    expected_predictions = [110, 110, 110]
    assert result['predictions'] == expected_predictions
    assert result['model'] == 'naive'
    
    # Check metrics exist
    assert 'mae' in result['metrics']
    assert 'rmse' in result['metrics']
    assert 'mape' in result['metrics']
    assert 'mase' in result['metrics']

def test_naive_forecast_insufficient_data():
    """Test naive forecast with insufficient data"""
    # Test with only one data point
    data = pd.DataFrame({'Close': [100]})
    
    with pytest.raises(ValueError, match="Need at least 2 data points for naive forecast"):
        naive_forecast(data, steps=1)
    
    # Test with zero steps
    data = pd.DataFrame({'Close': [105, 100]})  # Two points with newest first
    
    with pytest.raises(ValueError, match="Steps must be at least 1"):
        naive_forecast(data, steps=0)

def test_naive_forecast_metrics_calculation():
    """Test that metrics are calculated correctly"""
    # Create simple test data where we can predict the metrics
    data = pd.DataFrame({'Close': [110, 107, 103, 105, 100]})  # 110 is most recent
    
    result = naive_forecast(data, steps=1)
    
    # The historical naive forecast on this data should have specific metrics
    assert isinstance(result['metrics']['mae'], float)
    assert isinstance(result['metrics']['rmse'], float)
    assert isinstance(result['metrics']['mape'], float)
    assert isinstance(result['metrics']['mase'], float)

def test_naive_forecast_numpy_array():
    """Test naive forecast works with numpy array input"""
    # Array input treats first element as most recent (like our dataset)
    prices = np.array([110, 107, 103, 105, 100])  # 110 is most recent
    
    result = naive_forecast(prices, steps=1)
    
    assert result['predictions'] == [110]
    assert result['model'] == 'naive'

if __name__ == "__main__":
    # Run tests
    test_naive_forecast_single_step()
    test_naive_forecast_multi_step()
    test_naive_forecast_insufficient_data()
    test_naive_forecast_metrics_calculation()
    test_naive_forecast_numpy_array()
    print("All tests passed!")