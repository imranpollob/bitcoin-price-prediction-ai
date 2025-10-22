import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

def test_dense_model():
    """Test the dense model implementation"""
    try:
        from backend.models.dense_model import dense_model_1_prediction
        from backend.utils.data_loader import load_bitcoin_data
        
        print("Loading Bitcoin data...")
        df = load_bitcoin_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        print("Testing Dense Model (Window=7, Horizon=1)...")
        result = dense_model_1_prediction(df, window_size=7, horizon=1)
        
        print(f"Model: {result['model']}")
        print(f"Predictions: {result['predictions']}")
        print(f"Historical data points: {len(result['historical'])}")
        print(f"Metrics: {result['metrics']}")
        
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dense_model()