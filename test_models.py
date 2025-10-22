"""
Quick test script to verify all models work correctly
"""

import sys

sys.path.append(".")
from app import (
    model_0_naive_forecast,
    model_1_dense,
    model_4_conv1d,
    model_5_lstm,
    compare_all_models,
)
import numpy as np


def test_single_model(model_func, name):
    """Test a single model function"""
    print(f"\n{'='*60}")
    print(f"Testing {name}...")
    print("=" * 60)

    try:
        result = model_func()

        print(f"✓ Model trained successfully!")
        print(f"  - Model name: {result['model_name']}")
        print(f"  - Test predictions: {result['predictions'].shape}")
        print(f"  - Future predictions: {result.get('future_predictions', 'None')}")

        if result.get("future_predictions") is not None:
            avg_pred = np.mean(result["future_predictions"])
            print(f"  - Average 7-day prediction: ${avg_pred:.2f}")
            print(f"  - Day 1 prediction: ${result['future_predictions'][0]:.2f}")
            print(f"  - Day 7 prediction: ${result['future_predictions'][-1]:.2f}")

        print(f"  - MAE: {result['metrics'].get('mae', 0):.4f}")
        print(f"  - RMSE: {result['metrics'].get('rmse', 0):.4f}")

        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("BITCOIN PRICE PREDICTION - MODEL TESTING")
    print("=" * 60)

    # Test individual models
    models = [
        (model_0_naive_forecast, "Naive Forecast"),
        (model_1_dense, "Dense Model (Window=7)"),
        (model_4_conv1d, "Conv1D Model"),
        (model_5_lstm, "LSTM Model"),
    ]

    results = []
    for model_func, name in models:
        success = test_single_model(model_func, name)
        results.append((name, success))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print(f"\n🎉 All tests passed! The application is ready to use.")
        print(f"\nTo launch the Gradio interface, run:")
        print(f"  python app.py")
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
