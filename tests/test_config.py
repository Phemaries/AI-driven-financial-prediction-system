import pytest
import pandas as pd
from src.predictions.config.config import get_feature_keys

@pytest.fixture
def sample_dataframe():
    """Fixture that provides a dummy stock DataFrame for testing."""
    data = {
        "stock_open": [100, 102, 105],
        "stock_high": [101, 103, 106],
        "stock_low": [99, 101, 104],
        "stock_close": [100, 102, 105],
        "stock_volume": [1000, 1200, 1100],
        "stock_sma10": [101, 102, 103],
        "stock_sma20": [99, 100, 101],
        "growth_stock_1d": [1.01, 1.02, 1.03],
        "growth_stock_7d": [1.05, 1.06, 1.07],
        "rsi": [55, 60, 65],
    }
    return pd.DataFrame(data)


def test_get_feature_keys_output_type(sample_dataframe):
    """Ensure the function returns a dictionary."""
    result = get_feature_keys(sample_dataframe)
    assert isinstance(result, dict), "Expected output to be a dictionary"


def test_get_feature_keys_contains_required_groups(sample_dataframe):
    """Ensure the result dictionary contains key feature groups."""
    result = get_feature_keys(sample_dataframe)
    expected_keys = ["DATE_KEYS", "OHLCV_KEYS", "SMA_KEYS", "GROWTH_KEYS", "X_5_Feat", "X_MA_Feat"]
    for key in expected_keys:
        assert key in result, f"Missing expected key: {key}"


def test_ohlcv_keys_content(sample_dataframe):
    """Ensure OHLCV keys include expected columns."""
    result = get_feature_keys(sample_dataframe)
    ohlcv = result["OHLCV_KEYS"]
    assert any("close" in k for k in ohlcv), "Expected at least one close column in OHLCV_KEYS"
