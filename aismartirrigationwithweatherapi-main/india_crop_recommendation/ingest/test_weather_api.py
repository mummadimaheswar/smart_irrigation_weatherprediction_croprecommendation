"""
Weather API Test Suite
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

from weather_api import (
    fetch_historical_weather,
    fetch_forecast,
    exponential_backoff,
    request_with_retry,
    STATE_COORDS,
    _generate_synthetic_historical,
    _generate_synthetic_forecast,
)


class TestExponentialBackoff:
    def test_first_attempt(self):
        delay = exponential_backoff(0)
        assert delay == 1.0
    
    def test_second_attempt(self):
        delay = exponential_backoff(1)
        assert delay == 2.0
    
    def test_max_delay(self):
        delay = exponential_backoff(10, max_delay=60.0)
        assert delay == 60.0
    
    def test_custom_base(self):
        delay = exponential_backoff(2, base=2.0)
        assert delay == 8.0


class TestStateCoords:
    def test_major_states_exist(self):
        assert "Maharashtra" in STATE_COORDS
        assert "Karnataka" in STATE_COORDS
        assert "Tamil Nadu" in STATE_COORDS
        assert "Punjab" in STATE_COORDS
    
    def test_coords_are_valid(self):
        for state, (lat, lon) in STATE_COORDS.items():
            assert 6 <= lat <= 37, f"{state} latitude out of India range"
            assert 68 <= lon <= 98, f"{state} longitude out of India range"


class TestSyntheticData:
    def test_generate_historical(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        df = _generate_synthetic_historical("Maharashtra", start, end)
        
        assert len(df) == 10
        assert "date" in df.columns
        assert "t_min_c" in df.columns
        assert "t_max_c" in df.columns
        assert "precip_mm" in df.columns
        assert all(df["t_min_c"] < df["t_max_c"])
    
    def test_generate_forecast(self):
        df = _generate_synthetic_forecast("Maharashtra", 7)
        
        assert len(df) == 7
        assert "date" in df.columns
    
    def test_unknown_state_returns_empty(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        df = _generate_synthetic_historical("InvalidState", start, end)
        
        assert df.empty


class TestFetchHistoricalWeather:
    def test_returns_dataframe(self):
        """Test that fetch returns a DataFrame (using synthetic when no API key)"""
        start = datetime(2020, 6, 1)
        end = datetime(2020, 6, 7)
        df = fetch_historical_weather("Maharashtra", start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_unknown_state(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 7)
        df = fetch_historical_weather("InvalidState", start, end)
        
        assert df.empty
    
    def test_required_columns(self):
        start = datetime(2020, 6, 1)
        end = datetime(2020, 6, 7)
        df = fetch_historical_weather("Karnataka", start, end)
        
        required_cols = ["date", "state", "lat", "lon", "t_min_c", "t_max_c"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestFetchForecast:
    def test_returns_dataframe(self):
        df = fetch_forecast("Maharashtra", days=5)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_forecast_days(self):
        df = fetch_forecast("Punjab", days=7)
        
        assert len(df) <= 7


class TestRequestWithRetry:
    @patch('weather_api.requests.get')
    def test_successful_request(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response
        
        result = request_with_retry("http://test.com", {"key": "value"})
        
        assert result == {"data": "test"}
        mock_get.assert_called_once()
    
    @patch('weather_api.requests.get')
    def test_rate_limit_retry(self, mock_get):
        mock_rate_limited = MagicMock()
        mock_rate_limited.status_code = 429
        
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"data": "success"}
        
        mock_get.side_effect = [mock_rate_limited, mock_success]
        
        with patch('weather_api.time.sleep'):
            result = request_with_retry("http://test.com", {}, max_retries=3)
        
        assert result == {"data": "success"}
        assert mock_get.call_count == 2
    
    @patch('weather_api.requests.get')
    def test_invalid_api_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = request_with_retry("http://test.com", {})
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
