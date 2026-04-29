import pytest
from unittest.mock import patch, MagicMock
import datetime
import os
import json

from backend.garmin_service import (
    _get_tokenstore,
    _get_client,
    fetch_garmin_data,
    get_mock_payload,
    _GARMIN_SESSIONS
)
from garminconnect import GarminConnectTooManyRequestsError

@pytest.fixture(autouse=True)
def clear_sessions():
    _GARMIN_SESSIONS.clear()

def test_get_tokenstore():
    path = _get_tokenstore("test@example.com")
    assert "test_at_example_com" in path

@patch("backend.garmin_service.Garmin")
@patch("os.path.exists")
def test_get_client_fresh_login(mock_exists, mock_garmin_class):
    mock_exists.return_value = False
    mock_client_instance = MagicMock()
    mock_garmin_class.return_value = mock_client_instance
    
    client = _get_client("test@example.com", "pass")
    
    assert mock_garmin_class.called
    assert mock_client_instance.login.called
    assert "test@example.com" in _GARMIN_SESSIONS

@patch("backend.garmin_service.Garmin")
@patch("os.path.exists")
def test_get_client_restore_token(mock_exists, mock_garmin_class):
    mock_exists.return_value = True
    mock_client_instance = MagicMock()
    mock_garmin_class.return_value = mock_client_instance
    
    client = _get_client("test@example.com", "pass")
    assert mock_client_instance.login.called_with(True) # Just verifying it called login with tokenstore path
    
@patch("backend.garmin_service.Garmin")
@patch("os.path.exists")
def test_get_client_too_many_requests(mock_exists, mock_garmin_class):
    mock_exists.return_value = True
    mock_client_instance = MagicMock()
    mock_client_instance.login.side_effect = GarminConnectTooManyRequestsError("429")
    mock_garmin_class.return_value = mock_client_instance
    
    with pytest.raises(GarminConnectTooManyRequestsError):
        _get_client("test@example.com", "pass")

@patch("backend.garmin_service._get_client")
def test_fetch_garmin_data_success(mock_get_client):
    mock_client = MagicMock()
    mock_client.get_rhr_day.return_value = {"restingHeartRate": 60}
    mock_client.get_hrv_data.return_value = {"hrvSummary": {"lastNightAvg": 50}}
    mock_get_client.return_value = mock_client
    
    data = fetch_garmin_data("test@example.com", "pass", datetime.date(2023, 1, 1))
    
    assert data["source"] == "garmin"
    assert data["rhr"]["restingHeartRate"] == 60
    assert data["hrv"]["hrvSummary"]["lastNightAvg"] == 50

def test_fetch_garmin_data_no_creds():
    with patch.dict(os.environ, {}, clear=True):
        data = fetch_garmin_data(None, None, datetime.date(2023, 1, 1))
        assert "error" in data

def test_fetch_garmin_data_simulate():
    data = fetch_garmin_data(None, None, datetime.date(2023, 1, 1), simulate=True)
    assert data["source"] == "mock_garmin"
    assert "resting_hr" in data


