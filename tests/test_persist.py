import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Mock huggingface_hub before importing persist
sys.modules['huggingface_hub'] = MagicMock()

from backend.persist import persist_to_repo, persist_model

@patch("backend.persist.HF_TOKEN", "")
def test_persist_to_repo_no_token():
    res = persist_to_repo()
    assert res["status"] == "skipped"

@patch("backend.persist.HF_TOKEN", "fake_token")
@patch("backend.persist.SPACE_ID", "test/space")
@patch("backend.persist.HfApi")
@patch("backend.persist.glob.glob")
@patch("os.path.exists")
def test_persist_to_repo_success(mock_exists, mock_glob, mock_hf_api):
    mock_glob.return_value = ["models/user_1/model.pt"]
    mock_exists.return_value = True
    
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance
    
    res = persist_to_repo()
    
    assert res["status"] == "success"
    assert "models/user_1/model.pt" in res["uploaded"]
    assert "wellness.db" in res["uploaded"]
    
    assert mock_api_instance.upload_file.call_count == 2

@patch("backend.persist.HF_TOKEN", "fake_token")
@patch("backend.persist.SPACE_ID", "test/space")
@patch("backend.persist.HfApi")
@patch("backend.persist.glob.glob")
@patch("os.path.exists")
def test_persist_to_repo_upload_error(mock_exists, mock_glob, mock_hf_api):
    mock_glob.return_value = ["models/user_1/model.pt"]
    mock_exists.return_value = True
    
    mock_api_instance = MagicMock()
    mock_api_instance.upload_file.side_effect = Exception("Upload failed")
    mock_hf_api.return_value = mock_api_instance
    
    res = persist_to_repo()
    
    assert res["status"] == "success" # the function catches the exception and continues
    assert res["uploaded"] == [] # nothing actually uploaded successfully

@patch("backend.persist.HF_TOKEN", "fake_token")
@patch("backend.persist.SPACE_ID", "test/space")
@patch("backend.persist.HfApi")
def test_persist_to_repo_hf_init_error(mock_hf_api):
    mock_hf_api.side_effect = Exception("Auth failed")
    with pytest.raises(Exception, match="Auth failed"):
        persist_to_repo()

@patch("backend.persist.HF_TOKEN", "")
def test_persist_model_no_token():
    # Should just return None without throwing
    assert persist_model(1) is None

@patch("backend.persist.HF_TOKEN", "fake_token")
@patch("backend.persist.SPACE_ID", "test/space")
@patch("backend.persist.HfApi")
@patch("os.path.isdir")
@patch("os.path.isfile")
@patch("os.listdir")
def test_persist_model_success(mock_listdir, mock_isfile, mock_isdir, mock_hf_api):
    mock_isdir.return_value = True
    mock_isfile.return_value = True
    mock_listdir.return_value = ["model.pt", "calibrated.json"]
    
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance
    
    persist_model(1)
    
    assert mock_api_instance.upload_file.call_count == 2
