import os
import sqlite3
import json
import pytest
from backend.database import init_db, save_garmin_sync, add_manual_log, get_recent_history, get_user_profile, DATABASE_PATH

def test_database_persistence():
    # 1. Initialize DB
    init_db()
    assert os.path.exists(DATABASE_PATH)

    # 2. Test User Profile
    profile = get_user_profile()
    assert profile is not None
    assert profile['name'] == 'Digital Twin'

    # 3. Test Garmin Sync Persistence
    mock_data = {
        "hrv": {"lastNightAvg": 50},
        "rhr": {"restingHeartRate": 60},
        "body_battery": {"latestValue": 80}
    }
    save_garmin_sync("2026-04-12", 50, 60, 80, mock_data)
    
    # 4. Test Manual Log Persistence
    add_manual_log("2026-04-12", "food", 0.0, "Ate a healthy breakfast")
    
    # 5. Verify History retrieval
    history = get_recent_history()
    assert len(history['syncs']) >= 1
    assert len(history['logs']) >= 1
    
    # Check raw data storage
    sync = history['syncs'][0]
    assert sync['sync_date'] == "2026-04-12"
    assert json.loads(sync['raw_payload']) == mock_data
    
    log = history['logs'][0]
    assert log['raw_input'] == "Ate a healthy breakfast"
    
    print("Verification successful: Database persistence is working.")

if __name__ == "__main__":
    test_database_persistence()
