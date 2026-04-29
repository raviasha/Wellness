import os
import json
import pytest
from backend.database import init_db, create_user, save_wearable_sync, add_manual_log, get_recent_history, get_user_profile

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "wellness.db")
USER_ID = 1

def test_database_persistence():
    # 1. Initialize DB
    init_db()
    assert os.path.exists(DB_PATH)

    # 2. Ensure user exists (create if not present) and track the actual user id
    existing = get_user_profile(USER_ID)
    if existing is None:
        result = create_user(username=f"testuser_{USER_ID}", name="Test User")
        test_user_id = result["id"]
    else:
        test_user_id = USER_ID
    profile = get_user_profile(test_user_id)
    assert profile is not None

    # 3. Test Wearable Sync Persistence
    mock_data = {
        "hrv": {"lastNightAvg": 50},
        "rhr": {"restingHeartRate": 60},
        "recovery": {"score": 80}
    }
    save_wearable_sync(
        user_id=test_user_id,
        sync_date="2026-04-12",
        source="garmin",
        hrv_rmssd=50,
        resting_hr=60,
        recovery_score=80,
        raw_payload=mock_data
    )

    # 4. Test Manual Log Persistence
    add_manual_log(test_user_id, "2026-04-12", "food", 0.0, "Ate a healthy breakfast")

    # 5. Verify History retrieval
    history = get_recent_history(test_user_id)
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
