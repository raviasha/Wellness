import sqlite3
import json
import os
from datetime import datetime, timedelta
from backend.database import init_db, save_garmin_sync, add_manual_log
from backend.calibration import calibrate_user_persona

def verify_regression():
    print("Verification: Testing Regression Model Fidelity (Fixed Logic)...")
    
    db_path = "wellness.db"
    if os.path.exists(db_path): os.remove(db_path)
    init_db()
    
    # Ground Truth: 
    # Delta HRV = 4.0 * SleepProxy
    
    hrv = 40.0
    start_date = datetime.now() - timedelta(days=20)
    
    for i in range(15):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        
        # We vary body_battery between 30 and 80
        # In engine: sleep_proxy = 7.0 + (battery_delta / 10.0)
        # We will make battery_delta = 10 (sleep_proxy = 8) on even days, -10 (sleep_proxy = 6) on odd
        
        battery = 50 + (10 if i % 2 == 0 else -10)
        
        # Rule in engine logic: 
        # sleep_proxy_t = 7.0 + (battery_t - battery_{t-1}) / 10.0
        # hrv_t = hrv_{t-1} + (sleep_proxy_t * 4.0)
        
        # We simulate this sequence
        proxy = 8.0 if i % 2 == 0 else 6.0
        hrv += (proxy * 4.0)
        
        save_garmin_sync(
            sync_date=date_str,
            hrv_avg=int(hrv),
            resting_hr=60,
            body_battery=battery,
            raw_payload={"synthetic": True}
        )
        
        add_manual_log(date_str, "stress", 5, "test")
        add_manual_log(date_str, "food", 2000, "test")

    print("\nRunning Calibration Engine...")
    result = calibrate_user_persona()
    
    if "error" in result:
        print(f"FAILED: {result['error']}")
        return

    params = result["params"]
    print("\nRecovered Parameters:")
    print(f"  hrv_sleep_sensitivity: {params['hrv_sleep_sensitivity']} (Expected: ~4.0)")
    
    if 3.5 <= params['hrv_sleep_sensitivity'] <= 4.5:
        print("\n✅ SUCCESS: Regression recovered the rules from delta-based data.")
    else:
        print("\n❌ FAILED: Accuracy out of bounds.")

if __name__ == "__main__":
    verify_regression()
