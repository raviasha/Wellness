import sqlite3
import json
import os
import random
from datetime import datetime, timedelta
from backend.database import init_db, save_wearable_sync, add_manual_log

def bootstrap_high_fidelity():
    print("Bootstrapping High-Fidelity Sandbox Data (15 Days)...")
    
    db_path = "wellness.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    
    # Base Stats
    hrv = 45.0
    rhr = 60.0
    weight = 80.0
    start_date = datetime.now() - timedelta(days=15)
    
    for i in range(15):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Action Features for Day T
        sleep_quality = random.uniform(6.0, 9.0) # Hours
        stress_lvl = random.randint(1, 10)
        calories = random.randint(1800, 2800)
        intensity_mins = random.randint(0, 90)
        
        # Outcome Logic (Simplified Simulation)
        # HRV improves with sleep, drops with stress.
        hrv_delta = (sleep_quality * 3.5) - (stress_lvl * 1.5) - (intensity_mins * 0.05)
        hrv = max(30, min(100, hrv + (hrv_delta * 0.2))) # Slow convergence
        
        # Weight change
        weight_delta = (calories - 2200) / 7700.0 # 7700 kcal = 1kg
        weight += weight_delta
        
        # Save Wearable Sync
        save_wearable_sync(
            user_id=1,
            sync_date=date_str,
            source="garmin",
            hrv_rmssd=int(hrv),
            resting_hr=int(rhr),
            recovery_score=int(sleep_quality * 10),
            active_minutes=intensity_mins,
            active_calories=int(intensity_mins * 5),
            strain_score=intensity_mins * 0.8,
            raw_payload={"synthetic": True}
        )
        
        # Save Manual Logs
        add_manual_log(date_str, "stress", stress_lvl, "Synthetic Stress")
        add_manual_log(date_str, "food", calories, f"Synthetic {calories} kcal")
        add_manual_log(date_str, "weight", round(weight, 1), "Synthetic Weight")

    print("Success: 15 days of High-Fidelity history generated.")

if __name__ == "__main__":
    bootstrap_high_fidelity()
