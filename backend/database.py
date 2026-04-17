import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from cryptography.fernet import Fernet

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    db_path = os.path.join(os.path.dirname(__file__), "..", "wellness.db")
    DATABASE_URL = f"sqlite:///{db_path}"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Security Setup ---
# MASTER_KEY should be a base64 encoded Fernet key. 
# Generate one with Fernet.generate_key()
MASTER_KEY = os.environ.get("ENCRYPTION_KEY")
if not MASTER_KEY:
    # Fallback for dev only - but we'll log it so user can save it to .env
    MASTER_KEY = b'G0G1k198f7rWh7hX55y8oW-m_m-6U9IuJ6Z-q_Y6L7s=' # Placeholder
    # print("WARNING: ENCRYPTION_KEY not set. Using placeholder. DATA NOT SECURE.")

cipher_suite = Fernet(MASTER_KEY)

def encrypt_val(val: str) -> str:
    if not val: return None
    return cipher_suite.encrypt(val.encode()).decode()

def decrypt_val(val: str) -> str:
    if not val: return None
    return cipher_suite.decrypt(val.encode()).decode()

# --- Models ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True)
    garmin_email = Column(String)
    garmin_password_enc = Column(String)
    terra_user_id = Column(String)
    wearable_source = Column(String, default="garmin")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    syncs = relationship("WearableSync", back_populates="user")
    logs = relationship("ManualLog", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rec_date = Column(String) # YYYY-MM-DD
    sleep_rec = Column(String)
    exercise_rec = Column(String)
    nutrition_rec = Column(String)
    
    # Predictions
    expected_hrv_delta = Column(Float)
    expected_rhr_delta = Column(Float)
    expected_sleep_delta = Column(Float)
    expected_stress_delta = Column(Float)
    expected_weight_delta = Column(Float)
    expected_energy_delta = Column(Float)
    
    # Actuals (Filled after next sync)
    actual_hrv_delta = Column(Float)
    actual_rhr_delta = Column(Float)
    actual_sleep_delta = Column(Float)
    actual_stress_delta = Column(Float)
    actual_weight_delta = Column(Float)
    compliance_score = Column(Float) # 0.0 to 1.0
    fidelity_score = Column(Float) # 0.0 to 1.0
    
    # Per-input compliance breakdown
    compliance_sleep = Column(Float)
    compliance_exercise = Column(Float)
    compliance_nutrition = Column(Float)
    
    # LLM-generated long-term impact text
    long_term_impact = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="recommendations")

class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    name = Column(String)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    goal = Column(String)
    compliance_rate = Column(Float)
    simulator_approved = Column(Integer, default=0) # 0=False, 1=True
    last_calibration_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="profile")

# Legacy alias for migration compatibility
GarminSync = None  # replaced by WearableSync — downstream imports will be updated

class WearableSync(Base):
    __tablename__ = "wearable_syncs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sync_date = Column(String)
    source = Column(String, default="garmin")  # garmin, apple_health, oneplus, fitbit, etc.
    
    # Core biometrics (renamed to device-agnostic names)
    hrv_rmssd = Column(Integer)          # was hrv_avg — ms RMSSD
    resting_hr = Column(Integer)         # bpm (unchanged)
    recovery_score = Column(Integer)     # was body_battery — 0-100
    active_minutes = Column(Integer, default=0)   # was intensity_minutes
    active_calories = Column(Integer, default=0)   # kcal (unchanged)
    strain_score = Column(Float, default=0)        # was training_load
    sleep_score = Column(Integer)        # 0-100 (unchanged)
    stress_avg = Column(Integer)         # 0-100 (unchanged)
    steps = Column(Integer)              # was steps_total
    spo2 = Column(Float)                 # was spo2_avg — %
    respiration_rate = Column(Float)     # was respiration_avg — breaths/min
    
    # New columns — Terra-provided, not available from legacy Garmin scraping
    vo2_max = Column(Float)              # ml/kg/min
    sleep_deep_pct = Column(Float)       # % deep sleep
    sleep_rem_pct = Column(Float)        # % REM sleep
    sleep_light_pct = Column(Float)      # % light sleep
    sleep_duration_hours = Column(Float) # total hours
    skin_temp_delta = Column(Float)      # °C deviation from baseline
    avg_hr = Column(Integer)             # average heart rate bpm
    hr_max = Column(Integer)             # max HR during day bpm
    calories_total = Column(Integer)     # total calories (BMR + active)
    distance_meters = Column(Float)      # total distance
    floors_climbed = Column(Integer)     # count
    
    raw_payload = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('user_id', 'sync_date', name='_user_wearable_sync_uc'),)
    user = relationship("User", back_populates="syncs")

class ManualLog(Base):
    __tablename__ = "manual_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    log_date = Column(String)
    log_time = Column(String)
    log_type = Column(String) 
    value = Column(Float)
    raw_input = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="logs")

# --- Database API ---

def init_db():
    Base.metadata.create_all(bind=engine)
    
    # --- Schema Migrations (for existing databases) ---
    # create_all() only creates NEW tables; it doesn't add columns to existing ones.
    # We must ALTER TABLE manually for any columns added after initial deployment.
    from sqlalchemy import text, inspect
    inspector = inspect(engine)
    
    # Migration 1: Add simulator_approved + last_calibration_at to user_profile
    if "user_profile" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("user_profile")}
        with engine.begin() as conn:
            if "simulator_approved" not in existing_cols:
                conn.execute(text("ALTER TABLE user_profile ADD COLUMN simulator_approved INTEGER DEFAULT 0"))
                print("[Migration] Added simulator_approved to user_profile")
            if "last_calibration_at" not in existing_cols:
                conn.execute(text("ALTER TABLE user_profile ADD COLUMN last_calibration_at TIMESTAMP"))
                print("[Migration] Added last_calibration_at to user_profile")

    # Migration 1.5: Migrate garmin_syncs → wearable_syncs (if old table exists)
    if "garmin_syncs" in inspector.get_table_names() and "wearable_syncs" not in inspector.get_table_names():
        # Create new table first
        WearableSync.__table__.create(bind=engine, checkfirst=True)
        with engine.begin() as conn:
            # Copy existing data with column renames
            conn.execute(text("""
                INSERT INTO wearable_syncs (
                    user_id, sync_date, source, hrv_rmssd, resting_hr, recovery_score,
                    active_minutes, active_calories, strain_score, sleep_score, stress_avg,
                    steps, spo2, respiration_rate, raw_payload, created_at
                )
                SELECT 
                    user_id, sync_date, 'garmin', hrv_avg, resting_hr, body_battery,
                    intensity_minutes, active_calories, training_load, sleep_score, stress_avg,
                    steps_total, spo2_avg, respiration_avg, raw_payload, created_at
                FROM garmin_syncs
            """))
            print("[Migration] Copied garmin_syncs → wearable_syncs with column renames")

    # Migration 1.6: Add terra fields to users table
    if "users" in inspector.get_table_names():
        user_cols = {c["name"] for c in inspector.get_columns("users")}
        with engine.begin() as conn:
            if "terra_user_id" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN terra_user_id VARCHAR"))
                print("[Migration] Added terra_user_id to users")
            if "wearable_source" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN wearable_source VARCHAR DEFAULT 'garmin'"))
                print("[Migration] Added wearable_source to users")

    # Migration 3: Add log_time to manual_logs
    if 'log_time' not in [col['name'] for col in inspector.get_columns('manual_logs')]:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE manual_logs ADD COLUMN log_time VARCHAR;"))
            conn.commit()
        print("[Migration] Added log_time column to manual_logs")
    
    # Migration 2: Ensure recommendations table exists (create_all handles this, but be safe)
    if "recommendations" not in inspector.get_table_names():
        Recommendation.__table__.create(bind=engine, checkfirst=True)
        print("[Migration] Created recommendations table")

    # Migration 4: Add expanded delta, per-input compliance, and long-term impact columns to recommendations
    if "recommendations" in inspector.get_table_names():
        rec_cols = {c["name"] for c in inspector.get_columns("recommendations")}
        new_rec_cols = {
            "expected_sleep_delta": "FLOAT",
            "expected_stress_delta": "FLOAT",
            "expected_weight_delta": "FLOAT",
            "expected_energy_delta": "FLOAT",
            "actual_sleep_delta": "FLOAT",
            "actual_stress_delta": "FLOAT",
            "actual_weight_delta": "FLOAT",
            "compliance_sleep": "FLOAT",
            "compliance_exercise": "FLOAT",
            "compliance_nutrition": "FLOAT",
            "long_term_impact": "TEXT",
        }
        with engine.begin() as conn:
            for col_name, col_type in new_rec_cols.items():
                if col_name not in rec_cols:
                    conn.execute(text(f"ALTER TABLE recommendations ADD COLUMN {col_name} {col_type}"))
                    print(f"[Migration] Added {col_name} to recommendations")
    
    db = SessionLocal()
    try:
        # Create Demo Users A, B
        demo_users = [
            {"username": "Athlete_Alice", "name": "Alice (Endurance)", "age": 28, "goal": "athletic_performance"},
            {"username": "Developer_Dave", "name": "Dave (Desk Worker)", "age": 42, "goal": "stress_management"}
        ]
        
        for d in demo_users:
            if not db.query(User).filter(User.username == d["username"]).first():
                new_user = User(username=d["username"], garmin_email="")
                db.add(new_user)
                db.flush()
                
                profile = UserProfile(
                    user_id=new_user.id, name=d["name"], age=d["age"],
                    weight=70.0, height=170.0, goal=d["goal"], compliance_rate=0.85
                )
                db.add(profile)
        db.commit()
    finally:
        db.close()

def get_users():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        result = []
        for u in users:
            # Most recent wearable sync row for this user
            last_sync = (
                db.query(WearableSync)
                .filter(WearableSync.user_id == u.id)
                .order_by(WearableSync.created_at.desc())
                .first()
            )
            last_synced_at = last_sync.created_at.isoformat() if last_sync and last_sync.created_at else None
            last_sync_date = last_sync.sync_date if last_sync else None
            result.append({
                "id": u.id,
                "username": u.username,
                "wearable_source": u.wearable_source or "garmin",
                "last_synced_at": last_synced_at,
                "last_sync_date": last_sync_date,
            })
        return result
    finally:
        db.close()

def create_user(username, name=None, garmin_email=None, garmin_password=None, wearable_source="garmin"):
    """Create a new user + profile. Returns the new user dict."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return {"id": existing.id, "username": existing.username, "exists": True}
        
        new_user = User(
            username=username,
            garmin_email=garmin_email or "",
            garmin_password_enc=encrypt_val(garmin_password) if garmin_password else None,
            wearable_source=wearable_source
        )
        db.add(new_user)
        db.flush()
        
        profile = UserProfile(
            user_id=new_user.id,
            name=name or username,
            goal="overall_wellness",
            compliance_rate=0.85
        )
        db.add(profile)
        db.commit()
        return {"id": new_user.id, "username": new_user.username, "exists": False}
    finally:
        db.close()

def set_user_device(user_id, wearable_source):
    """Update the user's wearable device type."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.wearable_source = wearable_source
            db.commit()
            return True
        return False
    finally:
        db.close()

def update_garmin_creds(user_id, email, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.garmin_email = email
            user.garmin_password_enc = encrypt_val(password)
            user.wearable_source = "garmin"
            db.commit()
    finally:
        db.close()

def get_garmin_creds(user_id):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.garmin_email:
            return user.garmin_email, decrypt_val(user.garmin_password_enc)
        return None, None
    finally:
        db.close()

def update_terra_creds(user_id, terra_user_id, wearable_source):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.terra_user_id = terra_user_id
            user.wearable_source = wearable_source
            db.commit()
    finally:
        db.close()

def get_wearable_creds(user_id):
    """Returns (terra_user_id, wearable_source) for Terra users,
    or (garmin_email, garmin_password) for legacy Garmin users."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None, None, None
        if user.terra_user_id:
            return "terra", user.terra_user_id, user.wearable_source
        if user.garmin_email:
            return "garmin", user.garmin_email, decrypt_val(user.garmin_password_enc)
        return None, None, None
    finally:
        db.close()

def save_wearable_sync(user_id, sync_date, source, raw_payload,
                      hrv_rmssd=None, resting_hr=None, recovery_score=None,
                      active_minutes=None, active_calories=None, strain_score=None,
                      sleep_score=None, stress_avg=None, steps=None,
                      spo2=None, respiration_rate=None,
                      vo2_max=None, sleep_deep_pct=None, sleep_rem_pct=None,
                      sleep_light_pct=None, sleep_duration_hours=None,
                      skin_temp_delta=None, avg_hr=None, hr_max=None,
                      calories_total=None, distance_meters=None, floors_climbed=None):
    db = SessionLocal()
    try:
        existing = db.query(WearableSync).filter(
            WearableSync.user_id == user_id, 
            WearableSync.sync_date == sync_date
        ).first()
        
        if existing:
            # Smart Merge: Only update if the incoming value is non-zero, 
            # or if the existing value is currently null/zero. 
            # This protects high-fidelity data from being overwritten by partial syncs.
            def _merge(old, new):
                return new if (new is not None and new != 0) else old

            existing.source = source or existing.source
            existing.hrv_rmssd = _merge(existing.hrv_rmssd, hrv_rmssd)
            existing.resting_hr = _merge(existing.resting_hr, resting_hr)
            existing.recovery_score = _merge(existing.recovery_score, recovery_score)
            existing.active_minutes = _merge(existing.active_minutes, active_minutes)
            existing.active_calories = _merge(existing.active_calories, active_calories)
            existing.strain_score = _merge(existing.strain_score, strain_score)
            existing.sleep_score = _merge(existing.sleep_score, sleep_score)
            existing.stress_avg = _merge(existing.stress_avg, stress_avg)
            existing.steps = _merge(existing.steps, steps)
            existing.spo2 = _merge(existing.spo2, spo2)
            existing.respiration_rate = _merge(existing.respiration_rate, respiration_rate)
            # New extended fields
            existing.vo2_max = _merge(existing.vo2_max, vo2_max)
            existing.sleep_deep_pct = _merge(existing.sleep_deep_pct, sleep_deep_pct)
            existing.sleep_rem_pct = _merge(existing.sleep_rem_pct, sleep_rem_pct)
            existing.sleep_light_pct = _merge(existing.sleep_light_pct, sleep_light_pct)
            existing.sleep_duration_hours = _merge(existing.sleep_duration_hours, sleep_duration_hours)
            existing.skin_temp_delta = _merge(existing.skin_temp_delta, skin_temp_delta)
            existing.avg_hr = _merge(existing.avg_hr, avg_hr)
            existing.hr_max = _merge(existing.hr_max, hr_max)
            existing.calories_total = _merge(existing.calories_total, calories_total)
            existing.distance_meters = _merge(existing.distance_meters, distance_meters)
            existing.floors_climbed = _merge(existing.floors_climbed, floors_climbed)
            
            # Always update raw_payload to reflect latest attempt, but keep it composite
            try:
                old_raw = json.loads(existing.raw_payload) if existing.raw_payload else {}
                new_raw = raw_payload if raw_payload else {}
                combined = {**old_raw, **new_raw}
                existing.raw_payload = json.dumps(combined)
            except:
                existing.raw_payload = json.dumps(raw_payload)
        else:
            new_sync = WearableSync(
                user_id=user_id, sync_date=sync_date, source=source,
                hrv_rmssd=hrv_rmssd, resting_hr=resting_hr, recovery_score=recovery_score,
                active_minutes=active_minutes or 0, active_calories=active_calories or 0,
                strain_score=strain_score or 0, sleep_score=sleep_score, stress_avg=stress_avg,
                steps=steps, spo2=spo2, respiration_rate=respiration_rate,
                vo2_max=vo2_max, sleep_deep_pct=sleep_deep_pct, sleep_rem_pct=sleep_rem_pct,
                sleep_light_pct=sleep_light_pct, sleep_duration_hours=sleep_duration_hours,
                skin_temp_delta=skin_temp_delta, avg_hr=avg_hr, hr_max=hr_max,
                calories_total=calories_total, distance_meters=distance_meters,
                floors_climbed=floors_climbed,
                raw_payload=json.dumps(raw_payload)
            )
            db.add(new_sync)
        db.commit()
    finally:
        db.close()

# Legacy alias for backward compatibility during migration
def save_garmin_sync(user_id, sync_date, hrv_avg, resting_hr, body_battery, 
                     intensity_minutes, active_calories, training_load, raw_payload,
                     sleep_score=None, sleep_duration_hours=None, stress_avg=None,
                     steps_total=None, spo2_avg=None, respiration_avg=None):
    """Legacy wrapper — maps old Garmin column names to new WearableSync names."""
    save_wearable_sync(
        user_id=user_id, sync_date=sync_date, source="garmin", raw_payload=raw_payload,
        hrv_rmssd=hrv_avg, resting_hr=resting_hr, recovery_score=body_battery,
        active_minutes=intensity_minutes, active_calories=active_calories,
        strain_score=training_load, sleep_score=sleep_score,
        sleep_duration_hours=sleep_duration_hours,
        stress_avg=stress_avg,
        steps=steps_total, spo2=spo2_avg, respiration_rate=respiration_avg
    )

def add_manual_log(user_id, log_date, log_type, value, raw_input, log_time=None):
    """
    Insert or update a manual log entry.
    - food: LLM decides if the new text is the same meal (overwrite) or a new meal (append).
    - weight / note: upsert — one entry per day, always overwrites.
    """
    from backend.llm_nutrition import decide_food_action

    db = SessionLocal()
    try:
        if log_type == "weight" or log_type == "note":
            existing = (
                db.query(ManualLog)
                .filter(
                    ManualLog.user_id == user_id,
                    ManualLog.log_date == log_date,
                    ManualLog.log_type == log_type,
                )
                .first()
            )
            if existing:
                existing.value = value
                existing.raw_input = raw_input
                existing.log_time = log_time
                existing.created_at = datetime.utcnow()
            else:
                db.add(ManualLog(
                    user_id=user_id, log_date=log_date, log_time=log_time, log_type=log_type,
                    value=value, raw_input=raw_input
                ))
        else:
            # food: ask LLM to compare against today's existing entries
            existing_rows = (
                db.query(ManualLog)
                .filter(
                    ManualLog.user_id == user_id,
                    ManualLog.log_date == log_date,
                    ManualLog.log_type == "food",
                )
                .all()
            )

            # Extract text from each existing entry for the LLM to compare
            def _extract_text(row):
                try:
                    raw = json.loads(row.raw_input) if isinstance(row.raw_input, str) else row.raw_input
                    return (raw or {}).get("text") or row.raw_input or ""
                except Exception:
                    return row.raw_input or ""

            existing_for_llm = [{"id": r.id, "text": _extract_text(r)} for r in existing_rows]

            # Get the text of the new entry
            try:
                new_raw = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
                new_text = (new_raw or {}).get("text") or raw_input or ""
            except Exception:
                new_text = raw_input or ""

            decision = decide_food_action(new_text, existing_for_llm)

            if decision["action"] == "overwrite":
                target_id = decision["target_id"]
                target = next((r for r in existing_rows if r.id == target_id), None)
                if target:
                    target.value = value
                    target.raw_input = raw_input
                    target.log_time = log_time
                    target.created_at = datetime.utcnow()
                else:
                    db.add(ManualLog(
                        user_id=user_id, log_date=log_date, log_time=log_time, log_type=log_type,
                        value=value, raw_input=raw_input
                    ))
            else:
                db.add(ManualLog(
                    user_id=user_id, log_date=log_date, log_time=log_time, log_type=log_type,
                    value=value, raw_input=raw_input
                ))
        db.commit()
    finally:
        db.close()

def get_user_profile(user_id):
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile:
            return {c.name: getattr(profile, c.name) for c in profile.__table__.columns}
        return None
    finally:
        db.close()

def update_user_profile(user_id, data: dict):
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.add(profile)
        for key in ("name", "age", "weight", "height", "goal", "compliance_rate"):
            if key in data:
                setattr(profile, key, data[key])
        db.commit()
    finally:
        db.close()

def approve_simulator(user_id, approved: bool):
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile:
            profile.simulator_approved = 1 if approved else 0
            if approved:
                profile.last_calibration_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

def save_recommendation(user_id, rec_date, sleep_rec, exercise_rec, nutrition_rec, expected_hrv, expected_rhr,
                        expected_sleep=None, expected_stress=None, expected_weight=None, expected_energy=None,
                        long_term_impact=None):
    db = SessionLocal()
    try:
        # Only one rec per day - upsert
        existing = db.query(Recommendation).filter(
            Recommendation.user_id == user_id, 
            Recommendation.rec_date == rec_date
        ).first()
        
        if existing:
            existing.sleep_rec = sleep_rec
            existing.exercise_rec = exercise_rec
            existing.nutrition_rec = nutrition_rec
            existing.expected_hrv_delta = expected_hrv
            existing.expected_rhr_delta = expected_rhr
            existing.expected_sleep_delta = expected_sleep
            existing.expected_stress_delta = expected_stress
            existing.expected_weight_delta = expected_weight
            existing.expected_energy_delta = expected_energy
            if long_term_impact:
                existing.long_term_impact = long_term_impact
        else:
            new_rec = Recommendation(
                user_id=user_id, rec_date=rec_date,
                sleep_rec=sleep_rec, exercise_rec=exercise_rec,
                nutrition_rec=nutrition_rec,
                expected_hrv_delta=expected_hrv,
                expected_rhr_delta=expected_rhr,
                expected_sleep_delta=expected_sleep,
                expected_stress_delta=expected_stress,
                expected_weight_delta=expected_weight,
                expected_energy_delta=expected_energy,
                long_term_impact=long_term_impact,
            )
            db.add(new_rec)
        db.commit()
    finally:
        db.close()

def get_recommendations(user_id, limit=30):
    db = SessionLocal()
    try:
        recs = db.query(Recommendation).filter(
            Recommendation.user_id == user_id
        ).order_by(Recommendation.rec_date.desc()).limit(limit).all()
        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in recs]
    finally:
        db.close()

def get_recent_history(user_id, limit=30):
    db = SessionLocal()
    try:
        syncs = db.query(WearableSync).filter(WearableSync.user_id == user_id).order_by(WearableSync.sync_date.desc()).limit(limit).all()
        logs = db.query(ManualLog).filter(ManualLog.user_id == user_id).order_by(ManualLog.log_date.desc()).limit(limit).all()
        
        return {
            "syncs": [{c.name: getattr(s, c.name) for c in s.__table__.columns} for s in syncs],
            "logs": [{c.name: getattr(l, c.name) for c in l.__table__.columns} for l in logs]
        }
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Multi-tenant Database Initialized.")
