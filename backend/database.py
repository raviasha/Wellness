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
    created_at = Column(DateTime, default=datetime.utcnow)
    
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    syncs = relationship("GarminSync", back_populates="user")
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
    
    # Actuals (Filled after next sync)
    actual_hrv_delta = Column(Float)
    actual_rhr_delta = Column(Float)
    compliance_score = Column(Float) # 0.0 to 1.0
    fidelity_score = Column(Float) # 0.0 to 1.0
    
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

class GarminSync(Base):
    __tablename__ = "garmin_syncs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sync_date = Column(String)
    hrv_avg = Column(Integer)
    resting_hr = Column(Integer)
    body_battery = Column(Integer)
    intensity_minutes = Column(Integer, default=0)
    active_calories = Column(Integer, default=0)
    training_load = Column(Float, default=0)
    sleep_score = Column(Integer)
    stress_avg = Column(Integer)
    steps_total = Column(Integer)
    spo2_avg = Column(Float)
    respiration_avg = Column(Float)
    raw_payload = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('user_id', 'sync_date', name='_user_sync_uc'),)
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

    # Migration 1.5: Add extended Garmin biometrics
    if "garmin_syncs" in inspector.get_table_names():
        sync_cols = {c["name"] for c in inspector.get_columns("garmin_syncs")}
        with engine.begin() as conn:
            if "sleep_score" not in sync_cols:
                conn.execute(text("ALTER TABLE garmin_syncs ADD COLUMN sleep_score INTEGER"))
                conn.execute(text("ALTER TABLE garmin_syncs ADD COLUMN stress_avg INTEGER"))
                conn.execute(text("ALTER TABLE garmin_syncs ADD COLUMN steps_total INTEGER"))
                conn.execute(text("ALTER TABLE garmin_syncs ADD COLUMN spo2_avg FLOAT"))
                conn.execute(text("ALTER TABLE garmin_syncs ADD COLUMN respiration_avg FLOAT"))
                print("[Migration] Added 5 custom biometric columns to garmin_syncs")

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
        return [{"id": u.id, "username": u.username} for u in users]
    finally:
        db.close()

def create_user(username, name=None, garmin_email=None, garmin_password=None):
    """Create a new user + profile. Returns the new user dict."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return {"id": existing.id, "username": existing.username, "exists": True}
        
        new_user = User(
            username=username,
            garmin_email=garmin_email or "",
            garmin_password_enc=encrypt_val(garmin_password) if garmin_password else None
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

def update_garmin_creds(user_id, email, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.garmin_email = email
            user.garmin_password_enc = encrypt_val(password)
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

def save_garmin_sync(user_id, sync_date, hrv_avg, resting_hr, body_battery, 
                     intensity_minutes, active_calories, training_load, raw_payload,
                     sleep_score=None, stress_avg=None, steps_total=None,
                     spo2_avg=None, respiration_avg=None):
    db = SessionLocal()
    try:
        existing = db.query(GarminSync).filter(
            GarminSync.user_id == user_id, 
            GarminSync.sync_date == sync_date
        ).first()
        
        if existing:
            # Smart Merge: Only update if the incoming value is non-zero, 
            # or if the existing value is currently null/zero. 
            # This protects high-fidelity data from being overwritten by 429 partials.
            def _merge(old, new):
                return new if (new is not None and new != 0) else old

            existing.hrv_avg = _merge(existing.hrv_avg, hrv_avg)
            existing.resting_hr = _merge(existing.resting_hr, resting_hr)
            existing.body_battery = _merge(existing.body_battery, body_battery)
            existing.intensity_minutes = _merge(existing.intensity_minutes, intensity_minutes)
            existing.active_calories = _merge(existing.active_calories, active_calories)
            existing.training_load = _merge(existing.training_load, training_load)
            existing.sleep_score = _merge(existing.sleep_score, sleep_score)
            existing.stress_avg = _merge(existing.stress_avg, stress_avg)
            existing.steps_total = _merge(existing.steps_total, steps_total)
            existing.spo2_avg = _merge(existing.spo2_avg, spo2_avg)
            existing.respiration_avg = _merge(existing.respiration_avg, respiration_avg)
            
            # Always update raw_payload to reflect latest attempt, but keep it composite
            try:
                old_raw = json.loads(existing.raw_payload) if existing.raw_payload else {}
                new_raw = raw_payload if raw_payload else {}
                combined = {**old_raw, **new_raw}
                existing.raw_payload = json.dumps(combined)
            except:
                existing.raw_payload = json.dumps(raw_payload)
        else:
            new_sync = GarminSync(
                user_id=user_id, sync_date=sync_date, hrv_avg=hrv_avg, resting_hr=resting_hr,
                body_battery=body_battery, intensity_minutes=intensity_minutes,
                active_calories=active_calories, training_load=training_load,
                sleep_score=sleep_score, stress_avg=stress_avg, steps_total=steps_total,
                spo2_avg=spo2_avg, respiration_avg=respiration_avg,
                raw_payload=json.dumps(raw_payload)
            )
            db.add(new_sync)
        db.commit()
    finally:
        db.close()

def add_manual_log(user_id, log_date, log_type, value, raw_input, log_time=None):
    db = SessionLocal()
    try:
        new_log = ManualLog(
            user_id=user_id, log_date=log_date, log_time=log_time, log_type=log_type, 
            value=value, raw_input=raw_input
        )
        db.add(new_log)
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

def save_recommendation(user_id, rec_date, sleep_rec, exercise_rec, nutrition_rec, expected_hrv, expected_rhr):
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
        else:
            new_rec = Recommendation(
                user_id=user_id, rec_date=rec_date,
                sleep_rec=sleep_rec, exercise_rec=exercise_rec,
                nutrition_rec=nutrition_rec,
                expected_hrv_delta=expected_hrv,
                expected_rhr_delta=expected_rhr
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
        syncs = db.query(GarminSync).filter(GarminSync.user_id == user_id).order_by(GarminSync.sync_date.desc()).limit(limit).all()
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
