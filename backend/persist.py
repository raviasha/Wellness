"""Utility to persist critical files back to the HF Space repo.

After training a model or accumulating data, this uploads key files 
back to the repo so they survive container restarts/rebuilds.
"""

import os
import glob
from huggingface_hub import HfApi

SPACE_ID = os.environ.get("SPACE_ID", "raviasha/Fitness_Coach")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set in Space secrets


def persist_to_repo():
    """Upload models and database to the HF repo for persistence."""
    if not HF_TOKEN:
        print("[Persist] No HF_TOKEN set. Skipping persistence.")
        return {"status": "skipped", "reason": "No HF_TOKEN"}

    api = HfApi(token=HF_TOKEN)
    uploaded = []

    try:
        # 1. Persist all user models (calibrated_persona.json + ppo_wellness_lite.pt)
        model_files = glob.glob("models/user_*/*")
        for f in model_files:
            try:
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=f,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    commit_message=f"[auto-persist] {os.path.basename(f)}",
                )
                uploaded.append(f)
                print(f"[Persist] ✅ {f}")
            except Exception as e:
                print(f"[Persist] ⚠️ Failed {f}: {e}")

        # 2. Persist the database
        db_path = os.path.join(os.path.dirname(__file__), "..", "wellness.db")
        if os.path.exists(db_path):
            try:
                api.upload_file(
                    path_or_fileobj=db_path,
                    path_in_repo="wellness.db",
                    repo_id=SPACE_ID,
                    repo_type="space",
                    commit_message="[auto-persist] wellness.db snapshot",
                )
                uploaded.append("wellness.db")
                print("[Persist] ✅ wellness.db")
            except Exception as e:
                print(f"[Persist] ⚠️ Failed wellness.db: {e}")

        return {"status": "success", "uploaded": uploaded}

    except Exception as e:
        print(f"[Persist] Error: {e}")
        return {"status": "error", "message": str(e)}


def persist_model(user_id: int):
    """Persist a specific user's model files after training."""
    if not HF_TOKEN:
        print("[Persist] No HF_TOKEN set. Skipping.")
        return

    api = HfApi(token=HF_TOKEN)
    model_dir = os.path.join("models", f"user_{user_id}")

    if not os.path.isdir(model_dir):
        return

    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            try:
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=fpath,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    commit_message=f"[auto-persist] user_{user_id}/{fname}",
                )
                print(f"[Persist] ✅ {fpath}")
            except Exception as e:
                print(f"[Persist] ⚠️ Failed {fpath}: {e}")
