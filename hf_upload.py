"""Direct Upload to HuggingFace Space — token via prompt."""
import os
from huggingface_hub import HfApi

SPACE_ID = "raviasha/Fitness_Coach"
PROJECT_ROOT = "/Users/rampetaravishankar/Desktop/Wellness-Outcome"

WHITELIST = [
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "backend",
    "wellness_env",
    "webapp/package.json",
    "webapp/package-lock.json",
    "webapp/next.config.ts",
    "webapp/tsconfig.json",
    "webapp/src",
    "webapp/public",
    "rl_training",
]

def upload():
    token = input("Paste your HuggingFace WRITE token: ").strip()
    api = HfApi(token=token)

    print(f"\n🚀 Uploading to {SPACE_ID}...\n")
    for item in WHITELIST:
        full_path = os.path.join(PROJECT_ROOT, item)
        if not os.path.exists(full_path):
            print(f"  ⏭  Skipping (not found): {item}")
            continue

        print(f"  📤 Sending: {item} ...", end=" ", flush=True)
        try:
            if os.path.isdir(full_path):
                api.upload_folder(
                    folder_path=full_path,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    path_in_repo=item,
                    commit_message=f"Add {item}",
                )
            else:
                api.upload_file(
                    path_or_fileobj=full_path,
                    path_in_repo=item,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    commit_message=f"Add {item}",
                )
            print("✅")
        except Exception as e:
            print(f"❌ {e}")

    print(f"\n🎉 Done! Check: https://huggingface.co/spaces/{SPACE_ID}/tree/main")

if __name__ == "__main__":
    upload()
