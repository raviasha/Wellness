"""Copy source files to hf_deploy using Python (bypasses macOS fs issues)."""
import os, shutil

SRC = "/Users/rampetaravishankar/Desktop/Wellness-Outcome"
DST = os.path.join(SRC, "hf_deploy")

SKIP_DIRS = {"node_modules", ".next", ".git", "__pycache__", ".pytest_cache",
             "demo_output", "hf_deploy", ".git-rewrite", "scratch", "out", "models"}
SKIP_EXTS = {".pyc", ".pyo", ".pt", ".png", ".jpg", ".sst", ".meta", ".log"}
SKIP_FILES = {".env", "wellness.db", "hf_upload.py", "deploy_copy.py", ".DS_Store"}

count = 0
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if f in SKIP_FILES or any(f.endswith(e) for e in SKIP_EXTS):
            continue
        src_path = os.path.join(root, f)
        rel = os.path.relpath(src_path, SRC)
        dst_path = os.path.join(DST, rel)
        
        if os.path.getsize(src_path) == 0:
            continue
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(src_path, "rb") as fin:
            data = fin.read()
        with open(dst_path, "wb") as fout:
            fout.write(data)
        count += 1
        print(f"  [{count}] {rel}")

print(f"\nDone! Copied {count} files.")
