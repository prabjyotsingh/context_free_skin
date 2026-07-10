"""
Upload model .pth files to HuggingFace Model Hub.
Run: python upload_models.py
"""
import os
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from huggingface_hub import HfApi, create_repo

# ── Config ────────────────────────────────────────────────
HF_TOKEN  = input("Paste your HuggingFace WRITE token: ").strip()
REPO_ID   = "prabysingh996/gt-skin-models"
FILES     = [
    "xcp_best.pth",   # 87.8 MB
    "b5_best.pth",    # 122.8 MB
    "cvx_best.pth",   # 352.5 MB
]
# ──────────────────────────────────────────────────────────

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
print(f"\nCreating/verifying repo: {REPO_ID}")
try:
    create_repo(REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True)
    print("  OK - Repo ready")
except Exception as e:
    print(f"  WARNING: {e}")

# Upload each file
script_dir = os.path.dirname(os.path.abspath(__file__))
for fname in FILES:
    fpath = os.path.join(script_dir, fname)
    if not os.path.exists(fpath):
        print(f"\n  MISSING: {fname} - skipping")
        continue
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"\nUploading {fname} ({size_mb:.1f} MB)...")
    try:
        url = api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  DONE -> {url}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\n\nAll done! Check: https://huggingface.co/prabysingh996/gt-skin-models")
