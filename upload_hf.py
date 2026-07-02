"""
Upload dataset lên HuggingFace Hub.

Usage:
    python upload_hf.py --repo your-username/provedit-45class

Files được upload:
    data/combined_preprocessed_data.csv
    data/meta.json
    data/prepare_data.py
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

from huggingface_hub import HfApi, create_repo

DATA_DIR = Path(__file__).parent / "data"

UPLOAD_FILES = [
    DATA_DIR / "combined_preprocessed_data.csv",
    DATA_DIR / "meta.json",
    DATA_DIR / "prepare_data.py",
]


def main(repo_id: str, private: bool):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set in .env")

    api = HfApi(token=token)

    print(f"Creating/verifying repo: {repo_id} ...")
    create_repo(repo_id, repo_type="dataset", private=private,
                token=token, exist_ok=True)

    for fpath in UPLOAD_FILES:
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath.name}")
            continue
        print(f"  Uploading {fpath.name} ({fpath.stat().st_size // 1024} KB) ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. username/provedit-45class")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo (default: public)")
    args = parser.parse_args()
    main(args.repo, args.private)
