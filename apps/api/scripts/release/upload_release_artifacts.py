#!/usr/bin/env python3
"""
Upload release artifacts to a Hugging Face Hub repo under a versioned folder.

Example:
  export HF_TOKEN=...  # or HUGGINGFACE_TOKEN
  python3 apps/api/scripts/upload_release_artifacts.py --version v0.1.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "totoku/apex-studio-server"
DEFAULT_REPO_TYPE = "model"
DEFAULT_VERSION_FOLDER = "v0.1.0"

DEFAULT_API_TAR = "/Users/tosinkuye/apex-workspace/apex-studio/apps/api/dist/python-api-0.1.0-darwin-arm64-cpu-cp312.tar.zst"

DEFAULT_CODE_TAR = "/Users/tosinkuye/apex-workspace/apex-studio/apps/api/dist/python-code-0.1.0-darwin-arm64-cpu-cp312.tar.zst"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"File is empty: {path}")


def _token_from_env() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload Apex Studio server release tarballs to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target repo, default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=["model", "dataset", "space"],
        help=f"Repo type, default: {DEFAULT_REPO_TYPE}",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION_FOLDER,
        help=f"Folder in repo to upload into, default: {DEFAULT_VERSION_FOLDER}",
    )
    parser.add_argument(
        "--api-tar",
        default=DEFAULT_API_TAR,
        help="Path to python-api tarball (.tar.zst)",
    )
    parser.add_argument(
        "--code-tar",
        default=DEFAULT_CODE_TAR,
        help="Path to python-code tarball (.tar.zst)",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Apex Studio server release artifacts",
        help="Commit message to use for the upload",
    )
    args = parser.parse_args()

    # Opt into HF transfer acceleration when available.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = _token_from_env()
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN (preferred) or HUGGINGFACE_TOKEN."
        )

    api_tar = Path(args.api_tar).expanduser().resolve()
    code_tar = Path(args.code_tar).expanduser().resolve()
    _require_file(api_tar)
    _require_file(code_tar)

    version_folder = str(args.version).strip().strip("/")
    if not version_folder:
        raise SystemExit("--version must be a non-empty folder name, e.g. v0.1.0")

    api = HfApi(token=token)

    uploads = [
        (api_tar, f"{version_folder}/{api_tar.name}"),
        (code_tar, f"{version_folder}/{code_tar.name}"),
    ]

    for local_path, path_in_repo in uploads:
        print(f"Uploading {local_path} -> {args.repo_id}:{path_in_repo}")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=args.commit_message,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
