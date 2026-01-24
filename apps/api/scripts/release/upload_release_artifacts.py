#!/usr/bin/env python3
"""
Upload release artifacts to a Hugging Face Hub repo under a versioned folder.

Example:
  export HF_TOKEN=...  # or HUGGINGFACE_HUB_TOKEN
  python3 apps/api/scripts/release/upload_release_artifacts.py --version v0.1.0 --dist-dir ./dist
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "totoku/apex-studio-server"
DEFAULT_REPO_TYPE = "model"

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
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")  # legacy
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")  # legacy
    )


def _prompt_for_token() -> str:
    """
    Prompt interactively for a Hugging Face *write* token.
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment."
        )
    token = getpass.getpass(
        "Enter Hugging Face *write* token (won't echo). You can also set HF_TOKEN or HUGGINGFACE_HUB_TOKEN: "
    ).strip()
    if not token:
        raise SystemExit("No token provided; aborting upload.")
    return token


def _resolve_latest_tar(dist_dir: Path, prefix: str) -> Path:
    """
    Find the newest tarball under dist_dir matching <prefix>-*.tar.zst.
    """
    dist_dir = dist_dir.expanduser().resolve()
    if not dist_dir.exists() or not dist_dir.is_dir():
        raise SystemExit(f"dist dir not found: {dist_dir}")
    matches = sorted(dist_dir.glob(f"{prefix}-*.tar.zst"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(
            f"No {prefix} tarball found under {dist_dir}. Expected: {prefix}-*.tar.zst"
        )
    return matches[-1]


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
        required=True,
        help="Folder in repo to upload into (required), e.g. v0.1.0",
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory to search for tarballs when --api-tar/--code-tar are omitted (default: ./dist)",
    )
    parser.add_argument(
        "--api-tar",
        default=None,
        help="Path to python-api tarball (.tar.zst). If omitted, picks the newest python-api-*.tar.zst under --dist-dir.",
    )
    parser.add_argument(
        "--code-tar",
        default=None,
        help="Path to python-code tarball (.tar.zst). If omitted, picks the newest python-code-*.tar.zst under --dist-dir.",
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
        token = _prompt_for_token()

    dist_dir = Path(args.dist_dir)
    api_tar = (
        Path(args.api_tar).expanduser().resolve()
        if args.api_tar
        else _resolve_latest_tar(dist_dir, "python-api")
    )
    code_tar = (
        Path(args.code_tar).expanduser().resolve()
        if args.code_tar
        else _resolve_latest_tar(dist_dir, "python-code")
    )
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
