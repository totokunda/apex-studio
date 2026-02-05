#!/usr/bin/env python3
"""
Upload `apps/api/weights/Chroma1-HD` to Hugging Face.

Default target:
  - repo: totoku/apex-models
  - path_in_repo: Chroma1-HD

Auth:
  - set HF_TOKEN (recommended) or run `huggingface-cli login`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

def _api_root() -> Path:
    # .../apps/api/scripts/models/<this_file.py> -> parents[2] == .../apps/api
    return Path(__file__).resolve().parents[2]


def _default_local_dir() -> Path:
    return _api_root() / "weights" 


def _iter_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _total_bytes(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            # In case a file disappears mid-run.
            pass
    return total


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.2f} {u}" if u != "B" else f"{int(v)} {u}"
        v /= 1024
    return f"{n} B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload weights/Chroma1-HD to Hugging Face")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=_default_local_dir(),
        help="Local folder to upload.",
    )
    parser.add_argument(
        "--repo-id",
        default="totoku/apex-models",
        help="Target Hugging Face repo (e.g. org/name).",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repo type (default: model).",
    )
    parser.add_argument(
        "--path-in-repo",
        default='',
        help="Folder path inside the HF repo to upload into.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload weights",
        help="Commit message to use on HF.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional). If omitted, uses HF_TOKEN / cached login.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading.",
    )

    args = parser.parse_args()

    local_dir: Path = args.local_dir.expanduser().resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        raise SystemExit(f"Local dir not found: {local_dir}")

    files = _iter_files(local_dir)
    if not files:
        raise SystemExit(f"No files found under: {local_dir}")

    print(f"Local dir: {local_dir}")
    print(f"Files: {len(files)}  Total size: {_human_bytes(_total_bytes(files))}")
    print(f"HF repo: {args.repo_type}:{args.repo_id}")
    print(f"Path in repo: {args.path_in_repo}")
    print(f"Revision: {args.revision}")

    if args.dry_run:
        print("\nDry-run file list:")
        for p in files:
            rel = p.relative_to(local_dir)
            print(f" - {rel} ({_human_bytes(p.stat().st_size)})")
        return 0

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n\n"
            "Install it in this environment (pick one):\n"
            "  - pip install huggingface_hub\n"
            "  - pip install -r requirements/requirements.txt\n"
        ) from e

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = HfApi(token=token)

    # Ensure repo exists (no-op if it already does).
    try:
        api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            print("Repo does not exist yet; creating it...")
            api.create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                exist_ok=True,
            )
        else:
            raise

    print("\nUploading folder (this may take a while for large files)...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(local_dir),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        # Skip noise files if present
        ignore_patterns=[
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/__pycache__/**",
            "**/*.tmp",
        ],
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

