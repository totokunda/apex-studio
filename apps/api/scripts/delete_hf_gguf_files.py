#!/usr/bin/env python3
"""
Delete all files ending with `.gguf` from specific folders in a Hugging Face repo.

Defaults to DRY-RUN (prints what would be deleted). To actually delete, pass --apply.

Auth:
  - Provide a token via --token, or set env var HF_TOKEN.

Example:
  HF_TOKEN=... python3 scripts/delete_hf_gguf_files.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

from huggingface_hub import HfApi
from huggingface_hub._commit_api import CommitOperationDelete


DEFAULT_REPO_ID = "totoku/apex-models"
DEFAULT_REPO_TYPE = "model"
DEFAULT_REVISION = "main"

FOLDERS = [
    "Wan2.1-I2V-480P",
    "Wan2.1-T2V",
    "Wan2.1-VACE",
    "Wan2.2-Animate",
    "Wan2.2-I2V",
    "Wan2.2-S2V",
    "Wan2.2-SmoothMix-I2V",
    "Wan2.2-T2V-A14B",
    "Wan2.2-TI2V-5B",
    "Scail-Preview",
    "Ovi-5s",
    "Ovi-10s",
    "MOVA-720p",
    "MOVA-360p",
]


def _normalize_prefix(folder: str) -> str:
    # Accept either "Folder" or "Folder/" in the constant list.
    return folder if folder.endswith("/") else f"{folder}/"


def _iter_targets(all_files: Iterable[str], folders: List[str]) -> List[str]:
    prefixes = tuple(_normalize_prefix(f) for f in folders)
    targets: List[str] = []
    for p in all_files:
        if p.endswith(".gguf") and p.startswith(prefixes):
            targets.append(p)
    return sorted(set(targets))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete *.gguf files under selected folders in a Hugging Face repo."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Branch or commit SHA (default: main).")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="HF token (or set HF_TOKEN env var). Required for --apply.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this, runs in dry-run mode.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=FOLDERS,
        help="Folder prefixes to scan (default: the Wan/Ovi/Mova/Scail folders requested).",
    )
    parser.add_argument(
        "--commit-message",
        default="Delete gguf files",
        help="Commit message to use when applying deletions.",
    )
    args = parser.parse_args()

    api = HfApi(token=args.token)

    print(f"Repo: {args.repo_id} (type={args.repo_type}, revision={args.revision})")
    print("Scanning for *.gguf under folders:")
    for f in args.folders:
        print(f"  - {f}")

    all_files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision)
    targets = _iter_targets(all_files, args.folders)

    if not targets:
        print("No matching .gguf files found.")
        return 0

    print(f"\nFound {len(targets)} .gguf file(s) to delete:")
    for p in targets:
        print(f"  - {p}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete these files.")
        return 0

    if not args.token:
        print("\nERROR: --apply requires an auth token (set HF_TOKEN or pass --token).", file=sys.stderr)
        return 2

    ops = [CommitOperationDelete(path_in_repo=p) for p in targets]

    # Single commit for all deletions.
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        operations=ops,
        commit_message=args.commit_message,
    )

    print(f"\nDeleted {len(targets)} file(s) in commit: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

