#!/usr/bin/env python3
"""
Upload FlashAttention wheelhouse artifacts to Hugging Face Hub.

Uploads:
  apps/api/flash-attention/wheelhouse/sm80   -> flash-attention/sm80
  apps/api/flash-attention/wheelhouse/sm90   -> flash-attention/sm90
  apps/api/flash-attention/wheelhouse/sm100  -> flash-attention/sm100
  apps/api/flash-attention/wheelhouse/sm120  -> flash-attention/sm120
  apps/api/flash-attention/wheelhouse/flash_attn_3 -> flash-attention/flash-attention-3

Auth:
  Uses HF_TOKEN (preferred) or HUGGINGFACE_TOKEN / HUGGINGFACEHUB_API_TOKEN.
  Also calls load_dotenv() so a local .env can be used.

Speed:
  Opts into hf-transfer via HF_HUB_ENABLE_HF_TRANSFER=1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Opt into HF transfer acceleration (requires `hf-transfer` installed).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "totokunda/attention"
_API_DIR = Path(__file__).resolve().parents[2]  # .../apps/api
DEFAULT_BASE_DIR = _API_DIR / "flash-attention" / "wheelhouse"
DEFAULT_PATH_PREFIX = "flash-attention"


def _token_from_env() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


def _try_load_token_from_env_file(dotenv_path: str | Path) -> None:
    """
    Best-effort load HF token from an env file.

    Supports common formats:
      - HF_TOKEN=...
      - export HF_TOKEN=...
      - set HF_TOKEN=...            (cmd.exe)
      - $env:HF_TOKEN="..."         (PowerShell)
    """
    p = Path(dotenv_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return

    # First try python-dotenv's parser.
    try:
        load_dotenv(dotenv_path=str(p), override=False)
    except Exception:
        pass

    if _token_from_env():
        return

    # Fallback: manual parse for non-standard formats.
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return

    candidates = {
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        # PowerShell: $env:HF_TOKEN="..."
        if s.lower().startswith("$env:"):
            s = s[5:].strip()

        # bash: export HF_TOKEN=...
        if s.lower().startswith("export "):
            s = s[7:].strip()

        # cmd.exe: set HF_TOKEN=...
        if s.lower().startswith("set "):
            s = s[4:].strip()

        if "=" not in s:
            continue

        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key in candidates and val:
            # Prefer HF_TOKEN so downstream tools pick it up consistently.
            os.environ.setdefault("HF_TOKEN", val)
            return


def _detect_repo_type(api: HfApi, repo_id: str) -> str | None:
    for repo_type in ("model", "dataset", "space"):
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return repo_type
        except Exception as e:
            msg = str(e).lower()
            # Heuristic: treat "not found" / 404 as "try next type".
            if "404" in msg or "not found" in msg:
                continue
            # If it's something else (permissions, connectivity), bubble up.
            raise
    return None


def _require_dir(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload FlashAttention wheelhouse artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target repo, default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default=None,
        help="Repo type. If omitted, we auto-detect existing repo type; otherwise default to dataset when creating.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo if it doesn't exist. Requires permissions in the target namespace.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help=f"Wheelhouse directory, default: {DEFAULT_BASE_DIR.as_posix()}",
    )
    parser.add_argument(
        "--path-prefix",
        default=DEFAULT_PATH_PREFIX,
        help=f"Subfolder in repo to upload into, default: {DEFAULT_PATH_PREFIX}",
    )
    parser.add_argument(
        "--dotenv-path",
        default=None,
        help="Optional explicit path to a .env file containing HF_TOKEN.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FlashAttention wheelhouse artifacts",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    if args.dotenv_path:
        _try_load_token_from_env_file(args.dotenv_path)

    token = _token_from_env()
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Provide a .env with HF_TOKEN, or set HF_TOKEN in the environment."
        )

    base_dir = Path(args.base_dir).expanduser().resolve()
    _require_dir(base_dir)

    path_prefix = str(args.path_prefix).strip().strip("/")
    if not path_prefix:
        raise SystemExit("--path-prefix must be a non-empty folder name, e.g. flash-attention")

    api = HfApi(token=token)

    detected = _detect_repo_type(api, args.repo_id)
    repo_type = args.repo_type or detected or "dataset"
    if args.create_repo and not detected:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=repo_type,
            exist_ok=True,
        )

    uploads: list[tuple[Path, str]] = [
        (base_dir / "sm80", f"{path_prefix}/sm80"),
        (base_dir / "sm90", f"{path_prefix}/sm90"),
        (base_dir / "sm100", f"{path_prefix}/sm100"),
        (base_dir / "sm120", f"{path_prefix}/sm120"),
        (base_dir / "flash_attn_3", f"{path_prefix}/flash-attention-3"),
    ]

    for local_dir, path_in_repo in uploads:
        _require_dir(local_dir)
        print(f"Uploading {local_dir} -> {args.repo_id}:{path_in_repo} (repo_type={repo_type})")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=repo_type,
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            allow_patterns=["*.whl"],
            commit_message=args.commit_message,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

