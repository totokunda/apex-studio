#!/usr/bin/env python3
"""
Download all model config files referenced in v0.1.2 manifests to a local configs/ directory.

Usage:
    python scripts/download_model_configs.py

This script:
1. Parses all YAML manifests in manifest/v0.1.2/
2. Extracts all config_path, preprocessor_path, and vocoder_config_path values
3. Downloads each to configs/ using huggingface_hub
4. Preserves the directory structure from the HF path

After running, the configs/ directory will contain all model configs locally,
and the config_registry module can resolve config_ids to local paths.
"""

import os
import sys
import yaml
import json
import shutil
import requests
from pathlib import Path
from typing import Set, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_DIR = PROJECT_ROOT / "manifest" / "v0.1.2"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Prefix to strip from totoku HF paths to get the config_id
TOTOKU_PREFIX = "totoku/apex-models/"


def extract_config_paths_from_manifest(manifest_path: Path) -> Set[str]:
    """Extract all config_path, preprocessor_path, and vocoder_config_path values from a manifest."""
    paths = set()
    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)

    if not data or "spec" not in data:
        return paths

    components = data.get("spec", {}).get("components", [])
    if not components:
        return paths

    for component in components:
        if not isinstance(component, dict):
            continue

        # Direct config_path on the component
        if config_path := component.get("config_path"):
            paths.add(config_path)

        # preprocessor_path (directory containing preprocessor_config.json)
        if preprocessor_path := component.get("preprocessor_path"):
            paths.add(preprocessor_path)

        # vocoder_config_path inside config block
        config_block = component.get("config", {})
        if isinstance(config_block, dict):
            if vocoder_config_path := config_block.get("vocoder_config_path"):
                paths.add(vocoder_config_path)

        # scheduler_options may also have config_path
        scheduler_options = component.get("scheduler_options", [])
        if isinstance(scheduler_options, list):
            for option in scheduler_options:
                if isinstance(option, dict) and (cp := option.get("config_path")):
                    paths.add(cp)

    return paths


def hf_path_to_config_id(hf_path: str) -> str:
    """Convert an HF path to a config_id."""
    from urllib.parse import urlparse

    # Handle direct URLs
    if hf_path.startswith("http"):
        parsed = urlparse(hf_path)
        parts = parsed.path.strip("/").split("/")
        namespace = parts[0]
        repo = parts[1]
        subpath_parts = parts[4:]  # skip "resolve" and branch
        if subpath_parts and subpath_parts[-1].endswith(".json"):
            subpath_parts = subpath_parts[:-1]
        return f"{namespace}/{repo}/{'/'.join(subpath_parts)}"

    # Handle totoku/apex-models paths
    if hf_path.startswith(TOTOKU_PREFIX):
        remainder = hf_path[len(TOTOKU_PREFIX):]
        if remainder.endswith(".json"):
            remainder = str(Path(remainder).parent)
        return remainder

    # Handle other HF paths (apple/..., nvidia/...)
    if "/" in hf_path:
        if hf_path.endswith(".json"):
            return str(Path(hf_path).parent)
        return hf_path

    return hf_path


def parse_hf_path(hf_path: str):
    """Parse an HF path into (repo_id, subfolder, filename) or (url, None, filename).
    
    Returns: (repo_id_or_url, subfolder_or_None, filename)
    """
    from urllib.parse import urlparse

    # Handle direct URLs
    if hf_path.startswith("http"):
        filename = hf_path.split("/")[-1]
        return hf_path, None, filename

    parts = hf_path.split("/")
    
    # HF format: namespace/repo/subfolder.../filename.ext
    repo_id = f"{parts[0]}/{parts[1]}"
    
    if len(parts) > 2:
        remaining = "/".join(parts[2:])
        if remaining.endswith(".json"):
            # It's a file path
            subfolder = "/".join(parts[2:-1]) if len(parts) > 3 else None
            filename = parts[-1]
        else:
            # It's a directory (e.g., preprocessor_path)
            subfolder = remaining
            filename = None
    else:
        subfolder = None
        filename = None

    return repo_id, subfolder, filename


def download_hf_file(repo_id: str, subfolder: Optional[str], filename: str, local_dir: Path) -> bool:
    """Download a single file from HuggingFace."""
    from huggingface_hub import hf_hub_download, get_token

    try:
        token = get_token()
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            token=token,
        )
        # Copy from cache to our configs dir
        dest = local_dir / filename
        if os.path.exists(cached_path):
            shutil.copy2(cached_path, dest)
            return True
    except Exception as e:
        print(f"    HF download failed: {e}")
    return False


def download_hf_directory(repo_id: str, subfolder: str, local_dir: Path) -> bool:
    """Download all config files from a HuggingFace directory."""
    from huggingface_hub import list_repo_files, hf_hub_download, get_token

    try:
        token = get_token()
        all_files = list_repo_files(repo_id, token=token)
        
        # Filter files in the subfolder
        prefix = f"{subfolder}/" if subfolder else ""
        matching_files = [
            f for f in all_files
            if f.startswith(prefix) and f.endswith((".json", ".txt"))
        ]

        if not matching_files:
            print(f"    No config files found in {repo_id}/{subfolder}")
            return False

        for file_path in matching_files:
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    token=token,
                )
                # Determine relative path within subfolder
                rel_path = file_path[len(prefix):] if prefix else file_path
                dest = local_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, dest)
                print(f"    Downloaded: {file_path}")
            except Exception as e:
                print(f"    Failed to download {file_path}: {e}")

        return True
    except Exception as e:
        print(f"    HF directory listing failed: {e}")
    return False


def download_url(url: str, local_dir: Path) -> bool:
    """Download a file from a direct URL."""
    try:
        filename = url.split("/")[-1]
        dest = local_dir / filename
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"    URL download failed: {e}")
    return False


def download_config(hf_path: str, configs_dir: Path) -> bool:
    """Download a single config file/directory to the local configs directory."""
    config_id = hf_path_to_config_id(hf_path)
    local_dir = configs_dir / config_id
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Path:      {hf_path}")
    print(f"  Config ID: {config_id}")
    print(f"  Local:     {local_dir}")

    # Handle direct URLs
    if hf_path.startswith("http"):
        return download_url(hf_path, local_dir)

    repo_id, subfolder, filename = parse_hf_path(hf_path)

    if filename:
        # Single file download
        return download_hf_file(repo_id, subfolder, filename, local_dir)
    else:
        # Directory download (e.g., preprocessor_path)
        return download_hf_directory(repo_id, subfolder, local_dir)


def main():
    print(f"Manifest directory: {MANIFEST_DIR}")
    print(f"Configs directory:  {CONFIGS_DIR}")
    print()

    # Collect all unique config paths from all manifests
    all_paths = set()
    manifest_files = list(MANIFEST_DIR.rglob("*.yml"))
    print(f"Found {len(manifest_files)} manifest files")

    for manifest_path in sorted(manifest_files):
        paths = extract_config_paths_from_manifest(manifest_path)
        if paths:
            rel_path = manifest_path.relative_to(PROJECT_ROOT)
            print(f"  {rel_path}: {len(paths)} config paths")
            all_paths.update(paths)

    print(f"\nTotal unique config paths: {len(all_paths)}")
    print()

    # Create configs directory
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Download all configs
    downloaded = 0
    failed = 0
    failed_paths = []

    for i, hf_path in enumerate(sorted(all_paths), 1):
        print(f"\n[{i}/{len(all_paths)}]")
        success = download_config(hf_path, CONFIGS_DIR)
        if success:
            downloaded += 1
        else:
            failed += 1
            failed_paths.append(hf_path)

    print(f"\n{'='*60}")
    print(f"Download complete: {downloaded} succeeded, {failed} failed")
    print(f"Configs saved to: {CONFIGS_DIR}")

    if failed_paths:
        print(f"\nFailed paths:")
        for p in failed_paths:
            print(f"  - {p}")

    # Print config_id summary
    print(f"\nConfig ID mappings:")
    for hf_path in sorted(all_paths):
        config_id = hf_path_to_config_id(hf_path)
        print(f"  {config_id}")


if __name__ == "__main__":
    main()
