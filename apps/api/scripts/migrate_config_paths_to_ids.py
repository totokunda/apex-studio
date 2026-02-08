#!/usr/bin/env python3
"""
Migrate all v0.1.2 manifest YAML files from config_path to config_id.

This script:
1. Reads each YAML manifest
2. Replaces `config_path: <hf_path>` with `config_id: <registry_key>`
3. Replaces `preprocessor_path: <hf_path>` with `preprocessor_config_id: <registry_key>`
4. Replaces `vocoder_config_path` inside config blocks with `vocoder_config_id`
5. Writes the modified YAML back (preserving formatting as much as possible)

Usage:
    python scripts/migrate_config_paths_to_ids.py
    python scripts/migrate_config_paths_to_ids.py --dry-run
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "manifest" / "v0.1.2"
TOTOKU_PREFIX = "totoku/apex-models/"


def hf_path_to_config_id(hf_path: str) -> str:
    """Convert a HuggingFace path to a config_id."""
    from urllib.parse import urlparse

    if hf_path.startswith("http"):
        parsed = urlparse(hf_path)
        parts = parsed.path.strip("/").split("/")
        namespace = parts[0]
        repo = parts[1]
        subpath_parts = parts[4:]
        if subpath_parts and subpath_parts[-1].endswith(".json"):
            subpath_parts = subpath_parts[:-1]
        return f"{namespace}/{repo}/{'/'.join(subpath_parts)}"

    if hf_path.startswith(TOTOKU_PREFIX):
        remainder = hf_path[len(TOTOKU_PREFIX):]
        if remainder.endswith(".json"):
            remainder = str(Path(remainder).parent)
        return remainder

    if "/" in hf_path:
        if hf_path.endswith(".json"):
            return str(Path(hf_path).parent)
        return hf_path

    return hf_path


def migrate_yaml_file(filepath: Path, dry_run: bool = False) -> int:
    """Migrate a single YAML file. Returns number of replacements made."""
    with open(filepath, "r") as f:
        content = f.read()

    original_content = content
    replacements = 0

    # Pattern 1: config_path lines (but NOT vocoder_config_path)
    # Match lines like:    config_path: totoku/apex-models/Wan2.2-I2V/vae/config.json
    # or:    config_path: https://huggingface.co/...
    # But NOT: vocoder_config_path: ...
    def replace_config_path(match):
        nonlocal replacements
        indent = match.group(1)
        hf_path = match.group(2).strip()
        config_id = hf_path_to_config_id(hf_path)
        replacements += 1
        return f"{indent}config_id: {config_id}"

    # Replace config_path that is NOT preceded by vocoder_ or any other prefix
    # Use negative lookbehind for word characters before 'config_path'
    content = re.sub(
        r'^(\s+)config_path:\s+(.+)$',
        replace_config_path,
        content,
        flags=re.MULTILINE,
    )

    # Pattern 2: preprocessor_path lines
    # Match lines like:    preprocessor_path: totoku/apex-models/Wan2.1-I2V-480P/image_processor
    def replace_preprocessor_path(match):
        nonlocal replacements
        indent = match.group(1)
        hf_path = match.group(2).strip()
        config_id = hf_path_to_config_id(hf_path)
        replacements += 1
        return f"{indent}preprocessor_config_id: {config_id}"

    content = re.sub(
        r'^(\s+)preprocessor_path:\s+(.+)$',
        replace_preprocessor_path,
        content,
        flags=re.MULTILINE,
    )

    # Pattern 3: vocoder_config_path inside config blocks
    # Match lines like:      vocoder_config_path: nvidia/bigvgan_v2_44khz_128band_512x/config.json
    def replace_vocoder_config_path(match):
        nonlocal replacements
        indent = match.group(1)
        hf_path = match.group(2).strip()
        config_id = hf_path_to_config_id(hf_path)
        replacements += 1
        return f"{indent}vocoder_config_id: {config_id}"

    content = re.sub(
        r'^(\s+)vocoder_config_path:\s+(.+)$',
        replace_vocoder_config_path,
        content,
        flags=re.MULTILINE,
    )

    if content != original_content:
        if not dry_run:
            with open(filepath, "w") as f:
                f.write(content)
        rel_path = filepath.relative_to(PROJECT_ROOT)
        print(f"  {rel_path}: {replacements} replacements")
    
    return replacements


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN - no files will be modified")
        print()

    manifest_files = sorted(MANIFEST_DIR.rglob("*.yml"))
    print(f"Found {len(manifest_files)} manifest files in {MANIFEST_DIR}")
    print()

    total_replacements = 0
    files_modified = 0

    for filepath in manifest_files:
        count = migrate_yaml_file(filepath, dry_run=dry_run)
        if count > 0:
            files_modified += 1
            total_replacements += count

    print(f"\n{'='*60}")
    mode = "Would modify" if dry_run else "Modified"
    print(f"{mode} {files_modified} files with {total_replacements} total replacements")


if __name__ == "__main__":
    main()
