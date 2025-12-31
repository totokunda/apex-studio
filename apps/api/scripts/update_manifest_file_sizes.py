#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    from huggingface_hub import HfApi, hf_hub_url
except Exception:
    HfApi = None  # type: ignore
    hf_hub_url = None  # type: ignore


MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifest/engine"


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


_HF_RE = re.compile(r"^[\w\-.]+/[\w\-.]+(?:/.*)?$")


def is_hf_repo_path(path: str) -> bool:
    if is_url(path):
        return False
    # Heuristic: owner/repo[/optional/subpath]
    return bool(_HF_RE.match(path))


def sizeof_local(target: Path) -> int:
    if target.is_file():
        return target.stat().st_size
    if target.is_dir():
        total = 0
        for root, _dirs, files in os.walk(target):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    pass
        return total
    return 0


def sizeof_url(url: str, timeout: float = 15.0) -> int:
    if requests is None:
        return 0
    try:
        # Try HEAD first; allow redirects to get Content-Length
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        length = resp.headers.get("Content-Length") or resp.headers.get(
            "content-length"
        )
        if length is not None:
            return int(length)
        # Some endpoints don't give length on HEAD; try GET with stream=False but no download
        resp = requests.get(url, stream=True, timeout=timeout)
        length = resp.headers.get("Content-Length") or resp.headers.get(
            "content-length"
        )
        if length is not None:
            return int(length)
    except Exception:
        return 0
    return 0


def parse_hf_path(p: str) -> Tuple[str, str, str]:
    # owner/repo[/subpath]
    parts = p.strip("/").split("/", 2)
    owner, repo = parts[0], parts[1]
    subpath = parts[2] if len(parts) > 2 else ""
    return owner, repo, subpath


def sizeof_hf(path: str) -> int:
    """Compute size for a Hugging Face model path using the API tree listing.

    Supports both file and directory subpaths, summing sizes for directories.
    """

    if HfApi is None:
        return 0
    try:
        owner, repo, subpath = parse_hf_path(path)
        repo_id = f"{owner}/{repo}"
        # If subpath looks like a file path, try resolving the raw file URL and
        # using standard HTTP size detection. This avoids issues with the tree
        # API when pointing directly at a file.
        if subpath and "." in os.path.basename(subpath):
            if hf_hub_url is not None:
                try:
                    url = hf_hub_url(repo_id=repo_id, filename=subpath, revision="main")
                    size = sizeof_url(url)
                    if size:
                        return int(size)
                except Exception as e:
                    print(f"Error computing size for {path} via hf_hub_url: {e}")

        api = HfApi()
        # Otherwise, treat it as a directory or repo root and sum all entries.
        recursive = True
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=subpath or None,
            recursive=recursive,
            revision="main",
            repo_type="model",
        )
        total = 0
        for e in entries:
            size = getattr(e, "size", None)
            if isinstance(size, int):
                total += size
        # If nothing found, try listing at repo root and filtering by prefix
        if total == 0 and subpath:
            entries = api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=None,
                recursive=True,
                revision="main",
                repo_type="model",
            )
            prefix = subpath.rstrip("/") + "/"
            for e in entries:
                p = getattr(e, "path", "")
                if p == subpath or p.startswith(prefix):
                    size = getattr(e, "size", None)
                    if isinstance(size, int):
                        total += size
        return int(total)
    except Exception as e:
        print(f"Error computing size for {path}: {e}")

        return 0


def compute_size_for_model_path(path_value: str, file_root: Path) -> int:
    # 1) Direct URL
    if is_url(path_value):
        return sizeof_url(path_value)
    # 2) HF repo reference

    if is_hf_repo_path(path_value):
        size = sizeof_hf(path_value)
        if size:
            return size
    # 3) Local filesystem (absolute or relative to repo root)
    p = Path(path_value)
    if not p.is_absolute():
        # try relative to manifest root, then repo root
        candidates = [file_root / p, Path.cwd() / path_value]
    else:
        candidates = [p]
    for c in candidates:
        try:
            if c.exists():
                return sizeof_local(c)
        except Exception:
            continue
    return 0


def update_manifest_file(path: Path) -> bool:
    try:
        text = path.read_text()
        doc = yaml.load(text, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"[WARN] Failed to parse YAML {path}: {e}")
        return False

    if not isinstance(doc, dict):
        return False

    spec = doc.get("spec") or {}
    components: List[Dict[str, Any]] = spec.get("components") or []
    changed = False

    for comp in components:
        if not isinstance(comp, dict):
            continue
        mp = comp.get("model_path")
        if mp is None:
            continue
        # Normalize string → list[dict]
        if isinstance(mp, str):
            mp_items = [{"path": mp}]
            comp["model_path"] = mp_items
            changed = True
        elif isinstance(mp, list):
            mp_items = mp
        else:
            continue

        for item in mp_items:
            if isinstance(item, str):
                # convert to object
                new_item = {"path": item}
                idx = mp_items.index(item)
                mp_items[idx] = new_item
                item = new_item
                changed = True
            if not isinstance(item, dict):
                continue
            # Prefer explicit "path", but also allow "url" for remote artefacts
            path_value = item.get("path") or item.get("url")
            if not path_value or not isinstance(path_value, str):
                continue
            size_bytes = compute_size_for_model_path(path_value, path.parent)
            # Only set when we can determine a positive size; avoid flapping zeros
            if size_bytes and item.get("file_size") != int(size_bytes):
                item["file_size"] = int(size_bytes)
                changed = True

    if changed:
        # Write back with safe dumper preserving simple formatting
        try:
            path.write_text(yaml.dump(doc, sort_keys=False))
            print(f"[OK] Updated {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write {path}: {e}")
            return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate file_size for model_path items in manifests."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(MANIFEST_ROOT),
        help="Root directory containing manifests",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return 2

    updated = 0
    if root.is_file():
        if root.suffix in {".yml", ".yaml"} and update_manifest_file(root):
            updated += 1
    else:
        for yml in root.rglob("*.yml"):
            if update_manifest_file(yml):
                updated += 1

    print(f"Done. Updated {updated} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
