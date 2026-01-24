#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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


MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifest"


def _bytes_to_gib(size_bytes: int) -> float:
    return float(size_bytes) / (1024.0**3)


def _as_lower_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    return ""


def _guess_precision_bucket(item: Dict[str, Any]) -> str:
    # Prefer explicit precision, but fall back to variant naming.
    p = _as_lower_str(item.get("precision"))
    if p:
        return p
    v = _as_lower_str(item.get("variant"))
    return v


def estimate_resource_requirements_gb(
    *,
    model_type: str,
    component_type: str,
    item: Dict[str, Any],
    size_bytes: int,
) -> Tuple[int, int]:
    """Heuristic estimate of (min_vram_gb, recommended_vram_gb).

    Notes:
    - This is intentionally approximate: we primarily use on-disk weight size as a proxy.
    - We bias estimates to "pipeline-level" floors per component type (e.g. VAE still
      needs the rest of the pipeline resident during inference).
    """
    mt = (model_type or "").strip().lower()
    ct = (component_type or "").strip().lower()
    file_type = _as_lower_str(item.get("type"))
    precision = _guess_precision_bucket(item)
    size_gib = _bytes_to_gib(int(size_bytes))

    # Baseline VRAM floors (GB) — tuned for typical diffusion/T2I usage.
    # These act as "required_vram" floors even when the component weights are small.
    if mt in {"t2i", "i2i", "inpaint"}:
        base_by_component = {
            "transformer": 4.0,
            "unet": 4.0,
            "text_encoder": 6.0,
            "vae": 7.0,
            "scheduler": 2.0,
        }
    elif mt in {"t2v", "i2v", "video"}:
        base_by_component = {
            "transformer": 8.0,
            "unet": 8.0,
            "text_encoder": 8.0,
            "vae": 10.0,
            "scheduler": 3.0,
        }
    else:
        base_by_component = {
            "transformer": 4.0,
            "unet": 4.0,
            "text_encoder": 6.0,
            "vae": 7.0,
            "scheduler": 2.0,
        }

    base = float(base_by_component.get(ct, 4.0))

    # Weight-size multipliers by quantization/precision.
    # Idea: fp16 weights often load close to on-disk size; fp8 can have extra runtime
    # overhead; gguf quantization generally reduces active VRAM.
    mult = 0.8  # sensible default for fp16-ish
    if "fp32" in precision:
        mult = 1.3
    elif "bf16" in precision or "fp16" in precision:
        mult = 0.8
    elif "fp8" in precision:
        mult = 1.0
    elif file_type == "gguf" or precision.startswith("q"):
        # Common GGUF buckets
        if "q8" in precision:
            mult = 0.85
        elif "q6" in precision:
            mult = 0.65
        elif "q5" in precision:
            mult = 0.55
        elif "q4" in precision:
            mult = 0.45
        else:
            mult = 0.55

    required = int(max(2, math.ceil(base + (size_gib * mult))))
    extra = max(4, int(math.ceil(required * 0.33)))
    recommended = int(required + extra)
    return required, recommended


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
    model_type = spec.get("model_type") or ""
    components: List[Dict[str, Any]] = spec.get("components") or []
    changed = False

    for comp in components:
        if not isinstance(comp, dict):
            continue
        component_type = comp.get("type") or ""
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

            # Prefer existing size if present (allows offline VRAM updates).
            existing_size = item.get("file_size")
            if isinstance(existing_size, int) and existing_size > 0:
                size_bytes = int(existing_size)
            else:
                size_bytes = compute_size_for_model_path(path_value, path.parent)

            # Only set when we can determine a positive size; avoid flapping zeros
            if size_bytes and item.get("file_size") != int(size_bytes):
                item["file_size"] = int(size_bytes)
                changed = True

            # Update resource requirements when we have a positive size.
            if size_bytes:
                min_vram_gb, recommended_vram_gb = estimate_resource_requirements_gb(
                    model_type=str(model_type),
                    component_type=str(component_type),
                    item=item,
                    size_bytes=int(size_bytes),
                )
                rr = item.get("resource_requirements")
                if not isinstance(rr, dict):
                    rr = {}
                    item["resource_requirements"] = rr
                    changed = True
                # "min_vram_gb" is the "required" VRAM floor.
                if rr.get("min_vram_gb") != int(min_vram_gb):
                    rr["min_vram_gb"] = int(min_vram_gb)
                    changed = True
                if rr.get("recommended_vram_gb") != int(recommended_vram_gb):
                    rr["recommended_vram_gb"] = int(recommended_vram_gb)
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
        description="Populate file_size and resource_requirements for model_path items in manifests."
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
