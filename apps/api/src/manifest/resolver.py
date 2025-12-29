from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from loguru import logger

# Ensure the dummy !include is registered on FullLoader so exploratory loads don't fail
from src.utils.yaml import _dummy_include  # type: ignore  # noqa: F401


_INDEX_CACHE: Dict[str, str] | None = None
_LATEST_MAP: Dict[str, Tuple[int, int, int, str]] | None = None


def _project_root() -> Path:
    # src/manifest/resolver.py → apex/src/manifest → apex
    return Path(__file__).resolve().parents[2]


def _manifest_dir() -> Path:
    return _project_root() / "manifest"


def _iter_manifest_files() -> List[Path]:
    base = _manifest_dir()
    if not base.exists():
        return []
    return [p for p in base.rglob("*.yml") if p.is_file()]


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def _parse_semver(s: Optional[str]) -> Tuple[int, int, int]:
    if not s:
        return (0, 0, 0)
    m = _SEMVER_RE.match(str(s))
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _extract_minimal_info(path: Path) -> Tuple[str, str, str, Tuple[int, int, int]]:
    """
    Return (engine, model_type, slug, version_tuple)
    Best-effort for legacy manifests; missing values become empty/zero.
    """
    try:
        doc = yaml.load(path.read_text(), Loader=yaml.FullLoader) or {}
    except Exception:
        return ("", "", _slugify(path.stem), (0, 0, 0))

    engine = ""
    model_type = ""
    name = path.stem
    version = (0, 0, 0)

    if isinstance(doc, dict) and "apiVersion" in doc:  # v1
        md = doc.get("metadata", {}) or {}
        spec = doc.get("spec", {}) or {}
        name = md.get("name", name)
        engine = spec.get("engine", "") or engine
        model_type = spec.get("modelType", "") or model_type
        version = _parse_semver(md.get("version"))
    else:  # legacy
        engine = (doc or {}).get("engine", "") or engine
        model_type = (doc or {}).get("type", "") or model_type
        name = (doc or {}).get("name", name)
        # Try file-based version hint: ...-1.2.3[.v1].yml
        m = _SEMVER_RE.search(path.stem)
        if m:
            version = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    slug = _slugify(name)
    return (engine, model_type, slug, version)


def _build_index() -> Dict[str, str]:
    global _INDEX_CACHE, _LATEST_MAP
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    idx: Dict[str, str] = {}
    latest: Dict[str, Tuple[int, int, int, str]] = {}

    for path in _iter_manifest_files():
        engine, mtype, slug, version = _extract_minimal_info(path)
        ver_str = (
            f"{version[0]}.{version[1]}.{version[2]}"
            if version != (0, 0, 0)
            else "0.0.0"
        )
        # Construct keys and add
        candidates = []
        if engine and mtype and slug:
            candidates.append((f"{engine}/{mtype}/{slug}:{ver_str}", path))
            candidates.append((f"{engine}/{mtype}/{slug}:latest", path))
        if engine and slug:
            candidates.append((f"{engine}/{slug}:{ver_str}", path))
            candidates.append((f"{engine}/{slug}:latest", path))
        if slug:
            candidates.append((f"{slug}:{ver_str}", path))
            candidates.append((f"{slug}:latest", path))

        for key, p in candidates:
            idx.setdefault(key, str(p.resolve()))

        # Track latest per (engine, mtype, slug)
        scope_keys = [
            (engine, mtype, slug),
            (engine, "", slug),
            ("", "", slug),
        ]
        for sk in scope_keys:
            if not sk[2]:
                continue
            prev = latest.get(str(sk))
            if prev is None or version > prev[0:3]:
                latest[str(sk)] = (
                    version[0],
                    version[1],
                    version[2],
                    str(path.resolve()),
                )

    # Overwrite :latest to the true highest version for each scope
    for sk, (ma, mi, pa, pth) in latest.items():
        engine, mtype, slug = eval(sk) if isinstance(sk, str) and sk.startswith("(") else sk  # type: ignore
        if engine and mtype:
            idx[f"{engine}/{mtype}/{slug}:latest"] = pth
        if engine:
            idx[f"{engine}/{slug}:latest"] = pth
        idx[f"{slug}:latest"] = pth

    _INDEX_CACHE = idx
    _LATEST_MAP = latest
    return idx


def resolve_manifest_reference(ref: str) -> Optional[str]:
    """
    Resolve a Docker-style reference to a local manifest file.

    Supported forms (case-sensitive):
      - engine/modelType/slug:version
      - engine/slug:version
      - slug:version
      - same with :latest

    Returns an absolute file path string or None.
    """
    # Already a YAML path?
    p = Path(ref)
    if p.suffix in {".yml", ".yaml"} and p.exists():
        return str(p.resolve())

    idx = _build_index()

    # Allow implicit :latest
    if ":" not in ref:
        ref = ref + ":latest"

    # Try exact match
    if ref in idx:
        return idx[ref]

    # Try lower-cased
    ref_l = ref.lower()
    if ref_l in idx:
        return idx[ref_l]

    logger.debug(f"Manifest reference not found locally: {ref}")
    return None
