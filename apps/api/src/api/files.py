from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import Optional
import mimetypes
import hashlib

from src.utils.defaults import get_cache_path, get_components_path

router = APIRouter(prefix="/files", tags=["files"])


def _base_for_scope(scope: str) -> Path:
    s = (scope or "").lower().strip()
    if s == "apex-cache":
        return Path(get_cache_path()).expanduser().resolve()
    if s == "components":
        return Path(get_components_path()).expanduser().resolve()
    raise HTTPException(
        status_code=400, detail="Invalid scope; expected 'apex-cache' or 'components'"
    )


def _safe_join(base: Path, rel: str) -> Path:
    base = base.resolve()
    rel_input = (rel or "").strip()
    if not rel_input.startswith("/"):
        rel_input = "/" + rel_input

    # If rel already includes the base at its start, strip it to avoid duplication
    base_str = str(base)
    if rel_input.startswith(base_str):
        rel_stripped = rel_input[len(base_str) :].lstrip("/\\")
        rel_path = Path(rel_stripped)
    else:
        # Normalize leading separators for regular relative joins
        rel_path = Path(rel_input.lstrip("/\\"))
        # If an absolute path was provided, only allow it if it's under base
        if rel_path.is_absolute():
            try:
                rel_path = rel_path.resolve().relative_to(base)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid path")

    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_valid_sha256_hex(s: str) -> bool:
    if len(s) != 64:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False


@router.get("")
def get_file(request: Request, scope: str, path: str):
    base = _base_for_scope(scope)
    target = _safe_join(base, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    size = target.stat().st_size
    content_type, _ = mimetypes.guess_type(str(target))
    if not content_type:
        content_type = "application/octet-stream"

    range_header: Optional[str] = request.headers.get("range")

    def iter_range(start: int, end: int):
        with open(target, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            chunk = 1024 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        try:
            units, rng = range_header.split("=")
            if units.strip().lower() != "bytes":
                raise ValueError("Unsupported range unit")
            start_s, end_s = (rng.split("-") + [""])[:2]
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if start < 0 or end < start or end >= size:
                raise ValueError("Invalid range")
        except Exception:
            # Return full content if we can't parse the header
            start, end = 0, size - 1
            range_header = None

        if range_header:
            return StreamingResponse(
                iter_range(start, end),
                status_code=206,
                media_type=content_type,
                headers={
                    "Content-Length": str(end - start + 1),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Accept-Ranges": "bytes",
                },
            )

    # Fallback / full content
    return StreamingResponse(
        open(target, "rb"),
        media_type=content_type,
        headers={
            "Content-Length": str(size),
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/match")
def match_file(scope: str, path: str, sha256: str, size: Optional[int] = None):
    """
    Returns whether the remote file exists and whether its SHA-256 matches the provided hash.
    Useful for deduping uploads when the destination path is deterministic.
    """
    sha256_norm = (sha256 or "").strip().lower()
    if not _is_valid_sha256_hex(sha256_norm):
        raise HTTPException(
            status_code=400, detail="Invalid sha256; expected 64 hex chars"
        )

    base = _base_for_scope(scope)
    target = _safe_join(base, path)
    if not target.is_file():
        return {"exists": False, "matches": False}

    st = target.stat()
    if size is not None and size != st.st_size:
        return {
            "exists": True,
            "matches": False,
            "size": st.st_size,
            "reason": "size_mismatch",
        }

    try:
        computed = _sha256_file(target)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to hash file")

    return {
        "exists": True,
        "matches": computed == sha256_norm,
        "sha256": computed,
        "size": st.st_size,
    }


@router.post("/ingest")
async def ingest(scope: str, dest: str, file: UploadFile = File(...)):
    base = _base_for_scope(scope)
    target = _safe_join(base, dest)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    # Return the path relative to base
    rel = str(target.relative_to(base))
    return {"path": rel}
