from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
import mimetypes
import hashlib
import stat

from src.utils.defaults import get_cache_path, get_components_path

router = APIRouter(prefix="/files", tags=["files"])

# Larger chunks reduce Python/ASGI overhead and usually improve throughput for large downloads.
# Keep this reasonably sized to avoid memory spikes under heavy concurrency.
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


class TunedFileResponse(FileResponse):
    # Starlette's default is typically 64KiB; larger chunks often improve throughput on remote clients.
    chunk_size = DOWNLOAD_CHUNK_SIZE


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

    try:
        stat_result = target.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="Not found")

    if not stat.S_ISREG(stat_result.st_mode):
        raise HTTPException(status_code=404, detail="Not found")

    # Simple/fast weak ETag based on mtime and size
    etag = f'"{stat_result.st_mtime}-{stat_result.st_size}"'

    # Handle If-None-Match manually since Starlette's FileResponse doesn't (yet)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

    # Handle If-Range (RFC 7233):
    # If the client provides a conditional range request, verify the ETag.
    # If the ETag matches, we allow the 206 Partial Content (via FileResponse).
    # If it doesn't match, we must ignore the Range header and send 200 (full resource).
    if_range = request.headers.get("if-range")
    if if_range and if_range != etag and request.headers.get("range"):
        # Remove 'range' from the raw ASGI scope headers so Starlette ignores it
        # and serves a 200 OK with the full new file.
        request.scope["headers"] = [
            (k, v) for k, v in request.scope["headers"] if k.lower() != b"range"
        ]

    content_type, _ = mimetypes.guess_type(str(target))
    if not content_type:
        content_type = "application/octet-stream"

    return TunedFileResponse(
        path=target,
        status_code=206,
        stat_result=stat_result,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=14400",  # 4 hours
            "ETag": etag,
        },
    )


@router.get("/exists")
def exists_file(scope: str, path: str):
    base = _base_for_scope(scope)
    target = _safe_join(base, path)
    return {"exists": target.exists()}


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
