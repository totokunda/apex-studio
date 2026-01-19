#!/usr/bin/env python3
"""
Apply a `python-code-*.tar.zst` update bundle atomically (transaction-like) to an existing install.

This updater is designed to work with the "code-only update bundle" produced by `bundle_python.py`,
which contains (at minimum):
  - python-code/apex-engine/src/
  - python-code/apex-engine/assets/ (optional)
  - python-code/apex-engine/manifest/ (optional)
  - python-code/apex-engine/requirements-bundle.txt (optional)
  - python-code/apex-engine/apex-code-update-manifest.json (optional)

The install target can be either:
  - the full env bundle directory (apex-engine/) that also contains the venv (apex-studio/), OR
  - a "code overlay" directory that only contains the code/config files.

This script ONLY replaces a small allowlist of top-level entries; it does not touch the venv.
Updates are applied with per-path atomic renames + rollback on error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

ALLOWLIST_TOP_LEVEL = [
    "src",
    "assets",
    "manifest",
    "transformer_configs",
    "vae_configs",
    "requirements-bundle.txt",
    "requirements.lock",
    "apex-code-update-manifest.json",
]


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _acquire_lock(lock_path: Path) -> None:
    """
    Simple cross-platform lock: create a lock file exclusively.
    If it already exists, fail fast.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Another update is in progress (lock exists): {lock_path}")
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"started_at={_now_tag()}\n")


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)  # py3.8+: missing_ok
    except TypeError:
        # Python <3.8 compatibility not needed here, but keep it safe.
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def _safe_extract_tar_stream(tf: tarfile.TarFile, dest: Path) -> None:
    """
    Extract a tar stream safely (prevent path traversal / absolute paths / device files).
    """
    dest = dest.resolve()

    def is_within(p: Path, root: Path) -> bool:
        try:
            p = p.resolve()
            return p == root or str(p).startswith(str(root) + os.sep)
        except Exception:
            return False

    for m in tf:
        name = m.name or ""
        # Disallow absolute paths and parent traversal
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise RuntimeError(f"Unsafe path in archive: {name}")
        # Disallow special files (symlinks, devices) to keep updates safe.
        if m.issym() or m.islnk():
            continue
        if m.ischr() or m.isblk() or m.isfifo():
            continue

        out_path = (dest / name).resolve()
        if not is_within(out_path, dest):
            raise RuntimeError(f"Refusing to extract outside destination: {name}")

        # Python 3.12+ supports `filter=` to harden extraction behavior (and avoid deprecation warnings).
        try:
            tf.extract(m, path=str(dest), set_attrs=True, filter="data")  # type: ignore[arg-type]
        except TypeError:
            # Older Python: no filter kwarg.
            tf.extract(m, path=str(dest), set_attrs=True)


def _extract_tar_zst(archive_path: Path, dest_dir: Path) -> None:
    """
    Extract a .tar.zst into dest_dir.

    Strategy:
      - Prefer `zstd -d -c` piped into Python tarfile (safe extraction).
      - If `zstd` is not available, try importing `zstandard` (optional dependency).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    zstd_bin = shutil.which("zstd")
    if zstd_bin:
        p = subprocess.Popen(
            [zstd_bin, "-d", "-c", str(archive_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert p.stdout is not None
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                _safe_extract_tar_stream(tf, dest_dir)
        finally:
            # Drain stderr and wait so we can surface errors.
            stderr = b""
            try:
                if p.stderr:
                    stderr = p.stderr.read() or b""
            except Exception:
                stderr = b""
            code = p.wait()
            if code != 0:
                msg = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"zstd failed (exit {code}): {msg}".strip())
        return

    # Optional pure-Python fallback (requires `zstandard` installed)
    try:
        import zstandard as zstd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Cannot extract .tar.zst: `zstd` binary not found on PATH and Python package `zstandard` not installed.\n"
            "Install zstd (macOS: `brew install zstd`, Linux: `apt/yum install zstd`) or `pip install zstandard`."
        ) from e

    dctx = zstd.ZstdDecompressor()
    with archive_path.open("rb") as f:
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                _safe_extract_tar_stream(tf, dest_dir)


def _find_update_payload_root(extracted_root: Path) -> Path:
    """
    Find the `.../python-code/apex-engine` directory inside extracted_root.
    """
    candidates = []
    for p in extracted_root.rglob("apex-code-update-manifest.json"):
        # Expect: python-code/apex-engine/apex-code-update-manifest.json
        candidates.append(p.parent)

    # Fallback: look for a directory containing src/
    if not candidates:
        for p in extracted_root.rglob("src"):
            if p.is_dir() and (p.parent / "src").is_dir():
                candidates.append(p.parent)

    if not candidates:
        raise RuntimeError(
            "Could not locate update payload root (expected apex-engine/ with src/ inside the archive)."
        )

    # Prefer the shallowest path (closest to extracted root)
    candidates.sort(key=lambda x: len(x.parts))
    return candidates[0]


def _copy_to_staging(src: Path, staging_dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, staging_dest, symlinks=False)
    else:
        staging_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, staging_dest)


def _venv_python_path(venv_dir: Path) -> Path | None:
    if os.name == "nt":
        p = venv_dir / "Scripts" / "python.exe"
    else:
        p = venv_dir / "bin" / "python"
    return p if p.exists() else None


def _run_uv_pip_sync(*, python: Path, lockfile: Path, quiet: bool) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "`uv` not found on PATH; required for lockfile sync. Install uv or add it to PATH."
        )

    cmd = [uv, "pip", "sync", "-p", str(python), "--strict"]
    if quiet:
        cmd.append("--no-progress")
        cmd.append("-q")
    cmd.append(str(lockfile))

    proc = subprocess.run(cmd, capture_output=quiet, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(
            f"uv pip sync failed (exit {proc.returncode}). Output:\n{out.strip()}"
        )


def _smoke_test_imports(python: Path, *, quiet: bool) -> None:
    """
    Best-effort post-sync validation. Controlled via env var:
      APEX_CODE_UPDATE_SMOKE_IMPORTS="fastapi,uvicorn"
    """
    raw = os.environ.get(
        "APEX_CODE_UPDATE_SMOKE_IMPORTS", "fastapi,uvicorn,torch"
    ).strip()
    mods = [m.strip() for m in raw.split(",") if m.strip()]
    if not mods:
        return
    code = (
        "import importlib; "
        + "; ".join([f"importlib.import_module('{m}')" for m in mods])
        + "; print('ok')"
    )
    proc = subprocess.run([str(python), "-c", code], capture_output=quiet, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"Smoke test imports failed. Output:\n{out.strip()}")


def apply_update(
    code_archive: Path,
    target_dir: Path,
    *,
    keep_backup: bool = False,
    verbose: bool = True,
) -> None:
    code_archive = code_archive.resolve()
    target_dir = target_dir.resolve()
    if not code_archive.exists():
        raise SystemExit(f"Update archive not found: {code_archive}")
    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")

    lock_path = target_dir / ".apex-code-update.lock"
    _acquire_lock(lock_path)

    tmp_root = Path(tempfile.mkdtemp(prefix="apex-code-update-"))
    extracted_root = tmp_root / "extracted"
    staging_root = target_dir.parent / f".apex-code-update-staging-{uuid.uuid4().hex}"
    # Backups are required for rollback during the update. By default we delete them after success.
    backup_root = (
        target_dir.parent
        / ".apex-code-update-backups"
        / f"{_now_tag()}-{uuid.uuid4().hex}"
    )

    replaced: list[str] = []
    replaced_venv: bool = False
    env_sync_planned: bool = False
    env_sync_applied: bool = False
    new_venv_dir: Path | None = None

    try:
        if verbose:
            print(f"[update] extracting: {code_archive.name}")
        _extract_tar_zst(code_archive, extracted_root)
        payload_root = _find_update_payload_root(extracted_root)

        if verbose:
            print(f"[update] payload root: {payload_root}")

        # Optional: read manifest
        manifest_path = payload_root / "apex-code-update-manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if verbose:
                    print(f"[update] code version: {manifest.get('version')}")
            except Exception:
                pass

        # Prepare staging
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        # Stage all allowlisted entries that exist in the payload
        to_update: list[str] = []
        for name in ALLOWLIST_TOP_LEVEL:
            if (payload_root / name).exists():
                to_update.append(name)

        if "src" not in to_update:
            raise RuntimeError(
                "Update payload does not include `src/` - refusing to apply."
            )

        if verbose:
            print(f"[update] entries to update: {', '.join(to_update)}")

        # Decide whether to perform a two-phase env sync based on a lockfile.
        # - Only runs if payload includes requirements.lock AND target contains a venv.
        # - Creates a new venv copy, syncs it, smoke-tests it, then atomically swaps it in.
        lock_in_payload = payload_root / "requirements.lock"
        existing_lock = target_dir / "requirements.lock"
        venv_dir = target_dir / "apex-studio"
        venv_py = _venv_python_path(venv_dir)

        if lock_in_payload.exists() and venv_py is not None:
            try:
                payload_bytes = lock_in_payload.read_bytes()
                existing_bytes = (
                    existing_lock.read_bytes() if existing_lock.exists() else b""
                )
                env_sync_planned = payload_bytes != existing_bytes
            except Exception:
                env_sync_planned = True

        if env_sync_planned and verbose:
            print("[update] lockfile changed; preparing two-phase venv sync…")

        for name in to_update:
            _copy_to_staging(payload_root / name, staging_root / name)

        # Prepare the next venv BEFORE mutating the live install.
        # This gives us a clean rollback path if dependency resolution fails.
        if env_sync_planned:
            assert venv_py is not None
            if not venv_dir.exists():
                raise RuntimeError("Expected venv directory not found: apex-studio/")

            new_venv_dir = target_dir / f".apex-studio-staging-{uuid.uuid4().hex}"
            if new_venv_dir.exists():
                shutil.rmtree(new_venv_dir)
            if verbose:
                print(
                    f"[update] copying venv to staging: {new_venv_dir.name} (this may take a while)"
                )

            shutil.copytree(venv_dir, new_venv_dir, symlinks=False)
            new_py = _venv_python_path(new_venv_dir)
            if new_py is None:
                raise RuntimeError("Staged venv python not found after copy.")

            _run_uv_pip_sync(python=new_py, lockfile=lock_in_payload, quiet=not verbose)
            _smoke_test_imports(new_py, quiet=not verbose)

        # Transaction-like apply:
        # 1) move existing entries to backup (atomic per entry)
        # 2) move staged entries into place (atomic per entry)
        # 3) optionally swap the venv atomically (two-phase env update)
        backup_root.mkdir(parents=True, exist_ok=True)

        for name in to_update:
            dst = target_dir / name
            if dst.exists():
                (backup_root / name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(dst), str(backup_root / name))

        # Swap venv (after backup dir exists) so we can roll back.
        if env_sync_planned and new_venv_dir is not None:
            venv_backup = backup_root / "apex-studio"
            if (target_dir / "apex-studio").exists():
                os.replace(str(target_dir / "apex-studio"), str(venv_backup))
                replaced_venv = True
            os.replace(str(new_venv_dir), str(target_dir / "apex-studio"))
            env_sync_applied = True

        for name in to_update:
            src = staging_root / name
            dst = target_dir / name
            # Ensure destination parent exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(src), str(dst))
            replaced.append(name)

        # Write a small marker
        marker = target_dir / ".apex-code-update-last.json"
        marker.write_text(
            json.dumps(
                {
                    "applied_at": _now_tag(),
                    "archive": code_archive.name,
                    "entries": replaced,
                    "env_sync_planned": bool(env_sync_planned),
                    "env_sync_applied": bool(env_sync_applied),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if verbose:
            print("[update] update applied successfully")

        if not keep_backup:
            try:
                shutil.rmtree(backup_root)
            except Exception:
                pass

    except Exception as e:
        # Rollback: restore anything we moved to backup.
        if verbose:
            print(f"[update] ERROR: {e}")
            print("[update] attempting rollback…")
        try:
            # Roll back code entries
            for name in reversed(replaced):
                dst = target_dir / name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                b = backup_root / name
                if b.exists():
                    os.replace(str(b), str(dst))

            # Roll back venv swap
            if replaced_venv:
                cur = target_dir / "apex-studio"
                if cur.exists():
                    shutil.rmtree(cur)
                b = backup_root / "apex-studio"
                if b.exists():
                    os.replace(str(b), str(target_dir / "apex-studio"))
        except Exception as re:
            if verbose:
                print(f"[update] rollback failed: {re}")
        raise
    finally:
        try:
            if staging_root.exists():
                shutil.rmtree(staging_root)
        except Exception:
            pass
        try:
            if new_venv_dir and new_venv_dir.exists():
                shutil.rmtree(new_venv_dir)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
        _release_lock(lock_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a python-code tar.zst update bundle atomically."
    )
    parser.add_argument(
        "--code-archive",
        required=True,
        type=Path,
        help="Path to python-code-*.tar.zst",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Path to the existing apex-engine directory to update (will replace src/ and related config dirs).",
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep rollback backups after a successful update (uses disk; mostly for debugging).",
    )
    # Back-compat: older flag name (prefer --keep-backup).
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="(Deprecated) Kept for compatibility. Backups are not retained by default; use --keep-backup to retain.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    apply_update(
        code_archive=args.code_archive,
        target_dir=args.target_dir,
        keep_backup=bool(args.keep_backup) and not bool(args.no_backup),
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
