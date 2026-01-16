from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from src.utils.defaults import get_cache_path, get_config_store_path, DEFAULT_HEADERS
from src.utils.config_store import config_store_lock, read_json_dict, write_json_dict_atomic


def _now_iso() -> str:
    # ISO-ish without timezone for simple human readability.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_persisted_config() -> dict:
    p = Path(get_config_store_path())
    try:
        with config_store_lock(p):
            return read_json_dict(p)
    except Exception:
        return {}


def update_persisted_config(**updates: Any) -> None:
    """
    Best-effort persisted config update. Never raises: update flow should not crash the API.
    """
    try:
        p = Path(get_config_store_path())
        with config_store_lock(p):
            data = read_json_dict(p)
            for k, v in updates.items():
                if v is not None:
                    data[k] = v
            write_json_dict_atomic(p, data, indent=2)
    except Exception as e:
        # Avoid hard failure; log for debugging.
        print(f"Warning: failed to persist auto-update settings: {e}")


@dataclass(frozen=True)
class AutoUpdateConfig:
    enabled: bool
    interval_seconds: int
    repo_owner: str
    repo_name: str
    include_prerelease: bool


def get_auto_update_config() -> AutoUpdateConfig:
    persisted = read_persisted_config()

    # Defaults
    enabled = bool(persisted.get("auto_update_enabled", True))
    interval_hours_raw = persisted.get("auto_update_interval_hours", 4)
    try:
        interval_hours = float(interval_hours_raw)
    except Exception:
        interval_hours = 4.0
    interval_seconds = int(max(300, interval_hours * 3600))  # min 5 minutes

    repo_owner = str(
        persisted.get("auto_update_repo_owner")
        or os.environ.get("APEX_UPDATE_REPO_OWNER")
        or "totoku"
    ).strip()
    repo_name = str(
        persisted.get("auto_update_repo_name")
        or os.environ.get("APEX_UPDATE_REPO_NAME")
        or "apex-studio-server"
    ).strip()
    include_prerelease = bool(
        persisted.get(
            "auto_update_include_prerelease",
            os.environ.get("APEX_UPDATE_INCLUDE_PRERELEASE", "").strip().lower()
            in {"1", "true", "yes"},
        )
    )

    # Global kill switch (env) for emergency.
    if os.environ.get("APEX_DISABLE_AUTO_UPDATE", "").strip().lower() in {"1", "true", "yes"}:
        enabled = False

    return AutoUpdateConfig(
        enabled=enabled,
        interval_seconds=interval_seconds,
        repo_owner=repo_owner,
        repo_name=repo_name,
        include_prerelease=include_prerelease,
    )


def _parse_semver_triplet(v: str) -> Optional[tuple[int, int, int]]:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", (v or "").strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _is_semver_newer(a: str, b: str) -> bool:
    """
    Return True if a > b for X.Y.Z. Unknown versions are treated as not newer.
    """
    pa = _parse_semver_triplet(a)
    pb = _parse_semver_triplet(b)
    if not pa or not pb:
        return False
    return pa > pb


def _host_platform_candidates() -> set[str]:
    base = sys.platform.lower()
    out = {base}
    if base.startswith("win"):
        out |= {"win32", "windows"}
    if base == "darwin":
        out |= {"macos"}
    return out


def _host_arch_candidates() -> set[str]:
    import platform as _platform

    m = (_platform.machine() or "").lower()
    out: set[str] = {m} if m else set()
    # common aliases
    if m in {"x86_64", "amd64"}:
        out |= {"x64", "x86_64", "amd64"}
    if m in {"aarch64", "arm64"}:
        out |= {"arm64", "aarch64"}
    return out


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _resolve_apex_engine_root() -> Optional[Path]:
    """
    Best-effort locate the apex-engine root directory (the folder that contains `src/`).
    We intentionally avoid touching the user's workspace in dev unless explicitly allowed.
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").is_dir():
            # Expecting installed layout: <root>/src/api/...
            return p
    return None


def _read_installed_manifest(root: Path) -> dict:
    # Prefer the bundle manifest if present (created by bundler).
    m = root / "apex-engine-manifest.json"
    if m.exists():
        return _read_json_file(m)
    # Some setups may only ship the code-update manifest marker.
    m2 = root / "apex-code-update-manifest.json"
    if m2.exists():
        return _read_json_file(m2)
    return {}


def _current_installed_version(root: Path) -> str:
    persisted = read_persisted_config()
    v = str(persisted.get("auto_update_current_version") or "").strip()
    if _parse_semver_triplet(v):
        return v

    m = _read_installed_manifest(root)
    mv = str(m.get("version") or "").strip()
    if _parse_semver_triplet(mv):
        return mv

    # Final fallback: pyproject version may exist in source checkouts.
    try:
        from importlib.metadata import version as _pkg_version

        pv = str(_pkg_version("apex-engine") or "").strip()
        if _parse_semver_triplet(pv):
            return pv
    except Exception:
        pass

    return "0.0.0"


def _current_gpu_support(root: Path) -> str:
    m = _read_installed_manifest(root)
    gpu = str(m.get("gpu_support") or "").strip().lower()
    return gpu or "cpu"


@dataclass(frozen=True)
class RemoteAsset:
    tag: str
    tag_version: str
    asset_name: str
    download_url: str
    asset_version: str
    platform: str
    arch: str
    device: str
    python_tag: str
    prerelease: bool
    published_at: str | None


def _parse_python_code_asset(asset_name: str) -> Optional[dict]:
    """
    Accept:
      python-code-<version>-<platform>-<arch>-<device>-<pythonTag>[-extras].tar.zst
    """
    name = (asset_name or "").strip()
    if not name.lower().endswith(".tar.zst"):
        return None
    if not name.lower().startswith("python-code-"):
        return None
    core = name[:-len(".tar.zst")]
    parts = core.split("-")
    # parts: ["python", "code", <version>, <platform>, <arch>, <device>, <pythonTag>, ...extras]
    if len(parts) < 7:
        return None
    version = parts[2]
    plat = parts[3]
    arch = parts[4]
    device = parts[5]
    py = parts[6]
    return {
        "asset_version": version,
        "platform": plat,
        "arch": arch,
        "device": device,
        "python_tag": py,
    }


def _encode_hf_path(path: str) -> str:
    """
    Encode a repo-relative path for Hugging Face URLs.
    We must preserve "/" separators while percent-encoding each segment.
    """
    parts = [p for p in (path or "").split("/") if p]
    return "/".join(urllib.parse.quote(p, safe="") for p in parts)


def _hf_json(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "apex-engine",
            **(DEFAULT_HEADERS or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8"))


def _hf_list_tree(cfg: AutoUpdateConfig, path: str | None = None) -> list[dict]:
    base = f"https://huggingface.co/api/models/{cfg.repo_owner}/{cfg.repo_name}/tree/main"
    url = f"{base}/{_encode_hf_path(path)}" if path else base
    data = _hf_json(url)
    return data if isinstance(data, list) else []


def list_remote_python_code_assets(cfg: AutoUpdateConfig) -> list[RemoteAsset]:
    plat_ok = _host_platform_candidates()
    arch_ok = _host_arch_candidates()
    py_tag = _python_tag().lower()

    out: list[RemoteAsset] = []
    root = _hf_list_tree(cfg)
    version_dirs: list[str] = []
    for item in root:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "directory":
            continue
        p = str(item.get("path") or "")
        if not p:
            continue
        base = (p.split("/")[-1] or "").strip()
        if re.fullmatch(r"v?\d+\.\d+\.\d+", base, flags=re.IGNORECASE):
            version_dirs.append(p)

    def _dir_sort_key(p: str) -> tuple[int, int, int]:
        base = (p.split("/")[-1] or "").lstrip("vV")
        return _parse_semver_triplet(base) or (0, 0, 0)

    version_dirs.sort(key=_dir_sort_key, reverse=True)

    for dir_path in version_dirs:
        base = (dir_path.split("/")[-1] or "").strip()
        tag = base
        tag_ver = base.lstrip("vV")

        entries = _hf_list_tree(cfg, dir_path)
        for e in entries:
            if not isinstance(e, dict):
                continue
            if str(e.get("type") or "") != "file":
                continue
            full_path = str(e.get("path") or "")
            if not full_path:
                continue
            asset_name = full_path.split("/")[-1] or ""
            if not asset_name:
                continue

            parsed = _parse_python_code_asset(asset_name)
            if not parsed:
                continue
            if str(parsed["python_tag"]).lower() != py_tag:
                continue
            if str(parsed["platform"]).lower() not in plat_ok:
                continue
            if str(parsed["arch"]).lower() not in arch_ok:
                continue

            dl = (
                f"https://huggingface.co/{cfg.repo_owner}/{cfg.repo_name}/resolve/main/"
                f"{_encode_hf_path(full_path)}?download=true"
            )
            out.append(
                RemoteAsset(
                    tag=tag,
                    tag_version=tag_ver,
                    asset_name=asset_name,
                    download_url=dl,
                    asset_version=str(parsed["asset_version"]),
                    platform=str(parsed["platform"]),
                    arch=str(parsed["arch"]),
                    device=str(parsed["device"]),
                    python_tag=str(parsed["python_tag"]),
                    prerelease=False,
                    published_at=None,
                )
            )

    def sort_key(x: RemoteAsset) -> tuple:
        p = _parse_semver_triplet(x.tag_version) or (0, 0, 0)
        a = _parse_semver_triplet(x.asset_version) or (0, 0, 0)
        return (p, a, x.asset_name)

    out.sort(key=sort_key, reverse=True)
    return out


def _download_file(url: str, dest: Path) -> None:
    """
    Download a file to an explicit destination path.

    Uses the project's DownloadMixin for consistent behavior (resume support, .part files,
    retries, shared headers, optional Rust fast-path).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Delegate to shared download logic
    from src.mixins.download_mixin import DownloadMixin

    DownloadMixin()._download_from_url(  # noqa: SLF001 (internal helper, intentionally reused)
        url=str(url),
        save_path=str(dest.parent),
        progress_callback=None,
        filename=dest.name,
        dest_path=str(dest),
    )


def _run_apply_code_update(archive_path: Path, target_root: Path) -> tuple[int, str]:
    """
    Run scripts/apply_code_update.py in a subprocess.
    Returns (exit_code, combined_output_tail).
    """
    # Prefer scripts shipped with the bundle root (installed scenario).
    script = target_root / "scripts" / "apply_code_update.py"
    if not script.exists():
        # Dev fallback: repo layout (apps/api/scripts)
        repo_script = Path(__file__).resolve().parents[3] / "scripts" / "apply_code_update.py"
        if repo_script.exists():
            script = repo_script
    if not script.exists():
        raise RuntimeError(f"apply_code_update.py not found under {target_root}/scripts")

    import subprocess as _subprocess

    cmd = [
        sys.executable,
        str(script),
        "--code-archive",
        str(archive_path),
        "--target-dir",
        str(target_root),
        "--quiet",
    ]
    p = _subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    tail = out.strip()[-4000:]
    return p.returncode, tail


def _safe_to_update(root: Path) -> bool:
    """
    Safety gate: avoid mutating a dev checkout unless explicitly enabled.
    """
    # If it's clearly an installed bundle (has manifest), allow.
    if (root / "apex-engine-manifest.json").exists():
        return True
    # If env explicitly allows, allow.
    if os.environ.get("APEX_ALLOW_DEV_AUTO_UPDATE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return False


def _has_active_jobs() -> bool:
    """
    Best-effort: return True if any tracked Ray job is still running/queued.
    Never raises: auto-update should not crash the API.
    """
    try:
        import ray

        # Local import to avoid import-time Ray initialization side effects.
        from .job_store import job_store as unified_job_store

        if not ray.is_initialized():
            # If Ray isn't initialized, there shouldn't be active Ray jobs.
            return False

        for job_id in unified_job_store.all_job_ids():
            info = unified_job_store.get(job_id) or {}
            ref = info.get("ref")
            if ref is None:
                continue
            ready, _ = ray.wait([ref], timeout=0)
            if not ready:
                return True
        return False
    except Exception:
        return False


def _request_self_restart(reason: str) -> None:
    """
    Gracefully stop the API process so the supervisor (Apex Studio / systemd / etc.)
    can restart it. SIGTERM should trigger FastAPI lifespan shutdown and `shutdown_ray()`.
    """
    try:
        update_persisted_config(
            auto_update_restart_reason=reason,
            auto_update_restart_requested_at=_now_iso(),
        )
    except Exception:
        pass

    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # Last resort: hard exit
        os._exit(0)


def auto_update_once() -> None:
    """
    Synchronous "one-shot" update check + apply. Designed to be run in a thread.
    """
    cfg = get_auto_update_config()
    if not cfg.enabled:
        return

    root = _resolve_apex_engine_root()
    if not root:
        return
    if not _safe_to_update(root):
        return

    current_version = _current_installed_version(root)
    gpu_support = _current_gpu_support(root)

    update_persisted_config(
        auto_update_last_checked_at=_now_iso(),
        auto_update_current_version=current_version,
        auto_update_current_gpu_support=gpu_support,
    )

    assets = list_remote_python_code_assets(cfg)
    if not assets:
        return

    # Prefer assets matching our current gpu_support, but fall back to any that match host+python.
    preferred = [a for a in assets if str(a.device).lower() == str(gpu_support).lower()]
    candidate = preferred[0] if preferred else assets[0]

    if not _is_semver_newer(candidate.asset_version, current_version):
        return

    # Download into cache
    cache_root = Path(get_cache_path()).expanduser().resolve()
    updates_dir = cache_root / "updates"
    archive_path = updates_dir / candidate.asset_name

    update_persisted_config(
        auto_update_available_version=candidate.asset_version,
        auto_update_available_asset=candidate.asset_name,
        auto_update_available_tag=candidate.tag,
        auto_update_download_url=candidate.download_url,
    )

    if not archive_path.exists():
        _download_file(candidate.download_url, archive_path)

    # Apply update
    update_persisted_config(
        auto_update_last_apply_started_at=_now_iso(),
        auto_update_last_apply_asset=candidate.asset_name,
        auto_update_last_apply_version=candidate.asset_version,
        auto_update_last_apply_status="running",
    )
    code, tail = _run_apply_code_update(archive_path, root)
    if code == 0:
        update_persisted_config(
            auto_update_last_apply_finished_at=_now_iso(),
            auto_update_last_apply_status="success",
            auto_update_last_apply_output_tail=tail,
            auto_update_current_version=candidate.asset_version,
            auto_update_restart_pending=True,
        )
        # Restart immediately if we're idle; otherwise, defer to the loop to retry.
        if not _has_active_jobs():
            _request_self_restart(reason=f"Auto-update applied {candidate.asset_version}")
    else:
        update_persisted_config(
            auto_update_last_apply_finished_at=_now_iso(),
            auto_update_last_apply_status="failed",
            auto_update_last_apply_output_tail=tail,
        )


async def auto_update_loop() -> None:
    """
    Background task: run once on startup, then periodically (default 4 hours).
    Reads persisted config each cycle so users can disable it via API without restart.
    """
    while True:
        try:
            cfg = get_auto_update_config()
            # If an update was applied earlier, restart once we're idle.
            persisted = read_persisted_config()
            if persisted.get("auto_update_restart_pending") and not _has_active_jobs():
                update_persisted_config(auto_update_restart_pending=False)
                _request_self_restart(reason="Auto-update pending restart (idle)")
            if cfg.enabled:
                await asyncio.to_thread(auto_update_once)
            await asyncio.sleep(cfg.interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Keep the loop alive; next cycle will retry.
            try:
                update_persisted_config(auto_update_last_error=repr(e), auto_update_last_error_at=_now_iso())
            except Exception:
                pass
            await asyncio.sleep(60)


