#!/usr/bin/env python3
from __future__ import annotations

"""
Variant-matrix test runner for Apex Studio models using the JSON + assets in `apex-test-suite`.

What this does (high-level):
- Discovers all test JSON payloads under `<suite_dir>/{image,video,upscalers}/*.json`
- Maps each JSON test to its corresponding manifest YAML in `<manifest_dir>/{image,video,upscalers}/`
  (with a fallback that strips trailing semver-like suffixes, e.g. `-1.0.0.v1` or `.1.0.0.v1`)
- Runs sweeps in phases:
  1) Text-encoder variants: run all available text-encoder variants for each model
  2) Choose a default text-encoder variant per model (best pass rate; prefer "default" on ties)
  3) Transformer variants: run all transformer variants for each model using the chosen TE variant
  4) Choose a default transformer variant per model (same rule)
  5) Lynx (or any model with `type: extra_model_path` pseudo-components): sweep adapter variants,
     using the chosen TE + transformer variants.

Implementation details:
- Each (test, variant-selection) run is executed in a fresh subprocess ("--worker") to reduce
  VRAM fragmentation/leaks and to isolate crashes.
- Outputs are written under `<suite_dir>/outputs_variants/<run_id>/`.
- A JSONL of per-run records + a final `summary.json` are written for downstream analysis.

Notes:
- This script runs engines directly (no API server required).
- Ensure your Python environment can import `src.*` from the Apex repo (run from within `apps/api`
  or pass `--bundle-root`).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SEMVER_SUFFIX_RE = re.compile(r"(?P<base>.+?)(?:[.-])\d+\.\d+\.\d+\.v\d+$")


def _eprint(*args: Any) -> None:
    """
    Best-effort stderr printing.

    This script prints a machine-readable JSON status on stdout at the end, so
    human-facing errors must go to stderr to avoid breaking callers.
    """
    try:
        print(*args, file=sys.stderr, flush=True)
    except Exception:
        pass


def _tail(s: str, *, max_chars: int = 20000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


@dataclass
class FailFastError(Exception):
    """
    Raised when --fail-fast is enabled and a job fails.

    We carry enough context to:
    - write a partial summary
    - keep downloads for the failing model
    - allow resuming with --skip-successes
    """

    message: str
    download_root: Path
    worker_result_path: Path
    record: Dict[str, Any]
    results_so_far: List[Dict[str, Any]]


def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_on_syspath(bundle_root: Path) -> None:
    bundle_root = bundle_root.resolve()
    s = str(bundle_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _default_bundle_root() -> Path:
    # This file lives at apps/api/scripts/run_variant_matrix.py → bundle root is apps/api/
    return Path(__file__).resolve().parents[1]


def _default_suite_dir(bundle_root: Path) -> Path:
    # Prefer the canonical path used by other runners if present.
    cand = bundle_root / "test_suite"
    if cand.exists():
        return cand
    return bundle_root / "apex-test-suite"


def _default_manifest_dir(bundle_root: Path) -> Path:
    # Current manifests in this repo live under manifest/v0.1.2/
    return bundle_root / "manifest" / "v0.1.2"


def _slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "unknown"


def _iter_test_jsons(suite_dir: Path, *, kind: str = "all") -> List[Path]:
    kinds = ["image", "video", "upscalers", "audio"] if kind == "all" else [kind]
    out: List[Path] = []
    for k in kinds:
        d = suite_dir / k
        if not d.exists():
            continue
        out.extend(sorted(d.glob("*.json")))
    return out


def _strip_suffix(stem: str) -> str:
    m = SEMVER_SUFFIX_RE.match(stem)
    return m.group("base") if m else stem


def _resolve_manifest_for_test(
    *,
    test_json_path: Path,
    kind: str,
    manifest_dir: Path,
) -> Optional[Path]:
    """
    Map `<suite>/<kind>/<name>.json` → `<manifest_dir>/<kind>/<name>.yml`,
    with fallback that strips trailing semver-like suffixes.
    """
    name = test_json_path.stem
    candidates: List[str] = [name, _strip_suffix(name)]
    mdir = manifest_dir / kind
    for base in candidates:
        for ext in (".yml", ".yaml"):
            p = mdir / f"{base}{ext}"
            if p.exists():
                return p
    return None


def _safe_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    # Keep this lightweight (avoid importing torch, etc.) by only using PyYAML here.
    import yaml  # type: ignore

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


@dataclass(frozen=True)
class ManifestVariants:
    manifest_path: Path
    model_id: str
    kind: str
    # Per-component selectable variants. Key is component `name` if present else `type`.
    text_encoder_components: Tuple[Tuple[str, Tuple[str, ...]], ...]
    transformer_components: Tuple[Tuple[str, Tuple[str, ...]], ...]
    text_encoder_variants: Tuple[str, ...]
    transformer_variants: Tuple[str, ...]
    extra_components: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (selection_key, variants)
    is_lynx: bool
    fps: Optional[int]


def _extract_variants_from_manifest(
    manifest_path: Path, *, kind: str
) -> ManifestVariants:
    doc = _safe_read_yaml(manifest_path)
    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
    spec = doc.get("spec", {}) if isinstance(doc, dict) else {}
    components = spec.get("components", []) if isinstance(spec, dict) else []

    model_id = str(metadata.get("id") or manifest_path.stem)

    # Compute default fps if specified.
    fps = None
    try:
        fps_val = spec.get("fps")
        if isinstance(fps_val, int) and fps_val > 0:
            fps = fps_val
    except Exception:
        fps = None

    te_components: List[Tuple[str, Tuple[str, ...]]] = []
    xf_components: List[Tuple[str, Tuple[str, ...]]] = []
    te_union: List[str] = []
    xf_union: List[str] = []
    extra: List[Tuple[str, Tuple[str, ...]]] = []

    def _collect_model_path_variants(model_path: Any) -> List[str]:
        if isinstance(model_path, list):
            vs: List[str] = []
            for item in model_path:
                if isinstance(item, dict):
                    v = item.get("variant") or "default"
                    if isinstance(v, str) and v not in vs:
                        vs.append(v)
            return vs
        # Single path → treat as one implicit variant.
        return ["default"]

    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue
            ctype = comp.get("type")
            cname = comp.get("name")

            if ctype == "text_encoder":
                selection_key = (
                    cname.strip()
                    if isinstance(cname, str) and cname.strip()
                    else "text_encoder"
                )
                variants = tuple(_collect_model_path_variants(comp.get("model_path")))
                te_components.append((selection_key, variants))
                for v in variants:
                    if v not in te_union:
                        te_union.append(v)

            elif ctype == "transformer":
                selection_key = (
                    cname.strip()
                    if isinstance(cname, str) and cname.strip()
                    else "transformer"
                )
                variants = tuple(_collect_model_path_variants(comp.get("model_path")))
                xf_components.append((selection_key, variants))
                for v in variants:
                    if v not in xf_union:
                        xf_union.append(v)

            elif ctype == "extra_model_path":
                # Selection key must match BaseEngine's pre-pass lookup:
                # selected_label = pseudo.get("name") or pseudo.get("label")
                selection_key = comp.get("name") or comp.get("label")
                if isinstance(selection_key, str) and selection_key.strip():
                    selection_key = selection_key.strip()
                    raw_model_paths = comp.get("model_paths", comp.get("model_path"))
                    variants = tuple(_collect_model_path_variants(raw_model_paths))
                    extra.append((selection_key, variants))

    # Determine whether this is a Lynx model (used for reporting, but we don't special-case much).
    is_lynx = False
    try:
        tags = metadata.get("tags") or []
        if isinstance(tags, list) and any(str(t).lower() == "lynx" for t in tags):
            is_lynx = True
        if isinstance(model_id, str) and "lynx" in model_id.lower():
            is_lynx = True
        if str(spec.get("model_type") or "").lower() == "lynx":
            is_lynx = True
    except Exception:
        is_lynx = False

    return ManifestVariants(
        manifest_path=manifest_path,
        model_id=model_id,
        kind=kind,
        text_encoder_components=tuple(te_components)
        if te_components
        else (("text_encoder", ("default",)),),
        transformer_components=tuple(xf_components)
        if xf_components
        else (("transformer", ("default",)),),
        text_encoder_variants=tuple(te_union) if te_union else ("default",),
        transformer_variants=tuple(xf_union) if xf_union else ("default",),
        extra_components=tuple(extra),
        is_lynx=is_lynx,
        fps=fps,
    )


def _pick_best_variant(stats: Dict[str, Dict[str, int]]) -> str:
    """
    stats: variant -> {"ok": int, "fail": int}
    Returns best variant by (ok desc, fail asc), preferring literal "default" on ties.
    """
    best: Optional[str] = None
    best_key: Optional[Tuple[int, int, int]] = None
    for variant, s in stats.items():
        ok = int(s.get("ok", 0))
        fail = int(s.get("fail", 0))
        prefer_default = 1 if variant == "default" else 0
        key = (ok, -fail, prefer_default)
        if best_key is None or key > best_key:
            best_key = key
            best = variant
    return best or "default"


def _resolve_assets_in_payload(payload: Any, *, suite_dir: Path) -> Any:
    """
    Convert any string value starting with "assets/" into an absolute path under suite_dir/assets/.
    Applies recursively to dicts/lists.
    """
    assets_dir = suite_dir / "assets"

    def _map(x: Any) -> Any:
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("assets/") or s.startswith("assets\\"):
                rel = s.replace("\\", "/")
                p = (assets_dir / rel.split("/", 1)[1]).resolve()
                return str(p)
            return x
        if isinstance(x, list):
            return [_map(v) for v in x]
        if isinstance(x, dict):
            return {k: _map(v) for k, v in x.items()}
        return x

    return _map(payload)


def _worker_run_one(
    *,
    bundle_root: Path,
    suite_dir: Path,
    outputs_dir: Path,
    test_json_path: Path,
    manifest_path: Path,
    selected_components: Dict[str, Any],
    filename_prefix: str,
    kind: str,
    fps_hint: Optional[int],
) -> Dict[str, Any]:
    _ensure_on_syspath(bundle_root)

    # Imports that rely on bundle_root being on sys.path
    import numpy as np  # noqa: PLC0415

    from src.engine import UniversalEngine  # type: ignore
    from src.api.savers.engine_results import save_engine_output  # type: ignore
    from src.api.savers.audio_video import (  # type: ignore
        save_video_ovi,
        save_video_ltx2,
        save_video_mova,
    )

    t0 = time.time()
    payload_raw = _safe_read_json(test_json_path)
    payload = _resolve_assets_in_payload(payload_raw, suite_dir=suite_dir)

    # Allow per-test fps override, else manifest-provided hint, else default to 16.
    fps = 16
    if isinstance(payload, dict):
        if isinstance(payload.get("fps"), int) and int(payload["fps"]) > 0:
            fps = int(payload["fps"])
        elif isinstance(fps_hint, int) and fps_hint > 0:
            fps = int(fps_hint)

    # For upscalers, keep reference to input video for possible audio muxing.
    input_video_for_audio_mux = None
    if kind == "upscalers":
        for k in ("video", "input_video", "source_video", "video_path"):
            v = payload.get(k) if isinstance(payload, dict) else None
            if isinstance(v, str):
                input_video_for_audio_mux = v
                break

    # Run engine.
    # Note: BaseEngine will download / resolve selected component variants at init time.
    engine = UniversalEngine(
        yaml_path=str(manifest_path),
        selected_components=selected_components,
        should_download=True,
    )

    out_obj = engine.run(**payload)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Engines that generate audio+video may return a tuple (video, audio).
    # Use the dedicated muxing saver so audio is embedded into the MP4.
    if isinstance(out_obj, tuple) and len(out_obj) == 2:
        video_part, audio_part = out_obj
        try:
            # Heuristic:
            # - OVI: video is numpy (C, F, H, W)
            # - LTX2: video is torch tensor / iterator yielding (F, H, W, C)
            if isinstance(video_part, np.ndarray) and video_part.ndim == 4 and video_part.shape[0] in (1, 3):
                saved_path, media_type = save_video_ovi(
                    video_numpy=video_part,
                    audio_numpy=audio_part,
                    filename_prefix=filename_prefix,
                    fps=int(fps),
                    job_dir=outputs_dir,
                )
            elif "mova" in manifest_path:
                saved_path, media_type = save_video_mova(
                    video=video_part,
                    audio=audio_part,
                    filename_prefix=filename_prefix,
                    fps=int(fps),
                    job_dir=outputs_dir,
                )
            else:
                saved_path, media_type = save_video_ltx2(
                    video=video_part,
                    audio=audio_part,
                    filename_prefix=filename_prefix,
                    fps=int(fps),
                    job_dir=outputs_dir,
                )
        except Exception:
            import traceback
            traceback.print_exc()
            exit()
            # Fallback to generic saver if shape inference fails.
            saved_path, media_type = save_engine_output(
                output_obj=out_obj,
                job_dir=outputs_dir,
                filename_prefix=filename_prefix,
                final=True,
                fps=fps,
                audio_inputs=None,
                is_upscaler_engine=(kind == "upscalers"),
                input_video_for_audio_mux=input_video_for_audio_mux,
            )
    else:
        saved_path, media_type = save_engine_output(
            output_obj=out_obj[0],
            job_dir=outputs_dir,
            filename_prefix=filename_prefix,
            final=True,
            fps=fps,
            audio_inputs=None,
            is_upscaler_engine=(kind == "upscalers"),
            input_video_for_audio_mux=input_video_for_audio_mux,
        )
    dt = time.time() - t0

    return {
        "ok": True,
        "seconds": dt,
        "saved_path": saved_path,
        "media_type": media_type,
    }


def _build_selected_components(
    *,
    te_variant: Optional[str],
    xf_variant: Optional[str],
    te_components: Optional[Iterable[Tuple[str, Tuple[str, ...]]]] = None,
    xf_components: Optional[Iterable[Tuple[str, Tuple[str, ...]]]] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build the `selected_components` dict accepted by BaseEngine.
    Component-aware: only select a variant if that component declares it; otherwise
    fall back to that component's `default` (or first available entry).
    """
    selected: Dict[str, Any] = {}

    def _pick(desired: str, available: Tuple[str, ...]) -> str:
        if desired in available:
            return desired
        if "default" in available:
            return "default"
        return available[0] if available else "default"

    if te_variant is not None and te_components is not None:
        for key, variants in te_components:
            selected[key] = {"variant": _pick(te_variant, variants)}

    if xf_variant is not None and xf_components is not None:
        for key, variants in xf_components:
            selected[key] = {"variant": _pick(xf_variant, variants)}

    # Matrix metadata (safe: doesn't match any component keys)
    selected["_matrix"] = {
        "text_encoder": te_variant,
        "transformer": xf_variant,
        "extra": dict(extra or {}),
    }
    if extra:
        for selection_key, variant in extra.items():
            selected[selection_key] = {"variant": variant}

    return selected


@dataclass(frozen=True)
class JobSpec:
    phase: str
    kind: str
    test_json_path: Path
    manifest_path: Path
    model_id: str
    selected_components: Dict[str, Any]
    filename_prefix: str
    fps_hint: Optional[int]


def _run_jobs(
    *,
    script_path: Path,
    bundle_root: Path,
    suite_dir: Path,
    outputs_dir: Path,
    download_root: Path,
    jobs: List[JobSpec],
    runs_jsonl: Path,
    max_jobs: Optional[int],
    quiet: bool,
    progress_every: int,
    stream_worker_output: bool,
    skip_successes: bool,
    fail_fast: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    runs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for idx, job in enumerate(jobs):
        if max_jobs is not None and idx >= max_jobs:
            break

        if not quiet and (progress_every <= 1 or (idx % max(1, int(progress_every)) == 0)):
            # Compact “starting job” line. Avoid printing huge JSON blobs.
            try:
                print(
                    f"[{idx+1}/{min(len(jobs), max_jobs) if max_jobs else len(jobs)}] "
                    f"phase={job.phase} model={job.model_id} kind={job.kind} test={job.test_json_path.name}",
                    flush=True,
                )
            except Exception:
                pass

        # Worker writes a machine-readable result here so we can stream stdout/stderr
        # directly to the console without losing structured data.
        worker_results_dir = outputs_dir / "_worker_results"
        worker_results_dir.mkdir(parents=True, exist_ok=True)
        worker_result_path = worker_results_dir / f"{job.filename_prefix}.json"

        def _artifact_exists_for_prefix(prefix: str) -> bool:
            # Common outputs. (We also store the exact saved_path in worker JSON.)
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".wav"):
                if (outputs_dir / f"{prefix}{ext}").is_file():
                    return True
            return False

        def _already_ok() -> tuple[bool, Optional[dict]]:
            """
            A job is considered already successful if:
            - A prior `_worker_results/<prefix>.json` exists with ok=true, and
              - its saved_path exists, OR
              - an artifact with the expected prefix exists in outputs_dir.
            """
            try:
                if not worker_result_path.is_file():
                    return False, None
                obj = json.loads(worker_result_path.read_text(encoding="utf-8"))
                if not (isinstance(obj, dict) and obj.get("ok") is True):
                    return False, obj if isinstance(obj, dict) else None
                saved_path = obj.get("saved_path")
                if isinstance(saved_path, str) and Path(saved_path).is_file():
                    return True, obj
                if _artifact_exists_for_prefix(job.filename_prefix):
                    return True, obj
                # ok=true but no artifact found; treat as not ok so we can re-run.
                return False, obj
            except Exception:
                return False, None

        if skip_successes:
            is_ok, prior = _already_ok()
            if is_ok:
                rec: Dict[str, Any] = {
                    "phase": job.phase,
                    "kind": job.kind,
                    "model_id": job.model_id,
                    "test": str(job.test_json_path),
                    "manifest": str(job.manifest_path),
                    "selected_components": job.selected_components,
                    "filename_prefix": job.filename_prefix,
                    "returncode": None,
                    "wall_seconds": 0.0,
                    "worker_result_path": str(worker_result_path),
                    "worker": prior or {"ok": True},
                    "ok": True,
                    "skipped": True,
                    "skip_reason": "already_ok",
                }
                with runs_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                results.append(rec)
                if not quiet:
                    try:
                        saved_path = None
                        if isinstance(rec.get("worker"), dict):
                            saved_path = rec["worker"].get("saved_path")
                        extra = f" saved={Path(saved_path).name}" if isinstance(saved_path, str) else ""
                        print(f"  -> SKIP already_ok{extra}", flush=True)
                    except Exception:
                        pass
                continue

        # Shared per-model download/cache root (passed in). We intentionally set all
        # APEX_* paths (and HF caches) here so weights are reused across variants
        # within the model sweep, but can be cleaned up after the model completes.

        cmd = [
            sys.executable,
            "-u",  # unbuffered: stream denoising logs/progress promptly
            str(script_path),
            "--worker",
            "--bundle-root",
            str(bundle_root),
            "--suite-dir",
            str(suite_dir),
            "--outputs-dir",
            str(outputs_dir),
            "--kind",
            str(job.kind),
            "--json",
            str(job.test_json_path),
            "--manifest",
            str(job.manifest_path),
            "--phase",
            str(job.phase),
            "--filename-prefix",
            str(job.filename_prefix),
            "--selected-components-json",
            json.dumps(job.selected_components),
            "--result-json-path",
            str(worker_result_path),
        ]
        if job.fps_hint is not None:
            cmd.extend(["--fps-hint", str(int(job.fps_hint))])

        t0 = time.time()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        # Redirect all download/cache paths into the per-job download root.
        # This must be done via env vars because `src.utils.defaults` computes paths at import time.
        env["APEX_SAVE_PATH"] = str(download_root / "apex-save")
        env["APEX_COMPONENTS_PATH"] = str(download_root / "components")
        env["APEX_PREPROCESSOR_SAVE_PATH"] = str(download_root / "preprocessors")
        env["APEX_POSTPROCESSOR_SAVE_PATH"] = str(download_root / "postprocessors")
        env["APEX_CONFIG_SAVE_PATH"] = str(download_root / "configs")
        env["APEX_CACHE_PATH"] = str(download_root / "cache")
        env["APEX_LORA_SAVE_PATH"] = str(download_root / "loras")
        env["APEX_OFFLOAD_PATH"] = str(download_root / "offload")
        env["APEX_TORCH_COMPILE_PATH"] = str(download_root / "torch_compile")
        # HuggingFace caches (Hub + Transformers)
        env["APEX_HF_HOME"] = str(download_root / ".cache" / "huggingface")
        env["HF_HOME"] = env["APEX_HF_HOME"]
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = env["HF_HUB_CACHE"]
        env["TRANSFORMERS_CACHE"] = str(Path(env["HF_HOME"]) / "transformers")

        if stream_worker_output and not quiet:
            # Stream everything (denoising logs, tqdm, etc.) directly to this process' stdout/stderr.
            proc = subprocess.run(cmd, text=True, env=env)
            stdout = ""
            stderr = ""
        else:
            # Capture output (quiet / CI-style).
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        dt = time.time() - t0

        rec: Dict[str, Any] = {
            "phase": job.phase,
            "kind": job.kind,
            "model_id": job.model_id,
            "test": str(job.test_json_path),
            "manifest": str(job.manifest_path),
            "selected_components": job.selected_components,
            "filename_prefix": job.filename_prefix,
            "returncode": proc.returncode,
            "wall_seconds": dt,
            "worker_result_path": str(worker_result_path),
        }

        # Prefer the structured result file (works even when streaming worker output).
        worker_obj = None
        try:
            if worker_result_path.is_file():
                worker_obj = json.loads(worker_result_path.read_text(encoding="utf-8"))
        except Exception:
            worker_obj = None
        if isinstance(worker_obj, dict):
            rec["worker"] = worker_obj
        else:
            # Fallback to parsing captured stdout if available.
            if stdout.strip():
                try:
                    rec["worker"] = json.loads(stdout.strip().splitlines()[-1])
                except Exception:
                    rec["worker_stdout"] = stdout[-20000:]
            if stderr.strip():
                rec["worker_stderr"] = stderr[-20000:]

        rec["ok"] = bool(rec.get("worker", {}).get("ok")) if proc.returncode == 0 else False

        with runs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        results.append(rec)

        # Always send failures to the terminal (stderr), even in --quiet / captured mode.
        if not rec.get("ok"):
            try:
                _eprint(
                    "[variant-matrix] JOB FAILED "
                    f"phase={job.phase} model={job.model_id} kind={job.kind} "
                    f"test={job.test_json_path.name} prefix={job.filename_prefix}"
                )
                _eprint(
                    f"[variant-matrix] returncode={proc.returncode} "
                    f"worker_result={worker_result_path}"
                )

                worker = rec.get("worker")
                if isinstance(worker, dict):
                    werr = worker.get("error")
                    wtb = worker.get("traceback")
                    if isinstance(werr, str) and werr.strip():
                        _eprint(f"[variant-matrix] worker_error: {werr}")
                    if isinstance(wtb, str) and wtb.strip():
                        _eprint("[variant-matrix] worker_traceback:")
                        _eprint(_tail(wtb).rstrip())

                # If we captured stderr/stdout, include the tails (often where Python/CUDA errors land).
                if isinstance(stderr, str) and stderr.strip():
                    _eprint("[variant-matrix] worker_stderr (tail):")
                    _eprint(_tail(stderr).rstrip())
                if isinstance(stdout, str) and stdout.strip():
                    _eprint("[variant-matrix] worker_stdout (tail):")
                    _eprint(_tail(stdout).rstrip())
            except Exception:
                pass

        if not quiet:
            try:
                saved_path = None
                if isinstance(rec.get("worker"), dict):
                    saved_path = rec["worker"].get("saved_path")
                status = "OK" if rec.get("ok") else "FAIL"
                extra = f" saved={Path(saved_path).name}" if isinstance(saved_path, str) else ""
                print(
                    f"  -> {status} rc={rec.get('returncode')} wall={rec.get('wall_seconds'):.1f}s{extra}",
                    flush=True,
                )
            except Exception:
                pass

        # Fail-fast: stop entire matrix on first failure, and KEEP downloads
        # for the failing job so you can inspect/debug.
        if fail_fast and not rec.get("ok"):
            if not quiet:
                try:
                    print(
                        "[variant-matrix] FAIL-FAST: job failed; keeping downloads for debugging.",
                        flush=True,
                    )
                    print(f"[variant-matrix] downloads_dir={download_root}", flush=True)
                    print(f"[variant-matrix] worker_result={worker_result_path}", flush=True)
                    print(
                        "[variant-matrix] Tip: re-run with the same --run-id and "
                        "--skip-successes to resume from the first failure.",
                        flush=True,
                    )
                except Exception:
                    pass
            raise FailFastError(
                message="job_failed",
                download_root=download_root,
                worker_result_path=worker_result_path,
                record=rec,
                results_so_far=list(results),
            )

    return results


def _summarize_phase(
    results: List[Dict[str, Any]], *, phase: str
) -> Dict[str, Any]:
    per_model: Dict[str, Any] = {}
    for r in results:
        if r.get("phase") != phase:
            continue
        model_id = str(r.get("model_id") or "unknown")
        sel = r.get("selected_components") or {}
        per_model.setdefault(model_id, {"runs": 0, "ok": 0, "fail": 0, "by_variant": {}})
        per_model[model_id]["runs"] += 1
        if r.get("ok"):
            per_model[model_id]["ok"] += 1
        else:
            per_model[model_id]["fail"] += 1

        # Try to infer the variant under test for this phase.
        variant = None
        matrix = sel.get("_matrix") if isinstance(sel, dict) else None
        if phase == "text_encoder":
            if isinstance(matrix, dict):
                variant = matrix.get("text_encoder")
            if not variant:
                v = sel.get("text_encoder")
                if isinstance(v, dict):
                    variant = v.get("variant")
        elif phase == "transformer":
            if isinstance(matrix, dict):
                variant = matrix.get("transformer")
            if not variant:
                v = sel.get("transformer")
                if isinstance(v, dict):
                    variant = v.get("variant")
        elif phase.startswith("extra:"):
            # phase is like "extra:lynx"
            key = phase.split(":", 1)[1]
            if isinstance(matrix, dict):
                extra_map = matrix.get("extra")
                if isinstance(extra_map, dict):
                    variant = extra_map.get(key)
            if not variant:
                v = sel.get(key)
                if isinstance(v, dict):
                    variant = v.get("variant")

        if not isinstance(variant, str) or not variant:
            variant = "unknown"

        byv = per_model[model_id]["by_variant"].setdefault(variant, {"ok": 0, "fail": 0})
        if r.get("ok"):
            byv["ok"] += 1
        else:
            byv["fail"] += 1

    return {"phase": phase, "per_model": per_model}


def main() -> int:
    p = argparse.ArgumentParser(description="Run Apex test-suite variant matrix.")
    p.add_argument("--bundle-root", default=None, help="Path containing src/ and manifest/.")
    p.add_argument("--suite-dir", default=None, help="Path to apex-test-suite/test_suite dir.")
    p.add_argument("--manifest-dir", default=None, help="Path to manifest version dir (e.g. manifest/v0.1.2).")
    p.add_argument("--kind", default="all", choices=["all", "image", "video", "audio", "upscalers"])
    p.add_argument("--filter", default="", help="Substring filter applied to test JSON filename.")
    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated list of substrings to skip (matches against model id and test JSON filename).",
    )
    p.add_argument("--max-jobs", type=int, default=None, help="Limit number of jobs for quick smoke runs.")
    p.add_argument("--run-id", default=None, help="Override run id used for outputs dir naming.")
    p.add_argument("--no-transformer-sweep", action="store_true", help="Skip transformer variant sweep.")
    p.add_argument("--no-text-encoder-sweep", action="store_true", help="Skip text-encoder variant sweep.")
    p.add_argument("--no-extra-sweep", action="store_true", help="Skip extra_model_path variant sweeps (e.g. Lynx adapters).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-job progress printing.")
    p.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a 'starting job' line every N jobs (default: 1).",
    )
    p.add_argument(
        "--stream-worker-output",
        dest="stream_worker_output",
        action="store_true",
        default=True,
        help="Stream worker stdout/stderr live (default: enabled).",
    )
    p.add_argument(
        "--no-stream-worker-output",
        dest="stream_worker_output",
        action="store_false",
        help="Do not stream worker output (capture only).",
    )
    p.add_argument(
        "--keep-downloads",
        dest="cleanup_downloads",
        action="store_false",
        default=True,
        help="Keep downloaded files (debugging only). Default is to delete downloads after each full model sweep.",
    )
    p.add_argument(
        "--skip-successes",
        "--only-failed",
        dest="skip_successes",
        action="store_true",
        default=False,
        help="Skip jobs that already have ok=true in outputs_dir/_worker_results (resume mode).",
    )
    p.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        default=True,
        help="Exit immediately on first failed job (default: enabled).",
    )
    p.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help="Do not exit immediately on failures (keep going).",
    )

    # Worker args (internal)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", dest="json_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--manifest", dest="manifest_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--selected-components-json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--outputs-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--phase", default="", help=argparse.SUPPRESS)
    p.add_argument("--filename-prefix", default="result", help=argparse.SUPPRESS)
    p.add_argument("--fps-hint", default=None, type=int, help=argparse.SUPPRESS)
    p.add_argument("--result-json-path", default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

    script_path = Path(__file__).resolve()
    bundle_root = Path(args.bundle_root).resolve() if args.bundle_root else _default_bundle_root()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else _default_suite_dir(bundle_root)
    manifest_dir = (
        Path(args.manifest_dir).resolve() if args.manifest_dir else _default_manifest_dir(bundle_root)
    )

    if args.worker:
        # ------------------------------ Worker mode ------------------------------ #
        if not args.json_path or not args.manifest_path or not args.outputs_dir:
            _eprint("[variant-matrix][worker] missing required args")
            print(json.dumps({"ok": False, "error": "worker_missing_args"}))
            return 2
        try:
            selected_components = (
                json.loads(args.selected_components_json) if args.selected_components_json else {}
            )
            if not isinstance(selected_components, dict):
                selected_components = {}
            test_json_path = Path(args.json_path).resolve()
            manifest_path = Path(args.manifest_path).resolve()
            outputs_dir = Path(args.outputs_dir).resolve()
            kind = str(args.kind)
            res = _worker_run_one(
                bundle_root=bundle_root,
                suite_dir=suite_dir,
                outputs_dir=outputs_dir,
                test_json_path=test_json_path,
                manifest_path=manifest_path,
                selected_components=selected_components,
                filename_prefix=str(args.filename_prefix),
                kind=kind,
                fps_hint=args.fps_hint,
            )
            # Always persist a structured result if requested.
            if args.result_json_path:
                try:
                    rp = Path(args.result_json_path)
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    rp.write_text(json.dumps(res, indent=2), encoding="utf-8")
                except Exception:
                    pass
            return 0
        except Exception as e:
            # Keep stdout as JSON (caller expects to parse last line).
            err = {"ok": False, "error": str(e), "traceback": ""}
            try:
                import traceback

                err["traceback"] = traceback.format_exc()
            except Exception:
                pass
            # Always show the error in the terminal as well (stderr), not only in JSON files.
            try:
                _eprint("[variant-matrix][worker] ERROR:", str(e))
                if isinstance(err.get("traceback"), str) and err["traceback"].strip():
                    _eprint(err["traceback"].rstrip())
            except Exception:
                pass
            if args.result_json_path:
                try:
                    rp = Path(args.result_json_path)
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    rp.write_text(json.dumps(err, indent=2), encoding="utf-8")
                except Exception:
                    pass
            return 1

    # ---------------------------- Orchestrator mode ---------------------------- #
    if not suite_dir.exists():
        raise SystemExit(f"suite_dir not found: {suite_dir}")
    if not manifest_dir.exists():
        raise SystemExit(f"manifest_dir not found: {manifest_dir}")

    tests = _iter_test_jsons(suite_dir, kind=args.kind)
    if args.filter:
        filt = str(args.filter).lower()
        tests = [p for p in tests if filt in p.name.lower()]
    skip_tokens = [t.strip().lower() for t in str(args.skip or "").split(",") if t.strip()]
    if skip_tokens:
        def _skip_match(s: str) -> bool:
            sl = (s or "").lower()
            return any(tok in sl for tok in skip_tokens)

        tests = [p for p in tests if not (_skip_match(p.name) or _skip_match(p.stem))]

    # Build manifest → tests mapping and pre-extract variants.
    manifest_infos: Dict[str, Tuple[ManifestVariants, List[Path]]] = {}
    missing_manifests: List[str] = []
    for test_json in tests:
        kind = test_json.parent.name
        manifest_path = _resolve_manifest_for_test(
            test_json_path=test_json,
            kind=kind,
            manifest_dir=manifest_dir,
        )
        if manifest_path is None:
            missing_manifests.append(str(test_json))
            continue
        key = str(manifest_path.resolve())
        if key not in manifest_infos:
            manifest_infos[key] = (_extract_variants_from_manifest(manifest_path, kind=kind), [])
        manifest_infos[key][1].append(test_json)

    run_id = str(args.run_id or os.environ.get("APEX_VARIANT_RUN_ID") or _now_run_id())
    out_root = suite_dir / "outputs_variants" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    runs_jsonl = out_root / "runs.jsonl"
    summary_json = out_root / "summary.json"

    if not args.quiet:
        print(
            f"[variant-matrix] run_id={run_id} outputs_dir={out_root}",
            flush=True,
        )
        if missing_manifests:
            print(
                f"[variant-matrix] warning: {len(missing_manifests)} tests missing manifests (will be skipped)",
                flush=True,
            )

    # Run the variant pipeline *per model* so each model reaches transformer/extra sweeps
    # even if later models fail or the run is interrupted.
    all_results: List[Dict[str, Any]] = []
    te_choice: Dict[str, str] = {}
    xf_choice: Dict[str, str] = {}

    def _write_summary(*, status: str, failed: Optional[Dict[str, Any]] = None) -> None:
        summary: Dict[str, Any] = {
            "status": status,
            "run_id": run_id,
            "bundle_root": str(bundle_root),
            "suite_dir": str(suite_dir),
            "manifest_dir": str(manifest_dir),
            "outputs_dir": str(out_root),
            "missing_manifests": missing_manifests,
            "choices": {
                "text_encoder": te_choice,
                "transformer": xf_choice,
            },
            "failed": failed,
            "phases": {},
            "totals": {
                "runs": len(all_results),
                "ok": sum(1 for r in all_results if r.get("ok")),
                "fail": sum(1 for r in all_results if not r.get("ok")),
                "skipped": sum(1 for r in all_results if r.get("skipped")),
            },
        }

        summary["phases"]["text_encoder"] = _summarize_phase(
            all_results, phase="text_encoder"
        )
        summary["phases"]["transformer"] = _summarize_phase(
            all_results, phase="transformer"
        )
        extra_phases = sorted(
            {
                r.get("phase")
                for r in all_results
                if str(r.get("phase", "")).startswith("extra:")
            }
        )
        for ph in extra_phases:
            summary["phases"][ph] = _summarize_phase(all_results, phase=str(ph))

        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        for info, test_list in manifest_infos.values():
            mid = info.model_id
            if skip_tokens and any(tok in str(mid).lower() for tok in skip_tokens):
                if not args.quiet:
                    try:
                        print(
                            f"[variant-matrix] SKIP model={mid} (matched --skip)",
                            flush=True,
                        )
                    except Exception:
                        pass
                continue

            model_download_root = out_root / "_downloads" / _slug(mid)
            if not args.quiet:
                try:
                    print(
                        f"[variant-matrix] model={mid} tests={len(test_list)} "
                        f"te_variants={len(info.text_encoder_variants)} "
                        f"xf_variants={len(info.transformer_variants)} "
                        f"extra_components={len(info.extra_components)}",
                        flush=True,
                    )
                    print(
                        f"[variant-matrix] model_downloads_dir={model_download_root}",
                        flush=True,
                    )
                except Exception:
                    pass

            # ---- Text encoder sweep for this model ----
            chosen_te = te_choice.get(mid, "default")
            if not args.no_text_encoder_sweep:
                te_jobs: List[JobSpec] = []
                for te_variant in info.text_encoder_variants:
                    for test_json in test_list:
                        stem = test_json.stem
                        filename_prefix = _slug(f"{stem}__te-{te_variant}")
                        selected = _build_selected_components(
                            te_variant=te_variant,
                            xf_variant=None,
                            te_components=info.text_encoder_components,
                            xf_components=None,
                        )
                        te_jobs.append(
                            JobSpec(
                                phase="text_encoder",
                                kind=info.kind,
                                test_json_path=test_json,
                                manifest_path=info.manifest_path,
                                model_id=mid,
                                selected_components=selected,
                                filename_prefix=filename_prefix,
                                fps_hint=info.fps,
                            )
                        )

                te_results = _run_jobs(
                    script_path=script_path,
                    bundle_root=bundle_root,
                    suite_dir=suite_dir,
                    outputs_dir=out_root,
                    download_root=model_download_root,
                    jobs=te_jobs,
                    runs_jsonl=runs_jsonl,
                    max_jobs=args.max_jobs,
                    quiet=bool(args.quiet),
                    progress_every=int(args.progress_every or 1),
                    stream_worker_output=bool(args.stream_worker_output),
                    skip_successes=bool(getattr(args, "skip_successes", False)),
                    fail_fast=bool(getattr(args, "fail_fast", True)),
                )
                all_results.extend(te_results)

                te_stats: Dict[str, Dict[str, int]] = {}
                for r in te_results:
                    sel = r.get("selected_components") or {}
                    matrix = sel.get("_matrix") if isinstance(sel, dict) else None
                    variant = (
                        matrix.get("text_encoder")
                        if isinstance(matrix, dict)
                        else None
                    )
                    if not isinstance(variant, str) or not variant:
                        variant = "unknown"
                    te_stats.setdefault(variant, {"ok": 0, "fail": 0})
                    if r.get("ok"):
                        te_stats[variant]["ok"] += 1
                    else:
                        te_stats[variant]["fail"] += 1

                chosen_te = _pick_best_variant(te_stats) if te_stats else "default"
                te_choice[mid] = chosen_te

            # ---- Transformer sweep for this model (using chosen TE) ----
            chosen_xf = xf_choice.get(mid, "default")
            if not args.no_transformer_sweep:
                xf_jobs: List[JobSpec] = []
                # We already exercised the default transformer during the text-encoder sweep,
                # so avoid doing the redundant (TE chosen, XF default) job again.
                xf_variants = list(info.transformer_variants)
                if not args.no_text_encoder_sweep:
                    xf_variants = [v for v in xf_variants if v != "default"]

                for xf_variant in xf_variants:
                    for test_json in test_list:
                        stem = test_json.stem
                        filename_prefix = _slug(
                            f"{stem}__te-{chosen_te}__xf-{xf_variant}"
                        )
                        selected = _build_selected_components(
                            te_variant=chosen_te,
                            xf_variant=xf_variant,
                            te_components=info.text_encoder_components,
                            xf_components=info.transformer_components,
                        )
                        xf_jobs.append(
                            JobSpec(
                                phase="transformer",
                                kind=info.kind,
                                test_json_path=test_json,
                                manifest_path=info.manifest_path,
                                model_id=mid,
                                selected_components=selected,
                                filename_prefix=filename_prefix,
                                fps_hint=info.fps,
                            )
                        )

                xf_results = _run_jobs(
                    script_path=script_path,
                    bundle_root=bundle_root,
                    suite_dir=suite_dir,
                    outputs_dir=out_root,
                    download_root=model_download_root,
                    jobs=xf_jobs,
                    runs_jsonl=runs_jsonl,
                    max_jobs=args.max_jobs,
                    quiet=bool(args.quiet),
                    progress_every=int(args.progress_every or 1),
                    stream_worker_output=bool(args.stream_worker_output),
                    skip_successes=bool(getattr(args, "skip_successes", False)),
                    fail_fast=bool(getattr(args, "fail_fast", True)),
                )
                all_results.extend(xf_results)

                xf_stats: Dict[str, Dict[str, int]] = {}
                for r in xf_results:
                    sel = r.get("selected_components") or {}
                    matrix = sel.get("_matrix") if isinstance(sel, dict) else None
                    variant = (
                        matrix.get("transformer")
                        if isinstance(matrix, dict)
                        else None
                    )
                    if not isinstance(variant, str) or not variant:
                        variant = "unknown"
                    xf_stats.setdefault(variant, {"ok": 0, "fail": 0})
                    if r.get("ok"):
                        xf_stats[variant]["ok"] += 1
                    else:
                        xf_stats[variant]["fail"] += 1

                chosen_xf = _pick_best_variant(xf_stats) if xf_stats else "default"
                xf_choice[mid] = chosen_xf

            # ---- Extra components sweep (e.g. Lynx adapters) ----
            if not args.no_extra_sweep and info.extra_components:
                extra_jobs: List[JobSpec] = []
                for (selection_key, variants) in info.extra_components:
                    for extra_variant in variants:
                        for test_json in test_list:
                            stem = test_json.stem
                            filename_prefix = _slug(
                                f"{stem}__te-{chosen_te}__xf-{chosen_xf}__{selection_key}-{extra_variant}"
                            )
                            selected = _build_selected_components(
                                te_variant=chosen_te,
                                xf_variant=chosen_xf,
                                te_components=info.text_encoder_components,
                                xf_components=info.transformer_components,
                                extra={selection_key: extra_variant},
                            )
                            extra_jobs.append(
                                JobSpec(
                                    phase=f"extra:{selection_key}",
                                    kind=info.kind,
                                    test_json_path=test_json,
                                    manifest_path=info.manifest_path,
                                    model_id=mid,
                                    selected_components=selected,
                                    filename_prefix=filename_prefix,
                                    fps_hint=info.fps,
                                )
                            )

                extra_results = _run_jobs(
                    script_path=script_path,
                    bundle_root=bundle_root,
                    suite_dir=suite_dir,
                    outputs_dir=out_root,
                    download_root=model_download_root,
                    jobs=extra_jobs,
                    runs_jsonl=runs_jsonl,
                    max_jobs=args.max_jobs,
                    quiet=bool(args.quiet),
                    progress_every=int(args.progress_every or 1),
                    stream_worker_output=bool(args.stream_worker_output),
                    skip_successes=bool(getattr(args, "skip_successes", False)),
                    fail_fast=bool(getattr(args, "fail_fast", True)),
                )
                all_results.extend(extra_results)

            # Cleanup after the *full model sweep* (only if enabled and nothing failed).
            if bool(getattr(args, "cleanup_downloads", True)):
                try:
                    model_failed = False
                    for r in all_results:
                        if str(r.get("model_id")) != str(mid):
                            continue
                        if r.get("skipped"):
                            continue
                        if not r.get("ok"):
                            model_failed = True
                            break
                    if not model_failed:
                        shutil.rmtree(model_download_root, ignore_errors=True)
                except Exception:
                    pass
    except FailFastError as e:
        # Incorporate the partial results from the failing _run_jobs call so summary is accurate.
        try:
            all_results.extend(e.results_so_far)
        except Exception:
            pass

        failed_info = {
            "reason": e.message,
            "downloads_dir": str(e.download_root),
            "worker_result": str(e.worker_result_path),
            "record": e.record,
        }
        _write_summary(status="failed", failed=failed_info)
        print(
            json.dumps(
                {
                    "ok": False,
                    "outputs_dir": str(out_root),
                    "summary": str(summary_json),
                    "downloads_dir": str(e.download_root),
                }
            ),
            flush=True,
        )
        return 1

    _write_summary(status="ok")

    # Print a compact pointer for humans.
    print(json.dumps({"ok": True, "outputs_dir": str(out_root), "summary": str(summary_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

