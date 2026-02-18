#!/usr/bin/env python3
"""
Publish transformer weights (BF16 + FP8 + GGUF) to Hugging Face.

Goal
----
For every model manifest under `manifest/0.1.2/`, locate transformer components and ensure we
have the following artifacts staged for upload to `totoku/apex-models`:
  - BF16 safetensors (source weights)
  - FP8(E4M3FN) safetensors (+ per-weight `.scale_weight` tensors)
  - GGUF: Q8_0, Q6_K, Q4_K_M

This script is intentionally manifest-driven so the upload layout can follow the
manifest's intended `config_path` / `model_path` locations when those already point at
`totoku/apex-models`.

Notes / trade-offs
------------------
- FP8 generation uses `safetensors.torch.load_file()` which materializes a full safetensors
  shard in RAM. For very large transformers, prefer providing a sharded BF16 directory so
  we can FP8-quantize shard-by-shard.
- GGUF generation is memory-aware via `src/quantize/transformer.py` (lazy safetensors
  loading when RAM is insufficient).

Auth
----
Set `HF_TOKEN` (recommended) or use `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger


def _api_root() -> Path:
    # .../apps/api/scripts/models/<this_file.py> -> parents[2] == .../apps/api
    return Path(__file__).resolve().parents[2]


def _default_manifest_root() -> Path:
    return _api_root() / "manifest" / "v0.1.2"


def _default_staging_root() -> Path:
    return _api_root() / "weights" / "_publish" / "transformers"


def _iter_manifest_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for ext in ("*.yml", "*.yaml"):
        out.extend(sorted(root.rglob(ext)))
    return sorted(set(out))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a mapping: {path}")
    return data


def _is_hf_path(s: str) -> bool:
    # Heuristic: "org/repo/..." (at least 3 segments) and not an absolute path.
    if not s or os.path.isabs(s):
        return False
    parts = s.split("/")
    return len(parts) >= 3 and all(parts[:2])


@dataclass(frozen=True)
class HfLocation:
    repo_id: str
    path_in_repo: str  # file or directory


def _parse_hf_location(s: str) -> HfLocation:
    parts = s.split("/")
    if len(parts) < 3:
        raise ValueError(f"Not a Hub path: {s}")
    return HfLocation(repo_id="/".join(parts[:2]), path_in_repo="/".join(parts[2:]))


def _download_hf_path(loc: HfLocation, *, revision: str | None, cache_dir: str | None) -> Path:
    """
    Download a Hub file or directory and return the local path to the file/dir.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n\n"
            "Install it in this environment (pick one):\n"
            "  - pip install huggingface_hub\n"
            "  - pip install -r requirements/requirements.txt\n"
        ) from e

    # Directory-style path (no obvious file extension) → snapshot_download.
    is_probably_file = any(
        loc.path_in_repo.lower().endswith(suffix)
        for suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".json", ".txt", ".md")
    )
    if is_probably_file:
        # Avoid re-downloading if already cached locally.
        try:
            p = hf_hub_download(
                repo_id=loc.repo_id,
                filename=loc.path_in_repo,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            p = hf_hub_download(
                repo_id=loc.repo_id,
                filename=loc.path_in_repo,
                revision=revision,
                cache_dir=cache_dir,
            )
        return Path(p)

    # Snapshot download (folder). Try cache-only first.
    #
    # NOTE: allow_patterns handling can be finicky across huggingface_hub versions, especially
    # for patterns like "dir/**". Include both "dir/*" and "dir/**" so directory downloads
    # work reliably for cases like "transformer_2/".
    allow_patterns = [
        f"{loc.path_in_repo}",
        f"{loc.path_in_repo}/*",
        f"{loc.path_in_repo}/**",
    ]
    try:
        root = snapshot_download(
            repo_id=loc.repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
    except Exception:
        root = snapshot_download(
            repo_id=loc.repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
    out = Path(root) / loc.path_in_repo
    if not out.exists():
        # Fallback: the snapshot folder may already exist from a previous *filtered* download
        # (different allow_patterns), and huggingface_hub can return the cached snapshot dir
        # without fetching new matching files. Force a download for this prefix.
        root = snapshot_download(
            repo_id=loc.repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            force_download=True,
        )
        out = Path(root) / loc.path_in_repo
        if not out.exists():
            raise FileNotFoundError(
                "Downloaded snapshot does not contain expected path. "
                f"repo_id={loc.repo_id!r} path_in_repo={loc.path_in_repo!r} "
                f"revision={revision!r} resolved_root={str(root)!r}"
            )
    return out


def _copy_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        # python < 3.8
        try:
            if path.exists():
                path.unlink()
        except FileNotFoundError:
            pass


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        _safe_unlink(path)
        return
    shutil.rmtree(path, ignore_errors=True)


def _cleanup_empty_parents(path: Path, *, stop_at: Path) -> None:
    """
    Remove empty directories up the tree, stopping at `stop_at` (inclusive).
    """
    cur = path
    stop_at = stop_at.resolve()
    while True:
        if cur == stop_at:
            return
        parent = cur.parent
        try:
            if cur.exists() and cur.is_dir() and not any(cur.iterdir()):
                cur.rmdir()
        except OSError:
            return
        cur = parent


def _variant_source(component: dict[str, Any], variant: str) -> dict[str, Any] | None:
    paths = component.get("model_path") or []
    if not isinstance(paths, list):
        return None
    for entry in paths:
        if isinstance(entry, dict) and entry.get("variant") == variant:
            return entry
    return None


def _pick_bf16_source(component: dict[str, Any]) -> dict[str, Any] | None:
    paths = component.get("model_path") or []
    if not isinstance(paths, list):
        return None

    # Prefer explicit default.
    for entry in paths:
        if not isinstance(entry, dict):
            continue
        if entry.get("variant") == "default" and entry.get("type") == "safetensors":
            return entry

    # Fall back to bf16/fp16 safetensors.
    def _score(entry: dict[str, Any]) -> int:
        prec = str(entry.get("precision") or "").lower()
        if "bf16" in prec:
            return 0
        if "fp16" in prec or "f16" in prec:
            return 1
        return 10

    candidates = [e for e in paths if isinstance(e, dict) and e.get("type") == "safetensors"]
    if not candidates:
        return None
    return sorted(candidates, key=_score)[0]


def _base_family(base: str) -> str:
    return base.split(".", 1)[0].strip()


def _arch_from_transformer_base(base: str) -> str:
    """
    Map a manifest converter `base` value (e.g. 'cogvideox.base') to a GGUF architecture id.
    """
    base_lc = (base or "").strip().lower()
    # Sub-family overrides (the base family alone is not specific enough).
    if base_lc == "hunyuanvideo.foley":
        return "foley"

    family = _base_family(base).lower()
    family_map = {
        "cogvideox": "cogvideo",
    }
    return family_map.get(family, family)


BF16_FILENAME = "transformer-bf16.safetensors"
FP8_FILENAME = "transformer-fp8_e4m3fn.safetensors"
GGUF_Q8_0_FILENAME = "transformer-q8_0.gguf"
GGUF_Q6_K_FILENAME = "transformer-q6_k.gguf"
GGUF_Q4_K_M_FILENAME = "transformer-q4_k_m.gguf"


def _slugify_dir_name(s: str) -> str:
    """
    Create a stable, filesystem-friendly directory name from an arbitrary string.
    """
    s = (s or "").strip().lower()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s


def _dest_transformer_dir(
    *,
    manifest: dict[str, Any],
    transformer_component: dict[str, Any],
    target_repo_id: str,
) -> Path:
    """
    Decide the relative destination folder inside `target_repo_id` for this transformer's files.
    """
    meta = manifest.get("metadata") or {}
    meta_id = str(meta.get("id") or "unknown-model")

    # Best-effort: use config_path if it already targets our repo.
    config_path = transformer_component.get("config_path")
    if isinstance(config_path, str) and config_path.startswith(f"{target_repo_id}/"):
        rel = config_path[len(target_repo_id) + 1 :]
        # Usually: <ModelFolder>/transformer/config.json
        return Path(rel).parent

    # Next: if the BF16 weight is already targeting our repo, place alongside it.
    src = _pick_bf16_source(transformer_component)
    if src and isinstance(src.get("path"), str):
        p = str(src["path"])
        if p.startswith(f"{target_repo_id}/"):
            relp = p[len(target_repo_id) + 1 :]
            return Path(relp).parent

    # Fallback: stable-but-generic layout.
    return Path("UNSORTED") / meta_id / "transformer"


def _fp8_should_quantize_linear_weight(
    key: str, value, *, exclude_substrings_lc: tuple[str, ...]
) -> bool:
    if not key.endswith(".weight"):
        return False
    if not hasattr(value, "ndim") or value.ndim != 2:
        return False
    if hasattr(value, "is_floating_point") and not value.is_floating_point():
        return False
    k = key.lower()
    return not any(substr in k for substr in exclude_substrings_lc)


def _create_fp8_e4m3fn_file(
    *,
    bf16_path: Path,
    out_path: Path,
    exclude_substrings: Iterable[str],
) -> None:
    from safetensors.torch import load_file, save_file
    import torch
    from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval

    state_dict = load_file(str(bf16_path))
    new_state_dict: dict[str, torch.Tensor] = {}
    exclude_lc = tuple(s.lower() for s in exclude_substrings)

    for key, value in state_dict.items():
        if _fp8_should_quantize_linear_weight(key, value, exclude_substrings_lc=exclude_lc):
            maxval = get_fp_maxval()
            scale = torch.max(torch.abs(value.flatten())) / maxval
            linear_weight, scale, _log_scales = fp8_tensor_quant(value, scale)
            linear_weight = linear_weight.to(torch.float8_e4m3fn)
            new_state_dict[key] = linear_weight
            new_state_dict[key.replace(".weight", ".scale_weight")] = scale
        else:
            new_state_dict[key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(out_path))


def _save_safetensors_cast_floating_to_bf16(*, state, out_path: Path) -> None:
    """
    Save `state` to safetensors while ensuring all floating tensors are BF16.

    This is important because some upstream safetensors can contain FP32 tensors even when the
    checkpoint is advertised as BF16/FP16, and downstream tooling assumes the staged
    `transformer-bf16.safetensors` is truly BF16.
    """
    from safetensors.torch import save_file
    import torch

    new_state: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if hasattr(v, "is_floating_point") and v.is_floating_point() and v.dtype != torch.bfloat16:
            new_state[k] = v.to(torch.bfloat16)
        else:
            new_state[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        save_file(new_state, str(tmp_path))
        tmp_path.replace(out_path)
    finally:
        _safe_unlink(tmp_path)


def _consolidate_safetensors_shards_to_single_file(shards_dir: Path, out_path: Path) -> None:
    """
    Consolidate multiple safetensors shards into a single safetensors file.

    WARNING: This materializes the full state dict in RAM.
    """
    from safetensors.torch import safe_open
    import torch

    shards = sorted(shards_dir.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found under {shards_dir}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                # Enforce BF16 for the staged "transformer-bf16.safetensors".
                if t.is_floating_point() and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                state[k] = t

    _save_safetensors_cast_floating_to_bf16(state=state, out_path=out_path)


def _cast_safetensors_file_floating_to_bf16(src_path: Path, out_path: Path) -> None:
    """
    Load a single safetensors file and re-save with BF16 floating tensors.

    WARNING: This materializes the full state dict in RAM.
    """
    from safetensors.torch import load_file

    state = load_file(str(src_path))
    _save_safetensors_cast_floating_to_bf16(state=state, out_path=out_path)


def _stage_bf16_as_single_file(src_local: Path, transformer_out_dir: Path) -> Path:
    """
    Ensure BF16 weights are staged as a single file named `transformer-bf16.safetensors`.
    """
    dst = transformer_out_dir / BF16_FILENAME
    if dst.exists():
        return dst

    if src_local.is_dir():
        logger.warning(
            "BF16 source is a directory; consolidating shards into a single file. "
            "This can require a lot of RAM."
        )
        _consolidate_safetensors_shards_to_single_file(src_local, dst)
        return dst

    # If the source is a single safetensors file, re-save it so all floating tensors are BF16.
    # This ensures the staged `transformer-bf16.safetensors` is truly BF16, not FP32.
    if src_local.suffix.lower() == ".safetensors":
        logger.info("Ensuring staged BF16 weights are BF16 (casting floating tensors if needed)")
        _cast_safetensors_file_floating_to_bf16(src_local, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_local, dst)
    return dst


def _ensure_fp8_e4m3fn(
    *,
    bf16_src: Path,
    transformer_out_dir: Path,
    exclude_substrings: Iterable[str],
) -> list[Path]:
    """
    Returns paths created/ensured (file or directory contents).
    """
    out_file = transformer_out_dir / FP8_FILENAME
    if out_file.exists():
        return [out_file]

    logger.info("FP8 quantizing transformer")
    _create_fp8_e4m3fn_file(
        bf16_path=bf16_src,
        out_path=out_file,
        exclude_substrings=exclude_substrings,
    )
    return [out_file]


def _ensure_gguf(
    *,
    bf16_src: Path,
    transformer_out_dir: Path,
    architecture: str,
) -> list[Path]:
    from src.quantize import TransformerQuantizer
    from src.quantize.quants import QuantType
    from src.quantize.transformer import ModelArchitecture

    arch_enum = ModelArchitecture(architecture)
    created: list[Path] = []

    def _one(qt: QuantType, filename: str) -> Path:
        out_path = transformer_out_dir / filename
        out_prefix = str(out_path.with_suffix(""))  # TransformerQuantizer appends ".gguf"
        if out_path.exists():
            return out_path

        q = TransformerQuantizer(
            output_path=out_prefix,
            model_path=str(bf16_src),
            quantization=qt,
            architecture=arch_enum,
        )
        q.quantize()
        if not out_path.exists():
            raise FileNotFoundError(f"Expected GGUF output at {out_path}")
        created.append(out_path)
        return out_path

    _one(QuantType.Q8_0, GGUF_Q8_0_FILENAME)
    _one(QuantType.Q6_K, GGUF_Q6_K_FILENAME)
    _one(QuantType.Q4_K_M, GGUF_Q4_K_M_FILENAME)

    return created


def _ensure_hf_api(token: Optional[str]):
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n\n"
            "Install it in this environment (pick one):\n"
            "  - pip install huggingface_hub\n"
            "  - pip install -r requirements/requirements.txt\n"
        ) from e

    api = HfApi(token=token)
    return api, HfHubHTTPError


def _hf_file_exists(api, *, repo_id: str, revision: str, path_in_repo: str) -> bool:
    # Prefer method if present; fall back to functional API.
    try:
        return bool(api.file_exists(repo_id=repo_id, filename=path_in_repo, revision=revision))
    except Exception:
        try:
            from huggingface_hub import file_exists

            return bool(file_exists(repo_id=repo_id, filename=path_in_repo, revision=revision))
        except Exception:
            # If existence checks fail (permissions/network), be conservative and return False.
            return False


def _prompt_repo_dir(*, title: str, default_dir: str) -> str:
    """
    Ask interactively for a destination directory in the HF repo.
    Returns a normalized directory string (no leading slash, no trailing slash).
    """
    while True:
        raw = input(
            f"\n{title}\n"
            f"Enter repo directory path for this transformer's files\n"
            f"(blank = default, 'skip' = skip, 'quit' = exit)\n"
            f"default: {default_dir}\n"
            f"> "
        ).strip()
        if raw == "":
            raw = default_dir
        low = raw.lower()
        if low in {"skip", "quit"}:
            return low
        # normalize
        raw = raw.strip().lstrip("/").rstrip("/")
        if raw:
            return raw
        logger.warning("Repo directory cannot be empty.")


def _upload_transformer_to_repo_dir(
    *,
    api,
    repo_id: str,
    revision: str,
    commit_message: str,
    dry_run: bool,
    staging_root: Path,
    transformer_out_dir: Path,
    meta_id: str,
    base: str,
    architecture: str,
    repo_dir: str,
) -> None:
    """
    Upload missing files to `repo_dir/` and delete local weights after upload.
    """
    repo_dir = repo_dir.strip().lstrip("/").rstrip("/")
    if not repo_dir:
        raise ValueError("repo_dir cannot be empty")

    local_files: list[Path] = [
        transformer_out_dir / BF16_FILENAME,
        transformer_out_dir / FP8_FILENAME,
        transformer_out_dir / GGUF_Q8_0_FILENAME,
        transformer_out_dir / GGUF_Q6_K_FILENAME,
        transformer_out_dir / GGUF_Q4_K_M_FILENAME,
    ]
    pairs: list[tuple[Path, str]] = []
    for lf in local_files:
        if lf.exists() and lf.is_file():
            pairs.append((lf, f"{repo_dir}/{lf.name}"))

    if not pairs:
        logger.warning(f"No local transformer artifacts found in {transformer_out_dir}")
        return

    # No duplicates: if a file already exists remotely at the chosen path, do not upload it.
    for local_path, remote_path in pairs:
        if _hf_file_exists(api, repo_id=repo_id, revision=revision, path_in_repo=remote_path):
            logger.info(f"Remote exists; skipping upload: {remote_path}")
            # To conserve disk, delete our local copy.
            _safe_unlink(local_path)
            continue

        if dry_run:
            logger.info(f"DRY: would upload {local_path} -> {remote_path}")
            # In dry-run we do not delete local artifacts.
            continue

        logger.info(f"Uploading {local_path.name} -> {remote_path}")
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            revision=revision,
            commit_message=commit_message,
        )
        # Delete after successful upload to conserve space.
        _safe_unlink(local_path)

    # Cleanup empty directories.
    _cleanup_empty_parents(transformer_out_dir, stop_at=staging_root)


def _remote_artifacts_exist(
    *,
    api,
    repo_id: str,
    revision: str,
    repo_dir: str,
) -> dict[str, bool]:
    """
    Return existence map for our required transformer artifacts at `repo_dir/`.
    """
    targets = {
        BF16_FILENAME: f"{repo_dir}/{BF16_FILENAME}",
        FP8_FILENAME: f"{repo_dir}/{FP8_FILENAME}",
        GGUF_Q8_0_FILENAME: f"{repo_dir}/{GGUF_Q8_0_FILENAME}",
        GGUF_Q6_K_FILENAME: f"{repo_dir}/{GGUF_Q6_K_FILENAME}",
        GGUF_Q4_K_M_FILENAME: f"{repo_dir}/{GGUF_Q4_K_M_FILENAME}",
    }
    return {
        name: _hf_file_exists(api, repo_id=repo_id, revision=revision, path_in_repo=path)
        for name, path in targets.items()
    }


def _local_artifacts_exist(transformer_out_dir: Path) -> dict[str, bool]:
    targets = {
        BF16_FILENAME: transformer_out_dir / BF16_FILENAME,
        FP8_FILENAME: transformer_out_dir / FP8_FILENAME,
        GGUF_Q8_0_FILENAME: transformer_out_dir / GGUF_Q8_0_FILENAME,
        GGUF_Q6_K_FILENAME: transformer_out_dir / GGUF_Q6_K_FILENAME,
        GGUF_Q4_K_M_FILENAME: transformer_out_dir / GGUF_Q4_K_M_FILENAME,
    }
    return {name: p.exists() for name, p in targets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build + upload transformer BF16/FP8/GGUF weights to Hugging Face"
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=_default_manifest_root(),
        help="Root directory containing model manifests (new_manifest).",
    )
    parser.add_argument(
        "--target-repo-id",
        default="totoku/apex-models",
        help="Target Hugging Face repo (e.g. org/name).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Publish transformer weights",
        help="Commit message for the Hub commit.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=_default_staging_root(),
        help="Local staging directory that mirrors the Hub repo layout.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir override (passed to huggingface_hub).",
    )
    parser.add_argument(
        "--download-revision",
        default=None,
        help="Optional revision to download sources from (defaults to HF default).",
    )
    parser.add_argument(
        "--only-manifest",
        action="append",
        default=[],
        help="Process only manifests whose path contains this substring (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload; print actions. (Artifacts may still be generated unless --no-build is set.)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building/quantizing; only upload existing files in staging-root.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HF (build only).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional). If omitted, uses HF_TOKEN / cached login.",
    )

    args = parser.parse_args()

    manifest_root: Path = args.manifest_root.expanduser().resolve()
    staging_root: Path = args.staging_root.expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)

    manifests = _iter_manifest_files(manifest_root)
    if args.only_manifest:
        manifests = [m for m in manifests if any(s in str(m) for s in args.only_manifest)]

    if not manifests:
        logger.error(f"No manifests found under {manifest_root}")
        return 2

    logger.info(f"Found {len(manifests)} manifests under {manifest_root}")
    logger.info(f"Staging root: {staging_root}")
    logger.info(f"Target repo: {args.target_repo_id}")
    
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = None
    HfHubHTTPError = None
    if not args.no_upload:
        api, HfHubHTTPError = _ensure_hf_api(token)
        try:
            api.repo_info(repo_id=args.target_repo_id, repo_type="model")
        except HfHubHTTPError as e:  # type: ignore[misc]
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                logger.info("Repo does not exist; creating it...")
                api.create_repo(repo_id=args.target_repo_id, repo_type="model", exist_ok=True)
            else:
                raise

    # Avoid duplicates across manifests.
    seen: set[tuple[str, str]] = set()  # (base, src_path)

    if not args.no_build or not args.no_upload:
        for manifest_path in manifests:
            logger.info(f"Processing {manifest_path.relative_to(manifest_root)}")
            manifest = _load_yaml(manifest_path)

            spec = manifest.get("spec") or {}
            components = spec.get("components") or []
            if not isinstance(components, list):
                continue

            meta = manifest.get("metadata") or {}
            meta_id = str(meta.get("id") or manifest_path.stem)

            transformers: list[dict[str, Any]] = [
                c for c in components if isinstance(c, dict) and c.get("type") == "transformer"
            ]

            for t_idx, comp in enumerate(transformers):
                # If a manifest contains multiple transformer components, make the local staging
                # directory unique per transformer. Otherwise, the second transformer can wrongly
                # reuse "transformer-bf16.safetensors" staged for the first transformer simply
                # because the filenames are the same.
                transformer_name = str(comp.get("name") or comp.get("label") or "").strip()
                transformer_suffix = _slugify_dir_name(transformer_name) if len(transformers) > 1 else ""

                base = str(comp.get("base") or "").strip()
                if not base:
                    logger.warning("Skipping transformer with no base")
                    continue

                architecture = _arch_from_transformer_base(base)
                dest_rel = _dest_transformer_dir(
                    manifest=manifest,
                    transformer_component=comp,
                    target_repo_id=args.target_repo_id,
                )
                if transformer_suffix and (not dest_rel.parts or dest_rel.parts[-1] != transformer_suffix):
                    dest_rel = dest_rel / transformer_suffix
                transformer_out_dir = staging_root / dest_rel
                transformer_out_dir.mkdir(parents=True, exist_ok=True)

                src_entry = _pick_bf16_source(comp)
                if not src_entry or not isinstance(src_entry.get("path"), str):
                    logger.warning(f"No BF16/FP16 safetensors source found for {base}")
                    continue

                src_path_str = str(src_entry["path"])
                key = (base, src_path_str)
                if key in seen:
                    logger.info(f"Skipping duplicate transformer reference: {base} ({src_path_str})")
                    continue
                seen.add(key)

                bf16_staged = transformer_out_dir / BF16_FILENAME

                # Ask for destination path early so we can avoid download/build if remote already has it.
                repo_dir: str | None = None
                remote_exists: dict[str, bool] | None = None
                if not args.no_upload:
                    assert api is not None
                    default_repo_dir = str(dest_rel).replace("\\", "/").lstrip("/").rstrip("/")
                    title = (
                        f"Ready to upload transformer: model={meta_id} base={base} arch={architecture}"
                        + (f" name={transformer_name}" if transformer_name else "")
                    )
                    repo_dir = _prompt_repo_dir(title=title, default_dir=default_repo_dir)
                    if repo_dir == "quit":
                        raise SystemExit(0)
                    if repo_dir == "skip":
                        logger.info("Skipping this transformer.")
                        continue
                    remote_exists = _remote_artifacts_exist(
                        api=api,
                        repo_id=args.target_repo_id,
                        revision=args.revision,
                        repo_dir=repo_dir,
                    )

                    if all(remote_exists.values()):
                        logger.info(
                            "All transformer artifacts already exist remotely; skipping download/build/upload."
                        )
                        # If we happen to have local copies, delete them to conserve disk.
                        for name in remote_exists.keys():
                            _safe_unlink(transformer_out_dir / name)
                        _cleanup_empty_parents(transformer_out_dir, stop_at=staging_root)
                        continue

                local_exists = _local_artifacts_exist(transformer_out_dir)

                # If the target already has some artifacts, don't build them again.
                want_bf16 = True
                want_fp8 = True
                want_q8 = True
                want_q6 = True
                want_q4 = True
                if remote_exists is not None:
                    want_bf16 = not remote_exists[BF16_FILENAME]
                    want_fp8 = not remote_exists[FP8_FILENAME]
                    want_q8 = not remote_exists[GGUF_Q8_0_FILENAME]
                    want_q6 = not remote_exists[GGUF_Q6_K_FILENAME]
                    want_q4 = not remote_exists[GGUF_Q4_K_M_FILENAME]

                # Build phase: only generate missing remote artifacts.
                if not args.no_build:
                    # Stage BF16 weights (single file) only if needed for upload or for generating derived weights.
                    need_bf16_local = want_bf16 or want_fp8 or want_q8 or want_q6 or want_q4
                    if need_bf16_local and not bf16_staged.exists():
                        # Prefer local sources, avoid downloading when not necessary.
                        if _is_hf_path(src_path_str):
                            loc = _parse_hf_location(src_path_str)
                            logger.info(
                                f"Resolving source from HF (cache-first): {loc.repo_id}/{loc.path_in_repo}"
                            )
                            src_local = _download_hf_path(
                                loc, revision=args.download_revision, cache_dir=args.cache_dir
                            )
                        else:
                            src_local = Path(src_path_str).expanduser().resolve()
                            if not src_local.exists():
                                logger.warning(f"Source path does not exist: {src_local}")
                                continue

                        logger.info(
                            f"Staging BF16 weights -> {bf16_staged.relative_to(staging_root)}"
                        )
                        try:
                            bf16_staged = _stage_bf16_as_single_file(src_local, transformer_out_dir)
                        except MemoryError:
                            logger.error(
                                "Out of memory while consolidating safetensors shards. "
                                "Skipping this transformer."
                            )
                            continue
                    else:
                        if need_bf16_local:
                            logger.info("BF16 already staged; skipping copy")

                # FP8: exclude the same substrings we preserve in GGUF, plus any obvious norms.
                # (We err on the side of preserving more tensors in BF16.)
                from src.quantize.transformer import get_f32_weights_preserve_dtype, ModelArchitecture

                exclude = get_f32_weights_preserve_dtype(ModelArchitecture(architecture))
                if not args.no_build:
                    if want_fp8 and not (transformer_out_dir / FP8_FILENAME).exists():
                        _ensure_fp8_e4m3fn(
                            bf16_src=bf16_staged,
                            transformer_out_dir=transformer_out_dir,
                            exclude_substrings=exclude,
                        )

                    if (want_q8 or want_q6 or want_q4) and (
                        not (transformer_out_dir / GGUF_Q8_0_FILENAME).exists()
                        or not (transformer_out_dir / GGUF_Q6_K_FILENAME).exists()
                        or not (transformer_out_dir / GGUF_Q4_K_M_FILENAME).exists()
                    ):
                        # _ensure_gguf itself skips existing local files.
                        _ensure_gguf(
                            bf16_src=bf16_staged,
                            transformer_out_dir=transformer_out_dir,
                            architecture=architecture,
                        )

                if not args.no_upload:
                    assert api is not None
                    assert repo_dir is not None
                    # Use the already-selected `repo_dir` by passing it as the default and reusing the prompt,
                    # but the user can still override/skip/quit.
                    _upload_transformer_to_repo_dir(
                        api=api,
                        repo_id=args.target_repo_id,
                        revision=args.revision,
                        commit_message=args.commit_message,
                        dry_run=args.dry_run,
                        staging_root=staging_root,
                        transformer_out_dir=transformer_out_dir,
                        meta_id=meta_id,
                        base=base,
                        architecture=architecture,
                        repo_dir=repo_dir,
                    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

