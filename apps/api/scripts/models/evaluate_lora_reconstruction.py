import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running this script directly without requiring manual PYTHONPATH setup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logging
import torch
from safetensors.torch import load_file


def _local_setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_local_setup_logging()
logger = logging.getLogger(__name__)


def _resolve_weights_path(path: str, subfolder: Optional[str]) -> str:
    """
    If `path` is a directory and `subfolder` exists under it, return path/subfolder.
    Otherwise return `path` as-is.
    """
    if subfolder and os.path.isdir(path):
        candidate = os.path.join(path, subfolder)
        if os.path.exists(candidate):
            return candidate
    return path


def _parse_include_exclude(include_regex: Optional[str], exclude_regex: Optional[str]):
    include_re = None
    exclude_re = None
    if include_regex:
        import re

        include_re = re.compile(include_regex)
    if exclude_regex:
        import re

        exclude_re = re.compile(exclude_regex)
    return include_re, exclude_re


def _extract_lora_modules(
    lora_sd: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Return {module_prefix: {"A": tensor, "B": tensor, "alpha": optional tensor}}.
    module_prefix is the part before ".lora_A.weight" / ".lora_B.weight".
    """
    mods: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_A.weight"):
            p = k[: -len(".lora_A.weight")]
            mods.setdefault(p, {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            p = k[: -len(".lora_B.weight")]
            mods.setdefault(p, {})["B"] = v
        elif k.endswith(".alpha"):
            p = k[: -len(".alpha")]
            mods.setdefault(p, {})["alpha"] = v
    # Only keep complete A/B pairs
    mods = {p: d for p, d in mods.items() if "A" in d and "B" in d}
    return mods


def _maybe_remap_lora_to_distilled_layout(
    lora_sd: Dict[str, torch.Tensor],
    base_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Heuristic remap for LoRAs trained against an older HunyuanVideo-style transformer layout
    that uses fused qkv/proj modules (e.g. "img_attn.qkv", "img_attn.proj") and MLP blocks
    indexed as 0/2 (e.g. "img_mlp.0", "img_mlp.2").

    The distilled base checkpoint in this repo uses split projections:
      - img_attn_{q,k,v,proj}
      - txt_attn_{q,k,v,proj}
      - img_mlp.fc1/fc2, txt_mlp.fc1/fc2
      - img_mod.linear, txt_mod.linear

    This function rewrites LoRA keys accordingly and splits fused qkv LoRA_B into q/k/v.
    """
    # Only do this if it looks necessary and the base layout suggests split projections.
    has_fused_img_qkv = any("img_attn.qkv.lora_A.weight" in k for k in lora_sd.keys())
    base_has_split_img_q = any(k.endswith("img_attn_q.weight") for k in base_sd.keys())
    if not (has_fused_img_qkv and base_has_split_img_q):
        return lora_sd

    logger.info(
        "Detected fused qkv/proj LoRA keys; remapping to distilled split-projection layout."
    )

    def rename_simple(old_sub: str, new_sub: str) -> None:
        keys = list(lora_sd.keys())
        for k in keys:
            if old_sub not in k:
                continue
            v = lora_sd.pop(k)
            lora_sd[k.replace(old_sub, new_sub)] = v

    # Simple renames
    rename_simple(".img_mod.lin", ".img_mod.linear")
    rename_simple(".txt_mod.lin", ".txt_mod.linear")
    rename_simple(".img_mlp.0", ".img_mlp.fc1")
    rename_simple(".img_mlp.2", ".img_mlp.fc2")
    rename_simple(".txt_mlp.0", ".txt_mlp.fc1")
    rename_simple(".txt_mlp.2", ".txt_mlp.fc2")
    rename_simple(".img_attn.proj", ".img_attn_proj")
    rename_simple(".txt_attn.proj", ".txt_attn_proj")

    # Split fused qkv into q/k/v by splitting LoRA_B along dim=0.
    def split_qkv(prefix: str, out_prefix_base: str) -> None:
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"
        alpha_key = f"{prefix}.alpha"
        if a_key not in lora_sd or b_key not in lora_sd:
            return
        A = lora_sd.pop(a_key)
        B = lora_sd.pop(b_key)
        alpha = lora_sd.pop(alpha_key, None)

        if B.ndim != 2 or B.shape[0] % 3 != 0:
            # Can't split, put back and skip.
            lora_sd[a_key] = A
            lora_sd[b_key] = B
            if alpha is not None:
                lora_sd[alpha_key] = alpha
            return

        B_q, B_k, B_v = B.chunk(3, dim=0)
        for tag, B_part in (("q", B_q), ("k", B_k), ("v", B_v)):
            new_prefix = out_prefix_base + f"_{tag}"
            lora_sd[f"{new_prefix}.lora_A.weight"] = A
            lora_sd[f"{new_prefix}.lora_B.weight"] = B_part
            if alpha is not None:
                lora_sd[f"{new_prefix}.alpha"] = alpha

    # Example: double_blocks.0.img_attn.qkv -> double_blocks.0.img_attn_q / _k / _v
    prefixes = set()
    for k in lora_sd.keys():
        if k.endswith(".lora_A.weight") and ".img_attn.qkv" in k:
            prefixes.add(k[: -len(".lora_A.weight")])
        if k.endswith(".lora_A.weight") and ".txt_attn.qkv" in k:
            prefixes.add(k[: -len(".lora_A.weight")])

    for p in sorted(prefixes):
        if ".img_attn.qkv" in p:
            split_qkv(p, p.replace("img_attn.qkv", "img_attn"))
        elif ".txt_attn.qkv" in p:
            split_qkv(p, p.replace("txt_attn.qkv", "txt_attn"))

    return lora_sd


def _apply_lora_delta(
    w_base: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: Optional[torch.Tensor],
    *,
    scale: float,
) -> torch.Tensor:
    """
    Apply LoRA to a base weight tensor and return reconstructed weight:
        W = W_base + scale * (alpha/r) * (B @ A)
    Supports:
      - Linear: W (out,in), A (r,in), B (out,r)
      - Conv2d: W (out,in,kh,kw), A (r,in,kh,kw) or (r,in,1,1), B (out,r,1,1)
    """
    w_base_f = w_base.to(dtype=torch.float32)
    A_f = A.to(dtype=torch.float32)
    B_f = B.to(dtype=torch.float32)

    # Determine rank from A's first dim (PEFT convention)
    r = int(A_f.shape[0])
    if r <= 0:
        return w_base_f

    alpha_val: float
    if alpha is None:
        alpha_val = float(r)  # default => scaling=1.0
    else:
        alpha_val = float(alpha.detach().to("cpu").item())

    scaling = float(scale) * (alpha_val / float(r))

    if w_base_f.ndim == 2:
        # Linear
        # A: (r,in), B: (out,r)
        delta = B_f @ A_f
        return w_base_f + scaling * delta

    if w_base_f.ndim == 4:
        # Conv2d (PEFT uses B as (out,r,1,1), A as (r,in,kh,kw))
        out_ch, in_ch, kh, kw = map(int, w_base_f.shape)
        # Flatten to 2D for matmul.
        Bm = B_f.reshape(out_ch, r)
        Am = A_f.reshape(r, in_ch * kh * kw)
        delta_m = Bm @ Am
        delta = delta_m.reshape(out_ch, in_ch, kh, kw)
        return w_base_f + scaling * delta

    # Unsupported weight ndim
    return w_base_f


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how well a LoRA reconstructs a tuned checkpoint when applied to a base checkpoint."
    )
    parser.add_argument(
        "--base", required=True, help="Base weights path (file or dir of shards)."
    )
    parser.add_argument(
        "--tuned", required=True, help="Tuned weights path (file or dir of shards)."
    )
    parser.add_argument(
        "--lora", required=True, help="LoRA weights path (.safetensors)."
    )
    parser.add_argument(
        "--base_subfolder",
        default="transformer",
        help="Optional subfolder to use under --base if it exists.",
    )
    parser.add_argument(
        "--tuned_subfolder",
        default=None,
        help="Optional subfolder to use under --tuned if it exists.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Global LoRA scale multiplier (like adapter weight).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device for applying LoRA (cpu or cuda).",
    )
    parser.add_argument(
        "--include_regex",
        default=None,
        help="Only evaluate weight keys matching this regex.",
    )
    parser.add_argument(
        "--exclude_regex", default=None, help="Skip weight keys matching this regex."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Show the worst top-K layers by relative delta error.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Optional cap on number of layers evaluated.",
    )
    args = parser.parse_args()

    from src.converters.convert import load_state_dict, strip_common_prefix
    from src.lora.lora_converter import LoraConverter

    base_path = _resolve_weights_path(args.base, args.base_subfolder)
    tuned_path = _resolve_weights_path(args.tuned, args.tuned_subfolder)

    logger.info(f"Loading base weights from: {base_path}")
    base_sd = load_state_dict(base_path)
    logger.info(f"Loaded base state dict with {len(base_sd)} tensors.")

    logger.info(f"Loading tuned weights from: {tuned_path}")
    tuned_sd = load_state_dict(tuned_path)
    logger.info(f"Loaded tuned state dict with {len(tuned_sd)} tensors.")

    # Normalize prefixes so keys align
    tuned_sd = strip_common_prefix(tuned_sd, base_sd)
    base_sd = strip_common_prefix(base_sd, tuned_sd)

    logger.info(f"Loading LoRA weights from: {args.lora}")
    lora_sd = load_file(args.lora)
    logger.info(f"Loaded LoRA state dict with {len(lora_sd)} tensors.")

    # Normalize to PEFT-style keys and drop diff/diff_b, etc.
    LoraConverter().convert(lora_sd)
    # Strip common prefixes (e.g. 'diffusion_model.') against base keys
    lora_sd = strip_common_prefix(lora_sd, base_sd)
    # If this LoRA targets a fused-qkv layout but base is split, remap so we can compare apples-to-apples.
    lora_sd = _maybe_remap_lora_to_distilled_layout(lora_sd, base_sd)

    lora_modules = _extract_lora_modules(lora_sd)
    logger.info(f"LoRA contains {len(lora_modules)} module entries with A/B weights.")

    include_re, exclude_re = _parse_include_exclude(
        args.include_regex, args.exclude_regex
    )
    device = torch.device(args.device)

    # Evaluate
    eps = 1e-12
    rows: List[Tuple[float, float, float, str]] = (
        []
    )  # (rel_delta_err, rel_weight_err, delta_norm, key)
    missing_in_tuned = 0
    missing_in_base = 0
    evaluated = 0

    for module_prefix, ab in lora_modules.items():
        w_key = module_prefix + ".weight"
        if include_re and not include_re.search(w_key):
            continue
        if exclude_re and exclude_re.search(w_key):
            continue

        if w_key not in base_sd:
            missing_in_base += 1
            continue
        if w_key not in tuned_sd:
            missing_in_tuned += 1
            continue

        w_base = base_sd[w_key].to(device)
        w_tuned = tuned_sd[w_key].to(device)
        A = ab["A"].to(device)
        B = ab["B"].to(device)
        alpha = ab.get("alpha", None)
        if alpha is not None:
            alpha = alpha.to(device)

        with torch.no_grad():
            w_recon = _apply_lora_delta(w_base, A, B, alpha, scale=float(args.scale))
            # Errors
            delta_true = w_tuned.to(torch.float32) - w_base.to(torch.float32)
            delta_recon = w_recon.to(torch.float32) - w_base.to(torch.float32)
            rel_delta_err = float(
                (delta_recon - delta_true).norm() / (delta_true.norm() + eps)
            )
            rel_weight_err = float(
                (w_recon - w_tuned.to(torch.float32)).norm()
                / (w_tuned.to(torch.float32).norm() + eps)
            )
            delta_norm = float(delta_true.norm())
            rows.append((rel_delta_err, rel_weight_err, delta_norm, w_key))

        evaluated += 1
        if args.max_layers is not None and evaluated >= int(args.max_layers):
            break

    if evaluated == 0:
        raise RuntimeError(
            "No layers evaluated. Check regex filters and whether LoRA keys match base/tuned keys."
        )

    # Aggregate metrics
    rel_delta_errs = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    rel_weight_errs = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    delta_norms = torch.tensor([max(r[2], eps) for r in rows], dtype=torch.float64)

    weighted_mean_delta_err = float(
        (rel_delta_errs * delta_norms).sum() / delta_norms.sum()
    )
    weighted_mean_weight_err = float(
        (rel_weight_errs * delta_norms).sum() / delta_norms.sum()
    )

    logger.info("=== Summary ===")
    logger.info(f"Evaluated layers: {evaluated}")
    logger.info(f"LoRA modules missing in base: {missing_in_base}")
    logger.info(f"LoRA modules missing in tuned: {missing_in_tuned}")
    logger.info(f"Scale used: {args.scale}")
    logger.info(
        f"Rel delta error: mean={float(rel_delta_errs.mean()):.6f} weighted_mean={weighted_mean_delta_err:.6f} max={float(rel_delta_errs.max()):.6f}"
    )
    logger.info(
        f"Rel weight error: mean={float(rel_weight_errs.mean()):.6f} weighted_mean={weighted_mean_weight_err:.6f} max={float(rel_weight_errs.max()):.6f}"
    )

    # Show worst layers
    rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
    topk = int(args.topk)
    logger.info(f"=== Worst {topk} layers by relative delta error ===")
    for rel_d, rel_w, dnorm, k in rows_sorted[:topk]:
        logger.info(
            f"{k} | rel_delta_err={rel_d:.6f} rel_weight_err={rel_w:.6f} delta_norm={dnorm:.4g}"
        )


if __name__ == "__main__":
    main()
