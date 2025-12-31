import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from tqdm import tqdm

# Allow running this script directly without requiring manual PYTHONPATH setup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logging
import torch
from safetensors.torch import save_file


def _local_setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_local_setup_logging()
logger = logging.getLogger(__name__)

MIN_SV = 1e-6


def _strip_checkpoint_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("._fsdp_wrapped_module", "")
        nk = nk.replace("model.", "")
        out[nk] = v
    return out


def _iter_lora_candidate_weight_keys(
    state_dict: Dict[str, torch.Tensor],
    *,
    include_conv2d: bool = False,
) -> Iterable[str]:
    """
    Yield weight keys for modules we can LoRA-ize.
    Since this script can run without importing a model, we infer:
    - Linear-like weights: 2D tensors ending with `.weight`
    - Conv2d weights: 4D tensors ending with `.weight` (only if enabled)
    """
    for k, v in state_dict.items():
        if not k.endswith(".weight"):
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2:
            yield k
        elif include_conv2d and v.ndim == 4:
            yield k


# --- Singular value indexing helpers (Kohya-style options) ---
def _index_sv_cumulative(S: torch.Tensor, target: float) -> int:
    original_sum = float(torch.sum(S))
    if original_sum <= 0:
        return 1
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))
    return index


def _index_sv_fro(S: torch.Tensor, target: float) -> int:
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    if S_fro_sq <= 0:
        return 1
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S) - 1))
    return index


def _index_sv_ratio(S: torch.Tensor, target: float) -> int:
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item())
    index = max(1, min(index, len(S) - 1))
    return index


def _index_sv_knee(S: torch.Tensor, min_sv_threshold: float = 1e-8) -> int:
    n = len(S)
    if n < 3:
        return 1
    s_max, s_min = S[0], S[-1]
    if float(s_max - s_min) < float(min_sv_threshold):
        return 1
    s_normalized = (S - s_min) / (s_max - s_min)
    x_normalized = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (x_normalized + s_normalized - 1).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    rank = max(1, min(rank, n - 1))
    return rank


def _index_sv_cumulative_knee(S: torch.Tensor, min_sv_threshold: float = 1e-8) -> int:
    n = len(S)
    if n < 3:
        return 1
    s_sum = torch.sum(S)
    if float(s_sum) < float(min_sv_threshold):
        return 1
    y_values = torch.cumsum(S, dim=0) / s_sum
    y_min, y_max = y_values[0], y_values[n - 1]
    if float(y_max - y_min) < float(min_sv_threshold):
        return 1
    y_norm = (y_values - y_min) / (y_max - y_min)
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (y_norm - x_norm).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    rank = max(1, min(rank, n - 1))
    return rank


def _index_sv_rel_decrease(S: torch.Tensor, tau: float = 0.1) -> int:
    if len(S) < 2:
        return 1
    ratios = S[1:] / S[:-1]
    for k in range(len(ratios)):
        if float(ratios[k]) < float(tau):
            return k + 1
    return len(S)


def _determine_rank(
    S_values: torch.Tensor,
    dynamic_method_name: Optional[str],
    dynamic_param_value: Optional[float],
    max_rank_limit: int,
    module_eff_in_dim: int,
    module_eff_out_dim: int,
    min_sv_threshold: float = MIN_SV,
) -> int:
    if S_values.numel() == 0 or float(S_values[0]) <= float(min_sv_threshold):
        return 1
    rank = 0
    if dynamic_method_name == "sv_ratio":
        rank = _index_sv_ratio(S_values, float(dynamic_param_value))
    elif dynamic_method_name == "sv_cumulative":
        rank = _index_sv_cumulative(S_values, float(dynamic_param_value))
    elif dynamic_method_name == "sv_fro":
        rank = _index_sv_fro(S_values, float(dynamic_param_value))
    elif dynamic_method_name == "sv_knee":
        rank = _index_sv_knee(S_values, float(min_sv_threshold))
    elif dynamic_method_name == "sv_cumulative_knee":
        rank = _index_sv_cumulative_knee(S_values, float(min_sv_threshold))
    elif dynamic_method_name == "sv_rel_decrease":
        rank = _index_sv_rel_decrease(S_values, float(dynamic_param_value))
    else:
        rank = int(max_rank_limit)

    rank = min(
        int(rank),
        int(max_rank_limit),
        int(module_eff_in_dim),
        int(module_eff_out_dim),
        int(len(S_values)),
    )
    rank = max(1, int(rank))
    return rank


def _svd_decompose(
    x: torch.Tensor,
    *,
    method: str = "auto",
    max_rank_limit: int,
    oversample: int,
    niter: int,
    svd_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (U, S, Vh) such that x ~= U diag(S) Vh.
    For randomized mode we only compute up to q components.
    """
    if x.ndim != 2:
        raise ValueError(f"SVD expects 2D, got shape={tuple(x.shape)}")
    m, n = int(x.shape[0]), int(x.shape[1])
    k = min(m, n)
    if k <= 0:
        raise ValueError(f"Invalid SVD shape {tuple(x.shape)}")

    method = str(method).lower()
    if method not in {"auto", "exact", "randomized"}:
        raise ValueError("svd_method must be one of: auto, exact, randomized")

    # gate exact SVD by size; large matmuls can be painfully slow / memory-heavy
    elems = m * n
    use_exact = method == "exact"
    if method == "auto":
        use_exact = elems <= 2_000_000 and k <= 2048

    x = x.to(dtype=svd_dtype)
    if use_exact:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return U, S, Vh

    q = min(k, int(max_rank_limit) + int(oversample))
    U, S, V = torch.svd_lowrank(x, q=q, niter=int(niter))
    Vh = V.transpose(0, 1).contiguous()
    return U, S, Vh


def _construct_lora_factors_from_svd(
    U_full: torch.Tensor,
    S_all_values: torch.Tensor,
    Vh_full: torch.Tensor,
    *,
    rank: int,
    clamp_quantile: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kohya-style split:
      delta ~= (U_k * sqrt(S_k)) @ (sqrt(S_k) * Vh_k)
    We return (A, B) where:
      A = sqrt(S_k) * Vh_k     (rank, in)
      B = U_k * sqrt(S_k)     (out, rank)
    """
    S_k = S_all_values[:rank]
    U_k = U_full[:, :rank]
    Vh_k = Vh_full[:rank, :]

    S_k_non_negative = torch.clamp(S_k, min=0.0)
    s_sqrt = torch.sqrt(S_k_non_negative)
    B = (U_k * s_sqrt.unsqueeze(0)).to(dtype=torch.float32)
    A = (Vh_k * s_sqrt.unsqueeze(1)).to(dtype=torch.float32)

    if clamp_quantile is not None:
        q = float(clamp_quantile)
        q = min(max(q, 0.0), 1.0)
        dist = torch.cat([B.flatten(), A.flatten()])
        hi_val = torch.quantile(dist, q)
        # If quantile returns 0 on a non-zero dist, fall back to max-abs clamp.
        if float(hi_val) == 0.0 and float(torch.max(torch.abs(dist))) > 1e-9:
            hi_val = torch.max(torch.abs(dist))
        B = B.clamp(-hi_val, hi_val)
        A = A.clamp(-hi_val, hi_val)

    return A.contiguous(), B.contiguous()


def _construct_lora_factors_from_svd_comfy(
    U_full: torch.Tensor,
    S_all_values: torch.Tensor,
    Vh_full: torch.Tensor,
    *,
    rank: int,
    clamp_quantile: Optional[float],
    clamp_sample_size: int = 100_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Comfy/KJ-style factors (no sqrt split):
      diff ~= (U_k @ diag(S_k)) @ Vh_k

    We return (A, B) where:
      A = Vh_k            (rank, in)
      B = U_k @ diag(S_k) (out, rank)

    Clamping (when enabled) matches the Comfy approach: clamp both B and A by a quantile
    computed on concatenated values, optionally sampling for speed on very large tensors.
    """
    S_k = S_all_values[:rank]
    U_k = U_full[:, :rank]
    Vh_k = Vh_full[:rank, :]

    # B = U_k @ diag(S_k) == U_k * S_k (broadcast over columns)
    B = (U_k * S_k.unsqueeze(0)).to(dtype=torch.float32)
    A = Vh_k.to(dtype=torch.float32)

    if clamp_quantile is not None:
        q = float(clamp_quantile)
        q = min(max(q, 0.0), 1.0)
        dist = torch.cat([B.flatten(), A.flatten()])
        if dist.numel() > int(clamp_sample_size):
            idx = torch.randperm(dist.numel(), device=dist.device)[
                : int(clamp_sample_size)
            ]
            dist_sample = dist[idx]
            hi_val = torch.quantile(dist_sample, q)
        else:
            hi_val = torch.quantile(dist, q)
        low_val = -hi_val
        B = B.clamp(low_val, hi_val)
        A = A.clamp(low_val, hi_val)

    return A.contiguous(), B.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a PEFT-compatible LoRA by factoring (tuned - base) weight deltas for transformer modules.\n"
            "Outputs keys in APEX/PEFT schema: <module>.lora_A.weight, <module>.lora_B.weight, <module>.alpha"
        )
    )
    parser.add_argument(
        "--base_model",
        default="/home/tosin_coverquick_co/apex/Wan-22-TI2V-Base",
        help="Path to base weights (a .safetensors/.pt file OR a directory of weight shards).",
    )
    parser.add_argument(
        "--base_subfolder",
        default="transformer",
        help="If --base_model is a directory, and this subfolder exists, load weights from base_model/base_subfolder.",
    )
    parser.add_argument(
        "--new_ckpt",
        default="/home/tosin_coverquick_co/apex-diffusion/components/af18e595fc128bba84f88a92bd6e26bff7fb6e27659ab1846547518b75d2d3bb_Wan2_2-TI2V-5B-Turbo_fp16.safetensors",
        help="Path to tuned weights (a .safetensors/.pt file OR a directory of weight shards).",
    )
    parser.add_argument(
        "--out",
        default="/home/tosin_coverquick_co/apex/lora_delta.safetensors",
        help="Output LoRA file path (.safetensors recommended).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="Max LoRA rank for Linear layers (or fixed rank if --dynamic_method is not set).",
    )
    parser.add_argument(
        "--conv_rank",
        type=int,
        default=None,
        help="Max LoRA rank for Conv2d 3x3 layers (defaults to --rank). Only used when --include_conv2d is set.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha. If omitted, defaults to rank (so scaling alpha/rank = 1).",
    )
    parser.add_argument(
        "--include_conv2d",
        action="store_true",
        help="Also extract LoRA for nn.Conv2d weights. By default only nn.Linear is processed.",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=0.0,
        help="Skip layers whose max absolute delta <= this threshold.",
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=1.0,
        help="Quantile for clamping LoRA factors (Kohya-style). Set to 1.0 to disable clamping (exact reconstruction possible at full rank).",
    )
    parser.add_argument(
        "--factorization",
        choices=["kohya", "comfy"],
        default="kohya",
        help=(
            "How to split singular values into LoRA factors. "
            "'kohya' uses sqrt(S) split (Up=U*sqrt(S), Down=sqrt(S)*Vh). "
            "'comfy' matches Comfy/KJ extraction (Up=U*S, Down=Vh)."
        ),
    )
    parser.add_argument(
        "--comfy_mode",
        action="store_true",
        help="Convenience flag: set --factorization comfy and default --clamp_quantile 0.99 if you didn't override it.",
    )
    parser.add_argument(
        "--clamp_sample_size",
        type=int,
        default=100_000,
        help="When clamping is enabled, sample this many values to estimate quantile (Comfy-style speedup).",
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        choices=[
            None,
            "sv_ratio",
            "sv_fro",
            "sv_cumulative",
            "sv_knee",
            "sv_rel_decrease",
            "sv_cumulative_knee",
        ],
        default=None,
        help="Enable dynamic rank selection from singular values. If unset, uses fixed rank.",
    )
    parser.add_argument(
        "--dynamic_param",
        type=float,
        default=None,
        help="Parameter for the selected dynamic rank method (required for sv_ratio/sv_fro/sv_cumulative/sv_rel_decrease).",
    )
    parser.add_argument(
        "--svd_method",
        choices=["auto", "exact", "randomized"],
        default="auto",
        help="SVD method for factoring deltas. exact is most accurate but can be very slow.",
    )
    parser.add_argument(
        "--svd_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dtype used during SVD computation (factors are saved as float32). float64 can improve accuracy.",
    )
    parser.add_argument(
        "--svd_oversample",
        type=int,
        default=8,
        help="Randomized SVD oversampling (only used for svd_method=randomized/auto-fallback).",
    )
    parser.add_argument(
        "--svd_niter",
        type=int,
        default=2,
        help="Randomized SVD power iterations (only used for svd_method=randomized/auto-fallback).",
    )
    parser.add_argument(
        "--svd_exact_max_elements",
        type=int,
        default=2_000_000,
        help="In auto mode, only use exact SVD if out_dim*in_dim <= this threshold.",
    )
    parser.add_argument(
        "--svd_exact_max_rank",
        type=int,
        default=2048,
        help="In auto mode, only use exact SVD if min(out_dim,in_dim) <= this threshold.",
    )
    parser.add_argument(
        "--svd_device",
        default=None,
        help="Device for SVD computation (e.g. cpu, cuda, cuda:0). Defaults to cuda if available else cpu.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help=(
            "Optional model name hint to apply a transformer checkpoint key converter to BOTH base and tuned weights "
            "(e.g. 'Wan', 'WanAnimate', 'HunyuanVideo'). If unset, no conversion is applied."
        ),
    )
    parser.add_argument(
        "--converter_model_base",
        default=None,
        help=(
            "Optional model-base string for converter selection via `get_transformer_converter`, e.g. "
            "'wan.base', 'wan.animate', 'hunyuanvideo.base', 'cogvideox.base'. "
            "If set, this takes precedence over --model_name."
        ),
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Do not write metadata into safetensors header.",
    )
    parser.add_argument(
        "--skip_if_missing",
        action="store_true",
        help="Skip layers missing in new checkpoint instead of erroring.",
    )
    parser.add_argument(
        "--include_regex",
        default=None,
        help="Optional regex; only process linear weights whose key matches.",
    )
    parser.add_argument(
        "--exclude_regex",
        default=None,
        help="Optional regex; skip linear weights whose key matches.",
    )
    parser.add_argument(
        "--probe_ranks",
        default=None,
        help=(
            "Debug mode: comma-separated list of ranks to evaluate reconstruction error for each layer (does not write a LoRA). "
            "Example: --probe_ranks 2048,1024,512,256,128,64. "
            "In this mode, dynamic rank is ignored and clamping is disabled unless you set --clamp_quantile < 1."
        ),
    )
    parser.add_argument(
        "--probe_max_layers",
        type=int,
        default=10,
        help="In --probe_ranks mode, only evaluate up to this many matching layers (for speed).",
    )
    args = parser.parse_args()

    # Convenience behavior to mimic Comfy defaults without breaking full-rank exactness by default.
    if args.comfy_mode:
        args.factorization = "comfy"
        if args.clamp_quantile == 1.0:
            # Comfy default
            args.clamp_quantile = 0.99

    if args.conv_rank is None:
        args.conv_rank = int(args.rank)

    methods_requiring_param = {"sv_ratio", "sv_fro", "sv_cumulative", "sv_rel_decrease"}
    if args.dynamic_method in methods_requiring_param and args.dynamic_param is None:
        parser.error(
            f"Dynamic method '{args.dynamic_method}' requires --dynamic_param to be set."
        )

    if not args.dynamic_method and args.rank <= 0:
        parser.error(f"--rank must be > 0. Got {args.rank}")
    if (
        args.include_conv2d
        and (args.conv_rank is None or int(args.conv_rank) <= 0)
        and not args.dynamic_method
    ):
        parser.error(
            f"--conv_rank must be > 0 when --include_conv2d is set. Got {args.conv_rank}"
        )

    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32
    svd_device = args.svd_device or ("cuda" if torch.cuda.is_available() else "cpu")
    svd_device_t = torch.device(svd_device)
    logger.info(f"Using SVD device: {svd_device_t}")
    if (
        str(args.svd_method).lower() == "exact"
        and svd_dtype == torch.float32
        and str(svd_device_t).startswith("cuda")
    ):
        logger.warning(
            "You are running exact SVD in float32 on CUDA. This can be numerically inaccurate even at full rank. "
            "For near-zero full-rank reconstruction error, use --svd_dtype float64 (ideally with --svd_device cpu)."
        )

    # 1) Load base + tuned weights directly (no model import needed).
    from src.converters.convert import (
        get_transformer_converter,
        get_transformer_converter_by_model_name,
        load_state_dict,
        strip_common_prefix,
    )

    base_path = args.base_model
    if os.path.isdir(base_path):
        candidate = os.path.join(base_path, args.base_subfolder)
        if args.base_subfolder and os.path.exists(candidate):
            base_path = candidate

    logger.info(f"Loading base weights from: {base_path}")
    base_sd = load_state_dict(base_path)
    logger.info(f"Loaded base state dict with {len(base_sd)} tensors.")

    logger.info(f"Loading tuned weights from: {args.new_ckpt}")
    new_sd = load_state_dict(args.new_ckpt)
    logger.info(f"Loaded tuned state dict with {len(new_sd)} tensors.")

    # 2) Optional conversion into a consistent key-space.
    converter = None
    if args.converter_model_base:
        converter = get_transformer_converter(str(args.converter_model_base))
    elif args.model_name:
        converter = get_transformer_converter_by_model_name(str(args.model_name))

    if converter is not None:
        try:
            converter.convert(base_sd)
        except Exception as e:
            logger.warning(
                f"Base conversion raised {type(e).__name__}: {e}. Continuing with raw base keys."
            )
        try:
            converter.convert(new_sd)
        except Exception as e:
            logger.warning(
                f"Tuned conversion raised {type(e).__name__}: {e}. Continuing with raw tuned keys."
            )

    # Normalize common prefixes so both dicts share the same root layout.
    new_sd = strip_common_prefix(new_sd, base_sd)
    base_sd = strip_common_prefix(base_sd, new_sd)

    # If clamp_quantile is >= 1, disable clamping so full-rank reconstruction can be exact.
    clamp_q: Optional[float] = None
    if args.clamp_quantile is not None and float(args.clamp_quantile) < 1.0:
        clamp_q = float(args.clamp_quantile)

    # 3) Build LoRA dict
    lora_sd: Dict[str, torch.Tensor] = {}
    include_re = None
    exclude_re = None
    if args.include_regex:
        import re

        include_re = re.compile(args.include_regex)
    if args.exclude_regex:
        import re

        exclude_re = re.compile(args.exclude_regex)

    processed = 0
    skipped_missing = 0
    skipped_shape = 0
    skipped_zero = 0
    skipped_small = 0
    max_rel_err = 0.0

    probe_ranks: Optional[list[int]] = None
    if args.probe_ranks:
        try:
            probe_ranks = [
                int(x.strip()) for x in str(args.probe_ranks).split(",") if x.strip()
            ]
            if not probe_ranks:
                raise ValueError("empty")
        except Exception:
            raise ValueError(
                f"Invalid --probe_ranks {args.probe_ranks!r}. Expected comma-separated ints like '2048,1024,512'."
            )
        if args.dynamic_method:
            logger.warning(
                f"--probe_ranks is set; ignoring --dynamic_method={args.dynamic_method!r} (probe uses explicit ranks)."
            )
        if clamp_q is not None:
            logger.info(
                f"--probe_ranks will use clamping at quantile={clamp_q}. For exact full-rank reconstruction, use --clamp_quantile 1.0."
            )
        else:
            logger.info(
                "--probe_ranks will run with clamping disabled (--clamp_quantile >= 1.0)."
            )

    weight_keys_iter = _iter_lora_candidate_weight_keys(
        base_sd, include_conv2d=bool(args.include_conv2d)
    )
    for w_key in tqdm(weight_keys_iter):
        if include_re and not include_re.search(w_key):
            continue
        if exclude_re and exclude_re.search(w_key):
            continue

        if w_key not in base_sd:
            continue
        if w_key not in new_sd:
            if args.skip_if_missing:
                skipped_missing += 1
                continue
            raise KeyError(
                f"Missing {w_key!r} in new checkpoint after conversion. "
                f"Tip: use --skip_if_missing or adjust --include_regex/--exclude_regex."
            )

        w_base = base_sd[w_key]
        w_new = new_sd[w_key]
        if w_base.shape != w_new.shape:
            skipped_shape += 1
            continue

        delta = (w_new.to(torch.float32) - w_base.to(torch.float32)).contiguous()
        if torch.count_nonzero(delta).item() == 0:
            skipped_zero += 1
            continue
        if args.min_diff and float(torch.max(torch.abs(delta))) <= float(args.min_diff):
            skipped_small += 1
            continue

        # Determine module type from delta shape:
        # - Linear: (out, in)
        # - Conv2d: (out, in, kh, kw)
        is_conv2d = delta.ndim == 4
        if delta.ndim not in (2, 4):
            skipped_shape += 1
            continue

        # Prepare 2D matrix for SVD
        if is_conv2d:
            out_ch, in_ch, kh, kw = map(int, delta.shape)
            is_3x3 = (kh, kw) != (1, 1)
            mat_for_svd = (
                delta.flatten(start_dim=1) if is_3x3 else delta.reshape(out_ch, in_ch)
            )
            max_rank_limit = int(args.conv_rank) if is_3x3 else int(args.rank)
            eff_out_dim, eff_in_dim = int(mat_for_svd.shape[0]), int(
                mat_for_svd.shape[1]
            )
        else:
            out_dim, in_dim = map(int, delta.shape)
            mat_for_svd = delta
            max_rank_limit = int(args.rank)
            eff_out_dim, eff_in_dim = out_dim, in_dim

        # Compute SVD (exact or randomized) on the chosen device.
        mat_for_svd = mat_for_svd.to(device=svd_device_t, dtype=torch.float32)
        try:
            U_full, S_full, Vh_full = _svd_decompose(
                mat_for_svd,
                method=args.svd_method,
                max_rank_limit=max_rank_limit,
                oversample=args.svd_oversample,
                niter=args.svd_niter,
                svd_dtype=svd_dtype,
            )
        except Exception as e:
            logger.warning(
                f"SVD failed for {w_key} with shape {tuple(mat_for_svd.shape)}: {type(e).__name__}: {e}"
            )
            skipped_shape += 1
            continue

        # --- Debug rank sweep mode (does not create a LoRA file) ---
        if probe_ranks is not None:
            # Evaluate reconstruction error at requested ranks by truncating the SVD.
            # Note: For exact SVD and no clamping, rank == min(out,in) should yield ~0 error (up to float32 rounding).
            full_rank = int(min(eff_out_dim, eff_in_dim, int(S_full.numel())))
            ranks_to_eval = []
            for r in probe_ranks:
                r_eff = int(min(max(r, 1), full_rank))
                ranks_to_eval.append(r_eff)
            ranks_to_eval = sorted(set(ranks_to_eval), reverse=True)

            for r_eff in ranks_to_eval:
                # "Ideal" truncated-SVD reconstruction (does not quantize to float32 LoRA factors):
                U_k = U_full[:, :r_eff]
                S_k = S_full[:r_eff]
                Vh_k = Vh_full[:r_eff, :]
                approx_svd = (U_k * S_k.unsqueeze(0)) @ Vh_k
                rel_err_svd = (approx_svd - mat_for_svd).norm() / (
                    mat_for_svd.norm() + 1e-12
                )

                if args.factorization == "comfy":
                    A_2d, B_2d = _construct_lora_factors_from_svd_comfy(
                        U_full,
                        S_full,
                        Vh_full,
                        rank=r_eff,
                        clamp_quantile=clamp_q,
                        clamp_sample_size=int(args.clamp_sample_size),
                    )
                else:
                    A_2d, B_2d = _construct_lora_factors_from_svd(
                        U_full, S_full, Vh_full, rank=r_eff, clamp_quantile=clamp_q
                    )
                scale = 1.0  # alpha defaults to rank => alpha/rank == 1
                if is_conv2d:
                    out_ch, in_ch, kh, kw = map(int, delta.shape)
                    is_3x3 = (kh, kw) != (1, 1)
                    Bm = B_2d.reshape(out_ch, r_eff)
                    Am = (
                        A_2d.reshape(r_eff, in_ch * kh * kw)
                        if is_3x3
                        else A_2d.reshape(r_eff, in_ch)
                    )
                    approx_m = scale * (Bm @ Am)
                    approx = (
                        approx_m.reshape(out_ch, in_ch, kh, kw)
                        if is_3x3
                        else approx_m.reshape(out_ch, in_ch, 1, 1)
                    )
                else:
                    approx = scale * (B_2d @ A_2d)
                rel_err_probe = (approx - delta.to(approx.device)).norm() / (
                    delta.norm() + 1e-12
                )
                logger.info(
                    f"[probe] {w_key} shape={tuple(delta.shape)} rank={r_eff}/{full_rank} "
                    f"rel_err_svd={float(rel_err_svd):.6g} rel_err_lora_fp32={float(rel_err_probe):.6g}"
                )

            processed += 1
            if processed >= int(args.probe_max_layers):
                logger.info(
                    f"[probe] Reached --probe_max_layers={args.probe_max_layers}, stopping."
                )
                break
            continue

        # Decide rank (dynamic or fixed), bounded by matrix dims.
        if args.dynamic_method:
            rank_used = _determine_rank(
                S_full,
                args.dynamic_method,
                args.dynamic_param,
                max_rank_limit=max_rank_limit,
                module_eff_in_dim=eff_in_dim,
                module_eff_out_dim=eff_out_dim,
                min_sv_threshold=MIN_SV,
            )
        else:
            rank_used = int(
                min(max_rank_limit, eff_in_dim, eff_out_dim, int(S_full.numel()))
            )
            rank_used = max(1, rank_used)

        # The loader applies: w = w_base + (alpha/rank) * (B @ A)
        # If alpha is omitted, we set alpha=rank_used per-layer so scaling==1.0.
        alpha_layer = float(args.alpha) if args.alpha is not None else float(rank_used)

        if args.factorization == "comfy":
            A_2d, B_2d = _construct_lora_factors_from_svd_comfy(
                U_full,
                S_full,
                Vh_full,
                rank=rank_used,
                clamp_quantile=clamp_q,
                clamp_sample_size=int(args.clamp_sample_size),
            )
        else:
            A_2d, B_2d = _construct_lora_factors_from_svd(
                U_full,
                S_full,
                Vh_full,
                rank=rank_used,
                clamp_quantile=clamp_q,
            )

        # Move final weights back to CPU for saving.
        if is_conv2d:
            out_ch, in_ch, kh, kw = map(int, delta.shape)
            is_3x3 = (kh, kw) != (1, 1)
            B = (
                B_2d.reshape(out_ch, rank_used, 1, 1)
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            if is_3x3:
                A = (
                    A_2d.reshape(rank_used, in_ch, kh, kw)
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
            else:
                A = (
                    A_2d.reshape(rank_used, in_ch, 1, 1)
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
        else:
            A = A_2d.to(device="cpu", dtype=torch.float32).contiguous()
            B = B_2d.to(device="cpu", dtype=torch.float32).contiguous()

        # store in the format expected by APEX/PEFT loader:
        #   <module>.lora_A.weight  (rank, in) for Linear, or (rank, in, k, k) for Conv2d
        #   <module>.lora_B.weight  (out, rank) for Linear, or (out, rank, 1, 1) for Conv2d
        #   <module>.alpha          scalar
        lora_a_key = w_key.replace(".weight", ".lora_A.weight")
        lora_b_key = w_key.replace(".weight", ".lora_B.weight")
        lora_alpha_key = w_key.replace(".weight", ".alpha")

        lora_sd[lora_a_key] = A
        lora_sd[lora_b_key] = B
        lora_sd[lora_alpha_key] = torch.tensor(alpha_layer, dtype=torch.float32)

        # quick reconstruction check (scaled like loader does)
        scale = alpha_layer / float(rank_used)
        if is_conv2d:
            out_ch, in_ch, kh, kw = map(int, delta.shape)
            is_3x3 = (kh, kw) != (1, 1)
            Bm = B.reshape(out_ch, rank_used)
            Am = (
                A.reshape(rank_used, in_ch * kh * kw)
                if is_3x3
                else A.reshape(rank_used, in_ch)
            )
            approx_m = scale * (Bm @ Am)
            approx = (
                approx_m.reshape(out_ch, in_ch, kh, kw)
                if is_3x3
                else approx_m.reshape(out_ch, in_ch, 1, 1)
            )
        else:
            approx = scale * (B @ A)
        rel_err = (approx - delta.to(approx.device)).norm() / (delta.norm() + 1e-12)
        max_rel_err = max(max_rel_err, float(rel_err))

        processed += 1

    # Probe mode exits early without writing a LoRA.
    if probe_ranks is not None:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".safetensors"):
        metadata = None
        if not args.no_metadata:
            # Minimal Kohya-ish metadata to help tooling; safe for consumers to ignore.
            network_dim_meta = "Dynamic" if args.dynamic_method else str(int(args.rank))
            network_alpha_meta = (
                "Dynamic"
                if args.dynamic_method
                else str(
                    float(args.alpha) if args.alpha is not None else float(args.rank)
                )
            )
            metadata = {
                "ss_network_module": "networks.lora",
                "ss_network_dim": network_dim_meta,
                "ss_network_alpha": network_alpha_meta,
                "apex_base_model": str(args.base_model),
                "apex_base_subfolder": str(args.base_subfolder),
                "apex_new_ckpt": str(args.new_ckpt),
                "apex_model_name": str(args.model_name) if args.model_name else "",
            }
        save_file(lora_sd, args.out, metadata=metadata)
    else:
        torch.save(lora_sd, args.out)

    print(
        "Done.\n"
        f"- wrote: {args.out}\n"
        f"- processed weights: {processed}\n"
        f"- skipped (missing in new): {skipped_missing}\n"
        f"- skipped (shape mismatch / unsupported ndim): {skipped_shape}\n"
        f"- skipped (zero delta): {skipped_zero}\n"
        f"- skipped (max|delta|<=min_diff): {skipped_small}\n"
        f"- worst relative reconstruction error (scaled): {max_rel_err:.6f}\n"
        + (
            f"- alpha={float(args.alpha)} rank={args.rank} (scaling alpha/rank = {float(args.alpha)/float(args.rank):.6f})\n"
            if args.alpha is not None
            else f"- alpha=per-layer effective rank (scaling = 1.0)\n"
        )
    )


if __name__ == "__main__":
    main()
