"""
Train a Z-Image LoRA using precomputed VAE + text encodings.

This script expects:
- `vae_encodings.safetensors`: dict[str, Tensor] where each tensor is a normalized VAE latent
  for one image (typically shape (1, C, H, W) or (C, H, W)).
- `text_encodings.safetensors`: dict[str, Tensor] where each tensor is a per-token caption feature
  sequence (shape (seq_len, cap_feat_dim)).

We mirror the Z-Image inference loop (`apps/api/src/engine/zimage/t2i.py`):
- Model input `x` is a list of tensors, each shaped (C, 1, H, W) (frame dim is 1).
- Model input `t` is normalized time in [0, 1], computed as (1000 - timestep) / 1000.
- Model output is negated before being treated as "velocity".

Training objective:
- Sample sigma from the FlowMatch schedule (same scheduler used at inference).
- Use the rectified/flow-matching forward process (matches `ai-toolkit` Z-Image training):
    x_t = (1 - sigma) * x0 + sigma * noise
  where `noise ~ N(0, I)` and `sigma` monotonically decreases from ~1 to 0 at inference.
- The true velocity w.r.t sigma is:
    v = d x_t / d sigma = noise - x0
- Z-Image inference negates the transformer output before passing it to the scheduler
  (`noise_pred = -model_out`), so we train:
    noise_pred ≈ v   ⇔   model_out ≈ -v
""" 

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


# Make apps/api importable
_APPS_DIR = Path(__file__).resolve().parents[2]
_API_DIR = _APPS_DIR / "api"
sys.path.append(str(_API_DIR))

from src.engine import UniversalEngine  # noqa: E402


@dataclass(frozen=True)
class Sample:
    key: str
    latent: torch.Tensor  # (C, H, W) float32 on CPU
    cap: torch.Tensor  # (seq_len, cap_dim) float32/bf16/fp16 on CPU


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_captions_csv(path: str) -> Dict[str, str]:
    """
    Returns mapping: image_path -> caption.
    Accepts either:
    - header: image_path,caption
    - header: image_path,caption_text (etc)
    - no header: first col image_path, second col caption
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        # sniff header
        peek = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(peek)
        except Exception:
            has_header = True

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                img = row.get("image_path") or row.get("image") or row.get("path")
                cap = row.get("caption") or row.get("text") or row.get("prompt")
                if img is None or cap is None:
                    # fall back to first two columns
                    vals = list(row.values())
                    if len(vals) >= 2:
                        img, cap = vals[0], vals[1]
                    else:
                        continue
                mapping[str(img)] = str(cap)
        else:
            reader2 = csv.reader(f)
            rows = list(reader2)
            # skip empty
            for r in rows:
                if not r or len(r) < 2:
                    continue
                mapping[str(r[0])] = str(r[1])

    return mapping


def _to_chw(latent: torch.Tensor) -> torch.Tensor:
    """
    Normalize latent tensor shape to (C, H, W).
    Accepts:
    - (1, C, H, W)
    - (C, H, W)
    - (B, C, H, W) with B==1
    """
    if latent.ndim == 4:
        if latent.shape[0] != 1:
            raise ValueError(f"Expected latent batch dim == 1, got shape {tuple(latent.shape)}")
        latent = latent[0]
    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape (C,H,W), got {tuple(latent.shape)}")
    return latent


def _load_samples(
    vae_path: str,
    text_path: str,
    captions_csv: Optional[str],
) -> List[Sample]:
    vae_enc = load_file(vae_path)
    txt_enc = load_file(text_path)

    # First: direct key intersection
    keys = sorted(set(vae_enc.keys()) & set(txt_enc.keys()))
    samples: List[Sample] = []

    if keys:
        for k in keys:
            latent = _to_chw(vae_enc[k]).detach().cpu().to(torch.float32)
            cap = txt_enc[k].detach().cpu()
            samples.append(Sample(key=k, latent=latent, cap=cap))
        return samples

    # Second: join via captions csv (handles text_encodings keyed by caption text)
    if captions_csv is None:
        raise ValueError(
            "No matching keys between VAE and text encodings. "
            "Pass --captions_csv so we can join image_path -> caption."
        )

    img_to_cap = _parse_captions_csv(captions_csv)
    missing_latent = 0
    missing_text = 0
    for img_path, caption in img_to_cap.items():
        latent = vae_enc.get(img_path)
        if latent is None:
            missing_latent += 1
            continue

        cap = txt_enc.get(img_path)
        if cap is None:
            cap = txt_enc.get(caption)
        if cap is None:
            missing_text += 1
            continue

        samples.append(
            Sample(
                key=img_path,
                latent=_to_chw(latent).detach().cpu().to(torch.float32),
                cap=cap.detach().cpu(),
            )
        )

    if not samples:
        raise ValueError(
            "Could not build any samples. "
            f"missing_latent={missing_latent}, missing_text={missing_text}. "
            "Check that your captions.csv uses the same image_path strings as vae_encodings, "
            "and that text_encodings keys are either image_path or caption text."
        )

    return samples


def _bucket_by_shape(samples: Sequence[Sample]) -> Dict[Tuple[int, int, int], List[int]]:
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, s in enumerate(samples):
        c, h, w = s.latent.shape
        buckets.setdefault((c, h, w), []).append(i)
    return buckets


def _iter_batches(
    buckets: Dict[Tuple[int, int, int], List[int]],
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> Iterable[List[int]]:
    # We keep shapes separate so we can stack (B,C,H,W) efficiently and
    # compute a single `mu` for the batch.
    bucket_items = list(buckets.items())
    if shuffle:
        random.shuffle(bucket_items)

    for _, idxs in bucket_items:
        if shuffle:
            random.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batch = idxs[start : start + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            yield batch


def _set_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    # diffusers ModelMixin usually provides this helper
    if hasattr(model, "enable_gradient_checkpointing"):
        try:
            model.enable_gradient_checkpointing()
            return
        except Exception:
            pass
    # fallback: flip the attribute used by ZImageTransformer2DModel
    if hasattr(model, "gradient_checkpointing"):
        try:
            setattr(model, "gradient_checkpointing", True)
        except Exception:
            pass


def _build_optimizer(name: str, params, lr: float, betas: Tuple[float, float], weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
        except Exception as e:
            raise RuntimeError("adamw8bit requested but bitsandbytes is not available") from e
    raise ValueError(f"Unknown optimizer: {name}")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Optimizer-step scheduler with warmup then decay.
    - warmup: linear ramp 0 -> 1
    - decay: linear to 0 or cosine to 0, or constant
    """
    lr_scheduler = str(lr_scheduler).lower()
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)
    if warmup_steps > total_steps:
        warmup_steps = total_steps

    def lr_lambda(current_step: int) -> float:
        # current_step starts at 0 for the first scheduler.step() call
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        # after warmup
        denom = max(total_steps - warmup_steps, 1)
        progress = float(current_step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        if lr_scheduler == "constant":
            return 1.0
        if lr_scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _try_offload_non_train_components(engine: UniversalEngine) -> None:
    """
    Best-effort: free VRAM by offloading text_encoder + vae after sampling.
    Keeps the transformer resident for training.
    """
    try:
        eng = engine.engine  # BaseEngine instance
    except Exception:
        return

    for name in ("text_encoder", "vae"):
        try:
            if hasattr(eng, "_offload"):
                eng._offload(name, offload_type="discard")
        except Exception:
            pass
        try:
            # break strong references so weights can be freed
            if hasattr(eng, name):
                setattr(eng, name, None)
        except Exception:
            pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _sample_and_save(
    *,
    engine: UniversalEngine,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: str,
    step: int,
    prompt: str,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    # Ensure engine uses the LoRA-wrapped transformer for sampling.
    try:
        engine.engine.transformer = model
    except Exception:
        pass

    g = None
    if seed is not None:
        try:
            g = torch.Generator(device=device).manual_seed(int(seed))
        except Exception:
            g = None

    try:
        with torch.inference_mode():
            # Important: offload=False so we never offload the transformer mid-training.
            img = engine.run(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance),
                generator=g,
                offload=False,
                return_latents=False,
            )

        # zimage t2i returns PIL image; handle list-like just in case.
        if isinstance(img, list) and len(img) > 0:
            img0 = img[0]
        else:
            img0 = img

        out_path = os.path.join(out_dir, f"step_{int(step):06d}.png")
        try:
            img0.save(out_path)
        except Exception:
            # Last resort: skip saving if the object isn't PIL
            return
        tqdm.write(f"[sample] saved {out_path}")
    finally:
        # Keep sampling components lazy: drop text_encoder/vae back out of VRAM.
        _try_offload_non_train_components(engine)
        if was_training:
            model.train()


def main() -> None:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--vae_encodings", type=str, default="vae_encodings.safetensors")
    p.add_argument("--text_encodings", type=str, default="text_encodings.safetensors")
    p.add_argument("--captions_csv", type=str, default=None, help="Optional join file if safetensors keys don't match")

    # Model / engine
    p.add_argument(
        "--manifest",
        type=str,
        default=str(_API_DIR / "manifest" / "image" / "zimage-1.0.0.v1.yml"),
        help="Z-Image manifest yaml",
    )
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--f_patch_size", type=int, default=1)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=None, help="Defaults to lora_rank if unset")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0,w1,w2,w3",
        help="Comma-separated PEFT target module names",
    )

    # Training
    p.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="Name of this run. Outputs are written to lora/<run_name>/",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--caption_dropout",
        type=float,
        default=0.05,
        help="Probability of dropping caption conditioning per-sample (implemented by zeroing cap feats, keeping seq_len).",
    )
    p.add_argument("--max_steps", type=int, default=1000, help="If set >0, stops after this many optimizer steps")
    p.add_argument("--epochs", type=int, default=1, help="Used when max_steps<=0")
    p.add_argument("--num_scheduler_steps", type=int, default=1000, help="How many sigma steps to discretize for sampling")

    # Optimizer + precision
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, choices=["constant", "linear", "cosine"], default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", type=str, choices=["none", "fp16", "bf16"], default="bf16")

    # Logging / saving
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="Optional CSV path for step logs (default: lora/<run_name>/train_log.csv).",
    )
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_only_lora", action="store_true", default=True)

    # Sampling / qualitative monitoring
    p.add_argument("--sample_every", type=int, default=50, help="Generate a sample image every N optimizer steps.")
    p.add_argument("--sample_prompt", type=str, default="A woman with long, wavy hair, wearing a red dress, standing in a field of flowers.")
    p.add_argument("--sample_height", type=int, default=1024)
    p.add_argument("--sample_width", type=int, default=1024)
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--sample_guidance", type=float, default=4.0)
    p.add_argument("--sample_seed", type=int, default=42)

    args = p.parse_args()

    output_dir = os.path.join("lora", str(args.run_name))
    os.makedirs(output_dir, exist_ok=True)
    _seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _load_samples(args.vae_encodings, args.text_encodings, args.captions_csv)
    if len(samples) == 0:
        raise ValueError("No training samples found.")

    # Load ZImage transformer + scheduler via your UniversalEngine so we match config exactly.
    engine = UniversalEngine(yaml_path=args.manifest, components_to_load=["transformer", "scheduler"])
    transformer = engine.transformer
    scheduler = engine.scheduler

    # Enable grad checkpointing early (before PEFT wrapping).
    _set_gradient_checkpointing(transformer, args.gradient_checkpointing)

    # PEFT LoRA
    try:
        from peft import get_peft_model, LoraConfig
    except Exception as e:
        raise RuntimeError("peft is required to train LoRA. Please install peft.") from e

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    print(f"Training LoRA with rank {args.lora_rank}, alpha {lora_alpha}, target modules {target_modules}, dropout {args.lora_dropout}")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        
    )
    model = get_peft_model(transformer, lora_cfg)

    model.to(device)
    model.train()

    # Helpful sanity check: only LoRA params should be trainable.
    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    
    # Mixed precision
    use_autocast = args.mixed_precision != "none" and device.type == "cuda"
    if args.mixed_precision == "fp16":
        amp_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and amp_dtype == torch.float16))

    # Trainable params are only LoRA
    optim = _build_optimizer(
        args.optimizer,
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # Bucket by latent shape for efficient stacking
    buckets = _bucket_by_shape(samples)
    bucket_summary = {k: len(v) for k, v in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))}
    print(f"Loaded {len(samples)} samples across {len(bucket_summary)} latent shapes.")
    # Show top few buckets to explain batching behavior.
    for i, (shape, count) in enumerate(bucket_summary.items()):
        if i >= 6:
            break
        c, h, w = shape
        print(f"  bucket[{i}]: C={c} H={h} W={w} -> {count} samples")

    # Training schedule
    if args.max_steps > 0:
        total_optim_steps = args.max_steps
        total_epochs = 10**9  # effectively infinite; we stop by steps
    else:
        total_epochs = max(int(args.epochs), 1)
        # estimate steps
        num_batches = sum(
            math.ceil(len(idxs) / args.batch_size) for idxs in buckets.values()
        )
        total_optim_steps = math.ceil((num_batches * total_epochs) / args.gradient_accumulation_steps)

    lr_sched = _build_lr_scheduler(
        optim,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.lr_warmup_steps,
        total_steps=total_optim_steps,
    )

    global_step = 0  # optimizer steps
    micro_step = 0   # forward/backward steps
    start_time = time.time()

    progress = tqdm(total=total_optim_steps, desc="Training", dynamic_ncols=True)

    # Step logging
    log_csv_path = args.log_csv or os.path.join(output_dir, "train_log.csv")
    log_f = open(log_csv_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(
        [
            "global_step",
            "epoch",
            "loss",
            "loss_ma",
            "lr",
            "grad_norm",
            "sigma_mean",
            "t_norm_mean",
            "seconds_elapsed",
        ]
    )
    log_f.flush()
    loss_window = deque(maxlen=50)

    # Save samples here
    samples_dir = os.path.join(output_dir, "samples")

    # Qualitative baseline at step 0 (before any optimizer updates)
    if args.sample_every and int(args.sample_every) > 0:
        tqdm.write("[sample] running baseline sample at step 0")
        _sample_and_save(
            engine=engine,
            model=model,
            device=device,
            out_dir=samples_dir,
            step=0,
            prompt=args.sample_prompt,
            height=args.sample_height,
            width=args.sample_width,
            num_steps=args.sample_steps,
            guidance=args.sample_guidance,
            seed=args.sample_seed,
        )

    interrupted = False
    try:
        for epoch in range(total_epochs):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            for batch_idxs in _iter_batches(
                buckets, batch_size=args.batch_size, shuffle=True, drop_last=False
            ):
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

                batch = [samples[i] for i in batch_idxs]

                # Stack latents
                x0 = torch.stack([b.latent for b in batch], dim=0).to(device=device, dtype=torch.float32)  # (B,C,H,W)
                bsz, _, h, w = x0.shape

                # Build mu matching inference for this resolution
                image_seq_len = (h // 2) * (w // 2)
                mu = engine.calculate_shift(
                    image_seq_len,
                    scheduler.config.get("base_image_seq_len", 256),
                    scheduler.config.get("max_image_seq_len", 4096),
                    scheduler.config.get("base_shift", 0.5),
                    scheduler.config.get("max_shift", 1.15),
                )

                # Prepare a discrete sigma/timestep schedule and sample indices from it
                scheduler.sigma_min = 0.0
                scheduler.set_timesteps(args.num_scheduler_steps, device=device, mu=mu)

                # Random step per sample
                t_idx = torch.randint(0, len(scheduler.timesteps), (bsz,), device=device)
                t_int = scheduler.timesteps[t_idx]  # int64

                # scheduler.sigmas is on CPU (by design); index on CPU then move to GPU
                sigma = scheduler.sigmas[t_idx.detach().cpu()].to(device=device, dtype=torch.float32)  # (B,)
                sigma = sigma.view(bsz, 1, 1, 1)

                noise = torch.randn_like(x0, dtype=torch.float32)
                # Rectified / flow-matching noising:
                # x_t = (1 - sigma) * x0 + sigma * noise
                x_t = (1.0 - sigma) * x0 + sigma * noise

                # Model expects normalized t in [0,1] (see zimage/t2i.py)
                t_norm = (1000.0 - t_int.to(torch.float32)) / 1000.0  # (B,)

                # Caption dropout (keep seq_len so image positional ids distribution stays consistent)
                cap_list = []
                if args.caption_dropout and float(args.caption_dropout) > 0.0:
                    p_drop = float(args.caption_dropout)
                else:
                    p_drop = 0.0
                for b in batch:
                    cap = b.cap.to(device=device)
                    if p_drop > 0.0 and random.random() < p_drop:
                        cap = torch.zeros_like(cap)
                    cap_list.append(cap)  # list[(seq,dim)]

                # Forward
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=amp_dtype):
                    x_in = x_t.to(dtype=model.dtype).unsqueeze(2)  # (B,C,1,H,W) in model dtype
                    x_list = list(x_in.unbind(dim=0))
                    
                    out_list = model(
                        x_list,
                        t_norm,
                        cap_list,
                        return_dict=False,
                        patch_size=args.patch_size,
                        f_patch_size=args.f_patch_size,
                    )[0]

                    out = torch.stack([o.to(torch.float32) for o in out_list], dim=0).squeeze(2)  # (B,C,H,W)

                    # Z-Image inference passes velocity = -model_out into the scheduler.
                    # For x_t = (1 - sigma) * x0 + sigma * noise, the true velocity is v = noise - x0.
                    # Therefore we train: (-model_out) ≈ (noise - x0).
                    target_v = (noise - x0).detach()
                    noise_pred = (-out)
                    loss_unscaled = F.mse_loss(noise_pred, target_v)

                    loss = loss_unscaled / max(int(args.gradient_accumulation_steps), 1)

                # Backward
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1

                if micro_step % args.gradient_accumulation_steps == 0:
                    grad_norm = None
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optim)
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
                        try:
                            grad_norm = float(gn)
                        except Exception:
                            grad_norm = None

                    if scaler.is_enabled():
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    lr_sched.step()
                    optim.zero_grad(set_to_none=True)

                    global_step += 1
                    progress.update(1)

                    # Logging (use optimizer-step loss)
                    loss_val = float(loss_unscaled.detach().float().item())
                    loss_window.append(loss_val)
                    loss_ma = sum(loss_window) / max(len(loss_window), 1)
                    lr = float(optim.param_groups[0].get("lr", args.learning_rate))
                    sigma_mean = float(sigma.detach().float().mean().item())
                    t_norm_mean = float(t_norm.detach().float().mean().item())
                    seconds_elapsed = float(time.time() - start_time)

                    if args.log_every and global_step % args.log_every == 0:
                        msg = (
                            f"step={global_step} "
                            f"loss={loss_val:.5f} "
                            f"loss_ma={loss_ma:.5f} "
                            f"lr={lr:.2e} "
                            f"sigma_mean={sigma_mean:.4f} "
                            f"t_mean={t_norm_mean:.4f}"
                        )
                        if grad_norm is not None:
                            msg += f" grad_norm={grad_norm:.2f}"
                        tqdm.write(msg)
                        progress.set_postfix({"loss": f"{loss_val:.4f}", "ma": f"{loss_ma:.4f}"})

                    log_w.writerow(
                        [
                            global_step,
                            epoch,
                            f"{loss_val:.8f}",
                            f"{loss_ma:.8f}",
                            f"{lr:.8e}",
                            "" if grad_norm is None else f"{grad_norm:.6f}",
                            f"{sigma_mean:.8f}",
                            f"{t_norm_mean:.8f}",
                            f"{seconds_elapsed:.3f}",
                        ]
                    )
                    if global_step % 10 == 0:
                        log_f.flush()

                    if args.save_every and global_step % args.save_every == 0:
                        save_dir = os.path.join(output_dir, f"step_{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        # Save adapter weights
                        model.save_pretrained(save_dir)
                        # Save minimal trainer state
                        torch.save(
                            {
                                "global_step": global_step,
                                "micro_step": micro_step,
                                "optimizer": optim.state_dict(),
                                "lr_scheduler": lr_sched.state_dict(),
                                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                                "args": {**vars(args), "output_dir": output_dir},
                            },
                            os.path.join(save_dir, "trainer_state.pt"),
                        )

                    # Sample images every N optimizer steps
                    if args.sample_every and int(args.sample_every) > 0 and (global_step % int(args.sample_every) == 0):
                        _sample_and_save(
                            engine=engine,
                            model=model,
                            device=device,
                            out_dir=samples_dir,
                            step=global_step,
                            prompt=args.sample_prompt,
                            height=args.sample_height,
                            width=args.sample_width,
                            num_steps=args.sample_steps,
                            guidance=args.sample_guidance,
                            seed=args.sample_seed + global_step,
                        )

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write(f"\n[interrupt] Caught KeyboardInterrupt at step={global_step}. Saving checkpoint and exiting...")

    progress.close()
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass

    if interrupted:
        interrupt_dir = os.path.join(output_dir, f"interrupt_step_{global_step}")
        os.makedirs(interrupt_dir, exist_ok=True)
        model.save_pretrained(interrupt_dir)
        torch.save(
            {
                "global_step": global_step,
                "micro_step": micro_step,
                "optimizer": optim.state_dict(),
                "lr_scheduler": lr_sched.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "args": {**vars(args), "output_dir": output_dir},
            },
            os.path.join(interrupt_dir, "trainer_state.pt"),
        )
        return

    # Final save
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(
        {
            "global_step": global_step,
            "micro_step": micro_step,
            "optimizer": optim.state_dict(),
            "lr_scheduler": lr_sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "args": {**vars(args), "output_dir": output_dir},
        },
        os.path.join(final_dir, "trainer_state.pt"),
    )


if __name__ == "__main__":
    main()

