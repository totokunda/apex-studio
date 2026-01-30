"""
Convert a PEFT LoRA safetensors file to ComfyUI LoRA key format.

This script is intended for Z-Image Turbo LoRAs produced by PEFT/diffusers
which commonly look like:
  base_model.model.layers.0.attention.to_q.lora_A.weight
  base_model.model.layers.0.attention.to_q.lora_B.weight

ComfyUI LoRA loader expects:
  layers.0.attention.to_q.lora_down.weight
  layers.0.attention.to_q.lora_up.weight

We therefore:
  - strip the leading `base_model.model.` prefix
  - rename `lora_A` -> `lora_down`
  - rename `lora_B` -> `lora_up`
"""

from __future__ import annotations

import argparse
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


def _convert_key(k: str) -> str:
    if k.startswith("base_model.model."):
        k = k[len("base_model.model.") :]

    # PEFT -> ComfyUI naming
    if k.endswith(".lora_A.weight"):
        return k.replace(".lora_A.weight", ".lora_down.weight")
    if k.endswith(".lora_B.weight"):
        return k.replace(".lora_B.weight", ".lora_up.weight")

    # Fallback (shouldn't happen for PEFT LoRA-only checkpoints)
    return k


def convert_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        k2 = _convert_key(k)
        if k2 in out:
            raise ValueError(f"Key collision after conversion: '{k}' -> '{k2}' already exists")
        out[k2] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input PEFT LoRA safetensors path (e.g. adapter_model.safetensors).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output ComfyUI LoRA safetensors path (e.g. comfyui_adapter_model.safetensors).",
    )
    args = p.parse_args()

    sd = load_file(args.in_path)
    out = convert_state_dict(sd)
    # Keep metadata minimal and explicit.
    save_file(out, args.out_path, metadata={"format": "pt"})

    print(f"Wrote {len(out)} tensors to: {args.out_path}")


if __name__ == "__main__":
    main()

