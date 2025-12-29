from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def resolve_module(model: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    """
    Resolve a dotted module path on `model` (supports ModuleList/Sequential numeric indices).
    Returns None if any segment can't be resolved.
    """
    if not dotted_path:
        return None
    cur: nn.Module = model
    for part in dotted_path.split("."):
        # ModuleList/Sequential store children under string indices in `_modules`
        if hasattr(cur, "_modules") and part in cur._modules:  # type: ignore[attr-defined]
            cur = cur._modules[part]  # type: ignore[attr-defined]
            continue
        if part.isdigit() and isinstance(cur, (nn.ModuleList, nn.Sequential)):
            try:
                cur = cur[int(part)]
                continue
            except Exception:
                return None
        if hasattr(cur, part):
            nxt = getattr(cur, part)
            if not isinstance(nxt, nn.Module):
                return None
            cur = nxt
            continue
        return None
    return cur


def remap_embedding_lora_keys(
    state_dict: Dict[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    PEFT uses different parameter names for Embedding LoRA:
      - Linear LoRA:    <mod>.lora_A.weight / <mod>.lora_B.weight
      - Embedding LoRA: <mod>.lora_embedding_A / <mod>.lora_embedding_B

    When an Embedding LoRA checkpoint is stored using linear-style keys, we
    remap it to PEFT's embedding-style keys and drop the trailing ".weight".

    Note: the matrix roles AND orientations differ between linear-style exports and
    PEFT's Embedding adapter parameters, so we swap and transpose:
      - linear-style: lora_A.weight is (r, embed_dim)
      - linear-style: lora_B.weight is (num_embeddings, r)

      - PEFT Embedding params in our runtime expect:
          lora_embedding_A: (r, num_embeddings)
          lora_embedding_B: (embed_dim, r)

    Therefore:
      - lora_embedding_A = lora_B.T
      - lora_embedding_B = lora_A.T
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[-1] == "weight" and parts[-2] in (
            "lora_A",
            "lora_B",
            "lora_down",
            "lora_up",
        ):
            lora_tag = parts[-2]
            module_path = ".".join(parts[:-2])
            mod = resolve_module(model, module_path)
            if isinstance(mod, nn.Embedding):
                if lora_tag in ("lora_A", "lora_down"):
                    # (r, embed_dim) -> (embed_dim, r)
                    new_key = f"{module_path}.lora_embedding_B"
                    out[new_key] = value.transpose(0, 1).contiguous()
                else:
                    # (num_embeddings, r) -> (r, num_embeddings)
                    new_key = f"{module_path}.lora_embedding_A"
                    out[new_key] = value.transpose(0, 1).contiguous()
                continue
        out[key] = value
    return out


