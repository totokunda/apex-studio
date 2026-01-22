"""Configuration settings for memory management module."""

import dataclasses
from typing import Optional, Union, Dict, Any, List, Literal
import torch


@dataclasses.dataclass
class MemoryConfig:
    """Configuration for memory management system."""

    offload_mode: Literal["group", "budget"] = "group"

    # Group offloading behavior (diffusers-native offloading mechanism)
    group_offload_type: str = "leaf_level"
    group_offload_num_blocks_per_group: Optional[int] = None
    group_offload_use_stream: bool = False
    group_offload_record_stream: bool = False
    group_offload_non_blocking: bool = False
    group_offload_low_cpu_mem_usage: bool = True
    group_offload_offload_device: Union[str, torch.device] = "cpu"
    group_offload_disk_path: Optional[str] = None
    ignore_modules: Optional[List[str]] = None
    block_modules: Optional[List[str]] = None

    # Budget offloading behavior (Apex-native budget manager)
    budget_mb: Optional[Union[int, str]] = None
    async_transfers: bool = True
    prefetch: bool = True
    pin_cpu_memory: bool = False
    vram_safety_coefficient: float = 0.8
    offload_after_forward: bool = False

    def to_group_offload_kwargs(self, onload_device: torch.device) -> Dict[str, Any]:
        """
        Translate this config into keyword arguments expected by `model.enable_group_offload`.
        """
        offload_device = self.group_offload_offload_device
        try:
            if isinstance(offload_device, torch.device):
                resolved_offload_device = offload_device
            else:
                resolved_offload_device = torch.device(offload_device or "cpu")
        except Exception:
            resolved_offload_device = torch.device("cpu")

        kwargs: Dict[str, Any] = {
            "onload_device": onload_device,
            "offload_device": resolved_offload_device,
            "offload_type": self.group_offload_type,
            "non_blocking": self.group_offload_non_blocking,
            "use_stream": self.group_offload_use_stream,
            "record_stream": self.group_offload_record_stream,
            "low_cpu_mem_usage": self.group_offload_low_cpu_mem_usage,
            "ignore_modules": self.ignore_modules,
            "block_modules": self.block_modules,
        }

        if self.group_offload_num_blocks_per_group is not None:
            kwargs["num_blocks_per_group"] = self.group_offload_num_blocks_per_group

        if self.group_offload_disk_path:
            kwargs["offload_to_disk_path"] = self.group_offload_disk_path

        return kwargs

    @classmethod
    def for_block_level(cls) -> "MemoryConfig":
        """Create config optimized for block level offloading."""
        return cls(
            group_offload_type="block_level",
            group_offload_num_blocks_per_group=1,
            group_offload_use_stream=True,
            group_offload_record_stream=True,
            group_offload_non_blocking=True,
            group_offload_low_cpu_mem_usage=True,
        )
