"""
EasyCache implementation for MOVA dual-tower model.

This module implements EasyCache acceleration for the MOVA video-audio generation model.
EasyCache works by:
1. Tracking input changes between timesteps
2. Using a learned scaling factor (k) to predict output changes from input changes  
3. Caching residuals (output - input) and reusing them when accumulated error is below threshold
4. Separately tracking conditional (even) and unconditional (odd) CFG paths

Reference: https://github.com/Wan-Video/Wan2.2/blob/main/easycache_wan.py
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch

from src.utils.step_mem import step_mem

@dataclass
class MOVAEasyCacheState:
    """
    State management for MOVA EasyCache.
    
    Tracks cache state for both visual and audio branches across 
    conditional (even) and unconditional (odd) classifier-free guidance steps.
    """
    # Step counter
    cnt: int = 0
    num_steps: int = 100  # Total number of inference calls (steps * 2 for CFG)
    
    # Caching parameters
    thresh: float = 0.05  # Error threshold for cache reuse decision
    ret_steps: int = 14   # Number of initial steps to retain (compute fully)
    
    # Error tracking
    accumulated_error_even: float = 0.0
    k: Optional[float] = None  # Output/input change ratio
    
    # Decision flag
    should_calc_current_pair: bool = True
    
    # Visual cache - conditional (even) path
    prev_visual_input_even: Optional[torch.Tensor] = None
    prev_prev_visual_input_even: Optional[torch.Tensor] = None
    prev_visual_output_even: Optional[torch.Tensor] = None
    visual_cache_even: Optional[torch.Tensor] = None
    
    # Visual cache - unconditional (odd) path
    prev_visual_output_odd: Optional[torch.Tensor] = None
    visual_cache_odd: Optional[torch.Tensor] = None
    
    # Audio cache - conditional (even) path
    prev_audio_input_even: Optional[torch.Tensor] = None
    prev_prev_audio_input_even: Optional[torch.Tensor] = None
    prev_audio_output_even: Optional[torch.Tensor] = None
    audio_cache_even: Optional[torch.Tensor] = None
    
    # Audio cache - unconditional (odd) path
    prev_audio_output_odd: Optional[torch.Tensor] = None
    audio_cache_odd: Optional[torch.Tensor] = None
    
    def reset(self, num_steps: int, thresh: float = 0.05, ret_steps: int = 7):
        """Reset cache state for a new generation."""
        self.cnt = 0
        self.num_steps = num_steps * 2  # Double for CFG
        self.thresh = thresh
        self.ret_steps = ret_steps * 2  # Double for CFG
        self.accumulated_error_even = 0.0
        self.k = None
        self.should_calc_current_pair = True
        
        # Clear visual cache
        self.prev_visual_input_even = None
        self.prev_prev_visual_input_even = None
        self.prev_visual_output_even = None
        self.visual_cache_even = None
        self.prev_visual_output_odd = None
        self.visual_cache_odd = None
        
        # Clear audio cache
        self.prev_audio_input_even = None
        self.prev_prev_audio_input_even = None
        self.prev_audio_output_even = None
        self.audio_cache_even = None
        self.prev_audio_output_odd = None
        self.audio_cache_odd = None
    
    @property
    def is_even(self) -> bool:
        """Check if current step is conditional (even) or unconditional (odd)."""
        return self.cnt % 2 == 0
    
    def should_use_cache(self) -> bool:
        """Determine if we can use cached output."""
        if self.is_even:
            return (
                not self.should_calc_current_pair
                and self.prev_visual_output_even is not None
                and self.prev_audio_output_even is not None
            )
        else:
            return (
                not self.should_calc_current_pair
                and self.prev_visual_output_odd is not None
                and self.prev_audio_output_odd is not None
            )
    
    def get_cached_output(
        self,
        visual_input: torch.Tensor,
        audio_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached output by applying cached residuals to current input.
        
        Args:
            visual_input: Current visual latent input
            audio_input: Current audio latent input
            
        Returns:
            Tuple of (visual_output, audio_output)
        """
        if self.is_even:
            visual_output = visual_input + self.visual_cache_even
            audio_output = audio_input + self.audio_cache_even
        else:
            visual_output = visual_input + self.visual_cache_odd
            audio_output = audio_input + self.audio_cache_odd
        
        self.cnt += 1
        return visual_output.float(), audio_output.float()
    
    def decide_computation(
        self,
        visual_input: torch.Tensor,
        audio_input: torch.Tensor,
    ):
        """
        Decide whether to compute or use cache for current step.
        Only makes decisions on even (conditional) steps; odd steps follow even's decision.
        
        Args:
            visual_input: Current visual latent input
            audio_input: Current audio latent input
        """
        if not self.is_even:
            # Odd steps follow the decision made on even steps
            return
        
        # Always compute first ret_steps and last few steps
        if self.cnt < self.ret_steps or self.cnt >= (self.num_steps - 2):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
        else:
            # Check if we have previous data for comparison
            if (
                self.prev_visual_input_even is not None
                and self.prev_visual_output_even is not None
                and self.prev_audio_input_even is not None
                and self.prev_audio_output_even is not None
            ):
                # Calculate input change (combined visual + audio)
                visual_input_change = (visual_input - self.prev_visual_input_even).abs().mean()
                audio_input_change = (audio_input - self.prev_audio_input_even).abs().mean()
                raw_input_change = visual_input_change + audio_input_change
                
                if self.k is not None:
                    # Calculate output norm for relative comparison
                    visual_output_norm = self.prev_visual_output_even.abs().mean()
                    audio_output_norm = self.prev_audio_output_even.abs().mean()
                    output_norm = visual_output_norm + audio_output_norm
                    
                    # Predict change based on learned k factor
                    pred_change = self.k * (raw_input_change / output_norm)
                    self.accumulated_error_even += pred_change.item()
                    
                    # Decide based on accumulated error
                    if self.accumulated_error_even < self.thresh:
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0.0
                else:
                    # No k factor yet, must calculate
                    self.should_calc_current_pair = True
            else:
                # No previous data, must calculate
                self.should_calc_current_pair = True
        
        # Store current input state (only on even steps)
        self.prev_visual_input_even = visual_input.clone()
        self.prev_audio_input_even = audio_input.clone()
    
    def update_cache(
        self,
        visual_input: torch.Tensor,
        visual_output: torch.Tensor,
        audio_input: torch.Tensor,
        audio_output: torch.Tensor,
    ):
        """
        Update cache with computed outputs and calculate k factor.
        
        Args:
            visual_input: Visual latent input
            visual_output: Visual latent output
            audio_input: Audio latent input
            audio_output: Audio latent output
        """
        if self.is_even:
            # Conditional path - update k factor if we have previous output
            if (
                self.prev_visual_output_even is not None
                and self.prev_audio_output_even is not None
            ):
                # Calculate output change
                visual_output_change = (visual_output - self.prev_visual_output_even).abs().mean()
                audio_output_change = (audio_output - self.prev_audio_output_even).abs().mean()
                output_change = visual_output_change + audio_output_change
                
                # Calculate input change from previous inputs
                if (
                    self.prev_prev_visual_input_even is not None
                    and self.prev_prev_audio_input_even is not None
                ):
                    visual_input_change = (
                        self.prev_visual_input_even - self.prev_prev_visual_input_even
                    ).abs().mean()
                    audio_input_change = (
                        self.prev_audio_input_even - self.prev_prev_audio_input_even
                    ).abs().mean()
                    input_change = visual_input_change + audio_input_change
                    
                    if input_change > 0:
                        self.k = (output_change / input_change).item()
            
            # Update history
            self.prev_prev_visual_input_even = self.prev_visual_input_even
            self.prev_prev_audio_input_even = self.prev_audio_input_even
            
            # Store current output
            self.prev_visual_output_even = visual_output.clone()
            self.prev_audio_output_even = audio_output.clone()
            
            # Store residual cache
            self.visual_cache_even = visual_output - visual_input
            self.audio_cache_even = audio_output - audio_input
        else:
            # Unconditional path - just store output and cache
            self.prev_visual_output_odd = visual_output.clone()
            self.prev_audio_output_odd = audio_output.clone()
            self.visual_cache_odd = visual_output - visual_input
            self.audio_cache_odd = audio_output - audio_input
        
        self.cnt += 1


def create_easycache_state(
    num_steps: int,
    thresh: float = 0.05,
    ret_steps: int = 7,
) -> MOVAEasyCacheState:
    """
    Factory function to create a new EasyCache state.
    
    Args:
        num_steps: Number of inference steps
        thresh: Error threshold for cache decisions (default: 0.05)
        ret_steps: Number of initial steps to always compute (default: 7)
        
    Returns:
        Initialized MOVAEasyCacheState
    """
    state = MOVAEasyCacheState()
    state.reset(num_steps=num_steps, thresh=thresh, ret_steps=ret_steps)
    return state
