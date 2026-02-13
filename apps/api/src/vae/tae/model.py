#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple
from safetensors.torch import load_file
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import os
from diffusers.models.modeling_outputs import AutoencoderKLOutput
# create an attribute dict 


class FakeDiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.parameters = parameters
    def sample(self):
        return self.parameters
    def mode(self):
        return self.parameters
    def kl(self, other=None):
        return 0.0

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with parallelization over the time axis and iteration over blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N*T, C, H, W)

    # parallel over input timesteps, iterate over blocks
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)

def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i+1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N*b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i+1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT//b.stride, b.stride*C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i+1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i+1))
    return None

def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with iteration over timesteps as well as blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)

class TAEHV(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, checkpoint_path=None, encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16, latents_mean=None, latents_std=None):
        """Initialize pretrained TAEHV from the given checkpoint.

        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
        """
        
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        if checkpoint_path is not None and "taew2_2" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 48
        if checkpoint_path is not None and "taehv1_5" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 32
        if checkpoint_path is not None and "taeltx_2" in checkpoint_path:
            self.patch_size, self.latent_channels, encoder_time_downscale, decoder_time_upscale = 4, 128, (True, True, True), (True, True, True)
        self.encoder = nn.Sequential(
            conv(self.image_channels*self.patch_size**2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.image_channels*self.patch_size**2),
        )
        # computed properties
        self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2**sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            if checkpoint_path.endswith(".safetensors"):
                model = load_file(checkpoint_path, device="cpu")
            else:
                model = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(self.patch_tgrow_layers(model))

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1: x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1: x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=self.latent_channels) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        # check if x needs to be transposed
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        if skip_trim:
            # skip trimming for cogvideox to make frame counts match.
            # this still doesn't have correct temporal alignment for certain frame counts
            # (cogvideox seems to pad at the start?), but for multiple-of-4 it's fine.
            return x

        output_x = x[:, self.frames_to_trim:]
        return output_x
    
    @torch.no_grad()
    def denormalize_latents(self, latents: torch.Tensor):
        if self.latents_mean is None:
            return latents
        latents_mean = latents_mean = (
            torch.tensor(self.config.latents_mean)
            .view(1, self.config.latent_channels, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.config.latents_std).view(
            1, self.config.latent_channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents / latents_std) + latents_mean
        return latents

    @torch.no_grad()
    def normalize_latents(self, latents: torch.Tensor):
        if self.latents_mean is None:
            return latents
        latents_mean = (
            torch.tensor(self.config.latents_mean)
            .view(1, self.config.latent_channels, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.config.latents_std).view(
            1, self.config.latent_channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
        return latents
    
    def decode(self, latents, return_dict=None, cond=None):
        return (self.decode_video(latents.transpose(1, 2), parallel=False).transpose(1, 2).mul_(2).sub_(1),)
    
    def encode(self, x, return_dict=None):
        output_h = self.encode_video(x.transpose(1, 2), parallel=False).transpose(1, 2).contiguous()
        posterior = FakeDiagonalGaussianDistribution(output_h)
        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)
        else:
            return (posterior,)
