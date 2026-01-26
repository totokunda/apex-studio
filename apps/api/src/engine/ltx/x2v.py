from typing import Dict, Any, Callable, List, Union, Optional
import torch
import numpy as np
from PIL import Image
import math
from src.transformer.ltx.base.attention import SkipLayerStrategy
from src.scheduler.rf import TimestepShifter
import inspect
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from src.helpers.ltx.patchifier import Patchifier
import copy
from typing import Tuple
from torch.nn import functional as F
from src.engine.base_engine import BaseEngine
from .shared import LTXVideoCondition
from src.utils.progress import safe_emit_progress, make_mapped_progress


class LTXX2VEngine(BaseEngine):
    """LTX Any-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        # Delegate properties to main engine
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None) is not None
            else 32
        )

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None) is not None
            else 8
        )

        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size
            if getattr(self, "transformer", None) is not None
            else 1
        )

        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t
            if getattr(self, "transformer") is not None
            else 1
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        self.num_channels_latents: int = (
            self.vae.config.latent_channels if self.vae is not None else 128
        )

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        conditioning_mask: torch.Tensor | None,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).unsqueeze(-1)
        return torch.where(tokens_to_denoise_mask, denoised_latents, latents)

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    def _get_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.
            max_timestep ('float', *optional*, defaults to 1.0):
                The initial noising level for image-to-image/video-to-video. The list if timestamps will be
                truncated to start with a timestamp greater or equal to this.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor(timesteps, dtype=torch.float32, device=device)
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                timesteps=timesteps.float().cpu(), device=device, **kwargs
            )
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps

            if (
                skip_initial_inference_steps < 0
                or skip_final_inference_steps < 0
                or skip_initial_inference_steps + skip_final_inference_steps
                >= num_inference_steps
            ):
                raise ValueError(
                    "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
                )

            timesteps = timesteps[
                skip_initial_inference_steps : len(timesteps)
                - skip_final_inference_steps
            ]
            scheduler.set_timesteps(
                timesteps=timesteps.float().cpu(), device=device, **kwargs
            )
            num_inference_steps = len(timesteps)

        return timesteps, num_inference_steps

    def _get_latents(
        self,
        height: int,
        width: int,
        duration: int | str,
        fps: int = 16,
        num_videos: int = 1,
        shape: Tuple[int, int, int, int, int] = None,
        num_channels_latents: int = None,
        seed: int | None = None,
        dtype: torch.dtype = None,
        layout: torch.layout = None,
        generator: torch.Generator | None = None,
        return_generator: bool = False,
        parse_frames: bool = True,
    ):
        if parse_frames or isinstance(duration, str):
            num_frames = self._parse_num_frames(duration, fps)

            latent_num_frames = math.ceil(
                (num_frames + 3) / self.vae_scale_factor_temporal
            )
        else:
            latent_num_frames = duration

        latent_height = math.ceil(height / self.vae_scale_factor_spatial)
        latent_width = math.ceil(width / self.vae_scale_factor_spatial)

        if seed is not None and generator is not None:
            self.logger.warning(
                f"Both `seed` and `generator` are provided. `seed` will be ignored."
            )

        if generator is None:
            device = self.device
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
        else:
            device = generator.device

        if shape is not None:
            b, c, f, h, w = shape
        else:
            b, c, f, h, w = (
                num_videos,
                num_channels_latents or self.num_channels_latents,
                latent_num_frames,
                latent_height,
                latent_width,
            )

        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        if return_generator:
            return noise, generator
        else:
            return noise

    def latent_to_pixel_coords(
        self, latent_coords: torch.Tensor, causal_fix: bool = False
    ) -> torch.Tensor:
        """
        Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
        configuration.

        Args:
            latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
            containing the latent corner coordinates of each token.
            vae (AutoencoderKL): The VAE model
            causal_fix (bool): Whether to take into account the different temporal scale
                of the first frame. Default = False for backwards compatibility.
        Returns:
            Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
        """

        scale_factors = (
            self.vae_scale_factor_temporal,
            self.vae_scale_factor_spatial,
            self.vae_scale_factor_spatial,
        )
        pixel_coords = self.latent_to_pixel_coords_from_factors(
            latent_coords, scale_factors, causal_fix
        )
        return pixel_coords

    def latent_to_pixel_coords_from_factors(
        self,
        latent_coords: torch.Tensor,
        scale_factors: Tuple[int, int, int],
        causal_fix: bool = False,
    ) -> torch.Tensor:
        pixel_coords = (
            latent_coords
            * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
        )
        if causal_fix:
            # Fix temporal scale for first frame to 1 due to causality
            pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(
                min=0
            )
        return pixel_coords

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_output(
        self,
        latents: torch.Tensor,
        offload: bool = True,
        generator: torch.Generator | None = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        tone_map_compression_ratio: float = 0.0,
    ):

        batch_size = latents.shape[0]

        if not self.vae:
            self.load_component_by_type("vae")

        self.to_device(self.vae)

        latents = self.vae.denormalize_latents(latents)

        latents = latents.to(self.component_dtypes["vae"])

        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            timestep = torch.tensor(
                decode_timestep, device=self.device, dtype=latents.dtype
            )
            decode_noise_scale = torch.tensor(
                decode_noise_scale, device=self.device, dtype=latents.dtype
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = self.tone_map_latents(latents, tone_map_compression_ratio)
        *_, fl, hl, wl = latents.shape

        decoded_video = self.vae.decode(
            latents,
            target_shape=(
                1,
                3,
                fl * self.vae_scale_factor_temporal,
                hl * self.vae_scale_factor_spatial,
                wl * self.vae_scale_factor_spatial,
            ),
            timestep=timestep,
            return_dict=False,
        )[0]
        video = self._tensor_to_frames(decoded_video)

        if offload:
            self._offload("vae")

        return video

    @staticmethod
    def tone_map_latents(
        latents: torch.Tensor,
        compression: float,
    ) -> torch.Tensor:
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range
        in a perceptually smooth way using a sigmoid-based compression.

        This is useful for regularizing high-variance latents or for conditioning outputs
        during generation, especially when controlling dynamic behavior with a `compression` factor.

        Parameters:
        ----------
        latents : torch.Tensor
            Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
        compression : float
            Compression strength in the range [0, 1].
            - 0.0: No tone-mapping (identity transform)
            - 1.0: Full compression effect

        Returns:
        -------
        torch.Tensor
            The tone-mapped latent tensor of the same shape as input.
        """
        if not (0 <= compression <= 1):
            raise ValueError("Compression must be in the range [0, 1]")

        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered

    def prepare_conditioning(
        self,
        conditioning_items: Optional[List[LTXVideoCondition]],
        patchifier: Patchifier,
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        causal_fix: bool = False,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare conditioning tokens based on the provided conditioning items.

        This method encodes provided conditioning items (video frames or single frames) into latents
        and integrates them with the initial latent tensor. It also calculates corresponding pixel
        coordinates, a mask indicating the influence of conditioning latents, and the total number of
        conditioning latents.

        Args:
            conditioning_items (Optional[List[LTXVideoCondition]]): A list of LTXVideoCondition objects.
            patchifier: The patchifier to use.
            init_latents (torch.Tensor): The initial latent tensor of shape (b, c, f, h, w), where
                `f` is the number of latent frames, and `h` and `w` are latent spatial dimensions.
            num_frames, height, width: The dimensions of the generated video.
            generator: The random generator
            init_latents (torch.Tensor): The initial latent tensor of shape (b, c, f_l, h_l, w_l), where
                `f_l` is the number of latent frames, and `h_l` and `w_l` are latent spatial dimensions.
            num_frames, height, width: The dimensions of the generated video.
                Defaults to `False`.
            generator: The random generator

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - `init_latents` (torch.Tensor): The updated latent tensor including conditioning latents,
                  patchified into (b, n, c) shape.
                - `init_pixel_coords` (torch.Tensor): The pixel coordinates corresponding to the updated
                  latent tensor.
                - `conditioning_mask` (torch.Tensor): A mask indicating the conditioning-strength of each
                  latent token.
                - `num_cond_latents` (int): The total number of latent tokens added from conditioning items.

        Raises:
            AssertionError: If input shapes, dimensions, or conditions for applying conditioning are invalid.
        """

        if conditioning_items:
            batch_size, _, num_latent_frames = init_latents.shape[:3]

            init_conditioning_mask = torch.zeros(
                init_latents[:, 0, :, :, :].shape,
                dtype=torch.float32,
                device=init_latents.device,
            )

            extra_conditioning_latents = []
            extra_conditioning_pixel_coords = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0  # Number of extra conditioning latents added (should be removed before decoding)

            # Process each conditioning item
            for conditioning_item in conditioning_items:

                conditioning_item = self.resize_conditioning_item(
                    conditioning_item, height, width
                )
                media_item = conditioning_item.media_item
                media_frame_number = conditioning_item.frame_number
                strength = conditioning_item.conditioning_strength
                assert media_item.ndim == 5  # (b, c, f, h, w)
                b, c, n_frames, h, w = media_item.shape
                assert (
                    height == h and width == w
                ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {h}x{w} - allowed only when media_frame_number == 0"
                assert n_frames % 8 == 1
                assert (
                    media_frame_number >= 0
                    and media_frame_number + n_frames <= num_frames
                )

                # Encode the provided conditioning media item
                media_item_latents = self.vae_encode(
                    media_item, dtype=init_latents.dtype, sample_mode="mode"
                )

                # Handle the different conditioning cases
                if media_frame_number == 0:
                    # Get the target spatial position of the latent conditioning item
                    media_item_latents, l_x, l_y = self._get_latent_spatial_position(
                        media_item_latents,
                        conditioning_item,
                        height,
                        width,
                        strip_latent_border=True,
                    )
                    b, c_l, f_l, h_l, w_l = media_item_latents.shape

                    # First frame or sequence - just update the initial noise latents and the mask
                    init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
                        torch.lerp(
                            init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
                            media_item_latents,
                            strength,
                        )
                    )
                    init_conditioning_mask[
                        :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
                    ] = strength
                else:
                    # Non-first frame or sequence
                    if n_frames > 1:
                        # Handle non-first sequence.
                        # Encoded latents are either fully consumed, or the prefix is handled separately below.
                        (
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                        ) = self._handle_non_first_conditioning_sequence(
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                            media_frame_number,
                            strength,
                        )

                    # Single frame or sequence-prefix latents
                    if media_item_latents is not None:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )

                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                        # Patchify the extra conditioning latents and calculate their pixel coordinates
                        media_item_latents, latent_coords = patchifier.patchify(
                            latents=media_item_latents
                        )
                        pixel_coords = self.latent_to_pixel_coords(
                            latent_coords,
                            causal_fix=causal_fix,
                        )

                        # Update the frame numbers to match the target frame number
                        pixel_coords[:, 0] += media_frame_number
                        extra_conditioning_num_latents += media_item_latents.shape[1]

                        conditioning_mask = torch.full(
                            media_item_latents.shape[:2],
                            strength,
                            dtype=torch.float32,
                            device=init_latents.device,
                        )

                        extra_conditioning_latents.append(media_item_latents)
                        extra_conditioning_pixel_coords.append(pixel_coords)
                        extra_conditioning_mask.append(conditioning_mask)

        # Patchify the updated latents and calculate their pixel coordinates
        init_latents, init_latent_coords = patchifier.patchify(latents=init_latents)

        init_pixel_coords = self.latent_to_pixel_coords(
            init_latent_coords,
            causal_fix=causal_fix,
        )

        if not conditioning_items:
            return init_latents, init_pixel_coords, None, 0

        init_conditioning_mask, _ = patchifier.patchify(
            latents=init_conditioning_mask.unsqueeze(1)
        )

        init_conditioning_mask = init_conditioning_mask.squeeze(-1)

        if extra_conditioning_latents:
            # Stack the extra conditioning latents, pixel coordinates and mask
            init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
            init_pixel_coords = torch.cat(
                [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
            )
            init_conditioning_mask = torch.cat(
                [*extra_conditioning_mask, init_conditioning_mask], dim=1
            )

            if self.transformer.use_tpu_flash_attention:
                # When flash attention is used, keep the original number of tokens by removing
                #   tokens from the end.
                init_latents = init_latents[:, :-extra_conditioning_num_latents]
                init_pixel_coords = init_pixel_coords[
                    :, :, :-extra_conditioning_num_latents
                ]
                init_conditioning_mask = init_conditioning_mask[
                    :, :-extra_conditioning_num_latents
                ]

        return (
            init_latents,
            init_pixel_coords,
            init_conditioning_mask,
            extra_conditioning_num_latents,
        )

    def _get_latent_spatial_position(
        self,
        latents: torch.Tensor,
        conditioning_item: LTXVideoCondition,
        height: int,
        width: int,
        strip_latent_border,
    ):
        """
        Get the spatial position of the conditioning item in the latent space.
        If requested, strip the conditioning latent borders that do not align with target borders.
        (border latents look different then other latents and might confuse the model)
        """
        scale = self.vae_scale_factor_spatial
        h, w = conditioning_item.media_item.shape[-2:]
        assert (
            h <= height and w <= width
        ), f"Conditioning item size {h}x{w} is larger than target size {height}x{width}"
        assert h % scale == 0 and w % scale == 0

        # Compute the start and end spatial positions of the media item
        x_start, y_start = conditioning_item.media_x, conditioning_item.media_y
        x_start = (width - w) // 2 if x_start is None else x_start
        y_start = (height - h) // 2 if y_start is None else y_start
        x_end, y_end = x_start + w, y_start + h
        assert (
            x_end <= width and y_end <= height
        ), f"Conditioning item {x_start}:{x_end}x{y_start}:{y_end} is out of bounds for target size {width}x{height}"

        if strip_latent_border:
            # Strip one latent from left/right and/or top/bottom, update x, y accordingly
            if x_start > 0:
                x_start += scale
                latents = latents[:, :, :, :, 1:]

            if y_start > 0:
                y_start += scale
                latents = latents[:, :, :, 1:, :]

            if x_end < width:
                latents = latents[:, :, :, :, :-1]

            if y_end < height:
                latents = latents[:, :, :, :-1, :]

        return latents, x_start // scale, y_start // scale

    @staticmethod
    def _handle_non_first_conditioning_sequence(
        init_latents: torch.Tensor,
        init_conditioning_mask: torch.Tensor,
        latents: torch.Tensor,
        media_frame_number: int,
        strength: float,
        num_prefix_latent_frames: int = 2,
        prefix_latents_mode: str = "concat",
        prefix_soft_conditioning_strength: float = 0.15,
    ):
        """
        Special handling for a conditioning sequence that does not start on the first frame.
        The special handling is required to allow a short encoded video to be used as middle
        (or last) sequence in a longer video.
        Args:
            init_latents (torch.Tensor): The initial noise latents to be updated.
            init_conditioning_mask (torch.Tensor): The initial conditioning mask to be updated.
            latents (torch.Tensor): The encoded conditioning item.
            media_frame_number (int): The target frame number of the first frame in the conditioning sequence.
            strength (float): The conditioning strength for the conditioning latents.
            num_prefix_latent_frames (int, optional): The length of the sequence prefix, to be handled
                separately. Defaults to 2.
            prefix_latents_mode (str, optional): Special treatment for prefix (boundary) latents.
                - "drop": Drop the prefix latents.
                - "soft": Use the prefix latents, but with soft-conditioning
                - "concat": Add the prefix latents as extra tokens (like single frames)
            prefix_soft_conditioning_strength (float, optional): The strength of the soft-conditioning for
                the prefix latents, relevant if `prefix_latents_mode` is "soft". Defaults to 0.1.

        """
        f_l = latents.shape[2]
        f_l_p = num_prefix_latent_frames
        assert f_l >= f_l_p
        assert media_frame_number % 8 == 0
        if f_l > f_l_p:
            # Insert the conditioning latents **excluding the prefix** into the sequence
            f_l_start = media_frame_number // 8 + f_l_p
            f_l_end = f_l_start + f_l - f_l_p
            init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                init_latents[:, :, f_l_start:f_l_end],
                latents[:, :, f_l_p:],
                strength,
            )
            # Mark these latent frames as conditioning latents
            init_conditioning_mask[:, f_l_start:f_l_end] = strength

        # Handle the prefix-latents
        if prefix_latents_mode == "soft":
            if f_l_p > 1:
                # Drop the first (single-frame) latent and soft-condition the remaining prefix
                f_l_start = media_frame_number // 8 + 1
                f_l_end = f_l_start + f_l_p - 1
                strength = min(prefix_soft_conditioning_strength, strength)
                init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                    init_latents[:, :, f_l_start:f_l_end],
                    latents[:, :, 1:f_l_p],
                    strength,
                )
                # Mark these latent frames as conditioning latents
                init_conditioning_mask[:, f_l_start:f_l_end] = strength
            latents = None  # No more latents to handle
        elif prefix_latents_mode == "drop":
            # Drop the prefix latents
            latents = None
        elif prefix_latents_mode == "concat":
            # Pass-on the prefix latents to be handled as extra conditioning frames
            latents = latents[:, :, :f_l_p]
        else:
            raise ValueError(f"Invalid prefix_latents_mode: {prefix_latents_mode}")
        return (
            init_latents,
            init_conditioning_mask,
            latents,
        )

    @staticmethod
    def resize_conditioning_item(
        conditioning_item: LTXVideoCondition,
        height: int,
        width: int,
    ):
        if conditioning_item.media_x or conditioning_item.media_y:
            raise ValueError(
                "Provide media_item in the target size for spatial conditioning."
            )
        new_conditioning_item = copy.copy(conditioning_item)
        new_conditioning_item.media_item = LTXX2VEngine.resize_tensor(
            conditioning_item.media_item, height, width
        )
        return new_conditioning_item

    @staticmethod
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)
        return media_items

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        conditions: List[LTXVideoCondition] | LTXVideoCondition | None = None,
        initial_latents: Optional[torch.Tensor] = None,
        initial_image: Optional[Image.Image] = None,
        initial_video: Optional[List[Image.Image] | torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 25,
        num_inference_steps: int = 30,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        num_videos: int = 1,
        seed: int | None = None,
        eta: float = 0.0,
        fps: int = 30,
        text_encoder_kwargs: Dict[str, Any] = {},
        guidance_scale: float | List[float] = 3.0,
        stg_scale: float | List[float] = 1.0,
        rescaling_scale: float | List[float] = 1.0,
        image_cond_noise_scale: float = 0.15,
        offload: bool = True,
        render_on_step: bool = False,
        progress_callback: Callable | None = None,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        guidance_timesteps: List[int] | None = None,
        render_on_step_callback: Callable = None,
        cfg_star_rescale: bool = False,
        return_latents: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        stochastic_sampling: bool = False,
        tone_map_compression_ratio: float = 0.0,
        skip_block_list: Optional[List[List[int]]] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] | Optional[str] = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting LTX Any-to-Video pipeline")

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        if skip_layer_strategy is not None:
            if isinstance(skip_layer_strategy, str):
                skip_layer_strategy = SkipLayerStrategy(skip_layer_strategy)

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            return_attention_mask=True,
            **text_encoder_kwargs,
        )

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        if negative_prompt:
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    return_attention_mask=True,
                    **text_encoder_kwargs,
                )
            )
        else:
            negative_prompt_embeds, negative_prompt_attention_mask = torch.zeros_like(
                prompt_embeds
            ), torch.zeros_like(prompt_attention_mask)

        safe_emit_progress(
            progress_callback,
            0.13,
            (
                "Prepared negative prompt embeds"
                if negative_prompt
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        # load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        batch_size = prompt_embeds.shape[0]

        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )

        prompt_attention_mask_batch = torch.cat(
            [
                negative_prompt_attention_mask,
                prompt_attention_mask,
                prompt_attention_mask,
            ],
            dim=0,
        )

        if initial_image is not None:
            initial_image = self._load_image(initial_image)
            initial_image, height, width = self._aspect_ratio_resize(
                initial_image,
                max_area=height * width,
                mod_value=self.vae_scale_factor_spatial,
            )
            video_input = self.video_processor.preprocess(
                initial_image, height, width
            ).unsqueeze(2)
        elif initial_video is not None:
            video_input = self._load_video(initial_video, fps=fps)
            for i, frame in enumerate(video_input):
                frame, height, width = self._aspect_ratio_resize(
                    frame,
                    max_area=height * width,
                    mod_value=self.vae_scale_factor_spatial,
                )
                video_input[i] = frame
            video_input = self.video_processor.preprocess_video(
                video_input, height, width
            )
        else:
            video_input = None

        if video_input is not None:
            latent_input = self.vae_encode(
                video_input,
                sample_mode="mode",
                normalize_latents_dtype=torch.float32,
                offload=offload,
            )
            shape = latent_input.shape
        elif initial_latents is not None:
            latent_input = initial_latents
            shape = latent_input.shape
        else:
            latent_input = None
            shape = None

        num_frames = self._parse_num_frames(duration, fps)
        noise = self._get_latents(
            height,
            width,
            num_frames,
            fps,
            batch_size,
            shape=shape,
            dtype=transformer_dtype,
            seed=seed,
            generator=generator,
            parse_frames=(video_input is None and initial_latents is None),
        )
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            noise = noise * self.scheduler.init_noise_sigma

        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = noise.shape

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            skip_initial_inference_steps=skip_initial_inference_steps,
            skip_final_inference_steps=skip_final_inference_steps,
            **retrieve_timesteps_kwargs,
        )

        if latent_input is not None:
            latents = timesteps[0] * noise + (1 - timesteps[0]) * latent_input
        else:
            latents = noise

        latent_height = latents.shape[3]
        latent_width = latents.shape[4]

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if len(skip_block_list) == 0 or not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        # patch latents
        patchifier = self.helpers["ltx.patchifier"]
        causal_fix = getattr(
            self.transformer.config, "causal_temporal_positioning", False
        )

        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=conditions,
                causal_fix=causal_fix,
                patchifier=patchifier,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            )
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        init_latents = latents.clone()  # Used for image_cond_noise_update
        orig_conditioning_mask = conditioning_mask

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(progress_callback, 0.45, "Starting denoise phase")

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                do_classifier_free_guidance = guidance_scale[i] > 1.0
                do_spatio_temporal_guidance = stg_scale[i] > 0
                do_rescaling = rescaling_scale[i] != 1.0

                num_conds = 1
                if do_classifier_free_guidance:
                    num_conds += 1
                if do_spatio_temporal_guidance:
                    num_conds += 1

                if do_classifier_free_guidance and do_spatio_temporal_guidance:
                    indices = slice(num_videos * 0, num_videos * 3)
                elif do_classifier_free_guidance:
                    indices = slice(num_videos * 0, num_videos * 2)
                elif do_spatio_temporal_guidance:
                    indices = slice(num_videos * 1, num_videos * 3)
                else:
                    indices = slice(num_videos * 1, num_videos * 2)

                skip_layer_mask: Optional[torch.Tensor] = None
                if do_spatio_temporal_guidance:
                    if skip_block_list is not None:
                        skip_layer_mask = self.transformer.create_skip_layer_mask(
                            num_videos, num_conds, num_conds - 1, skip_block_list[i]
                        )

                batch_pixel_coords = torch.cat([pixel_coords] * num_conds)
                conditioning_mask = orig_conditioning_mask

                if conditioning_mask is not None:
                    assert num_videos == 1
                    conditioning_mask = torch.cat([conditioning_mask] * num_conds)
                fractional_coords = batch_pixel_coords.to(torch.float32)
                fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / fps)

                if conditioning_mask is not None and image_cond_noise_scale > 0.0:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                if conditioning_mask is not None:
                    # Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
                    # and will start to be denoised when the current timestep is lower than their conditioning timestep.
                    current_timestep = torch.min(
                        current_timestep, 1.0 - conditioning_mask
                    )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input.to(transformer_dtype),
                    video_coords=fractional_coords,
                    encoder_hidden_states=prompt_embeds_batch[indices].to(
                        transformer_dtype
                    ),
                    encoder_attention_mask=prompt_attention_mask_batch[indices].to(
                        self.device
                    ),
                    timestep=current_timestep,
                    skip_layer_mask=skip_layer_mask,
                    skip_layer_strategy=skip_layer_strategy,
                    return_dict=False,
                )[0]

                init_noise_pred = noise_pred.clone()

                # perform guidance
                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(
                        num_conds
                    )[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]

                    if cfg_star_rescale:
                        # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                        # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                        # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                        positive_flat = noise_pred_text.view(num_videos, -1)
                        negative_flat = noise_pred_uncond.view(num_videos, -1)
                        dot_product = torch.sum(
                            positive_flat * negative_flat, dim=1, keepdim=True
                        )
                        squared_norm = (
                            torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        )
                        alpha = dot_product / squared_norm
                        noise_pred_uncond = alpha * noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale[i] * (
                        noise_pred_text - noise_pred_uncond
                    )
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale[i] * (
                        noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling and stg_scale[i] > 0.0:
                        noise_pred_text_std = noise_pred_text.view(num_videos, -1).std(
                            dim=1, keepdim=True
                        )
                        noise_pred_std = noise_pred.view(num_videos, -1).std(
                            dim=1, keepdim=True
                        )

                        factor = noise_pred_text_std / noise_pred_std
                        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                        noise_pred = noise_pred * factor.view(num_videos, 1, 1)

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    orig_conditioning_mask,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    # Emit progress during denoising
                    step_progress = (i + 1) / len(timesteps)
                    safe_emit_progress(
                        denoise_progress_callback,
                        step_progress,
                        f"Denoising step {i + 1}/{len(timesteps)}",
                    )

                # if render_on_step and render_on_step_callback and i != len(timesteps) - 1:
                #     self._render_step(latents, render_on_step_callback)

        if offload:
            self._offload("transformer")

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        latents = latents[:, num_cond_latents:]

        latents = patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(patchifier.patch_size),
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.95, "Decoding latents to video")
        output = self.prepare_output(
            latents=latents,
            offload=offload,
            generator=generator,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            tone_map_compression_ratio=tone_map_compression_ratio,
        )
        safe_emit_progress(
            progress_callback, 1.0, "Completed LTX Any-to-Video pipeline"
        )
        return output
