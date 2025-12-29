import torch
from typing import Literal, List
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from src.postprocess.base import (
    BasePostprocessor,
    PostprocessorCategory,
    postprocessor_registry,
)
from src.utils.cache import empty_cache
import numpy as np
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor


@postprocessor_registry("ltx.latent_upsampler")
class LatentUpsamplerPostprocessor(BasePostprocessor):
    def __init__(self, engine, **kwargs):
        super().__init__(engine, PostprocessorCategory.UPSCALER, **kwargs)

        # Get configuration from component_conf
        self.config = self.component_conf
        self.device = engine.device
        self.component_dtypes = getattr(engine, "component_dtypes", {})

        # Default dtype for latent upsampler
        self.dtype = kwargs.get("dtype", torch.float32)

        # Initialize the latent upsampler model
        self.latent_upsampler = None
        self._load_latent_upsampler()

    def _load_latent_upsampler(self):
        """Load the latent upsampler model following engine patterns"""
        try:
            # Check if model_path is provided
            model_path = self.config.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for latent upsampler")

            model_path = self._download(model_path, DEFAULT_PREPROCESSOR_SAVE_PATH)
            # Get configuration
            config_path = self.config.get("config_path")

            upsampler_config = self.config.get("config", {})

            if config_path:
                fetched_config = self.engine.fetch_config(config_path)
                upsampler_config = {**fetched_config, **upsampler_config}

            self.engine.logger.info(f"Loading latent upsampler from {model_path}")

            # Load model using proper loading mechanics
            if upsampler_config:
                # Load with custom config
                self.latent_upsampler = self._load_model(
                    component={
                        "base": "LTXLatentUpsamplerModel",
                        "model_path": model_path,
                        "config": upsampler_config,
                        "type": "latent_upsampler",
                    },
                    module_name="diffusers.pipelines.ltx.modeling_latent_upsampler",
                    load_dtype=self.dtype,
                )
            else:
                # Load using from_pretrained
                self.latent_upsampler = LTXLatentUpsamplerModel.from_pretrained(
                    model_path, torch_dtype=self.dtype
                )

            # Move to device and set dtype
            self.latent_upsampler = self.latent_upsampler.to(
                device=self.device, dtype=self.dtype
            )

            self.engine.logger.info("Latent upsampler loaded successfully")
            empty_cache()

        except Exception as e:
            self.engine.logger.error(f"Failed to load latent upsampler: {e}")
            raise

    def _upsample_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Perform latent upsampling using the loaded model.

        Note: The LTX latent upsampler expects denormalized latents and returns upsampled latents directly.
        """

        # Match the upsampler's parameter dtype
        try:
            upsampler_dtype = next(self.latent_upsampler.parameters()).dtype
        except Exception:
            upsampler_dtype = self.dtype

        latents = latents.to(device=self.device, dtype=upsampler_dtype)

        with torch.no_grad():
            upsampled_latents = self.latent_upsampler(latents)

        return upsampled_latents

    def _adain_filter_latent(
        self,
        latents: torch.Tensor,
        reference_latents: torch.Tensor,
        factor: float = 1.0,
    ) -> torch.Tensor:
        result = latents.clone()
        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)
                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean
        result = torch.lerp(latents, result, factor)
        return result

    @torch.no_grad()
    def __call__(
        self,
        video: str | list[str] | list[Image.Image] | list[np.ndarray] | None = None,
        latents: torch.Tensor | None = None,
        return_latents: bool = False,
        adain_factor: float = 1.0,
        offload: bool = False,
        decode_timestep: float = 0.05,
        decode_noise_scale: float | None = 0.025,
        tone_map_compression_ratio: float = 0.0,
        output_type: Literal["pil", "np"] = "pil",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | List[List[Image.Image]]:
        """
        Upsample latents using the LTX latent upsampler

        Args:
            latents: Input latents to upsample
            video: Video to upsample
            output_type: Type of output to return
            return_latents: Whether to return latents or decoded images
            **kwargs: Additional arguments

        Returns:
            Upsampled latents or decoded images
        """

        # Ensure the latent upsampler is loaded
        if self.latent_upsampler is None:
            self._load_latent_upsampler()

        # Optionally denormalize latents using VAE stats if available
        vae = self.engine.vae
        if not vae:
            self.engine.load_component_by_type("vae")
            vae = self.engine.vae
        self.engine.to_device(vae)

        if latents is not None and video is not None:
            raise ValueError("Either latents or video must be provided, not both")

        if latents is None and video is None:
            raise ValueError("Either latents or video must be provided")

        if latents is None:
            video = self.engine._load_video(video)
            video = self.engine.video_processor.preprocess_video(video)
            latents = self.engine.vae_encode(video, sample_mode="mode")

        prepared_latents = vae.denormalize_latents(latents)
        upsampled_latents = self._upsample_latents(prepared_latents)
        upsampled_latents = vae.normalize_latents(upsampled_latents)
        upsampled_latents = self._adain_filter_latent(
            upsampled_latents, latents, adain_factor
        )

        if return_latents:
            return upsampled_latents

        batch_size = latents.shape[0]

        latents = vae.denormalize_latents(latents)

        latents = latents.to(self.component_dtypes["vae"])

        if not vae.config.timestep_conditioning:
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

        if hasattr(self.engine.implementation_engine, "tone_map_latents"):
            latents = self.engine.implementation_engine.tone_map_latents(
                latents, tone_map_compression_ratio
            )

        *_, fl, hl, wl = latents.shape
        decoded_video = vae.decode(
            latents,
            target_shape=(
                1,
                3,
                fl * self.engine.vae_scale_factor_temporal,
                hl * self.engine.vae_scale_factor_spatial,
                wl * self.engine.vae_scale_factor_spatial,
            ),
            timestep=timestep,
            return_dict=False,
        )[0]

        video = self.engine._tensor_to_frames(decoded_video, output_type=output_type)

        if offload:
            self.engine._offload("vae")

        return video

    def __str__(self):
        return f"LatentUpsamplerPostprocessor(device={self.device}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()
