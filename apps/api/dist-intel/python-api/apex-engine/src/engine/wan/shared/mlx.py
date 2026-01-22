from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING
import mlx.core as mx
from loguru import logger
from tqdm import tqdm
from src.utils.mlx import convert_dtype_to_mlx, torch_to_mlx, to_torch
from src.utils.progress import safe_emit_progress

if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine  # noqa: F401

    BaseClass = BaseEngine  # type: ignore
else:
    BaseClass = object


class WanMLXDenoise(BaseClass):

    def _get_logger(self):
        return getattr(self, "logger", logger)

    def _maybe_to_dtype(self, array: mx.array, dtype):
        if dtype is None:
            return array
        if array.dtype == dtype:
            return array

        return array.astype(dtype)

    def _concat_if_needed(
        self, latents: mx.array, latent_condition: Optional[mx.array]
    ) -> mx.array:
        if latent_condition is None:
            return latents
        return mx.concatenate([latents, latent_condition], axis=1)

    def mlx_moe_denoise(self, *args, **kwargs) -> mx.array:

        kwargs = torch_to_mlx(kwargs)

        timesteps: Iterable[mx.array] = kwargs.get("timesteps", None)
        latents: mx.array = kwargs.get("latents", None)
        latent_condition: Optional[mx.array] = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance: bool = kwargs.get("use_cfg_guidance", True)
        render_on_step: bool = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        boundary_timestep = kwargs.get("boundary_timestep", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        transformer_dtype = convert_dtype_to_mlx(transformer_dtype)

        log = self._get_logger()

        steps_list = list(timesteps) if not isinstance(timesteps, list) else timesteps
        total_steps = len(steps_list)
        for i, t in enumerate(tqdm(steps_list, desc="Sampling MOE (MLX)")):
            if latent_condition is not None:
                latent_model_input = self._concat_if_needed(latents, latent_condition)
                latent_model_input = self._maybe_to_dtype(
                    latent_model_input, transformer_dtype
                )
            else:
                latent_model_input = self._maybe_to_dtype(latents, transformer_dtype)

            timestep = mx.broadcast_to(t, (latents.shape[0],))

            # Match PyTorch WAN MOE behavior:
            # - For boundary_timestep is not None and t >= boundary_timestep:
            #     use the "main" transformer and guidance_scale[0]
            # - Otherwise:
            #     use the alternate transformer_2 and guidance_scale[1]
            if boundary_timestep is not None and t >= boundary_timestep:
                if (
                    hasattr(self, "high_noise_transformer")
                    and self.high_noise_transformer
                ):
                    self._offload("high_noise_transformer")
                    setattr(self, "high_noise_transformer", None)
                if not self.high_noise_transformer:
                    self.load_component_by_name("high_noise_transformer")
                transformer = self.high_noise_transformer
                if isinstance(guidance_scale, list):
                    guidance_scale = guidance_scale[0]
            else:
                if self.low_noise_transformer:
                    self._offload("low_noise_transformer")
                    setattr(self, "low_noise_transformer", None)
                if (
                    not hasattr(self, "low_noise_transformer")
                    or not self.low_noise_transformer
                ):
                    self.load_component_by_name("low_noise_transformer")
                transformer = self.low_noise_transformer
                if isinstance(guidance_scale, list):
                    guidance_scale = guidance_scale[1]
                    # Standard denoising

            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                return_dict=False,
                **kwargs.get("transformer_kwargs", {}),
            )[0]

            mx.eval(noise_pred)

            if use_cfg_guidance and kwargs.get(
                "unconditional_transformer_kwargs", None
            ):
                uncond_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("unconditional_transformer_kwargs", {}),
                )[0]

                noise_pred = uncond_noise_pred + guidance_scale * (
                    noise_pred - uncond_noise_pred
                )
            mx.eval(noise_pred)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            mx.eval(latents)

            if (
                render_on_step
                and render_on_step_callback
                and ((i + 1) % render_on_step_interval == 0 or i == 0)
                and i != len(timesteps) - 1
            ):
                try:
                    self._render_step(to_torch(latents), render_on_step_callback)
                except Exception as e:
                    log.warning(f"Render-on-step callback failed: {e}")

            safe_emit_progress(
                denoise_progress_callback,
                float(i + 1) / float(max(1, total_steps)),
                f"Denoising step {i + 1}/{total_steps}",
            )

        log.info("Denoising completed.")
        return to_torch(latents)

    def mlx_base_denoise(self, *args, **kwargs) -> mx.array:
        kwargs = torch_to_mlx(kwargs)

        timesteps: Iterable[mx.array] = kwargs.get("timesteps", None)
        latents: mx.array = kwargs.get("latents", None)
        latent_condition: Optional[mx.array] = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance: bool = kwargs.get("use_cfg_guidance", True)
        render_on_step: bool = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        expand_timesteps: bool = kwargs.get("expand_timesteps", False)
        first_frame_mask: Optional[mx.array] = kwargs.get("first_frame_mask", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        transformer_dtype = convert_dtype_to_mlx(transformer_dtype)

        if not hasattr(self, "transformer") or self.transformer is None:
            self.load_component_by_type("transformer")
        transformer = self.transformer

        log = self._get_logger()

        steps_list = list(timesteps) if not isinstance(timesteps, list) else timesteps
        total_steps = len(steps_list)
        for i, t in enumerate(tqdm(steps_list, desc=f"Sampling WAN (MLX)")):
            if expand_timesteps and first_frame_mask is not None:
                mask = mx.ones_like(latents)
            else:
                mask = None

            if expand_timesteps:
                if latent_condition is not None and first_frame_mask is not None:
                    latent_model_input = (
                        1 - first_frame_mask
                    ) * latent_condition + first_frame_mask * latents
                    latent_model_input = self._maybe_to_dtype(
                        latent_model_input, transformer_dtype
                    )
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                else:
                    latent_model_input = self._maybe_to_dtype(
                        latents, transformer_dtype
                    )
                    # mask is ensured non-None above when expand_timesteps
                    temp_ts = (
                        (mask[0][0][:, ::2, ::2] * t).flatten()
                        if mask is not None
                        else t.flatten()
                    )
                timestep = mx.broadcast_to(
                    temp_ts, (latents.shape[0], temp_ts.shape[0])
                )
            else:
                timestep = mx.broadcast_to(t, (latents.shape[0],))
                if latent_condition is not None:
                    latent_model_input = self._concat_if_needed(
                        latents, latent_condition
                    )
                    latent_model_input = self._maybe_to_dtype(
                        latent_model_input, transformer_dtype
                    )
                else:
                    latent_model_input = self._maybe_to_dtype(
                        latents, transformer_dtype
                    )

            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                return_dict=False,
                **kwargs.get("transformer_kwargs", {}),
            )[0]

            mx.eval(noise_pred)

            if use_cfg_guidance and kwargs.get(
                "unconditional_transformer_kwargs", None
            ):
                uncond_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("unconditional_transformer_kwargs", {}),
                )[0]
                noise_pred = uncond_noise_pred + guidance_scale * (
                    noise_pred - uncond_noise_pred
                )
            mx.eval(noise_pred)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            mx.eval(latents)

            if (
                render_on_step
                and render_on_step_callback
                and ((i + 1) % render_on_step_interval == 0 or i == 0)
                and i != len(timesteps) - 1
            ):
                try:
                    # convert to torch
                    self._render_step(to_torch(latents), render_on_step_callback)
                except Exception as e:
                    log.warning(f"Render-on-step callback failed: {e}")

            safe_emit_progress(
                denoise_progress_callback,
                float(i + 1) / float(max(1, total_steps)),
                f"Denoising step {i + 1}/{total_steps}",
            )

        if (
            expand_timesteps
            and first_frame_mask is not None
            and latent_condition is not None
        ):
            latents = (
                1 - first_frame_mask
            ) * latent_condition + first_frame_mask * latents

        log.info("Denoising completed.")

        return to_torch(latents)
