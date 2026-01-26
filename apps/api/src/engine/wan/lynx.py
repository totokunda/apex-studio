import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from src.utils.cache import empty_cache
from .shared import WanShared
from src.helpers.wan.lynx import WanLynxHelper
from src.types import InputImage
from src.utils.progress import make_mapped_progress, safe_emit_progress

class LynxEngine(WanShared):
    """Personalized Wan (Lynx) engine with IPA and Ref adapters."""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, auto_apply_loras=False, **kwargs)
        self._adapter_path = self._resolve_adapter_path()
        self._lynx_helper = WanLynxHelper(adapter_path=self._adapter_path)

    # --------------------------- Internal utilities --------------------------- #
    def _resolve_adapter_path(self, override: str | None = None) -> str | None:
        if override:
            return override
        env_path = os.getenv("LYNX_ADAPTER_PATH")
        if env_path:
            return env_path
        cfg = getattr(self, "config", {}) or {}
        transformer_component = next(
            (x for x in cfg.get("components", []) if x.get("type") == "transformer"),
            None,
        )
        extra_model_path = transformer_component.get("extra_model_paths", [])[0]
        return (
            extra_model_path
            or cfg.get("adapter_path")
            or cfg.get("lynx_adapter_path")
            or cfg.get("adapter_dir")
            or override
        )

    # ------------------------------- Main entry ------------------------------- #
    def run(
        self,
        subject_image: InputImage,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        guidance_scale_i: float | None = 2.0,
        ip_scale: float = 1.0,
        ref_scale: float = 1.0,
        adapter_path: str | None = None,
        face_embeds: Optional[np.ndarray | torch.Tensor] = None,
        face_token_embeds: Optional[torch.FloatTensor] = None,
        landmarks: Optional[np.ndarray | torch.Tensor] = None,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable | None = None,
        render_on_step: bool = False,
        render_on_step_interval: int = 3,
        offload: bool = True,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        progress_callback: Callable | None = None,
        denoise_progress_callback: Callable | None = None,
        output_type: str = "pil",
        rope_on_cpu: bool = False,
        chunking_profile: str = "none",
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting Lynx pipeline")
        num_frames = self._parse_num_frames(duration, fps)

        use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None

        helper = self._lynx_helper
        safe_emit_progress(progress_callback, 0.02, "Resolving Lynx adapter path")
        adapter_root = helper.resolve_adapter_path(
            config=getattr(self, "config", {}),
            override=adapter_path or self._adapter_path,
        )
        safe_emit_progress(progress_callback, 0.04, "Preparing face data")
        embeds, face_landmarks, face_image = helper.prepare_face_data(
            subject_image,
            face_embeds,
            landmarks,
            device=self.device,
            load_image_fn=self._load_image,
        )
        
        helper.face_encoder = None
        empty_cache()

        safe_emit_progress(progress_callback, 0.08, "Prepared face embeddings")

        safe_emit_progress(progress_callback, 0.08, "Encoding prompts")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            num_videos=num_videos,
            max_sequence_length=512,
            progress_callback=make_mapped_progress(progress_callback, 0.08, 0.18),
            text_encoder_kwargs=text_encoder_kwargs,
            offload=False,
        )
        safe_emit_progress(progress_callback, 0.18, "Prompts ready")

        batch_size = prompt_embeds.shape[0]

        if offload:
            safe_emit_progress(progress_callback, 0.19, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.20, "Text encoder ready")

        if self.scheduler is None:
            safe_emit_progress(progress_callback, 0.21, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.22, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.23, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.24, "Scheduler on device")
        scheduler = self.scheduler

        safe_emit_progress(progress_callback, 0.25, "Configuring scheduler timesteps")
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        safe_emit_progress(progress_callback, 0.26, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )
        safe_emit_progress(progress_callback, 0.26, "Scheduler prepared")

        transformer_dtype = self.component_dtypes["transformer"]
        safe_emit_progress(
            progress_callback, 0.28, "Moving prompt embeddings to device"
        )
        
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
        safe_emit_progress(progress_callback, 0.30, "Prompt embeddings ready")

        if generator is None and seed is not None:
            safe_emit_progress(progress_callback, 0.31, "Seeding RNG")
            generator = torch.Generator(device=self.device).manual_seed(seed)

        safe_emit_progress(progress_callback, 0.32, "Initializing latent noise")

        latents = self._get_latents(
            height,
            width,
            num_frames,
            fps=fps,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            dtype=transformer_dtype,
            generator=generator,
        )

        safe_emit_progress(progress_callback, 0.34, "Latent noise initialized")

        if self.transformer is None:
            safe_emit_progress(progress_callback, 0.35, "Loading transformer")
            self.load_component_by_type("transformer")
            safe_emit_progress(progress_callback, 0.36, "Transformer loaded")
        safe_emit_progress(progress_callback, 0.37, "Moving transformer to device")
        
        safe_emit_progress(progress_callback, 0.38, "Transformer on device")
        # Preserve any deferred offloading config across adapter wrapping.
        pending_group_offload = getattr(
            self.transformer, "_apex_pending_group_offloading", None
        )
        pending_budget_offload = getattr(
            self.transformer, "_apex_pending_budget_offloading", None
        )
        safe_emit_progress(progress_callback, 0.39, "Loading Lynx adapters")
        
        self.transformer = helper.load_adapters(
            self.transformer, adapter_root, device=torch.device("cpu"), dtype=transformer_dtype
        )
        safe_emit_progress(progress_callback, 0.40, "Lynx adapters loaded")
        if (
            pending_group_offload is not None
            and getattr(self.transformer, "_apex_pending_group_offloading", None)
            is None
        ):
            setattr(
                self.transformer, "_apex_pending_group_offloading", pending_group_offload
            )
        if (
            pending_budget_offload is not None
            and getattr(self.transformer, "_apex_pending_budget_offloading", None)
            is None
        ):
            setattr(
                self.transformer,
                "_apex_pending_budget_offloading",
                pending_budget_offload,
            )

        # Apply any preloaded LoRAs (transformer-only) after adapters are attached.
        # This mirrors BaseEngine's sequencing (mutations first, then offloading).
        preloaded_transformer_loras = [
            (lora.source, lora.scale, lora.name)
            for lora in self.preloaded_loras.values()
            if lora.component is None or lora.component == "transformer"
        ]
        if preloaded_transformer_loras:
            safe_emit_progress(
                progress_callback,
                0.41,
                f"Applying {len(preloaded_transformer_loras)} LoRAs",
            )
            self.logger.info(
                f"Applying {len(preloaded_transformer_loras)} loras to Lynx transformer"
            )
            self.apply_loras(
                [(src, scale) for (src, scale, _name) in preloaded_transformer_loras],
                adapter_names=[name for (_src, _scale, name) in preloaded_transformer_loras],
                model=self.transformer,
            )
            safe_emit_progress(progress_callback, 0.42, "LoRAs applied")

        # Offloading for transformers must be enabled *after* any post-load mutations
        # (adapters and/or LoRAs). Lynx always wraps the transformer, so ensure any
        # deferred configuration (and any pre-wrap enabled offloading) is applied now.
        self._apply_pending_group_offloading(self.transformer)
        self._apply_pending_budget_offloading(self.transformer)
  
        # Fallback: if this transformer was offloaded during initial load but got
        # replaced/wrapped by Lynx adapters, re-apply the configured offloading mode.
        try:
            transformer_component = self.get_component_by_type("transformer")
            mm_config = (
                self._resolve_memory_config_for_component(transformer_component)
                if transformer_component is not None
                else None
            )
            if mm_config is not None and transformer_component is not None:
                label = (
                    transformer_component.get("name")
                    or transformer_component.get("type")
                    or "transformer"
                )
                offloading_module = transformer_component.get("offloading_module", None)
                ignore_offloading_modules = transformer_component.get(
                    "ignore_offloading_modules", None
                )
                block_modules = transformer_component.get("block_modules", None)
                mm_config.ignore_modules = ignore_offloading_modules
                mm_config.block_modules = block_modules
                offload_mode = getattr(mm_config, "offload_mode", "budget") or "budget"
                model_to_offload = (
                    self.transformer.get_submodule(offloading_module)
                    if offloading_module
                    else self.transformer
                )
                if offload_mode == "budget":

                    self._apply_budget_offloading(
                        model_to_offload, mm_config, module_label=label
                    )
                else:
                    print("Applying group offloading")
                    exit()
                    self._apply_group_offloading(
                        model_to_offload, mm_config, module_label=label
                    )
        except Exception:
            # Best-effort only; Lynx should still run without reapplying here.
            pass
        
        if chunking_profile != "none":
            self.transformer.set_chunking_profile(chunking_profile)
        
 
        self.to_device(self.transformer)
        self.to_device(helper.resampler)
        

        safe_emit_progress(progress_callback, 0.43, "Building IP states")
        ip_states, ip_states_uncond = helper.build_ip_states(
            embeds, device=self.device, dtype=transformer_dtype
        )
        safe_emit_progress(progress_callback, 0.44, "IP states ready")

        ref_buffer = ref_buffer_uncond = None
        if helper.ref_loaded and ref_scale is not None:
            safe_emit_progress(progress_callback, 0.45, "Preparing reference buffer")

            aligned_face = helper.align_face(face_image, face_landmarks, face_size=256)
            safe_emit_progress(
                progress_callback, 0.48, "Encoding reference buffer (cond/uncond)"
            )
            ref_gen = (
                torch.Generator(device=self.device).manual_seed(seed + 1)
                if seed is not None
                else None
            )

            ref_buffer = helper.encode_reference_buffer(
                self,
                aligned_face,
                device=self.device,
                dtype=transformer_dtype,
                drop=False,
                generator=ref_gen,
            )
            ref_buffer_uncond = helper.encode_reference_buffer(
                self,
                aligned_face,
                device=self.device,
                dtype=transformer_dtype,
                drop=True,
                generator=ref_gen,
            )
            safe_emit_progress(progress_callback, 0.49, "Reference buffer ready")

        merged_attention_kwargs = {
            **attention_kwargs,
            "ip_hidden_states": ip_states,
            "ip_scale": ip_scale,
        }
        if ref_buffer is not None:
            merged_attention_kwargs.update(
                {"ref_buffer": ref_buffer, "ref_scale": ref_scale}
            )
        merged_attention_kwargs_uncond = {
            "ip_hidden_states": ip_states_uncond,
            "ip_scale": ip_scale,
        }
        if ref_buffer_uncond is not None:
            merged_attention_kwargs_uncond.update(
                {"ref_buffer": ref_buffer_uncond, "ref_scale": ref_scale}
            )

        if face_token_embeds is None:
            transformer_dtype = self.component_dtypes["transformer"]
            # 3.1 Encoder input face embedding
            safe_emit_progress(progress_callback, 0.49, "Encoding face embedding")
            face_token_embeds = helper.encode_face_embedding(
                embeds, use_cfg_guidance, device=self.device, dtype=transformer_dtype
            )
            safe_emit_progress(progress_callback, 0.50, "Face embedding ready")
        else:
            face_token_embeds = torch.cat(
                [torch.zeros_like(face_token_embeds), face_token_embeds], dim=0
            )

        face_token_embeds = face_token_embeds.to(
            device=self.device, dtype=transformer_dtype
        )

        merged_attention_kwargs.update({"image_embed": face_token_embeds})
        merged_attention_kwargs_uncond.update({"image_embed": face_token_embeds})

        if offload:
            del helper
            safe_emit_progress(progress_callback, 0.50, "Offloading Lynx helper")
            self._offload("helper")

        do_cfg = use_cfg_guidance and negative_prompt_embeds is not None
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        denoise_progress_callback = denoise_progress_callback or make_mapped_progress(
            progress_callback, 0.50, 0.90
        )
        safe_emit_progress(
            progress_callback,
            0.50,
            f"Starting denoise phase (CFG: {'on' if do_cfg else 'off'})",
        )

        total_steps = len(timesteps)
        with self._progress_bar(total=total_steps, desc="Denoising steps") as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=merged_attention_kwargs,
                    return_dict=False,
                    rope_on_cpu=rope_on_cpu,
                )[0]

                if do_cfg:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=merged_attention_kwargs_uncond,
                        return_dict=False,
                        rope_on_cpu=rope_on_cpu,
                    )[0]

                    if guidance_scale_i is not None:
                        noise_i = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=merged_attention_kwargs,
                            return_dict=False,
                            rope_on_cpu=rope_on_cpu,
                        )[0]
                        noise_pred = (
                            noise_uncond
                            + guidance_scale_i * (noise_i - noise_uncond)
                            + guidance_scale * (noise_pred - noise_i)
                        )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_pred - noise_uncond
                        )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != total_steps - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps) if total_steps else 1.0,
                    f"Denoising step {i + 1}/{total_steps}",
                )
                pbar.update(1)

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")
        if offload:
            safe_emit_progress(progress_callback, 0.93, "Offloading transformer")
            self._offload("transformer")
            safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.95, "Decoding latents")
        video = self.vae_decode(latents, offload=offload)
        safe_emit_progress(progress_callback, 0.98, "Formatting output frames")
        frames = self._tensor_to_frames(video, output_type=output_type)
        safe_emit_progress(progress_callback, 1.0, "Lynx generation complete")
        return frames
