from src.engine.wan.shared import WanShared, _DualNoiseTransformerSelectState
from typing import Optional, Union, List, Tuple, Callable
import torch
from diffusers.utils.torch_utils import randn_tensor
from src.types import InputImage
import copy
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.transformer.wan.mova.model import MOVAWanModel as WanModel, sinusoidal_embedding_1d
from src.transformer.wan.mova_audio.model import MOVAWanAudioModel as WanAudioModel
from src.transformer.wan.mova.easy_cache import MOVAEasyCacheState, create_easycache_state

class WanMOVAEngine(WanShared):
    """WAN MOVA Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.audio_vae_scale_factor = int(self.audio_vae.hop_length) if getattr(self, "audio_vae", None) else 960
        self.audio_sample_rate = self.audio_vae.sample_rate if getattr(self, "audio_vae", None) else 48000
        self.audio_latent_dim = self.audio_vae.latent_dim if getattr(self, "audio_vae", None) else 128
    
    def prepare_latents(
        self,
        image: InputImage,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        offload: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        vae_dtype = self.component_dtypes["vae"]

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=vae_dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=vae_dtype)

        if isinstance(generator, list):
            latent_condition = [
                self.vae_encode(video_condition, sample_mode="sample", sample_generator=g, offload=offload) for g in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = self.vae_encode(video_condition, sample_mode="mode", offload=offload)
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(vae_dtype)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)
    
    def prepare_audio_latents(
        self,
        audio: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
        num_samples: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        shape = (batch_size, num_channels, latent_t)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents
    
    
    def inference_single_step(
        self,
        visual_transformer: WanModel,
        audio_transformer: WanAudioModel,
        dual_tower_bridge,
        visual_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor],
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        video_fps: float,
        offload: bool = True,
        rope_on_cpu: bool = False,
    ):
        """
        Args:
            visual_latents:
                shape=[B, C=16, T // 4 + 1, H // 8, W // 8]
                dtype=bf16
            audio_latents:
                shape=[B, 128, 403]
                dtype=bf16
            y: first frame embedding
                shape=[B, C=20, T // 4 + 1, H // 8, W // 8]
                dtype=bf16
        """
        audio_context = visual_context = context  # [B, 512, C=4096]

        if audio_latents is None:
            raise ValueError("audio_latents is required for MOVA dual-tower inference.")

        if audio_timestep is None:
            audio_timestep = timestep

        # Prepare model-specific embeddings / patches / RoPE via each model's `forward`.
        # This keeps chunking + RoPE-on-CPU logic centralized in the model implementations.

        visual_prepared = visual_transformer(
            visual_latents,
            timestep,
            visual_context,
            use_gradient_checkpointing=False,
            rope_on_cpu=rope_on_cpu,
            return_prepared=True,
        )
        audio_prepared = audio_transformer(
            audio_latents,
            audio_timestep,
            audio_context,
            use_gradient_checkpointing=False,
            rope_on_cpu=rope_on_cpu,
            return_prepared=True,
        )

        visual_x = visual_prepared["x"]
        audio_x = audio_prepared["x"]
        visual_context_emb = visual_prepared["context"]
        audio_context_emb = audio_prepared["context"]
        visual_t = visual_prepared["t"]
        audio_t = audio_prepared["t"]
        visual_t_mod = visual_prepared["t_mod"]
        audio_t_mod = audio_prepared["t_mod"]
        visual_freqs = visual_prepared["freqs"]
        audio_freqs = audio_prepared["freqs"]
        grid_size = tuple(visual_prepared["grid_size"])
        (f,) = tuple(audio_prepared["grid_size"])
        visual_rotary_emb_chunk_size = visual_prepared.get("rotary_emb_chunk_size", None)
        audio_rotary_emb_chunk_size = audio_prepared.get("rotary_emb_chunk_size", None)

        # visual_transformer + audio_transformer + bridge forward
        visual_x, audio_x = self.forward_dual_tower_dit(
            visual_transformer=visual_transformer,
            audio_transformer=audio_transformer,
            dual_tower_bridge=dual_tower_bridge,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=visual_freqs,
            audio_freqs=audio_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            visual_rotary_emb_chunk_size=visual_rotary_emb_chunk_size,
            audio_rotary_emb_chunk_size=audio_rotary_emb_chunk_size,
        )
        

        visual_output = visual_transformer.head(
            visual_x,
            visual_t,
            modulated_norm_chunk_size=getattr(
                visual_transformer, "_out_modulated_norm_chunk_size", None
            ),
        )
        visual_output = visual_transformer.unpatchify(visual_output, grid_size)  # shape=[B, C=16, T // 4 + 1, H // 8, W // 8]
        
        audio_output = audio_transformer.head(
            audio_x,
            audio_t,
            modulated_norm_chunk_size=getattr(
                audio_transformer, "_out_modulated_norm_chunk_size", None
            ),
        )
        audio_output = audio_transformer.unpatchify(audio_output, (f, ))  # [1, 128, 403]

        return visual_output, audio_output
    
    def forward_dual_tower_dit(
        self,
        visual_transformer: WanModel,
        audio_transformer: WanAudioModel,
        dual_tower_bridge,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: Optional[torch.Tensor],
        visual_freqs: torch.Tensor,
        audio_freqs: torch.Tensor,
        grid_size: tuple[int, int, int],
        video_fps: float,
        condition_scale: Optional[float] = 1.0,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        visual_rotary_emb_chunk_size: Optional[int] = None,
        audio_rotary_emb_chunk_size: Optional[int] = None,
    ):
        min_layers = min(len(visual_transformer.blocks), len(audio_transformer.blocks))
        visual_layers = len(visual_transformer.blocks)

        

        # Cross-modal RoPE (cos, sin) can be very large. Compute it lazily only if
        # at least one interaction layer is used, and free it immediately after the
        # last interaction layer to reduce peak VRAM pressure.
        visual_rope_cos_sin = None
        audio_rope_cos_sin = None
        last_interact_layer: int = -1
        if dual_tower_bridge.apply_cross_rope:
            for i in range(min_layers):
                if dual_tower_bridge.should_interact(i, "a2v") or dual_tower_bridge.should_interact(i, "v2a"):
                    last_interact_layer = i

        # forward dit blocks and bridges
        for layer_idx in range(min_layers):
            # Only call the bridge when either direction is enabled at this layer.
            do_interact = dual_tower_bridge.should_interact(layer_idx, "a2v") or dual_tower_bridge.should_interact(
                layer_idx, "v2a"
            )
            if do_interact:
                # Lazily materialize aligned (cos, sin) pairs once, on the correct device/dtype.
                if dual_tower_bridge.apply_cross_rope and visual_rope_cos_sin is None:
                    (visual_rope_cos_sin, audio_rope_cos_sin) = dual_tower_bridge.build_aligned_freqs(
                        video_fps=video_fps,
                        grid_size=grid_size,
                        audio_steps=audio_x.shape[1],
                        device=visual_x.device,
                        dtype=visual_x.dtype,
                    )
                visual_x, audio_x = dual_tower_bridge(
                    layer_idx,
                    visual_x,
                    audio_x,
                    x_freqs=visual_rope_cos_sin,
                    y_freqs=audio_rope_cos_sin,
                    a2v_condition_scale=a2v_condition_scale,
                    v2a_condition_scale=v2a_condition_scale,
                    condition_scale=condition_scale,
                    video_grid_size=grid_size,
                )
            
            visual_block = visual_transformer.blocks[layer_idx]
            visual_x = visual_block(
                visual_x,
                visual_context,
                visual_t_mod,
                visual_freqs,
                rotary_emb_chunk_size=visual_rotary_emb_chunk_size,
            )
            # Drop per-layer refs early to help GC release buffers sooner.
            del visual_block

            audio_block = audio_transformer.blocks[layer_idx]
            audio_x = audio_block(
                audio_x,
                audio_context,
                audio_t_mod,
                audio_freqs,
                rotary_emb_chunk_size=audio_rotary_emb_chunk_size,
            )
            del audio_block
            del do_interact

            # If we precomputed large cross-modal RoPE tensors, free them as soon
            # as we're past the final interaction layer.
            if (
                dual_tower_bridge.apply_cross_rope
                and last_interact_layer >= 0
                and layer_idx == last_interact_layer
                and visual_rope_cos_sin is not None
            ):
                del visual_rope_cos_sin
                del audio_rope_cos_sin
                visual_rope_cos_sin = None
                audio_rope_cos_sin = None
        
        # Audio-side embeddings are no longer needed after the shared depth.
        # Drop references to allow immediate reclamation (especially helpful when
        # running close to VRAM limits).
        audio_context = None
        audio_t_mod = None
        audio_freqs = None
        grid_size = None  # no longer needed beyond bridging

        # forward remaining visual blocks
        assert visual_layers >= min_layers, "visual_layers must be greater than min_layers"
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_transformer.blocks[layer_idx]
            visual_x = visual_block(
                visual_x,
                visual_context,
                visual_t_mod,
                visual_freqs,
                rotary_emb_chunk_size=visual_rotary_emb_chunk_size,
            )
            del visual_block
        
        # Drop local reference (bridge can remain resident/offloaded by caller policy).
        return visual_x, audio_x

    def inference_single_step_with_easycache(
        self,
        visual_transformer: WanModel,
        audio_transformer: WanAudioModel,
        dual_tower_bridge,
        visual_latents: torch.Tensor,
        visual_latents_raw: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        video_fps: float,
        easycache_state: MOVAEasyCacheState,
        offload: bool = True,
        rope_on_cpu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference single step with EasyCache acceleration.
        
        This method wraps inference_single_step with caching logic to skip
        redundant computations when input changes are small.
        
        Args:
            visual_transformer: Visual DiT model
            audio_transformer: Audio DiT model
            visual_latents: Visual latent input with condition [B, C=36, T, H, W]
            visual_latents_raw: Raw visual latents without condition [B, C=16, T, H, W]
            audio_latents: Audio latent input [B, C, T]
            context: Text embeddings
            timestep: Current timestep
            audio_timestep: Audio timestep (can differ from visual)
            video_fps: Video frames per second
            easycache_state: EasyCache state for tracking
            offload: Whether to offload models after computation
            
        Returns:
            Tuple of (visual_output, audio_output)
        """
        # Decide whether to compute or use cache based on raw latents (same shape as output)
        easycache_state.decide_computation(visual_latents_raw, audio_latents)
        
        # Check if we can use cached output
        if easycache_state.should_use_cache():
            return easycache_state.get_cached_output(visual_latents_raw, audio_latents)
        
        # Compute normally
        visual_output, audio_output = self.inference_single_step(
            visual_transformer=visual_transformer,
            audio_transformer=audio_transformer,
            dual_tower_bridge=dual_tower_bridge,
            visual_latents=visual_latents,
            audio_latents=audio_latents,
            context=context,
            timestep=timestep,
            audio_timestep=audio_timestep,
            video_fps=video_fps,
            offload=offload,
            rope_on_cpu=rope_on_cpu,
        )
        
        # Update cache with results using raw latents (same shape as output)
        easycache_state.update_cache(
            visual_input=visual_latents_raw,
            visual_output=visual_output,
            audio_input=audio_latents,
            audio_output=audio_output,
        )
        
        return visual_output, audio_output

    def run(self,
        prompt: str,
        image: InputImage,
        negative_prompt="",
        seed=42,
        height: int = 360,
        width: int = 640,
        duration: int = 193,
        fps: float = 24.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        high_noise_guidance_scale: float = 5.0,
        low_noise_guidance_scale: float = 5.0,
        offload: bool = True,
        progress_callback: Callable = None,
        max_sequence_length: int = 512,
        boundary_ratio: float = 0.875,
        cfg_mode: str = "text",
        cfg_merge: bool = False,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        denoise_progress_callback: Callable = None,
        # EasyCache parameters
        use_easycache: bool = False,
        easycache_thresh: float = 0.05,
        easycache_ret_steps: int = 7,
        chunking_profile: str = "none",
        rope_on_cpu: bool = True,
        **kwargs,
    ):
        safe_emit_progress(
            progress_callback,
            0.0,
            "Starting MOVA image-to-video + audio generation pipeline",
        )

        use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0
        num_frames = self._parse_num_frames(duration, fps)
        if high_noise_guidance_scale is not None and low_noise_guidance_scale is not None:
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(
                progress_callback, 0.01, "Using high/low-noise guidance scales"
            )
        device = self.device

        safe_emit_progress(progress_callback, 0.02, "Loading input image")
        image = self._load_image(image)
        safe_emit_progress(
            progress_callback,
            0.04,
            f"Preprocessing input image (target: ~{height}x{width}, frames: {num_frames}, fps: {fps:g})",
        )
        image, height, width = self._aspect_ratio_resize(image, max_area=height * width, mod_value=16)
        image = self.video_processor.preprocess(image, height=height, width=width).to(self.device, dtype=torch.float32)
        audio_num_samples = int(self.audio_sample_rate * num_frames / fps)
        
        if seed is not None:
            safe_emit_progress(progress_callback, 0.06, f"Seeding RNG (seed: {seed})")
            generator = torch.Generator(device=device).manual_seed(seed)
        
        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.08, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.09, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.10, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.11, "Configuring scheduler timesteps")
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_support_audio = hasattr(self.scheduler, "get_pairs")
        if scheduler_support_audio:
            safe_emit_progress(progress_callback, 0.12, "Using paired video/audio timesteps")
            audio_scheduler = self.scheduler
            paired_timesteps = self.scheduler.get_pairs()
        else:
            safe_emit_progress(progress_callback, 0.12, "Preparing separate audio scheduler")
            audio_scheduler = copy.deepcopy(self.scheduler)
            paired_timesteps = torch.stack([self.scheduler.timesteps, self.scheduler.timesteps], dim=1)
        
        safe_emit_progress(
            progress_callback,
            0.14,
            f"Preparing video latents + conditioning (resolution: {height}x{width}, frames: {num_frames})",
        )
        latents, condition = self.prepare_latents(
            image,
            1,
            self.num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator=generator,
            latents=latents,
            last_image=last_image,
            offload=offload,
        )
        
        safe_emit_progress(
            progress_callback,
            0.18,
            f"Preparing audio noise latents (samples: {audio_num_samples}, sample_rate: {self.audio_sample_rate})",
        )
        audio_latents = self.prepare_audio_latents(
            None,
            1,
            self.audio_latent_dim,
            audio_num_samples,
            torch.float32,
            device,
            generator=generator,
            latents=audio_latents,
        )
        
        safe_emit_progress(progress_callback, 0.20, "Encoding prompt embeddings")
        encode_progress_callback = make_mapped_progress(progress_callback, 0.20, 0.32)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            progress_callback=encode_progress_callback,
            max_sequence_length=max_sequence_length,
            offload=offload,
        )
        safe_emit_progress(progress_callback, 0.33, "Prompt embeddings ready")
        
        total_steps = paired_timesteps.shape[0]
        boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps
        denoise_progress_callback = denoise_progress_callback or make_mapped_progress(
            progress_callback, 0.40, 0.92
        )
        
        # Initialize EasyCache state if enabled
        easycache_state = None
        if use_easycache:
            safe_emit_progress(
                progress_callback,
                0.34,
                f"Initializing EasyCache (thresh: {easycache_thresh:g}, ret_steps: {easycache_ret_steps})",
            )
            easycache_state = create_easycache_state(
                num_steps=total_steps,
                thresh=easycache_thresh,
                ret_steps=easycache_ret_steps,
            )
        
        dual_transformer_state = _DualNoiseTransformerSelectState()
        if not getattr(self, "audio_transformer", None):
            safe_emit_progress(progress_callback, 0.35, "Loading audio transformer")
            self.load_component_by_name("audio_transformer")

        safe_emit_progress(progress_callback, 0.36, "Moving audio transformer to device")
        self.to_device(self.audio_transformer)
        if chunking_profile != "none":
            if hasattr(self.audio_transformer, "set_chunking_profile"):
                safe_emit_progress(
                    progress_callback,
                    0.37,
                    f"Enabling audio transformer chunking profile '{chunking_profile}' (RoPE on CPU: {'on' if rope_on_cpu else 'off'})",
                )
                self.audio_transformer.enable_memory_efficient_inference(chunking_profile=chunking_profile, rope_on_cpu=rope_on_cpu)
        

        dual_tower_bridge = self.helpers["dual_tower_bridge"]
        safe_emit_progress(progress_callback, 0.38, "Moving audio/video bridge to device")
        self.to_device(dual_tower_bridge)

        safe_emit_progress(
            progress_callback,
            0.40,
            (
                "Starting denoising "
                f"(steps: {total_steps}, CFG: {'on' if use_cfg_guidance else 'off'}, "
                f"mode: {cfg_mode}{', merged' if cfg_merge else ''}, "
                f"EasyCache: {'on' if use_easycache else 'off'})"
            ),
        )
        
        with self._progress_bar(total=total_steps, desc="Denoising") as pbar:
            
            for idx_step in range(total_steps):
                timestep, audio_timestep = paired_timesteps[idx_step]
                

                transformer = self._select_dual_noise_transformer(
                    t=timestep,
                    boundary_timestep=boundary_timestep,
                    step_idx=idx_step,
                    total_steps=total_steps,
                    state=dual_transformer_state,
                    denoise_progress_callback=denoise_progress_callback,
                    chunking_profile=chunking_profile,
                )

      
                latent_model_input = torch.cat([latents, condition], dim=1)
                timestep = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
                audio_timestep = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
                
                current_guidance_scale = self._select_dual_noise_guidance_scale(
                    t=timestep,
                    boundary_timestep=boundary_timestep,
                    guidance_scale=guidance_scale
                )

                # Use EasyCache-accelerated inference if enabled
                if use_easycache and easycache_state is not None:
                    noise_pred_posi = self.inference_single_step_with_easycache(
                        visual_transformer=transformer,
                        audio_transformer=self.audio_transformer,
                        dual_tower_bridge=dual_tower_bridge,
                        visual_latents=latent_model_input,
                        visual_latents_raw=latents,  # Raw latents without condition (same shape as output)
                        audio_latents=audio_latents,
                        context=prompt_embeds,
                        timestep=timestep,
                        audio_timestep=audio_timestep,
                        video_fps=fps,
                        easycache_state=easycache_state,
                        offload=offload,
                        rope_on_cpu=rope_on_cpu,
                    )
                else:
                    noise_pred_posi = self.inference_single_step(
                        visual_transformer=transformer,
                        audio_transformer=self.audio_transformer,
                        dual_tower_bridge=dual_tower_bridge,
                        visual_latents=latent_model_input,  # video noise
                        audio_latents=audio_latents,  # audio noise
                        context=prompt_embeds,  # prompt embedding
                        timestep=timestep,
                        audio_timestep=audio_timestep,
                        video_fps=fps,
                        offload=offload,
                        rope_on_cpu=rope_on_cpu,
                    )

                if guidance_scale == 1.0 and "dual" not in cfg_mode:
                    visual_noise_pred = noise_pred_posi[0].float()
                    audio_noise_pred = noise_pred_posi[1].float()
                elif "dual" not in cfg_mode:
                    if cfg_merge:
                        visual_noise_pred_posi, visual_noise_pred_nega = noise_pred_posi[0].float().chunk(2, dim=0)
                        audio_noise_pred_posi, audio_noise_pred_nega = noise_pred_posi[1].float().chunk(2, dim=0)
                    else:
                        # Use EasyCache-accelerated inference for negative prompt if enabled
                        if use_easycache and easycache_state is not None:
                            noise_pred_nega = self.inference_single_step_with_easycache(
                                visual_transformer=transformer,
                                audio_transformer=self.audio_transformer,
                                dual_tower_bridge=dual_tower_bridge,
                                visual_latents=latent_model_input,
                                visual_latents_raw=latents,  # Raw latents without condition (same shape as output)
                                audio_latents=audio_latents,
                                context=negative_prompt_embeds,
                                timestep=timestep,
                                audio_timestep=audio_timestep,
                                video_fps=fps,
                                easycache_state=easycache_state,
                                offload=offload,
                            )
                        else:
                            noise_pred_nega = self.inference_single_step(
                                visual_transformer=transformer,
                                audio_transformer=self.audio_transformer,
                                dual_tower_bridge=dual_tower_bridge,
                                visual_latents=latent_model_input,
                                audio_latents=audio_latents,
                                context=negative_prompt_embeds,
                                timestep=timestep,
                                audio_timestep=audio_timestep,
                                video_fps=fps,
                                offload=offload,
                            )
                        visual_noise_pred_nega, audio_noise_pred_nega = noise_pred_nega[0].float(), noise_pred_nega[1].float()
                        visual_noise_pred_posi, audio_noise_pred_posi = noise_pred_posi[0].float(), noise_pred_posi[1].float()
                    visual_noise_pred = visual_noise_pred_nega + current_guidance_scale * (visual_noise_pred_posi - visual_noise_pred_nega)
                    audio_noise_pred = audio_noise_pred_nega + current_guidance_scale * (audio_noise_pred_posi - audio_noise_pred_nega)
                else:
                    raise NotImplementedError

                # move a step
                if scheduler_support_audio:
                    next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
                    next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
                    latents = self.scheduler.step_from_to(
                        visual_noise_pred,
                        timestep,
                        next_timestep,
                        latents,
                    )
                    audio_latents = audio_scheduler.step_from_to(
                        audio_noise_pred,
                        audio_timestep,
                        next_audio_timestep,
                        audio_latents,
                    )
                else:
                    latents = self.scheduler.step(visual_noise_pred, timestep, latents, return_dict=False)[0]
                    audio_latents = audio_scheduler.step(audio_noise_pred, audio_timestep, audio_latents, return_dict=False)[0]

                if total_steps > 0:
                    safe_emit_progress(
                        denoise_progress_callback,
                        (idx_step + 1) / total_steps,
                        f"Denoising step {idx_step + 1}/{total_steps}",
                    )
                
                pbar.update(1)

        if offload:
            safe_emit_progress(progress_callback, 0.93, "Offloading denoising models")
            self._offload("audio_transformer")
            self._offload("high_noise_transformer")
            self._offload("low_noise_transformer")
            self._offload("dual_tower_bridge")
            
        # decode video
        safe_emit_progress(progress_callback, 0.94, "Decoding video latents")
        video = self.vae_decode(latents, offload=offload)
        video = self._tensor_to_frames(video)
        safe_emit_progress(progress_callback, 0.96, "Decoded video frames")

        # decode audio
        if not getattr(self, "audio_vae", None):
            safe_emit_progress(progress_callback, 0.965, "Loading audio decoder")
            self.load_component_by_name("audio_vae")
        safe_emit_progress(progress_callback, 0.97, "Moving audio decoder to device")
        self.to_device(self.audio_vae)
        
        safe_emit_progress(progress_callback, 0.98, "Decoding audio latents")
        audio_latents = audio_latents.to(self.audio_vae.dtype)
        audio = self.audio_vae.decode(audio_latents).to(torch.float32)
        audio = audio[0].cpu().squeeze()
        
        if offload:
            safe_emit_progress(progress_callback, 0.99, "Offloading audio decoder")
            self._offload("audio_vae")

        safe_emit_progress(progress_callback, 1.0, "Completed MOVA generation pipeline")
        return video, audio

    