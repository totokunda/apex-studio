import torch
from typing import Dict, Any, Callable, List, Optional
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared
import re
import torch.nn.functional as F
from src.utils.cache import empty_cache
from diffusers.models import ModelMixin


class WanHoloCineEngine(WanShared):
    """WAN HoloCine Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = 4

    def prepare_multishot_inputs(
        self,
        global_caption: str,
        shot_captions: list[str],
        duration: int | str,
        custom_shot_cut_frames: list[int] = None,
        fps: int = 15,
    ) -> dict:
        """
        (Helper for Mode 1)
        Prepares the inference parameters from user-friendly segmented inputs.
        """

        num_shots = len(shot_captions)

        # 1. Prepare 'prompt'
        if "This scene contains" not in global_caption:
            global_caption = (
                global_caption.strip() + f" This scene contains {num_shots} shots."
            )
        per_shot_string = " [shot cut] ".join(shot_captions)
        prompt = (
            f"[global caption] {global_caption} [per shot caption] {per_shot_string}"
        )

        # 2. Prepare 'num_frames'
        processed_total_frames = self._parse_num_frames(duration, fps=fps)

        # 3. Prepare 'shot_cut_frames'
        num_cuts = num_shots - 1
        processed_shot_cuts = []

        if custom_shot_cut_frames:
            # User provided custom cuts
            print(
                f"Using {len(custom_shot_cut_frames)} user-defined shot cuts (enforcing 4t+1)."
            )
            for frame in custom_shot_cut_frames:
                processed_shot_cuts.append(self._parse_num_frames(frame, fps=fps))
        else:
            # Auto-calculate cuts
            print(f"Auto-calculating {num_cuts} shot cuts.")
            if num_cuts > 0:
                ideal_step = processed_total_frames / num_shots
                for i in range(1, num_shots):
                    approx_cut_frame = i * ideal_step
                    processed_shot_cuts.append(
                        self._parse_num_frames(round(approx_cut_frame), fps=fps)
                    )

        processed_shot_cuts = sorted(list(set(processed_shot_cuts)))
        processed_shot_cuts = [
            f for f in processed_shot_cuts if f > 0 and f < processed_total_frames
        ]

        return {
            "prompt": prompt,
            "shot_cut_frames": processed_shot_cuts,
            "num_frames": processed_total_frames,
        }

    def _process_shot_cut_frames(self, shot_cut_frames, num_frames):
        if shot_cut_frames is None:
            return {}

        num_latent_frames = (num_frames - 1) // 4 + 1

        # Convert frame cut indices to latent cut indices
        shot_cut_latents = [0]
        for frame_idx in sorted(shot_cut_frames):
            if frame_idx > 0:
                latent_idx = (frame_idx - 1) // 4 + 1
                if latent_idx < num_latent_frames:
                    shot_cut_latents.append(latent_idx)

        cuts = sorted(list(set(shot_cut_latents))) + [num_latent_frames]

        shot_indices = torch.zeros(num_latent_frames, dtype=torch.long)
        for i in range(len(cuts) - 1):
            start_latent, end_latent = cuts[i], cuts[i + 1]
            shot_indices[start_latent:end_latent] = i

        shot_indices = shot_indices.unsqueeze(0).to(device=self.device)

        return shot_indices

    def _run_inference_step(
        self,
        transformer: ModelMixin,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        shot_indices: Optional[torch.Tensor] = None,
        shot_mask_type: Optional[str] = None,
        text_cut_positions: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Port of the core HoloCine WAN model function (`model_fn_wan_video`),
        simplified to only handle the arguments that are actually used in the
        current bash / pipeline setup.

        Removed features:
        - Sliding‑window tiling
        - TeaCache
        - VACE conditioning
        - Motion controller & sequence parallel distributed logic
        """

        dit = transformer
        if dit is None:
            raise RuntimeError(
                "Transformer (HoloCine DiT) is not loaded on the engine."
            )

        x = latents

        # Optional per‑shot mask channel
        if shot_mask_type is not None and shot_indices is not None:
            num_shots = shot_indices.max() + 1
            if shot_mask_type == "id":
                shot_mask_tensor = shot_indices.to(x.dtype)
            elif shot_mask_type == "normalized":
                shot_mask_tensor = (
                    shot_indices.to(x.dtype) / 20
                    if num_shots > 1
                    else torch.zeros_like(shot_indices, dtype=x.dtype)
                )
            elif shot_mask_type == "alternating":
                shot_mask_tensor = (shot_indices % 2).to(x.dtype)
            else:
                shot_mask_tensor = None

            if shot_mask_tensor is not None:
                b, c, f, h, w = x.shape
                mask = (
                    shot_mask_tensor.view(b, 1, f, 1, 1)
                    .expand(b, 1, f, h, w)
                    .to(x.dtype)
                )
                x = torch.cat([x, mask], dim=1)

        t = dit.time_embedding(self.sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        # Text embedding
        context = dit.text_embedding(context)

        # Merged CFG batching (pos/neg)
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        # Camera control & patchify
        x, (f, h, w) = dit.patchify(x, None)

        # Optional per‑shot token embeddings
        if (
            getattr(dit, "shot_embedding", None) is not None
            and shot_indices is not None
        ):
            assert (
                shot_indices.shape[0] == x.shape[0]
            ), f"Batch size mismatch between latents ({x.shape[0]}) and shot_indices ({shot_indices.shape[0]})"
            assert (
                shot_indices.shape[-1] == f
            ), f"Shot indices length mismatch. Expected {f}, got {shot_indices.shape[-1]}"
            shot_ids = shot_indices.repeat_interleave(h * w, dim=1)
            shot_embs = dit.shot_embedding(shot_ids)
            x = x + shot_embs

        # Positional frequencies
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        # Cross‑attention mask based on `text_cut_positions`
        attn_mask = None
        if (
            text_cut_positions is not None
            and text_cut_positions.get("global") is not None
        ):
            try:
                B, S_q = x.shape[0], x.shape[1]
                L_text_ctx = context.shape[1]

                g0, g1 = map(int, text_cut_positions["global"])
                shot_ranges = text_cut_positions.get("shots", [])
                S_shots = len(shot_ranges)

                if shot_indices is not None and S_shots > 0:
                    max_shot_id = shot_indices.max()
                    num_defined_shots = S_shots
                    assert max_shot_id < num_defined_shots, (
                        f"Error: Shot index out of bounds! The maximum shot ID in the data is {max_shot_id.item()}, "
                        f"but only {num_defined_shots} shots were defined (valid IDs are from 0 to {num_defined_shots - 1}). "
                        f"Please check your `shot_indices` and `text_cut_positions` inputs! "
                        f"prompt: {prompt}"
                    )

                max_end = (
                    max([g1] + [int(r[1]) for r in shot_ranges]) if shot_ranges else g1
                )
                L_text_pos = max_end + 1

                device, dtype = x.device, x.dtype

                global_mask = torch.zeros(L_text_ctx, dtype=torch.bool, device=device)
                global_mask[g0 : g1 + 1] = True

                if shot_ranges:
                    shot_table = torch.zeros(
                        S_shots, L_text_ctx, dtype=torch.bool, device=device
                    )
                    for sid, (s0, s1) in enumerate(shot_ranges):
                        s0 = int(s0)
                        s1 = int(s1)
                        shot_table[sid, s0 : s1 + 1] = True

                    if shot_indices is not None:
                        vid_shot = shot_indices.repeat_interleave(h * w, dim=1)
                        allow_shot = shot_table[vid_shot]
                        allow = allow_shot | global_mask.view(1, 1, L_text_ctx)
                    else:
                        allow = global_mask.view(1, 1, L_text_ctx)
                else:
                    allow = global_mask.view(1, 1, L_text_ctx)

                pad_mask = torch.zeros(L_text_ctx, dtype=torch.bool, device=device)
                if L_text_pos < L_text_ctx:
                    pad_mask[L_text_pos:] = True
                allow = allow | pad_mask.view(1, 1, L_text_ctx)

                block_value = -1e4
                bias = torch.zeros(B, S_q, L_text_ctx, dtype=dtype, device=device)
                bias = bias.masked_fill(~allow, block_value)
                attn_mask = bias.unsqueeze(1)
            except Exception as e:
                self.logger.error(f"Error in attention mask construction: {e}")
                attn_mask = None

        # Optional sparse self‑attention by shots
        use_sparse_self_attn = getattr(dit, "use_sparse_self_attn", False)
        if use_sparse_self_attn and shot_indices is not None:
            shot_latent_indices = shot_indices.repeat_interleave(h * w, dim=1)
            shot_latent_indices = self.labels_to_cuts(shot_latent_indices)
        else:
            shot_latent_indices = None

        # Main DiT blocks
        for block in dit.blocks:
            x = block(x, context, t_mod, freqs, attn_mask, shot_latent_indices, h * w)

        # Final head & unpatchify
        x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
        """
        1D sinusoidal positional embedding matching the HoloCine WanVideo implementation.
        """
        sinusoid = torch.outer(
            position.to(torch.float64),
            torch.pow(
                10000,
                -torch.arange(
                    dim // 2, dtype=torch.float64, device=position.device
                ).div(dim // 2),
            ),
        )
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    @staticmethod
    def labels_to_cuts(batch_labels: torch.Tensor):
        """
        Convert per‑token shot labels into cut indices per batch item.
        Ported from the original HoloCine pipeline implementation.
        """
        assert batch_labels.dim() == 2, "expect [b, s]"
        b, s = batch_labels.shape
        labs = batch_labels.to(torch.long)

        diffs = torch.zeros((b, s), dtype=torch.bool, device=labs.device)
        diffs[:, 1:] = labs[:, 1:] != labs[:, :-1]

        cuts_list = []
        for i in range(b):
            change_pos = torch.nonzero(diffs[i], as_tuple=False).flatten()
            cuts = [0]
            cuts.extend(change_pos.tolist())
            if cuts[-1] != s:
                cuts.append(s)
            cuts_list.append(cuts)
        return cuts_list

    def encode_prompt(self, prompt, positive=True, max_sequence_length: int = 512):

        device = self.device
        if self.text_encoder is None:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model(no_weights=False)
            self.text_encoder.model_loaded = True

        cleaned_prompt = self.text_encoder.prompt_clean(prompt)
        tokenizer = self.text_encoder.tokenizer
        prompt_parts = []

        global_match = re.search(r"\[global caption\]", cleaned_prompt)
        per_shot_match = re.search(r"\[per shot caption\]", cleaned_prompt)
        shot_cut_matches = list(re.finditer(r"\[shot cut\]", cleaned_prompt))

        if global_match is None:
            output = tokenizer(
                cleaned_prompt,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
            )

            ids = output["input_ids"].to(device)
            mask = output["attention_mask"].to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.text_encoder.model(ids, mask)[0]
            for i, v in enumerate(seq_lens):
                prompt_emb[:, v:] = 0
            return prompt_emb, {"global": None, "shots": []}

        if global_match:
            start_pos = global_match.start()
            end_pos = per_shot_match.start() if per_shot_match else len(cleaned_prompt)
            global_text = cleaned_prompt[start_pos:end_pos].strip()
            if global_text:
                prompt_parts.append({"id": -1, "text": global_text})

        if per_shot_match:
            current_start_pos = per_shot_match.start()
            shot_id = 0

            for shot_cut_match in shot_cut_matches:
                end_pos = shot_cut_match.start()
                shot_text = cleaned_prompt[current_start_pos:end_pos].strip()
                if shot_text:
                    prompt_parts.append({"id": shot_id, "text": shot_text})

                current_start_pos = shot_cut_match.start()
                shot_id += 1

            last_shot_text = cleaned_prompt[current_start_pos:].strip()
            if last_shot_text:
                prompt_parts.append({"id": shot_id, "text": last_shot_text})

        embeddings_list = []
        positions = {"global": None, "shots": {}}
        current_token_idx = 0

        for part in prompt_parts:
            text = part["text"]
            shot_id = part["id"]

            enc_output = tokenizer(
                text,
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=max_sequence_length,
                return_tensors="pt",
            )
            ids = enc_output["input_ids"].to(device)
            mask = enc_output["attention_mask"].to(device)

            part_emb = self.text_encoder.model(ids, mask)[0]

            seq_len = mask.sum().item()

            start_idx = current_token_idx
            end_idx = current_token_idx + seq_len

            if shot_id == -1:  # Global prompt
                positions["global"] = [start_idx, end_idx]
            else:  # Per-shot prompt
                positions["shots"][shot_id] = [start_idx, end_idx]

            embeddings_list.append(part_emb[0, :seq_len, :])

            current_token_idx += seq_len

        if not embeddings_list:
            return torch.zeros(
                1,
                max_sequence_length,
                self.text_encoder.model.config.hidden_size,
                device=device,
            ), {"global": None, "shots": []}

        concatenated_emb = torch.cat(
            embeddings_list, dim=0
        )  # shape: (total_seq_len, hidden_dim)

        total_len = concatenated_emb.shape[0]
        if total_len > max_sequence_length:
            concatenated_emb = concatenated_emb[:max_sequence_length, :]
            total_len = max_sequence_length

        pad_len = max_sequence_length - total_len

        prompt_emb = F.pad(concatenated_emb, (0, 0, 0, pad_len), "constant", 0)
        prompt_emb = prompt_emb.unsqueeze(0)

        final_positions = {"global": positions["global"], "shots": []}
        if positions["shots"]:

            sorted_shots = sorted(positions["shots"].items())

            max_shot_id = sorted_shots[-1][0]
            shot_map = dict(sorted_shots)
            for i in range(max_shot_id + 1):
                final_positions["shots"].append(shot_map.get(i, None))

        return prompt_emb, final_positions

    def check_sparse_self_attn(self, transformer_name: str):
        comp = self.get_component_by_name(transformer_name) or {}
        transformer = getattr(self, transformer_name)
        if transformer is not None:
            if comp.get("extra_kwargs", {}).get("sparse_self_attn", False):
                setattr(transformer, "use_sparse_self_attn", True)
            else:
                setattr(transformer, "use_sparse_self_attn", False)
        else:
            self.logger.warning(
                f"Transformer {transformer_name} not found. Skipping sparse self-attention check."
            )

    def run(
        self,
        prompt: str = None,
        global_caption: str | None = None,
        shot_captions: List[str] | None = None,
        duration: int | str = 241,
        shot_cut_frames: List[int] = None,
        shot_cut_points: List[float] = None,
        negative_prompt: str = None,
        fps: int = 15,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        tiled: bool = True,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float | List[float] = 5.0,
        boundary_ratio: float = 0.875,
        progress_callback: Callable = None,
        render_on_step_callback: Callable = None,
        render_on_step: bool = False,
        render_on_step_interval: int = 3,
        timesteps: List[int] | None = None,
        return_latents: bool = False,
        offload: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the inference pipeline.
        """
        # num_frames = None
        # shot_cut_frames = None
        safe_emit_progress(progress_callback, 0.0, "Starting holocine pipeline")
        safe_emit_progress(progress_callback, 0.05, "Preparing inputs")
        if global_caption and shot_captions:
            inputs = self.prepare_multishot_inputs(
                global_caption, shot_captions, duration, shot_cut_frames, fps
            )
            prompt = inputs["prompt"]
            shot_cut_frames = inputs["shot_cut_frames"]
            num_frames = inputs["num_frames"]
        elif prompt:
            num_frames = self._parse_num_frames(duration, fps)
            if shot_cut_points:
                shot_cut_frames = [int(point * fps) for point in shot_cut_points]
            else:
                shot_cut_frames = [
                    self._parse_num_frames(frame, fps) for frame in shot_cut_frames
                ]
            # remove any frames that are greater than the number of frames
            shot_cut_frames = [frame for frame in shot_cut_frames if frame < num_frames]

        safe_emit_progress(progress_callback, 0.10, "Preparing shot indices")

        shot_indices = self._process_shot_cut_frames(shot_cut_frames, num_frames)
        positive_context, positive_text_cut_positions = self.encode_prompt(
            prompt, positive=True
        )
        if negative_prompt:
            negative_context, _ = self.encode_prompt(negative_prompt, positive=False)
            negative_text_cut_positions = {"global": None, "shots": []}
        else:
            negative_context = None
            negative_text_cut_positions = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Preparing latents")

        latents = self._get_latents(
            height,
            width,
            num_frames,
            batch_size=1,
            dtype=torch.float32,
            generator=generator,
        )

        if self.scheduler is None:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        scheduler = self.scheduler
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(progress_callback, 0.20, "Preparing transformer")
        transformer_dtype = self.component_dtypes.get("transformer")
        boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.25, 0.90)
        total_steps = len(timesteps) if timesteps is not None else 0
        safe_emit_progress(progress_callback, 0.30, "Starting denoise")

        with self._progress_bar(total_steps, desc="Denoising") as pbar:
            for i, timestep in enumerate(timesteps):
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i) / float(total_steps) if total_steps else 0.0,
                    f"Denoising step {i + 1}/{total_steps}",
                )
                timestep = timestep.unsqueeze(0).to(
                    dtype=transformer_dtype, device=self.device
                )
                latent_model_input = latents.to(transformer_dtype)

                positive_input_kwargs = {
                    "latents": latent_model_input,
                    "timestep": timestep,
                    "context": positive_context.to(transformer_dtype),
                    "shot_indices": shot_indices,
                    "text_cut_positions": positive_text_cut_positions,
                    "prompt": prompt,
                }

                negative_input_kwargs = (
                    dict(
                        {
                            "latents": latent_model_input,
                            "timestep": timestep,
                            "context": negative_context.to(transformer_dtype),
                            "text_cut_positions": negative_text_cut_positions,
                            "prompt": negative_prompt,
                        }
                    )
                    if negative_context is not None
                    else None
                )

                if boundary_timestep is not None and timestep >= boundary_timestep:

                    if (
                        hasattr(self, "low_noise_transformer")
                        and self.low_noise_transformer
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Offloading previous transformer",
                        )
                        self._offload("low_noise_transformer")

                    if (
                        not hasattr(self, "high_noise_transformer")
                        or not self.high_noise_transformer
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading new transformer",
                        )

                        self.load_component_by_name("high_noise_transformer")
                        self.to_device(self.high_noise_transformer)
                        comp = (
                            self.get_component_by_name("high_noise_transformer") or {}
                        )
                        self.check_sparse_self_attn("high_noise_transformer")
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "New transformer ready",
                        )

                    transformer = self.high_noise_transformer

                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[0]
                else:
                    if (
                        hasattr(self, "high_noise_transformer")
                        and self.high_noise_transformer
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Switching model boundary, offloading previous transformer",
                        )
                        self._offload("high_noise_transformer")

                    if (
                        not hasattr(self, "low_noise_transformer")
                        or not self.low_noise_transformer
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading alternate transformer",
                        )
                        self.load_component_by_name("low_noise_transformer")
                        self.to_device(self.low_noise_transformer)
                        self.check_sparse_self_attn("low_noise_transformer")
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Alternate transformer ready",
                        )

                    transformer = self.low_noise_transformer
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[1]

                # Inference
                noise_pred_posi = self._run_inference_step(
                    **positive_input_kwargs, transformer=transformer
                )
                if guidance_scale != 1.0 and negative_context is not None:
                    noise_pred_nega = self._run_inference_step(
                        **negative_input_kwargs, transformer=transformer
                    )
                    noise_pred = noise_pred_nega + guidance_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred, timestep, latents, return_dict=False
                )[0]
                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != total_steps - 1
                ):
                    self._render_step(latents, render_on_step_callback)
                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(
                            min((i + 1) / total_steps, 1.0),
                            f"Denoising step {i + 1}/{total_steps}",
                        )
                    except Exception:
                        pass
                pbar.update(1)

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")
        if offload:
            if hasattr(self, "high_noise_transformer") and self.high_noise_transformer:
                self._offload("high_noise_transformer")
            if hasattr(self, "low_noise_transformer") and self.low_noise_transformer:
                self._offload("low_noise_transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to image")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(progress_callback, 1.0, "Completed holocine pipeline")
            return postprocessed_video
