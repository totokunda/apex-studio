import torch
from src.utils.type import EnumType
from typing import List, Dict, Any, Tuple
from collections import Counter


class UnconditionGuard:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.prev_state = {
            "range_num": kwargs["range_num"],
            "denoising_range_num": kwargs["denoising_range_num"],
            "slice_point": kwargs["slice_point"],
            "fwd_extra_1st_chunk": kwargs["fwd_extra_1st_chunk"],
        }

    def __enter__(self):
        if self.kwargs.get("fwd_extra_1st_chunk", False):
            self.kwargs["denoising_range_num"] -= 1
            self.kwargs["slice_point"] += 1
            self.kwargs["fwd_extra_1st_chunk"] = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kwargs["range_num"] = self.prev_state["range_num"]
        self.kwargs["denoising_range_num"] = self.prev_state["denoising_range_num"]
        self.kwargs["slice_point"] = self.prev_state["slice_point"]
        self.kwargs["fwd_extra_1st_chunk"] = self.prev_state["fwd_extra_1st_chunk"]


def check_forward(module, inputs, outputs):
    for i in inputs:
        if isinstance(i, torch.Tensor):
            if not torch.isfinite(i).all():
                raise RuntimeError(f"NaN/Inf in forward of {module.__class__.__name__}")
    if isinstance(outputs, torch.Tensor):
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"NaN/Inf in forward of {module.__class__.__name__}")
    else:
        for o in outputs:
            if isinstance(o, torch.Tensor):
                if not torch.isfinite(o).all():
                    raise RuntimeError(
                        f"NaN/Inf in forward of {module.__class__.__name__}"
                    )


class MagiDenoise:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        latents = kwargs.get("latents", None)
        num_chunks = kwargs.get("num_chunks", None)
        time_interval = kwargs.get("time_interval", None)
        prompt_embeds = kwargs.get("prompt_embeds", None)
        prompt_embeds_mask = kwargs.get("prompt_embeds_mask", None)
        num_inference_steps = kwargs.get("num_inference_steps", 64)
        chunk_width = kwargs.get("chunk_width", 6)
        window_size = kwargs.get("window_size", 4)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        prefix_video = kwargs.get("prefix_video", None)
        noise2clean_kvrange = kwargs.get("noise2clean_kvrange", [5, 4, 3, 2])
        clean_chunk_kvrange = kwargs.get("clean_chunk_kvrange", 1)
        kv_offload = kwargs.get("kv_offload", True)

        distill_nearly_clean_chunk_threshold = kwargs.get(
            "distill_nearly_clean_chunk_threshold", 0.3
        )

        cfg_number = kwargs.get("cfg_number", 3)
        text_scales = kwargs.get("text_scales", [7.5, 7.5, 7.5, 0.0, 0.0])
        prev_chunk_scales = kwargs.get("prev_chunk_scales", [1.5, 1.5, 1.5, 1.0, 1.0])
        cfg_t_range = kwargs.get("cfg_t_range", [0.0, 0.0217, 0.1, 0.3, 0.999])
        transformer_dtype = kwargs.get("transformer_dtype", torch.bfloat16)
        distill = kwargs.get("distill", False)

        denoise_step_per_stage = num_inference_steps // window_size
        total_steps = self.total_steps(
            denoise_step_per_stage, num_chunks, chunk_width, window_size, prefix_video
        )

        post_patch_height = latents.size(3) // self.transformer.config.patch_size
        post_patch_width = latents.size(4) // self.transformer.config.patch_size
        chunk_token_nums = chunk_width * post_patch_height * post_patch_width

        max_sequence_length = (
            latents.shape[2]
            * (latents.shape[3] // self.transformer.config.patch_size)
            * (latents.shape[4] // self.transformer.config.patch_size)
        )

        kv_cache_params = self._init_kv_cache_params(
            max_batch_size=latents.size(0) // 2,
            max_sequence_length=max_sequence_length,
        )

        self.scheduler.set_scheduler_params(chunk_width, denoise_step_per_stage)
        self.transformer.set_distill(distill)

        for name, module in self.transformer.named_modules():
            module.register_forward_hook(check_forward)

        default_model_kwargs = {
            "chunk_width": chunk_width,
            "num_steps": num_inference_steps,
            "fwd_extra_1st_chunk": False,
            "transformer_dtype": transformer_dtype,
            "kv_offload": kv_offload,
        }

        chunk_denoise_count = Counter()
        clean_latents = []

        with self._progress_bar(total_steps, desc=f"Sampling Magi") as pbar:
            for denoise_step in range(total_steps):

                (
                    denoise_idx,
                    chunk_offset,
                    chunk_start,
                    chunk_end,
                    t_start,
                    t_end,
                ) = self.generate_denoise_status_and_sequences(
                    denoise_step,
                    denoise_step_per_stage,
                    num_chunks,
                    chunk_width,
                    window_size,
                    prefix_video,
                )

                model_kwargs = default_model_kwargs.copy()

                if chunk_offset > 0 and denoise_step == 0:
                    self.cache_prefix_video(
                        prefix_video,
                        prompt_embeds,
                        prompt_embeds_mask,
                        chunk_offset,
                        chunk_width,
                        clean_chunk_kvrange,
                        noise2clean_kvrange,
                        kv_cache_params,
                        cfg_number,
                        text_scales,
                        prev_chunk_scales,
                        cfg_t_range,
                        model_kwargs,
                    )

                latents_chunk = latents[
                    :, :, chunk_start * chunk_width : chunk_end * chunk_width
                ].clone()

                prompt_embeds_chunk = prompt_embeds[:, chunk_start:chunk_end]
                prompt_embeds_mask_chunk = prompt_embeds_mask[:, chunk_start:chunk_end]

                model_kwargs.update(
                    {
                        "slice_point": chunk_start,
                        "range_num": chunk_end,
                        "denoising_range_num": chunk_end - chunk_start,
                    }
                )

                fwd_extra_1st_chunk = chunk_start > chunk_offset and denoise_idx == 0

                if fwd_extra_1st_chunk:
                    clean_x = latents[
                        :,
                        :,
                        (chunk_start - 1) * chunk_width : chunk_start * chunk_width,
                    ].clone()
                    latents_chunk = torch.cat([clean_x, latents_chunk], dim=2)

                    # clean feature without y embedding
                    prompt_embeds_chunk = torch.cat(
                        [
                            prompt_embeds[1:2, 0:1].expand(
                                prompt_embeds_chunk.size(0), -1, -1, -1
                            ),
                            prompt_embeds_chunk,
                        ],
                        dim=1,
                    )

                    prompt_embeds_mask_chunk = torch.cat(
                        [
                            prompt_embeds_mask[1:2, 1:2].expand(
                                prompt_embeds_mask_chunk.size(0), -1, -1
                            ),
                            prompt_embeds_mask_chunk,
                        ],
                        dim=1,
                    )

                    model_kwargs["slice_point"] = chunk_start - 1
                    model_kwargs["denoising_range_num"] = chunk_end - chunk_start + 1
                    model_kwargs["fwd_extra_1st_chunk"] = True

                prompt_embeds_chunk_flatten = prompt_embeds_chunk.flatten(
                    start_dim=0, end_dim=1
                ).unsqueeze(1)
                prompt_embeds_mask_chunk_flatten = prompt_embeds_mask_chunk.flatten(
                    start_dim=0, end_dim=1
                ).unsqueeze(1)

                denoise_step_of_each_chunk = self.get_denoise_step_of_each_chunk(
                    num_inference_steps,
                    denoise_step_per_stage,
                    t_start,
                    t_end,
                    denoise_idx,
                    has_clean_t=fwd_extra_1st_chunk,
                )

                timestep = self.scheduler.get_timestep(
                    t_start,
                    t_end,
                    denoise_idx,
                    has_clean_t=fwd_extra_1st_chunk,
                )

                timestep = timestep.unsqueeze(0).repeat(latents_chunk.size(0), 1)

                kv_range = self.generate_kvrange_for_denoising_video(
                    batch_size=latents_chunk.shape[0] // 2,
                    chunk_token_nums=chunk_token_nums,
                    num_steps=num_inference_steps,
                    slice_point=model_kwargs["slice_point"],
                    denoising_range_num=model_kwargs["denoising_range_num"],
                    denoise_step_of_each_chunk=denoise_step_of_each_chunk,
                    noise2clean_kvrange=noise2clean_kvrange,
                    clean_chunk_kvrange=clean_chunk_kvrange,
                )

                if prefix_video is not None:
                    latents_chunk, timestep = self.prepare_prefix_video(
                        latents_chunk,
                        timestep,
                        prefix_video,
                        model_kwargs["slice_point"] * chunk_width,
                        chunk_width,
                    )

                nearly_clean_chunk_t = timestep[
                    0, int(model_kwargs["fwd_extra_1st_chunk"])
                ].item()
                model_kwargs["distill_nearly_clean_chunk"] = (
                    nearly_clean_chunk_t > distill_nearly_clean_chunk_threshold
                )
                model_kwargs["distill_interval"] = time_interval[denoise_idx]

                flow_pred = self.forward(
                    hidden_states=latents_chunk,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_chunk_flatten,
                    encoder_hidden_states_mask=prompt_embeds_mask_chunk_flatten,
                    kv_range=kv_range,
                    kv_cache_params=kv_cache_params,
                    cfg_number=cfg_number,
                    text_scales=text_scales,
                    prev_chunk_scales=prev_chunk_scales,
                    cfg_t_range=cfg_t_range,
                    **model_kwargs,
                )

                if fwd_extra_1st_chunk:
                    latents_chunk = latents_chunk[:, :, chunk_width:]
                    flow_pred = flow_pred[:, :, chunk_width:]

                latents_chunk = self.scheduler.step(
                    latents_chunk,
                    flow_pred,
                    t_start,
                    t_end,
                    denoise_idx,
                    return_dict=False,
                )[0]

                for chunk_index in range(chunk_start, chunk_end):
                    chunk_denoise_count[chunk_index] += 1

                if render_on_step_callback is not None:
                    self._render_step(latents_chunk, render_on_step_callback)

                latents[:, :, chunk_start * chunk_width : chunk_end * chunk_width] = (
                    latents_chunk
                )

                if chunk_denoise_count[chunk_start] == num_inference_steps:
                    if prefix_video is not None:
                        prefix_video_length = prefix_video.size(2)
                        if (chunk_start + 1) * chunk_width <= prefix_video_length:
                            return None, None

                        real_start = max(chunk_start * chunk_width, prefix_video_length)

                        # Keep the first 4-frames only for I2V Job
                        if chunk_start == 0 and prefix_video_length == 1:
                            real_start = 0

                        clean_latents_chunk, _ = latents[
                            :, :, real_start : (chunk_start + 1) * chunk_width
                        ].chunk(2, dim=0)
                    else:
                        clean_latents_chunk, _ = latents[
                            :,
                            :,
                            chunk_start * chunk_width : (chunk_start + 1) * chunk_width,
                        ].chunk(2, dim=0)

                    clean_latents.append(clean_latents_chunk)

                pbar.update(1)

            self.logger.info("Denoising completed.")

        return clean_latents

    def total_steps(
        self,
        denoise_step_per_stage: int,
        chunk_num: int,
        chunk_width: int,
        window_size: int,
        prefix_video: torch.Tensor | None = None,
    ) -> int:

        chunk_offset = 0
        if prefix_video is not None:
            chunk_offset = prefix_video.size(2) // chunk_width

        total_forward_steps = denoise_step_per_stage * (
            chunk_num + window_size - 1 - chunk_offset
        )

        return total_forward_steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        kv_range: torch.Tensor,
        kv_cache_params: Dict[str, Any],
        cfg_number: int,
        text_scales: List[float],
        prev_chunk_scales: List[float],
        cfg_t_range: List[float],
        **kwargs,
    ) -> torch.Tensor:
        """Forward method for transformer"""
        if cfg_number == 3:
            out_cond_pre_and_text, out_cond_pre, out_uncond, denoise_width = (
                self.forward_3cfg(
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    kv_range,
                    kv_cache_params,
                    **kwargs,
                )
            )
            prev_chunk_scale_s = torch.tensor(prev_chunk_scales).to(self.device)
            text_scale_s = torch.tensor(text_scales).to(self.device)
            cfg_t_range = torch.tensor(cfg_t_range).to(self.device)
            applied_cfg_range_num, chunk_width = (
                kwargs["denoising_range_num"],
                kwargs["chunk_width"],
            )
            if kwargs["fwd_extra_1st_chunk"]:
                applied_cfg_range_num -= 1
            cfg_timestep = timestep[0, -applied_cfg_range_num:]

            assert len(prev_chunk_scale_s) == len(
                cfg_t_range
            ), "prev_chunks_scale and t_range should have the same length"
            assert len(text_scale_s) == len(
                cfg_t_range
            ), "text_scale and t_range should have the same length"

            cfg_output_list = []

            for chunk_idx in range(applied_cfg_range_num):
                prev_chunk_scale, text_scale = self.get_cfg_scale(
                    cfg_timestep[chunk_idx],
                    cfg_t_range,
                    prev_chunk_scale_s,
                    text_scale_s,
                )
                l = chunk_idx * chunk_width
                r = (chunk_idx + 1) * chunk_width
                cfg_output = (
                    (1 - prev_chunk_scale) * out_uncond[:, :, l:r]
                    + (prev_chunk_scale - text_scale)
                    * out_cond_pre[:, :, -denoise_width:][:, :, l:r]
                    + text_scale
                    * out_cond_pre_and_text[:, :, -denoise_width:][:, :, l:r]
                )
                cfg_output_list.append(cfg_output)

            cfg_output = torch.cat(cfg_output_list, dim=2)

            hidden_states = torch.cat(
                [hidden_states[0:1, :, :-denoise_width], cfg_output], dim=2
            )
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            return hidden_states
        elif cfg_number == 1:
            assert hidden_states.shape[0] == 2
            hidden_states = torch.cat([hidden_states[0:1], hidden_states[0:1]], dim=0)

            kwargs["caption_dropout_mask"] = torch.tensor(
                [False], dtype=torch.bool, device=hidden_states.device
            )

            kv_cache_params["update_kv_cache"] = True
            if kwargs.get("distill_nearly_clean_chunk", False):
                prev_chunks_scale = float(prev_chunk_scales[0])
                slice_start = 1 if kwargs["fwd_extra_1st_chunk"] else 0
                cond_pre_and_text_channel = hidden_states.shape[2]
                new_hidden_states_chunk = hidden_states[
                    0:1,
                    :,
                    slice_start
                    * kwargs["chunk_width"] : (slice_start + 1)
                    * kwargs["chunk_width"],
                ]
                new_kvrange = self._generate_kv_range_for_uncondition(
                    new_hidden_states_chunk
                )
                kwargs["denoising_range_num"] += 1
                cat_hidden_states_chunk = torch.cat(
                    [hidden_states[0:1], new_hidden_states_chunk], dim=2
                )
                new_kvrange = new_kvrange + kv_range.max()
                cat_kvrange = torch.cat([kv_range, new_kvrange], dim=0)
                cat_timestep = torch.cat(
                    [timestep[0:1], timestep[0:1, slice_start : slice_start + 1]], dim=1
                )
                cat_encoder_hidden_states = torch.cat(
                    [
                        encoder_hidden_states[0 : encoder_hidden_states.shape[0] // 2],
                        encoder_hidden_states[slice_start : slice_start + 1],
                    ],
                    dim=0,
                )
                cat_encoder_hidden_states_mask = torch.cat(
                    [
                        encoder_hidden_states_mask[
                            0 : encoder_hidden_states_mask.shape[0] // 2
                        ],
                        encoder_hidden_states_mask[slice_start : slice_start + 1],
                    ],
                    dim=0,
                )

                cat_out = self.transformer(
                    cat_hidden_states_chunk,
                    cat_timestep,
                    cat_encoder_hidden_states,
                    cat_encoder_hidden_states_mask,
                    kv_range=cat_kvrange,
                    kv_cache_params=kv_cache_params,
                    **kwargs,
                    return_dict=False,
                )[0]

                near_clean_out_cond_pre_and_text = cat_out[
                    :,
                    :,
                    slice_start
                    * kwargs["chunk_width"] : (slice_start + 1)
                    * kwargs["chunk_width"],
                ]
                near_clean_out_cond_text = cat_out[:, :, cond_pre_and_text_channel:]
                near_out_cond_pre_and_text = (
                    near_clean_out_cond_pre_and_text * prev_chunks_scale
                    + near_clean_out_cond_text * (1 - prev_chunks_scale)
                )
                cat_out[
                    :,
                    :,
                    slice_start
                    * kwargs["chunk_width"] : (slice_start + 1)
                    * kwargs["chunk_width"],
                ] = near_out_cond_pre_and_text
                out_cond_pre_and_text = cat_out[:, :, :cond_pre_and_text_channel]
            else:
                out_cond_pre_and_text = self.transformer(
                    hidden_states[0:1],
                    timestep[0:1],
                    encoder_hidden_states[0 : encoder_hidden_states.shape[0] // 2],
                    encoder_hidden_states_mask[
                        0 : encoder_hidden_states_mask.shape[0] // 2
                    ],
                    kv_range=kv_range,
                    kv_cache_params=kv_cache_params,
                    **kwargs,
                    return_dict=False,
                )[0]

            denoise_width = kwargs["chunk_width"] * kwargs["denoising_range_num"]
            if kwargs["fwd_extra_1st_chunk"]:
                denoise_width -= kwargs["chunk_width"]

            hidden_states = torch.cat(
                [
                    hidden_states[0:1, :, :-denoise_width],
                    out_cond_pre_and_text[:, :, -denoise_width:],
                ],
                dim=2,
            )
            hidden_states = torch.cat([hidden_states[0:1], hidden_states[0:1]], dim=0)
            return hidden_states
        else:
            raise NotImplementedError(f"Invalid cfg_number: {cfg_number}")

    def forward_3cfg(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        kv_range,
        kv_cache_params,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        assert hidden_states.shape[0] == 2
        assert (
            encoder_hidden_states_mask.shape[0] % 2 == 0
        )  # mask should be a multiple of 2
        hidden_states = torch.cat([hidden_states[0:1], hidden_states[0:1]], dim=0)
        caption_dropout_mask = torch.tensor(
            [False, True], dtype=torch.bool, device=hidden_states.device
        )

        kv_cache_params["update_kv_cache"] = False

        out_cond_pre_and_text = self.transformer(
            hidden_states[0:1],
            timestep[0:1],
            encoder_hidden_states[0 : encoder_hidden_states.shape[0] // 2],
            encoder_hidden_states_mask[0 : encoder_hidden_states_mask.shape[0] // 2],
            caption_dropout_mask=caption_dropout_mask[0:1],
            kv_range=kv_range,
            kv_cache_params=kv_cache_params,
            **kwargs,
            return_dict=False,
        )[0]

        kv_cache_params["update_kv_cache"] = True

        out_cond_pre = self.transformer(
            hidden_states[1:2],
            timestep[1:2],
            encoder_hidden_states[
                encoder_hidden_states.shape[0] // 2 : encoder_hidden_states.shape[0]
            ],
            caption_dropout_mask=caption_dropout_mask[1:2],
            encoder_hidden_states_mask=encoder_hidden_states_mask[
                encoder_hidden_states_mask.shape[0]
                // 2 : encoder_hidden_states_mask.shape[0]
            ],
            kv_range=kv_range,
            kv_cache_params=kv_cache_params,
            **kwargs,
            return_dict=False,
        )[0]

        def chunk_to_batch(input, denoising_range_num):
            input = input.squeeze(0)
            input = input.reshape(
                -1, denoising_range_num, kwargs["chunk_width"], *input.shape[2:]
            )
            return input.transpose(
                0, 1
            )  # (denoising_range_num, chn, chunk_width, h, w)

        def batch_to_chunk(input, denoising_range_num):
            input = input.transpose(0, 1)
            input = input.reshape(
                1, -1, denoising_range_num * kwargs["chunk_width"], *input.shape[3:]
            )
            return input

        with UnconditionGuard(kwargs):
            denoising_range_num = kwargs["denoising_range_num"]
            denoise_width = kwargs["chunk_width"] * denoising_range_num
            uncond_hidden_states = chunk_to_batch(
                hidden_states[0:1, :, -denoise_width:], denoising_range_num
            )
            timestep = timestep[0:1, -denoising_range_num:].transpose(0, 1)
            uncond_encoder_hidden_states = encoder_hidden_states[
                encoder_hidden_states.shape[0] // 2 : encoder_hidden_states.shape[0]
            ][-denoising_range_num:]
            caption_dropout_mask = torch.tensor(
                [True], dtype=torch.bool, device=hidden_states.device
            )
            uncond_encoder_hidden_states_mask = encoder_hidden_states_mask[
                encoder_hidden_states_mask.shape[0]
                // 2 : encoder_hidden_states_mask.shape[0]
            ][-denoising_range_num:]
            uncond_kv_range = self._generate_kv_range_for_uncondition(
                uncond_hidden_states
            )

            kwargs["range_num"] = 1
            kwargs["denoising_range_num"] = 1
            kwargs["slice_point"] = 0

            out_uncond = self.transformer(
                uncond_hidden_states,
                timestep,
                uncond_encoder_hidden_states,
                caption_dropout_mask=caption_dropout_mask,
                encoder_hidden_states_mask=uncond_encoder_hidden_states_mask,
                kv_range=uncond_kv_range,
                kv_cache_params=kv_cache_params,
                **kwargs,
                return_dict=False,
            )[0]
            out_uncond = batch_to_chunk(out_uncond, denoising_range_num)

        return out_cond_pre_and_text, out_cond_pre, out_uncond, denoise_width

    def get_denoise_step_of_each_chunk(
        self,
        num_inference_steps: int,
        denoise_step_per_stage: int,
        t_start: int,
        t_end: int,
        denoise_idx: int,
        has_clean_t: bool = False,
    ):
        denoise_step_of_each_chunk = []
        for i in range(t_start, t_end):
            denoise_step_of_each_chunk.append(i * denoise_step_per_stage + denoise_idx)
        denoise_step_of_each_chunk.reverse()
        if has_clean_t:
            denoise_step_of_each_chunk = [
                num_inference_steps
            ] + denoise_step_of_each_chunk
        return denoise_step_of_each_chunk

    def get_cfg_scale(self, t, cfg_t_range, prev_chunk_scale_s, text_scale_s):
        indices = torch.searchsorted(cfg_t_range - 1e-7, t) - 1
        assert indices.min() >= 0 and indices.max() < len(prev_chunk_scale_s)
        return prev_chunk_scale_s[indices], text_scale_s[indices]

    def generate_denoise_status_and_sequences(
        self,
        cur_denoise_step: int,
        denoise_step_per_stage: int,
        chunk_num: int,
        chunk_width: int,
        window_size: int,
        prefix_video: torch.Tensor | None = None,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int]]:
        """Const Method"""
        chunk_offset = 0
        if prefix_video is not None:
            chunk_offset = prefix_video.size(2) // chunk_width

        denoise_stage, denoise_idx = (
            cur_denoise_step // denoise_step_per_stage,
            cur_denoise_step % denoise_step_per_stage,
        )
        chunk_start_s, chunk_end_s, t_start_s, t_end_s = self.generate_sequences(
            chunk_num, window_size, chunk_offset
        )
        chunk_start, chunk_end, t_start, t_end = (
            chunk_start_s[denoise_stage],
            chunk_end_s[denoise_stage],
            t_start_s[denoise_stage],
            t_end_s[denoise_stage],
        )

        return (
            denoise_idx,
            chunk_offset,
            chunk_start,
            chunk_end,
            t_start,
            t_end,
        )

    def generate_sequences(self, chunk_num, window_size, chunk_offset):
        # Adjust range to include the offset
        start_index = chunk_offset
        end_index = chunk_num + window_size - 1

        # Generate clip_start and clip_end
        clip_start = [
            max(chunk_offset, i - window_size + 1)
            for i in range(start_index, end_index)
        ]
        clip_end = [min(chunk_num, i + 1) for i in range(start_index, end_index)]

        # Generate t_start and t_end
        t_start = [max(0, i - chunk_num + 1) for i in range(start_index, end_index)]
        t_end = [
            (
                min(window_size, i - chunk_offset + 1)
                if i - chunk_offset < window_size
                else window_size
            )
            for i in range(start_index, end_index)
        ]

        return clip_start, clip_end, t_start, t_end

    def generate_noise2clean_kvrange(
        self,
        batch_size: int,
        chunk_token_nums: int,
        num_steps: int,
        slice_point: int,
        denoising_range_num: int,
        noise2clean_kvrange: List[int],
        clean_chunk_kvrange: int,
        denoise_step_of_each_chunk: List[int],
    ) -> torch.Tensor:
        """Const Method"""
        assert len(denoise_step_of_each_chunk) == denoising_range_num
        assert len(noise2clean_kvrange) > 0

        if clean_chunk_kvrange == -1:
            clean_chunk_kvrange = noise2clean_kvrange[-1]

        assert num_steps % len(noise2clean_kvrange) == 0
        denoise_step_per_stage = num_steps // len(noise2clean_kvrange)
        denoise_kv_range = []

        for cur_chunk_denoise_step in denoise_step_of_each_chunk:
            if cur_chunk_denoise_step == num_steps:
                denoise_kv_range.append(clean_chunk_kvrange)
            else:
                denoise_kv_range.append(
                    noise2clean_kvrange[
                        cur_chunk_denoise_step // denoise_step_per_stage
                    ]
                )

        range_num = slice_point + denoising_range_num

        k_ranges = []
        for i in range(batch_size):
            k_batch_start = i * range_num
            for j in range(denoising_range_num):
                k_chunk_end = slice_point + j + 1
                k_chunk_start = max(0, k_chunk_end - denoise_kv_range[j])
                k_ranges.append(
                    torch.Tensor(
                        [
                            (k_batch_start + k_chunk_start) * chunk_token_nums,
                            (k_batch_start + k_chunk_end) * chunk_token_nums,
                        ]
                    )
                    .reshape(1, 2)
                    .to(self.device)
                )
        k_range = torch.concat(k_ranges, dim=0).to(torch.int32).to(self.device)
        return k_range

    def generate_kvrange_for_denoising_video(
        self,
        batch_size: int,
        chunk_token_nums: int,
        num_steps: int,
        slice_point: int,
        denoising_range_num: int,
        denoise_step_of_each_chunk: List[int],
        noise2clean_kvrange: List[int],
        clean_chunk_kvrange: int,
    ) -> torch.Tensor:
        """Const Method"""
        if len(noise2clean_kvrange) == 0:
            k_range = self.generate_default_kvrange(
                batch_size, chunk_token_nums, slice_point, denoising_range_num
            )
        else:
            k_range = self.generate_noise2clean_kvrange(
                batch_size,
                chunk_token_nums,
                num_steps,
                slice_point,
                denoising_range_num,
                noise2clean_kvrange,
                clean_chunk_kvrange,
                denoise_step_of_each_chunk,
            )
        return k_range

    def generate_default_kvrange(
        self,
        batch_size: int,
        chunk_token_nums: int,
        slice_point: int,
        denoising_range_num: int,
    ) -> torch.Tensor:
        """Const Method"""
        range_num = slice_point + denoising_range_num

        k_chunk_end = torch.linspace(
            slice_point + 1, range_num, steps=denoising_range_num
        ).reshape((denoising_range_num, 1))
        k_chunk_start = torch.Tensor([0] * denoising_range_num).reshape(
            (denoising_range_num, 1)
        )
        k_chunk_range = torch.concat([k_chunk_start, k_chunk_end], dim=1)
        k_batch_range = (
            torch.concat(
                [k_chunk_range + i * range_num for i in range(batch_size)], dim=0
            )
            .to(torch.int32)
            .to(self.device)
        )
        return k_batch_range * chunk_token_nums

    def _init_kv_cache_params(
        self, max_batch_size: int, max_sequence_length: int
    ) -> Dict[str, Any]:
        """Const Method"""
        return {
            "key_value_memory_dict": {},
            "update_kv_cache": False,
            "sequence_length_offset": 0,
            "max_batch_size": max_batch_size,
            "max_sequence_length": max_sequence_length,
        }

    def _generate_kv_range_for_uncondition(self, uncond_x) -> torch.Tensor:
        B, C, T, H, W = uncond_x.shape
        chunk_token_nums = (
            (T // self.transformer.config.t_patch_size)
            * (H // self.transformer.config.patch_size)
            * (W // self.transformer.config.patch_size)
        )

        k_chunk_start = torch.linspace(0, (B - 1) * chunk_token_nums, steps=B).reshape(
            (B, 1)
        )
        k_chunk_end = torch.linspace(
            chunk_token_nums, B * chunk_token_nums, steps=B
        ).reshape((B, 1))
        return (
            torch.concat([k_chunk_start, k_chunk_end], dim=1)
            .to(torch.int32)
            .to(self.device)
        )

    def generate_kvrange_for_prefix_video(
        self,
        batch_size,
        chunk_token_nums,
        clean_chunk_kvrange,
        noise2clean_kvrange,
        range_num,
    ):
        """Const Method"""
        if clean_chunk_kvrange != -1:
            prev_chunk_num = clean_chunk_kvrange
        elif len(noise2clean_kvrange) > 0:
            prev_chunk_num = noise2clean_kvrange[-1]
        else:
            prev_chunk_num = 8

        k_chunk_end = torch.linspace(1, range_num, steps=range_num).reshape(
            (range_num, 1)
        )
        k_chunk_start = torch.clamp(k_chunk_end - prev_chunk_num, min=0).reshape(
            (range_num, 1)
        )
        k_chunk_range = torch.concat([k_chunk_start, k_chunk_end], dim=1)
        k_batch_range = (
            torch.concat(
                [k_chunk_range + i * range_num for i in range(batch_size)], dim=0
            )
            .to(torch.int32)
            .to(self.device)
        )
        return k_batch_range * chunk_token_nums

    def cache_prefix_video(
        self,
        prefix_video,
        prompt_embeds,
        prompt_embeds_mask,
        chunk_offset,
        chunk_width,
        clean_chunk_kvrange,
        noise2clean_kvrange,
        kv_cache_params,
        cfg_number,
        text_scales,
        prev_chunk_scales,
        cfg_t_range,
        model_kwargs,
    ):
        hidden_states_chunk = prefix_video[:, :, : chunk_offset * chunk_width]
        hidden_states_chunk = torch.cat(
            [hidden_states_chunk, hidden_states_chunk], dim=0
        )

        null_prompt_embeds = prompt_embeds[1:2, :chunk_offset]
        null_prompt_embeds = torch.cat([null_prompt_embeds, null_prompt_embeds], dim=0)
        null_prompt_embeds_mask = prompt_embeds_mask[1:2, :chunk_offset]
        null_prompt_embeds_mask = torch.cat(
            [null_prompt_embeds_mask, null_prompt_embeds_mask], dim=0
        )

        null_prompt_embeds = null_prompt_embeds.flatten(
            start_dim=0, end_dim=1
        ).unsqueeze(1)
        null_prompt_embeds_mask = null_prompt_embeds_mask.flatten(
            start_dim=0, end_dim=1
        ).unsqueeze(1)

        timestep = torch.ones(chunk_offset, device=self.device) * self.scheduler.clean_t
        timestep = timestep.unsqueeze(0).repeat(hidden_states_chunk.size(0), 1)

        model_kwargs.update(
            {
                "slice_point": 0,
                "range_num": chunk_offset,
                "denoising_range_num": chunk_offset,
                "fwd_extra_1st_chunk": False,
                "extract_prefix_video_feature": True,
                "transformer_dtype": self.component_dtypes.get(
                    "transformer", torch.bfloat16
                ),
                "distill_interval": self.scheduler.time_interval[0],
            }
        )

        kv_range = self.generate_kvrange_for_prefix_video(
            prefix_video.shape[0],
            chunk_width,
            clean_chunk_kvrange,
            noise2clean_kvrange,
            chunk_offset,
        )

        self.forward(
            hidden_states=hidden_states_chunk,
            timestep=timestep,
            encoder_hidden_states=null_prompt_embeds,
            encoder_hidden_states_mask=null_prompt_embeds_mask,
            kv_range=kv_range,
            kv_cache_params=kv_cache_params,
            cfg_number=cfg_number,
            text_scales=text_scales,
            prev_chunk_scales=prev_chunk_scales,
            cfg_t_range=cfg_t_range,
            **model_kwargs,
        )

    def prepare_prefix_video(
        self,
        x_chunk: torch.Tensor,
        t: torch.Tensor,
        prefix_video: torch.Tensor,
        prefix_video_start: int,
        chunk_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-Const Method"""
        prefix_length = prefix_video.size(2)

        if prefix_length <= prefix_video_start:
            return x_chunk, t

        padding_length = min(prefix_length - prefix_video_start, x_chunk.size(2))
        prefix_video_end = prefix_video_start + padding_length
        ret = x_chunk.clone()

        ret[:, :, :padding_length] = prefix_video[
            :, :, prefix_video_start:prefix_video_end
        ]

        num_clean_t = (prefix_length - prefix_video_start) // chunk_width
        if num_clean_t > 0:
            t[:, :num_clean_t] = 1.0
        return ret, t
