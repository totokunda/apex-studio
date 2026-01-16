from src.types import InputVideo, InputImage
import torch
from PIL import Image
from src.engine.flashvsr.shared import FlashVSRShared
from typing import Optional, Callable
from src.transformer.wan.flashvsr.model import sinusoidal_embedding_1d
import numpy as np
from tqdm import tqdm
from src.utils.cache import empty_cache
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.utils.assets import get_asset_path

# Don't hard-fail at import time; fail when the engine is actually used.
context_path = get_asset_path("flashvsr", "posi_prompt.pth", must_exist=False)
from src.vae.tiny_wan.model import AutoencoderKLTinyWan


class FlashVSRUpscaleEngine(FlashVSRShared):
    """FlashVSR Upscale Engine Implementation"""

    @staticmethod
    def largest_8n1_leq(n):  # 8n+1
        return 0 if n < 1 else ((n - 1) // 8) * 8 + 1

    def pil_to_tensor_neg1_1(self, img: Image.Image, dtype=torch.bfloat16):
        device = self.device
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(
            device=device, dtype=torch.float32
        )  # HWC
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
        return t.to(dtype)

    def prepare_input_tensor(
        self,
        video: InputVideo = None,
        image: InputImage = None,
        height: int = None,
        width: int = None,
        dtype=torch.bfloat16,
        progress_callback: Callable[[float, str], None] | None = None,
        fps: float = 24.0,
    ):

        if image is not None:
            image = self._load_image(image)
            image, tH, tW = self._aspect_ratio_resize(
                image, max_area=height * width, mod_value=128
            )
            paths = [image] * 25
            F = self.largest_8n1_leq(len(paths))
            paths = paths[:F]
            frames = []
            prep_progress = make_mapped_progress(progress_callback, 0.0, 1.0)
            for p in tqdm(paths, desc="Preparing input tensor"):
                frames.append(self.pil_to_tensor_neg1_1(p, dtype))
                prep_progress(
                    len(frames) / max(1, len(paths)), "Preparing input tensor"
                )
            vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
            return vid, tH, tW, F

        elif video is not None:
            video = self._load_video(video, fps=fps)
            total = len(video)
            idx = list(range(total)) + [total - 1] * 4
            F = self.largest_8n1_leq(len(idx))
            idx = idx[:F]
            frames = []
            prep_progress = make_mapped_progress(progress_callback, 0.0, 1.0)
            for i in tqdm(idx, desc="Preparing input tensor"):
                image, tH, tW = self._aspect_ratio_resize(
                    video[i], max_area=height * width, mod_value=128
                )
                frames.append(self.pil_to_tensor_neg1_1(image, dtype))
                prep_progress(len(frames) / max(1, len(idx)), "Preparing input tensor")
            vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
            return vid, tH, tW, F
        else:
            raise ValueError("video or image is required")

    def inference_step(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        tea_cache=None,
        use_unified_sequence_parallel: bool = False,
        LQ_latents: Optional[torch.Tensor] = None,
        is_full_block: bool = False,
        is_stream: bool = False,
        pre_cache_k: Optional[list[torch.Tensor]] = None,
        pre_cache_v: Optional[list[torch.Tensor]] = None,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        cur_process_idx: int = 0,
        t_mod: torch.Tensor = None,
        t: torch.Tensor = None,
        local_range: int = 9,
        **kwargs,
    ):
        # patchify
        x, (f, h, w) = self.transformer.patchify(x)

        win = (2, 8, 8)
        seqlen = f // win[0]
        local_num = seqlen
        window_size = win[0] * h * w // 128
        square_num = window_size * window_size
        topk = int(square_num * topk_ratio) - 1
        kv_len = int(kv_ratio)

        # RoPE 位置（分段）
        if cur_process_idx == 0:
            freqs = (
                torch.cat(
                    [
                        self.transformer.freqs[0][:f]
                        .view(f, 1, 1, -1)
                        .expand(f, h, w, -1),
                        self.transformer.freqs[1][:h]
                        .view(1, h, 1, -1)
                        .expand(f, h, w, -1),
                        self.transformer.freqs[2][:w]
                        .view(1, 1, w, -1)
                        .expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )
        else:
            freqs = (
                torch.cat(
                    [
                        self.transformer.freqs[0][
                            4 + cur_process_idx * 2 : 4 + cur_process_idx * 2 + f
                        ]
                        .view(f, 1, 1, -1)
                        .expand(f, h, w, -1),
                        self.transformer.freqs[1][:h]
                        .view(1, h, 1, -1)
                        .expand(f, h, w, -1),
                        self.transformer.freqs[2][:w]
                        .view(1, 1, w, -1)
                        .expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )

        # TeaCache（默认不启用）
        tea_cache_update = (
            tea_cache.check(self.transformer, x, t_mod)
            if tea_cache is not None
            else False
        )

        # Block 堆叠
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            for block_id, block in enumerate(self.transformer.blocks):
                if LQ_latents is not None and block_id < len(LQ_latents):
                    x = x + LQ_latents[block_id]
                x, last_pre_cache_k, last_pre_cache_v = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    f,
                    h,
                    w,
                    local_num,
                    topk,
                    block_id=block_id,
                    kv_len=kv_len,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=(
                        pre_cache_k[block_id] if pre_cache_k is not None else None
                    ),
                    pre_cache_v=(
                        pre_cache_v[block_id] if pre_cache_v is not None else None
                    ),
                    local_range=local_range,
                )
                if pre_cache_k is not None:
                    pre_cache_k[block_id] = last_pre_cache_k
                if pre_cache_v is not None:
                    pre_cache_v[block_id] = last_pre_cache_v

        x = self.transformer.head(x, t)

        x = self.transformer.unpatchify(x, (f, h, w))
        return x, pre_cache_k, pre_cache_v

    def run(
        self,
        video: InputVideo = None,
        image: InputImage = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 1,
        kv_ratio: float = 3.0,
        local_range: int = 11,
        color_fix: bool = True,
        sparse_ratio: float = 2.0,
        seed: int = None,
        buffer: bool = True,
        is_full_block: bool = False,
        height: int = None,
        width: int = None,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        offload: bool = True,
        tile_sample_min_height: int = 60,
        tile_sample_min_width: int = 104,
        tile_sample_stride_height: int = 30,
        tile_sample_stride_width: int = 52,
        continuous_decode: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        **kwargs,
    ):

        assert video is not None or image is not None, "video or image is required"

        safe_emit_progress(progress_callback, 0.0, "Starting FlashVSR upscale")
        safe_emit_progress(progress_callback, 0.02, "Preparing input tensor")

        # Reserve a span for input preparation [0.02, 0.18]
        input_prep_progress = make_mapped_progress(progress_callback, 0.02, 0.18)
        vid, tH, tW, num_frames = self.prepare_input_tensor(
            video,
            image,
            height=height,
            width=width,
            progress_callback=input_prep_progress,
        )

        if num_frames % 4 != 1:
            self.logger.warning(f"num_frames % 4 != 1, padding to {num_frames}")
            num_frames = (num_frames + 2) // 4 * 4 + 1
        topk_ratio = sparse_ratio * 768 * 1280 / (tH * tW)
        transformer_dtype = self.component_dtypes["transformer"]
        if buffer:
            input_num_frames = (num_frames - 1) // 4
        else:
            input_num_frames = (num_frames - 1) // 4 + 1

        safe_emit_progress(progress_callback, 0.20, "Initializing latent noise")
        noise = self._get_latents(
            height=tH,
            width=tW,
            duration=input_num_frames,
            seed=seed,
            dtype=torch.float32,
            device=self.device,
            parse_frames=False,
        )
        noise = noise.to(dtype=transformer_dtype, device=self.device)
        latents = noise

        process_total_num = (num_frames - 1) // 8 - 2
        is_stream = True
        if self.transformer is None:
            safe_emit_progress(progress_callback, 0.22, "Loading transformer")
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        safe_emit_progress(progress_callback, 0.24, "Loading prompt context")
        context = torch.load(context_path, map_location="cpu", weights_only=False).to(
            device=self.device, dtype=transformer_dtype
        )
        self.prompt_emb_posi = {}
        self.prompt_emb_posi["context"] = context

        if hasattr(self.transformer, "reinit_cross_kv"):
            self.transformer.reinit_cross_kv(context)

        self.transformer.LQ_proj_in.clear_cache()
        latents_total = []
        frames_total = []
        if self.scheduler is None:
            safe_emit_progress(progress_callback, 0.26, "Loading scheduler")
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        safe_emit_progress(progress_callback, 0.28, "Configuring timesteps")
        self.timestep = torch.tensor(
            [1000.0], device=self.device, dtype=transformer_dtype
        )
        self.t = self.transformer.time_embedding(
            sinusoidal_embedding_1d(self.transformer.freq_dim, self.timestep)
        )
        self.t_mod = self.transformer.time_projection(self.t).unflatten(
            1, (6, self.transformer.dim)
        )
        self.scheduler.set_timesteps(
            1, denoising_strength=denoising_strength, shift=shift
        )

        LQ_pre_idx = 0
        LQ_cur_idx = 0
        clean_mem = True

        # Reserve inference progress span [0.30, 0.88]
        infer_progress = make_mapped_progress(progress_callback, 0.30, 0.88)
        infer_progress(0.0, "Starting FlashVSR inference")

        with self._progress_bar(total=process_total_num) as progress_bar:

            for cur_process_idx in range(process_total_num):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.transformer.blocks)
                    pre_cache_v = [None] * len(self.transformer.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        cur = (
                            self.transformer.LQ_proj_in.stream_forward(
                                vid[
                                    :,
                                    :,
                                    max(0, inner_idx * 4 - 3) : (inner_idx + 1) * 4 - 3,
                                    :,
                                    :,
                                ]
                            )
                            if vid is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                    LQ_cur_idx = (inner_loop_num - 1) * 4 - 3
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        cur = (
                            self.transformer.LQ_proj_in.stream_forward(
                                vid[
                                    :,
                                    :,
                                    cur_process_idx * 8
                                    + 17
                                    + inner_idx * 4 : cur_process_idx * 8
                                    + 21
                                    + inner_idx * 4,
                                    :,
                                    :,
                                ]
                            )
                            if vid is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                    LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    cur_latents = latents[
                        :, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2, :, :
                    ]

                # 推理（无 motion_controller / vace）
                noise_pred_posi, pre_cache_k, pre_cache_v = self.inference_step(
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range=local_range,
                )

                # 更新 latent
                cur_latents = cur_latents - noise_pred_posi

                if continuous_decode:
                    cond = vid[:, :, LQ_pre_idx:LQ_cur_idx, :, :].to(self.device)
                    cur_frames = self.vae_decode(
                        cur_latents,
                        cond=cond,
                        clean_mem=clean_mem,
                        offload=offload,
                        offload_type="cpu",
                    )
                    try:
                        if color_fix:
                            cur_frames = self.color_corrector(
                                cur_frames.to(device=self.device),
                                cond,
                                clip_range=(-1, 1),
                                chunk_size=None,
                                method="adain",
                            )
                    except:
                        pass

                    frames_total.append(cur_frames.cpu())
                else:
                    latents_total.append(cur_latents)
                clean_mem = False
                LQ_pre_idx = LQ_cur_idx
                progress_bar.update(1)
                infer_progress(
                    (cur_process_idx + 1) / max(1, process_total_num),
                    f"FlashVSR Inference {int(((cur_process_idx + 1) / max(1, process_total_num)) * 100)}%",
                )

        # Decode
        if offload:
            safe_emit_progress(progress_callback, 0.89, "Offloading transformer")
            self._offload("transformer")

        empty_cache()

        if not continuous_decode:
            safe_emit_progress(progress_callback, 0.92, "Decoding latents")
            latents = torch.cat(latents_total, dim=2)
            vae_dtype = self.component_dtypes["vae"]
            cond = vid[:, :, :LQ_cur_idx, :, :].to(dtype=vae_dtype, device=self.device)
            frames = self.vae_decode(latents, cond=cond, offload=offload)
            try:
                if color_fix:
                    safe_emit_progress(
                        progress_callback, 0.95, "Applying color correction"
                    )
                    frames = self.color_corrector(
                        frames.to(device=vid.device),
                        vid[:, :, : frames.shape[2], :, :],
                        clip_range=(-1, 1),
                        chunk_size=16,
                        method="adain",
                    )
            except:
                pass
        else:
            safe_emit_progress(progress_callback, 0.92, "Collecting decoded frames")
            frames = torch.cat(frames_total, dim=2)

        safe_emit_progress(progress_callback, 0.98, "Converting output")
        if image is not None:
            frames = frames[:, :, :1, :, :]
            safe_emit_progress(progress_callback, 1.0, "FlashVSR upscale complete")
            return self._tensor_to_frame(frames)
        else:
            safe_emit_progress(progress_callback, 1.0, "FlashVSR upscale complete")
            return self._tensor_to_frames(frames)

    def vae_decode(
        self,
        latents: torch.Tensor,
        cond: torch.Tensor = None,
        offload: bool = False,
        clean_mem: bool = True,
        offload_type: str = "discard",
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        if isinstance(self.vae, AutoencoderKLTinyWan):
            if clean_mem:
                self.vae.clean_mem()
            frames = self.vae.stream_decode_with_cond(latents, cond=cond)

            if clean_mem:
                self.vae.clean_mem()
            if offload:
                self._offload("vae", offload_type=offload_type)
            return frames
        else:
            return super().vae_decode(latents, offload=offload)

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable, timestep: Optional[torch.Tensor] = None, image: Optional[bool] = False):
        self.logger.warning("Rendering step not supported for FlashVSR upscale")