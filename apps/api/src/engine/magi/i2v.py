import torch
from typing import Dict, Any, Callable, List, Union
from PIL import Image
import math
from einops import rearrange
from tqdm import tqdm
from .shared import MagiShared


class MagiI2VEngine(MagiShared):
    """Magi Image-to-Video Engine Implementation"""

    def run(
        self,
        image: Union[str, torch.Tensor, Image.Image],
        prompt: Union[List[str], str],
        height: int = 512,
        width: int = 512,
        duration: str | int = 96,
        num_inference_steps: int = 64,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 24,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        chunk_width: int = 6,
        noise2clean_kvrange: List[int] = [5, 4, 3, 2],
        clean_chunk_kvrange: int = 1,
        cfg_number: int = 3,
        cfg_t_range: List[float] = [0.0, 0.0217, 0.1, 0.3, 0.999],
        text_scales: List[float] = [7.5, 7.5, 7.5, 0.0, 0.0],
        prev_chunk_scales: List[float] = [1.5, 1.5, 1.5, 1.0, 1.0],
        distill_nearly_clean_chunk_threshold: float = 0.3,
        window_size: int = 4,
        distill: bool = False,
        kv_offload: bool = True,
        **kwargs,
    ):
        """Image-to-video generation using MAGI's chunk-based approach"""

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds, prompt_embeds_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            return_attention_mask=True,
            **text_encoder_kwargs,
        )

        batch_size = prompt_embeds.shape[0]

        loaded_image = self._load_image(image)

        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        transformer_dtype = self.component_dtypes.get("transformer", torch.bfloat16)

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).unsqueeze(2)

        prefix_video = self.vae_encode(
            preprocessed_image,
            offload=offload,
            dtype=transformer_dtype,
            normalize_latents_dtype=transformer_dtype,
            sample_mode="mode",
        )

        if offload:
            self._offload("text_encoder")

        num_frames = self._parse_num_frames(duration, fps)
        num_chunks = math.ceil(
            (num_frames // self.vae_scale_factor_temporal * 1.0 + prefix_video.size(2))
            / chunk_width
        )

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        null_caption_embeds = self.transformer.get_null_caption_embeds(
            device=self.device,
            num_videos_per_prompt=num_videos,
        )

        prompt_embeds, prompt_embeds_mask = self._process_txt_embeddings(
            caption_embs=prompt_embeds,
            emb_masks=prompt_embeds_mask,
            null_emb=null_caption_embeds,
            infer_chunk_num=num_chunks,
            clean_chunk_num=0,
        )

        null_emb_masks = torch.zeros_like(prompt_embeds_mask)
        null_embs, null_emb_masks = self.process_null_embeddings(
            null_caption_embedding=null_caption_embeds,
            null_emb_masks=null_emb_masks,
            infer_chunk_num=num_chunks,
        )

        if prompt_embeds_mask.sum() == 0:
            prompt_embeds_mask = torch.cat([null_emb_masks, null_emb_masks], dim=0)
            prompt_embeds = torch.cat([null_embs, null_embs])
        else:
            prompt_embeds_mask = torch.cat([prompt_embeds_mask, null_emb_masks], dim=0)
            prompt_embeds = torch.cat([prompt_embeds, null_embs])

        latent = self._get_latents(
            height=height,
            width=width,
            duration=chunk_width * num_chunks,
            batch_size=batch_size,
            parse_frames=False,
            generator=generator,
            seed=seed,
        )

        latent = torch.cat([latent, latent], dim=0)

        timesteps = self._get_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps
        )

        time_interval = self.scheduler.set_time_interval(
            num_inference_steps, self.device
        )

        latents = self.denoise(
            latents=latent,
            timesteps=timesteps,
            num_chunks=num_chunks,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_inference_steps=num_inference_steps,
            chunk_width=chunk_width,
            render_on_step_callback=render_on_step_callback,
            render_on_step=render_on_step,
            time_interval=time_interval,
            noise2clean_kvrange=noise2clean_kvrange,
            cfg_number=cfg_number,
            text_scales=text_scales,
            prev_chunk_scales=prev_chunk_scales,
            cfg_t_range=cfg_t_range,
            window_size=window_size,
            distill_nearly_clean_chunk_threshold=distill_nearly_clean_chunk_threshold,
            transformer_dtype=transformer_dtype,
            prefix_video=prefix_video,
            clean_chunk_kvrange=clean_chunk_kvrange,
            kv_offload=kv_offload,
            distill=distill,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return torch.cat(latents, dim=2)
        else:
            videos = []
            for latent in tqdm(latents, desc="Decoding latents"):
                video = self.vae_decode(latent, offload=False)
                video = self._tensor_to_frames(video, output_type="pil")[0]
                videos.extend(video)
            return [videos]
