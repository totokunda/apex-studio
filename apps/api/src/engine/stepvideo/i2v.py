import torch
from typing import Dict, Any, Callable, List
from PIL import Image
from torchvision import transforms as T
from .shared import StepVideoShared


class StepVideoI2VEngine(StepVideoShared):
    """StepVideo Image-to-Video Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: str | Image.Image | torch.Tensor,
        negative_prompt: List[str] | str = None,
        height: int = 550,
        width: int = 992,
        duration: int | str = 34,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 24,
        guidance_scale: float = 9.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_interval: int = 3,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        motion_score: float = 5.0,
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        use_cfg_guidance = (
            use_cfg_guidance and negative_prompt is not None and guidance_scale > 1.0
        )

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload("text_encoder")

        llm_preprocessor = self.helpers["stepvideo.text_encoder"]
        self.to_device(llm_preprocessor)
        llm_prompt_embeds, llm_mask = llm_preprocessor(
            prompt, with_mask=True, max_length=320
        )
        len_clip = prompt_embeds.shape[1]
        llm_mask = torch.nn.functional.pad(llm_mask, (len_clip, 0), value=1)

        if use_cfg_guidance:
            llm_negative_prompt_embeds, llm_negative_mask = llm_preprocessor(
                negative_prompt, with_mask=True, max_length=320
            )
            len_clip = negative_prompt_embeds.shape[1]
            llm_negative_mask = torch.nn.functional.pad(
                llm_negative_mask, (len_clip, 0), value=1
            )

        if offload:
            self._offload("stepvideo.text_encoder")

        transformer_dtype = self.component_dtypes["transformer"]
        batch_size = prompt_embeds.shape[0]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        self.to_device(self.scheduler)
        scheduler = self.scheduler

        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        num_frames = self._parse_num_frames(duration, fps)
        ## ensure its divisible by 17
        latent_num_frames = max(num_frames // 17 * 3, 1)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        num_channels_latents = self.transformer.config.in_channels

        latents = self._get_latents(
            height,
            width,
            latent_num_frames,
            num_channels_latents=num_channels_latents,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
            parse_frames=False,
            order="BFC",
        )

        loaded_image = self._load_image(image)
        preprocessed_image = T.ToTensor()(loaded_image) * 2 - 1
        preprocessed_image = self.resize_to_desired_aspect_ratio(
            preprocessed_image[None], aspect_size=[(height, width)]
        )[None]

        img_emb = self.vae_encode(preprocessed_image).repeat(num_videos, 1, 1, 1, 1)
        padding_tensor = torch.zeros(
            (
                num_videos,
                max(num_frames // 17 * 3, 1) - 1,
                num_channels_latents,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            ),
            device=self.device,
        )
        condition_hidden_states = torch.cat([img_emb, padding_tensor], dim=1)
        condition_hidden_states = condition_hidden_states.repeat(
            2 if use_cfg_guidance else 1, 1, 1, 1, 1
        ).to(self.device, dtype=transformer_dtype)

        if use_cfg_guidance:
            encoder_hidden_states = torch.cat(
                [llm_prompt_embeds, llm_negative_prompt_embeds], dim=0
            )
            encoder_attention_mask = torch.cat([llm_mask, llm_negative_mask], dim=0)
            encoder_hidden_states_2 = torch.cat(
                [prompt_embeds, negative_prompt_embeds], dim=0
            )
        else:
            encoder_hidden_states = llm_prompt_embeds
            encoder_attention_mask = llm_mask
            encoder_hidden_states_2 = prompt_embeds

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            transformer_kwargs=dict(
                encoder_hidden_states=encoder_hidden_states.to(
                    self.device, dtype=transformer_dtype
                ),
                encoder_hidden_states_2=encoder_hidden_states_2.to(
                    self.device, dtype=transformer_dtype
                ),
                encoder_attention_mask=encoder_attention_mask.to(self.device),
                condition_hidden_states=condition_hidden_states,
                motion_score=motion_score,
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            render_on_step_interval=render_on_step_interval,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            video = video.permute(0, 2, 1, 3, 4)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
