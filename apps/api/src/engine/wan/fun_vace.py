import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from src.helpers.wan.fun_camera import Camera
from src.types import InputImage, InputVideo
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared
from einops import rearrange
import inspect
import math
from diffusers.utils.torch_utils import randn_tensor
import gc
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor
import math


class WanFunVACEEngine(WanShared):
    """WAN Control Engine Implementation for camera control and video guidance"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.num_channels_latents = 16

    def _get_sampling_sigmas(self, sampling_steps, shift):
        sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
        sigma = shift * sigma / (1 + (shift - 1) * sigma)

        return sigma

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        offload: bool = True,
        progress_callback: Callable | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device
        safe_emit_progress(progress_callback, 0.05, "Preparing prompt encoding")

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.10, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.20, "Text encoder loaded")
        safe_emit_progress(progress_callback, 0.25, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.30, "Text encoder on device")

        if prompt_embeds is None:
            safe_emit_progress(progress_callback, 0.40, "Encoding prompt embeddings")
            prompt_embeds = self.text_encoder.encode(
                text=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                use_attention_mask=True,
                device=device,
                dtype=dtype,
            )
            safe_emit_progress(progress_callback, 0.65, "Prompt embeddings ready")

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            safe_emit_progress(
                progress_callback, 0.75, "Encoding negative prompt embeddings"
            )
            negative_prompt_embeds = self.text_encoder.encode(
                text=negative_prompt,
                use_attention_mask=True,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            safe_emit_progress(
                progress_callback, 0.90, "Negative prompt embeddings ready"
            )

        if offload:
            safe_emit_progress(progress_callback, 0.95, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")

        return prompt_embeds, negative_prompt_embeds

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        num_length_latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (
                (num_frames - 1) // self.vae.temporal_compression_ratio + 1
                if num_length_latents is None
                else num_length_latents
            ),
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        weight_dtype = frames.dtype
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae_encode(frames, sample_mode="mode")
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0).to(weight_dtype) for m in masks]
            inactive = torch.stack(
                [i * (1 - m) + 0 * m for i, m in zip(frames, masks)], dim=0
            )
            reactive = torch.stack(
                [i * m + 0 * (1 - m) for i, m in zip(frames, masks)], dim=0
            )
            inactive = self.vae_encode(inactive, sample_mode="mode")
            reactive = self.vae_encode(reactive, sample_mode="mode")
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                refs = torch.stack(refs, dim=0)
                if masks is None:
                    ref_latent = self.vae_encode(refs, sample_mode="mode")
                else:
                    ref_latent = self.vae_encode(refs, sample_mode="mode")
                    ref_latent = [
                        torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent
                    ]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=[4, 8, 8]):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(
                mask.unsqueeze(0), size=(new_depth, height, width), mode="nearest-exact"
            ).squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
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

    def padding_image(self, images, new_width, new_height):
        new_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))

        aspect_ratio = images.width / images.height
        if new_width / new_height > 1:
            if aspect_ratio > new_width / new_height:
                new_img_width = new_width
                new_img_height = int(new_img_width / aspect_ratio)
            else:
                new_img_height = new_height
                new_img_width = int(new_img_height * aspect_ratio)
        else:
            if aspect_ratio > new_width / new_height:
                new_img_width = new_width
                new_img_height = int(new_img_width / aspect_ratio)
            else:
                new_img_height = new_height
                new_img_width = int(new_img_height * aspect_ratio)

        resized_img = images.resize((new_img_width, new_img_height))

        paste_x = (new_width - new_img_width) // 2
        paste_y = (new_height - new_img_height) // 2

        new_image.paste(resized_img, (paste_x, paste_y))

        return new_image

    def get_image_latent(self, ref_image=None, sample_size=None, padding=False):
        if ref_image is not None:
            ref_image = self._load_image(ref_image)
            if padding:
                ref_image = self.padding_image(
                    ref_image, sample_size[1], sample_size[0]
                )
            ref_image = self._aspect_ratio_resize(
                ref_image, max_area=math.prod(sample_size)
            )[0]
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

        return ref_image

    def get_image_to_video_latent(
        self, validation_image_start, validation_image_end, video_length, sample_size
    ):
        if validation_image_start is not None and validation_image_end is not None:

            image_start = clip_image = self._load_image(validation_image_start)
            image_start = self._aspect_ratio_resize(
                image_start, max_area=math.prod(sample_size)
            )[0]
            clip_image = self._aspect_ratio_resize(
                clip_image, max_area=math.prod(sample_size)
            )[0]

            image_end = self._load_image(validation_image_end)
            image_end = self._aspect_ratio_resize(
                image_end, max_area=math.prod(sample_size)
            )[0]

            input_video = torch.tile(
                torch.from_numpy(np.array(image_start))
                .permute(2, 0, 1)
                .unsqueeze(1)
                .unsqueeze(0),
                [1, 1, video_length, 1, 1],
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

            image_end = self._aspect_ratio_resize(
                image_end, max_area=math.prod(sample_size)
            )[0]
            input_video[:, :, -1:] = (
                torch.from_numpy(np.array(image_end))
                .permute(2, 0, 1)
                .unsqueeze(1)
                .unsqueeze(0)
            )
            input_video_mask[:, :, -1:] = 0

            input_video = input_video / 255

        elif validation_image_start is not None:

            image_start = clip_image = self._load_image(validation_image_start)
            image_start = self._aspect_ratio_resize(
                image_start, max_area=math.prod(sample_size)
            )[0]
            clip_image = self._aspect_ratio_resize(
                clip_image, max_area=math.prod(sample_size)
            )[0]
            image_end = None
            input_video = (
                torch.tile(
                    torch.from_numpy(np.array(image_start))
                    .permute(2, 0, 1)
                    .unsqueeze(1)
                    .unsqueeze(0),
                    [1, 1, video_length, 1, 1],
                )
                / 255
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[
                :,
                :,
                1:,
            ] = 255
        else:
            image_start = None
            image_end = None
            input_video = torch.zeros(
                [1, 3, video_length, sample_size[0], sample_size[1]]
            )
            input_video_mask = (
                torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
            )
            clip_image = None

        del image_start
        del image_end
        gc.collect()

        return input_video, input_video_mask, clip_image

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def get_video_to_video_latent(
        self,
        input_video_path,
        video_length,
        sample_size,
        fps=16,
        validation_video_mask=None,
        ref_image=None,
    ):
        if input_video_path is not None:
            input_video = self._load_video(
                input_video_path, num_frames=video_length, fps=fps
            )
            input_video = [
                self._aspect_ratio_resize(frame, max_area=math.prod(sample_size))[0]
                for frame in input_video
            ]
            input_video = torch.from_numpy(np.array(input_video))[:video_length]
            input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

            if validation_video_mask is not None:
                validation_video_mask = (
                    Image.open(validation_video_mask)
                    .convert("L")
                    .resize((sample_size[1], sample_size[0]))
                )
                input_video_mask = np.where(
                    np.array(validation_video_mask) < 240, 0, 255
                )

                input_video_mask = (
                    torch.from_numpy(np.array(input_video_mask))
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .permute([3, 0, 1, 2])
                    .unsqueeze(0)
                )
                input_video_mask = torch.tile(
                    input_video_mask, [1, 1, input_video.size()[2], 1, 1]
                )
                input_video_mask = input_video_mask.to(
                    input_video.device, input_video.dtype
                )
            else:
                input_video_mask = torch.zeros_like(input_video[:, :1])
                input_video_mask[:, :, :] = 255
        else:
            input_video, input_video_mask = None, None

        if ref_image is not None:
            clip_image = self._load_image(ref_image)
        else:
            clip_image = None

        if ref_image is not None:
            ref_image = self._load_image(ref_image)
            ref_image = self._aspect_ratio_resize(
                ref_image, max_area=math.prod(sample_size)
            )[0]
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

        return input_video, input_video_mask, ref_image, clip_image

    def run(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        video: InputVideo = None,
        start_image: InputImage | str = None,
        end_image: InputImage | str = None,
        subject_ref_images: List[InputImage | str] = None,
        mask_video: InputVideo = None,
        control_video: InputVideo = None,
        control_camera_poses: List[float] | str | List[Camera] | Camera = None,
        duration: int | str = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float | List[float] = 6.0,
        high_noise_guidance_scale: float = 6.0,
        low_noise_guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_latents: bool = False,
        max_sequence_length: int = 512,
        boundary_ratio: float = 0.875,
        render_on_step: bool = False,
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        sigmas: Optional[List[float]] = None,
        vace_context_scale: float = 1.0,
        offload: bool = True,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting FunVACE pipeline")
        safe_emit_progress(progress_callback, 0.02, "Preparing inputs")
        video_length = self._parse_num_frames(duration, fps)
        height = height // 16 * 16
        width = width // 16 * 16
        sample_size = [height, width]
        if subject_ref_images is not None:
            safe_emit_progress(
                progress_callback, 0.03, "Loading subject reference images"
            )
            subject_ref_images = [
                self.get_image_latent(
                    _subject_ref_image, sample_size=sample_size, padding=True
                )
                for _subject_ref_image in subject_ref_images
            ]
            subject_ref_images = torch.cat(subject_ref_images, dim=2)

        if video is not None:
            if mask_video is None:
                raise ValueError("mask_video is required when video is provided")
            safe_emit_progress(
                progress_callback, 0.05, "Loading input video + mask video"
            )
            video, _, _, _ = self.get_video_to_video_latent(
                video,
                video_length=video_length,
                sample_size=sample_size,
                fps=fps,
                ref_image=None,
            )
            mask_video, _, _, _ = self.get_video_to_video_latent(
                mask_video,
                video_length=video_length,
                sample_size=sample_size,
                fps=fps,
                ref_image=None,
            )
            mask_video = mask_video[:, :1]
        else:
            safe_emit_progress(
                progress_callback, 0.05, "Preparing start/end images to video latents"
            )
            video, mask_video, clip_image = self.get_image_to_video_latent(
                start_image,
                end_image,
                video_length=video_length,
                sample_size=sample_size,
            )

        safe_emit_progress(progress_callback, 0.06, "Loading control video")
        control_video, _, _, _ = self.get_video_to_video_latent(
            control_video,
            video_length=video_length,
            sample_size=sample_size,
            fps=fps,
            ref_image=None,
        )

        num_videos_per_prompt = 1

        if (
            high_noise_guidance_scale is not None
            and low_noise_guidance_scale is not None
        ):
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(
                progress_callback, 0.01, "Using high/low-noise guidance scales"
            )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        dtype = self.component_dtypes["text_encoder"]
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if isinstance(guidance_scale, list):
            do_classifier_free_guidance = any(
                [guidance_scale_i > 1.0 for guidance_scale_i in guidance_scale]
            )
        else:
            do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        encode_progress_callback = make_mapped_progress(progress_callback, 0.06, 0.18)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            progress_callback=encode_progress_callback,
        )
        safe_emit_progress(progress_callback, 0.18, "Prompts ready")

        # 4. Prepare timesteps
        if self.scheduler is None:
            safe_emit_progress(progress_callback, 0.19, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.20, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.21, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.22, "Scheduler on device")
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        safe_emit_progress(progress_callback, 0.23, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
        )
        safe_emit_progress(progress_callback, 0.24, "Timesteps ready")
        self._num_timesteps = len(timesteps)
        weight_dtype = self.component_dtypes["transformer"]

        # 5. Prepare latents.
        if mask_video is not None:
            safe_emit_progress(progress_callback, 0.25, "Preprocessing mask video")
            bs, _, video_length, height, width = video.size()
            mask_condition = self.mask_processor.preprocess(
                rearrange(mask_video, "b c f h w -> (b f) c h w"),
                height=height,
                width=width,
            )
            mask_condition = mask_condition.to(dtype=torch.float32)
            mask_condition = rearrange(
                mask_condition, "(b f) c h w -> b c f h w", f=video_length
            )
            mask_condition = torch.tile(mask_condition, [1, 3, 1, 1, 1]).to(
                dtype=weight_dtype, device=device
            )

        if control_video is not None:
            safe_emit_progress(progress_callback, 0.27, "Preprocessing control video")
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(
                rearrange(control_video, "b c f h w -> (b f) c h w"),
                height=height,
                width=width,
            )
            control_video = control_video.to(dtype=torch.float32)
            input_video = rearrange(
                control_video, "(b f) c h w -> b c f h w", f=video_length
            )

            input_video = input_video.to(dtype=weight_dtype, device=device)

        elif video is not None:
            safe_emit_progress(progress_callback, 0.27, "Preprocessing input video")
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(
                rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width
            )
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(
                init_video, "(b f) c h w -> b c f h w", f=video_length
            ).to(dtype=weight_dtype, device=device)

            input_video = init_video * (mask_condition < 0.5)
            input_video = input_video.to(dtype=weight_dtype, device=device)

        if subject_ref_images is not None:
            safe_emit_progress(
                progress_callback, 0.29, "Preprocessing subject reference images"
            )
            video_length = subject_ref_images.shape[2]
            subject_ref_images = self.image_processor.preprocess(
                rearrange(subject_ref_images, "b c f h w -> (b f) c h w"),
                height=height,
                width=width,
            )
            subject_ref_images = subject_ref_images.to(dtype=torch.float32)
            subject_ref_images = rearrange(
                subject_ref_images, "(b f) c h w -> b c f h w", f=video_length
            )
            subject_ref_images = subject_ref_images.to(
                dtype=weight_dtype, device=device
            )

            bs, c, f, h, w = subject_ref_images.size()
            new_subject_ref_images = []
            for i in range(bs):
                new_subject_ref_images.append([])
                for j in range(f):
                    new_subject_ref_images[i].append(
                        subject_ref_images[i, :, j : j + 1]
                    )
            subject_ref_images = new_subject_ref_images

        safe_emit_progress(progress_callback, 0.31, "Encoding VACE context")
        vace_latents = self.vace_encode_frames(
            input_video, subject_ref_images, masks=mask_condition, vae=self.vae
        )
        mask_latents = self.vace_encode_masks(mask_condition, subject_ref_images)
        vace_context = self.vace_latent(vace_latents, mask_latents)
        safe_emit_progress(progress_callback, 0.34, "VACE context ready")

        # 5. Prepare latents.
        safe_emit_progress(progress_callback, 0.35, "Initializing latent noise")
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            self.num_channels_latents,
            video_length,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
            num_length_latents=vace_latents[0].size(1),
        )
        safe_emit_progress(progress_callback, 0.38, "Latents ready")

        safe_emit_progress(progress_callback, 0.39, "Loading transformer config")
        self.transformer_config = self.load_config_by_name(
            "high_noise_transformer", "transformer"
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        safe_emit_progress(progress_callback, 0.40, "Preparing scheduler step kwargs")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        transformer_dtype = self.component_dtypes["transformer"]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        if boundary_ratio is not None:
            safe_emit_progress(progress_callback, 0.41, "Computing boundary timestep")
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        safe_emit_progress(progress_callback, 0.42, "Computing denoise sequence length")
        transformer_config = self.load_config_by_name(
            "high_noise_transformer", "transformer"
        )
        target_shape = (
            self.num_channels_latents,
            vace_latents[0].size(1),
            vace_latents[0].size(2),
            vace_latents[0].size(3),
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (
                transformer_config["patch_size"][1]
                * transformer_config["patch_size"][2]
            )
            * target_shape[1]
        )
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoise (CFG: {'on' if do_classifier_free_guidance else 'off'})",
        )
        # 7. Denoising loop
        transformer_dtype = self.component_dtypes["transformer"]
        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            transformer_kwargs=dict(
                context=prompt_embeds,
                vace_context=vace_context,
                seq_len=seq_len,
                vace_context_scale=vace_context_scale,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    context=negative_prompt_embeds,
                    vace_context=vace_context,
                    seq_len=seq_len,
                    vace_context_scale=vace_context_scale,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            denoise_progress_callback=denoise_progress_callback,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        if subject_ref_images is not None:
            safe_emit_progress(
                progress_callback, 0.91, "Trimming subject reference frames"
            )
            len_subject_ref_images = len(subject_ref_images[0])
            latents = latents[:, :, len_subject_ref_images:, :, :]

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            safe_emit_progress(progress_callback, 0.94, "Decoding latents to video")
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(progress_callback, 1.0, "Completed control pipeline")
            return postprocessed_video
