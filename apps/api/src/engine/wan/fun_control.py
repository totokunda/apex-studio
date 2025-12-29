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

class WanFunControlEngine(WanShared):
    """WAN Fun Control Engine Implementation for camera control and video guidance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    
    def _get_sampling_sigmas(self, sampling_steps, shift):
        sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
        sigma = (shift * sigma / (1 + (shift - 1) * sigma))

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
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

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

            safe_emit_progress(progress_callback, 0.75, "Encoding negative prompt embeddings")
            negative_prompt_embeds = self.text_encoder.encode(
                text=negative_prompt,
                use_attention_mask=True,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            safe_emit_progress(progress_callback, 0.90, "Negative prompt embeddings ready")
            
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
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        
    
        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae_encode(mask_bs, sample_mode="mode")
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae_encode(mask_pixel_values_bs, sample_mode="mode")
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae_encode(control_bs, sample_mode="mode")
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae_encode(control_pixel_values_bs, sample_mode="mode")
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents
    
    def _resize_mask(self, mask, latent, process_first_frame_only=True):
        latent_size = latent.size()
        batch_size, channels, num_frames, height, width = mask.shape

        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )

            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )
                resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
            else:
                resized_mask = first_frame_resized
        else:
            target_size = list(latent_size[2:])
            resized_mask = F.interpolate(
                mask,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
        return resized_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    
    def padding_image(images, new_width, new_height):
        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

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
                ref_image = self.padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

        return ref_image

    def get_image_to_video_latent(self, validation_image_start, validation_image_end, video_length, sample_size):
        if validation_image_start is not None and validation_image_end is not None:
            
            image_start = clip_image = self._load_image(validation_image_start)
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
           
            image_end = self._load_image(validation_image_end)
            image_end = image_end.resize([sample_size[1], sample_size[0]])

            input_video = torch.tile(
                    torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                    [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

            input_video = input_video / 255

        elif validation_image_start is not None:
            
            image_start = clip_image = self._load_image(validation_image_start)
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
            image_end = None
            input_video = torch.tile(
                    torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                    [1, 1, video_length, 1, 1]
                ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
        else:
            image_start = None
            image_end = None
            input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
            input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
            clip_image = None

        del image_start
        del image_end
        gc.collect()

        return  input_video, input_video_mask, clip_image

    def get_video_to_video_latent(self, input_video_path, video_length, sample_size, fps=16, validation_video_mask=None, ref_image=None):
        if input_video_path is not None:
            input_video = self._load_video(input_video_path, num_frames=video_length, fps=fps)
            input_video = torch.from_numpy(np.array(input_video))[:video_length]
            input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

            if validation_video_mask is not None:
                validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
                input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)

                input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
                input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
                input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
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
            if isinstance(ref_image, str):
                ref_image = Image.open(ref_image).convert("RGB")
                ref_image = ref_image.resize((sample_size[1], sample_size[0]))
                ref_image = torch.from_numpy(np.array(ref_image))
                ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
            else:
                ref_image = torch.from_numpy(np.array(ref_image))
                ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        return input_video, input_video_mask, ref_image, clip_image

    def run(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: InputVideo = None,
        start_image: InputImage | str = None,
        end_image: InputImage | str = None,
        ref_image: InputImage | str = None,
        mask_video: InputVideo = None,
        control_video: InputVideo = None,
        control_camera_poses: List[float] | str | List[Camera] | Camera = None,
        duration: int | str = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        high_noise_guidance_scale: float = 6,
        low_noise_guidance_scale: float = 6,
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
        offload: bool = True,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting FunControl pipeline")
        if (high_noise_guidance_scale is not None and low_noise_guidance_scale is not None):
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(progress_callback, 0.01, "Using high/low-noise guidance scales")
        
        safe_emit_progress(progress_callback, 0.02, "Preparing inputs")
        num_frames = self._parse_num_frames(duration, fps)
        
        height = height // 16 * 16
        width = width // 16 * 16
        sample_size = [height, width]
        safe_emit_progress(progress_callback, 0.03, "Preparing start/end images to video latents")
        video, mask_video, _ = self.get_image_to_video_latent(start_image, end_image, video_length=num_frames, sample_size=sample_size)

        if ref_image is not None:
            safe_emit_progress(progress_callback, 0.04, "Loading reference image")
            ref_image = self.get_image_latent(ref_image, sample_size=sample_size)

        if control_camera_poses is not None:
            video, mask_video = None, None
            if isinstance(control_camera_poses, Camera):
                control_camera_poses = [control_camera_poses]
            safe_emit_progress(progress_callback, 0.05, "Preparing camera control video")
            camera_preprocessor = self.helpers["wan.fun_camera"]
            control_camera_video = camera_preprocessor(
                control_camera_poses, H=height, W=width, device=self.device
            )
        else:
            safe_emit_progress(progress_callback, 0.05, "Loading control video")
            control_video, _, _, _ = self.get_video_to_video_latent(control_video, video_length=num_frames, sample_size=sample_size, fps=fps, ref_image=None)
            control_camera_video = None

        
        num_videos_per_prompt = 1

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
            do_classifier_free_guidance = any([guidance_scale_i > 1.0 for guidance_scale_i in guidance_scale])
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

        # 5. Prepare latents.
        if video is not None:
            safe_emit_progress(progress_callback, 0.25, "Preprocessing input video")
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None
        

        
        self.vae_config = self.load_config_by_type("vae")

        latent_channels = self.num_channels_latents
        weight_dtype = self.component_dtypes["transformer"]
        safe_emit_progress(progress_callback, 0.35, "Initializing latent noise")
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        safe_emit_progress(progress_callback, 0.38, "Latents ready")
        

        # Prepare mask latent variables
        if init_video is not None:
            if (mask_video == 255).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(device, weight_dtype), [1, 4, 1, 1, 1]
                )
                masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
                if self.vae_scale_factor_spatial >= 16:
                    mask = torch.ones_like(latents).to(device, weight_dtype)[:, :1].to(device, weight_dtype)
                else:
                    mask = None
            else:
                safe_emit_progress(progress_callback, 0.39, "Preprocessing mask video")
                bs, _, video_length, height, width = video.size()
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                _, masked_video_latents = self.prepare_mask_latents(
                    None,
                    masked_video,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    noise_aug_strength=None,
                )
                
                mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), 
                        mask_condition[:, :, 1:]
                    ], dim=2
                )
                mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, height, width)
                mask_condition = mask_condition.transpose(1, 2)
                mask_latents = self._resize_mask(1 - mask_condition, masked_video_latents, True).to(device, weight_dtype) 

                if self.vae_scale_factor_spatial >= 16:
                    mask = F.interpolate(mask_condition[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(device, weight_dtype)
                    if not mask[:, :, 0, :, :].any():
                        mask[:, :, 1:, :, :] = 1
                        latents = (1 - mask) * masked_video_latents + mask * latents
                else:
                    mask = None
        else:
            mask = None
            masked_video_latents = None

        # Prepare mask latent variables
        if control_camera_video is not None:
            control_latents = None
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        elif control_video is not None:
            safe_emit_progress(progress_callback, 0.40, "Preprocessing control video")
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            control_video = control_video.to(dtype=torch.float32)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
            control_camera_latents = None
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_camera_latents = None

        if start_image is not None:
            safe_emit_progress(progress_callback, 0.41, "Preprocessing start image")
            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)
            
            start_image_latentes = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            start_image_latentes_conv_in = torch.zeros_like(latents)
            if latents.size()[2] != 1:
                start_image_latentes_conv_in[:, :, :1] = start_image_latentes
        else:
            start_image_latentes_conv_in = torch.zeros_like(latents)
            
        
        self.transformer_config = self.load_config_by_name("high_noise_transformer", "transformer")

        if self.transformer_config.get("add_ref_conv", False):
            if ref_image is not None:
                safe_emit_progress(progress_callback, 0.42, "Preprocessing reference image latents")
                video_length = ref_image.shape[2]
                ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
                ref_image = ref_image.to(dtype=torch.float32)
                ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)
                
                ref_image_latentes = self.prepare_control_latents(
                    None,
                    ref_image,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance
                )[1]
                ref_image_latentes = ref_image_latentes[:, :, 0]
            else:
                ref_image_latentes = torch.zeros_like(latents)[:, :, 0]
        else:
            if ref_image is not None:
                raise ValueError("The add_ref_conv is False, but ref_image is not None")
            else:
                ref_image_latentes = None


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        safe_emit_progress(progress_callback, 0.43, "Preparing scheduler step kwargs")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # Prepare mask latent variables
        if control_camera_video is not None:
            control_latents_input = None
            control_camera_latents_input = (control_camera_latents).to(device, weight_dtype)
        else:
            control_latents_input = (control_video_latents).to(device, weight_dtype)
            control_camera_latents_input = None
        if init_video is not None:
            mask_input = mask_latents
            masked_video_latents_input = (masked_video_latents)
            y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(device, weight_dtype) 
            control_latents_input = y if control_latents_input is None else \
                torch.cat([control_latents_input, y], dim = 1)
        else:
            start_image_latentes_conv_in_input = (start_image_latentes_conv_in).to(device, weight_dtype)
            control_latents_input = start_image_latentes_conv_in_input if control_latents_input is None else \
                torch.cat([control_latents_input, start_image_latentes_conv_in_input], dim = 1)
        if ref_image_latentes is not None:
            full_ref = (ref_image_latentes).to(device, weight_dtype)
        else:
            full_ref = None

        
        transformer_dtype = self.component_dtypes["transformer"]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        
        if boundary_ratio is not None:
            safe_emit_progress(progress_callback, 0.44, "Computing boundary timestep")
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None
        
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoise (CFG: {'on' if do_classifier_free_guidance else 'off'})",
        )
        

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=control_latents_input,
            mask_kwargs=dict(
                mask=mask,
                masked_video_latents=masked_video_latents,
            ),
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_camera=(
                    control_camera_latents_input.to(transformer_dtype)
                    if control_camera_latents_input is not None
                    else None
                ),
                encoder_hidden_states_full_ref=(
                    full_ref.to(transformer_dtype)
                    if full_ref is not None
                    else None
                ),
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_camera=(
                        control_camera_latents_input.to(transformer_dtype)
                        if control_camera_latents_input is not None
                        else None
                    ),
                    encoder_hidden_states_full_ref=(
                        full_ref.to(transformer_dtype)
                        if full_ref is not None
                        else None
                    ),
                    attention_kwargs=attention_kwargs,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=do_classifier_free_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            denoise_progress_callback=denoise_progress_callback,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        

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
