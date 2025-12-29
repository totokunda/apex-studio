from typing import Union, List, Optional, Callable, Dict, Any, Tuple
import torch
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import ZImageShared
from PIL import Image
import numpy as np
from torch.nn import functional as F

class ZImageControlEngine(ZImageShared):
    """ZImage Control Engine Implementation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_in_dim = 33

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def _padding_image(self, images, new_width, new_height):
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

    def prepare_image(self, ref_image, sample_size: Tuple[int, int], padding=False):
        ref_image = self._load_image(ref_image)
        if padding:
            ref_image = self._padding_image(ref_image, sample_size[1], sample_size[0])
        ref_image = ref_image.resize((sample_size[1], sample_size[0]))
        ref_image = torch.from_numpy(np.array(ref_image))
        ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

        return ref_image
    
    def get_image_latent(self, ref_image=None, sample_size=None, padding=False):
        if ref_image is not None:
            ref_image = self._load_image(ref_image)
            if padding:
                ref_image = self.padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

        return ref_image

    def run(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        inpaint_image: Union[torch.FloatTensor] = None,
        control_image: Union[torch.FloatTensor] = None,
        mask_image: Union[torch.FloatTensor] = None,
        control_context_scale: float = 0.75,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        return_latents: bool = False,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: Optional[List[torch.FloatTensor]] = None,
        render_on_step_callback: Optional[Callable] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            height = height - (height % vae_scale)
        if width % vae_scale != 0:
            width = width - (width % vae_scale)
        
        sample_size = [height, width]
        if inpaint_image is not None:
            image = self.get_image_latent(inpaint_image, sample_size=sample_size)[:, :, 0]
        else:
            image = torch.zeros([1, 3, sample_size[0], sample_size[1]])

        if mask_image is not None:
            mask_image = self.get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
        else:
            mask_image = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255

        if control_image is not None:
            control_image = self.get_image_latent(control_image, sample_size=sample_size)[:, :, 0]

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        device = self.device
        weight_dtype = self.component_dtypes["transformer"]
        num_channels_latents = self.num_channels_latents

        # Prepare mask latent variables
        if num_channels_latents != self.control_in_dim:
            if mask_image is not None:
                mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width) 
                mask_condition = torch.tile(mask_condition, [1, 3, 1, 1]).to(dtype=weight_dtype, device=device)
            
            if image is not None:
                init_image = self.image_processor.preprocess(image, height=height, width=width)
                init_image = init_image.to(dtype=weight_dtype, device=device) * (mask_condition < 0.5)
                inpaint_latent = self.vae_encode(init_image)
            else:
                inpaint_latent = torch.zeros((batch_size, num_channels_latents, 2 * (int(height) // (self.vae_scale_factor * 2)), 2 * (int(width) // (self.vae_scale_factor * 2)))).to(device, weight_dtype)

        if control_image is not None:
            control_image = self.image_processor.preprocess(control_image, height=height, width=width) 
            control_image = control_image.to(dtype=weight_dtype, device=device)
            control_latents = self.vae_encode(control_image)
        else:
            control_latents = torch.zeros_like(inpaint_latent)

        # Unsqueeze
        if num_channels_latents != self.control_in_dim:
            inpaint_latent = inpaint_latent.unsqueeze(2)
            mask_condition = F.interpolate(1 - mask_condition[:, :1], size=inpaint_latent.size()[-2:], mode='nearest').to(device, weight_dtype)
            mask_condition = mask_condition.unsqueeze(2)

        control_latents = control_latents.unsqueeze(2)

        # Concat
        if num_channels_latents != self.control_in_dim:
            control_context = torch.concat([control_latents, mask_condition, inpaint_latent], dim=1)
        else:
            control_context = control_latents
        
        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            encode_progress_callback = make_mapped_progress(progress_callback, 0.02, 0.18)
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
                progress_callback=encode_progress_callback,
            )
            safe_emit_progress(progress_callback, 0.18, "Prompts ready")

        # 4. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        
        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.19, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.20, "Scheduler loaded")
            safe_emit_progress(progress_callback, 0.21, "Moving scheduler to device")
            self.to_device(self.scheduler)
            safe_emit_progress(progress_callback, 0.22, "Scheduler on device")

        # 5. Prepare timesteps
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        
        if not self.transformer:
            safe_emit_progress(progress_callback, 0.23, "Loading transformer")
            self.load_component_by_type("transformer")
            safe_emit_progress(progress_callback, 0.24, "Transformer loaded")
            safe_emit_progress(progress_callback, 0.25, "Moving transformer to device")
            self.to_device(self.transformer)
            safe_emit_progress(progress_callback, 0.26, "Transformer on device")
            
 
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.40, 0.92)
        # 6. Denoising loop
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    control_context=control_context,
                    control_context_scale=control_context_scale,
                )[0]

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    
                if denoise_progress_callback is not None and len(timesteps) > 0:
                    try:
                        denoise_progress_callback(min((i + 1) / len(timesteps), 1.0), f"Denoising step {i + 1}/{len(timesteps)}")
                    except Exception:
                        pass
        
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")
        
        if offload:
            self._offload("transformer")
            safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")
        
        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            image = self.vae_decode(latents, offload=offload)
            image = self._tensor_to_frame(image)
            safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
            return image
