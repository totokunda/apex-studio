from src.engine.flux2.shared import Flux2Shared
from typing import Union, List, Optional, Dict, Any, Callable
from PIL import Image
import numpy as np
import torch
from src.types import InputImage
from typing import Tuple
from torch.nn import functional as F


class Flux2Control(Flux2Shared):

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

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

    def run(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image: Optional[InputImage] = None,
        inpaint_image: Optional[InputImage] = None,
        mask_image: Optional[InputImage] = None,
        control_image: Optional[InputImage] = None,
        control_context_scale: float = 1.0,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        seed: int = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        attention_kwargs: Dict[str, Any] = {},
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        offload: bool = True,
        **kwargs,
    ):

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        weight_dtype = self.component_dtypes.get("text_encoder", None)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        transformer_config = self.load_config_by_type("transformer")
        num_channels_latents = transformer_config.in_channels // 4

        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        # Prepare mask latent variables
        if mask_image is not None:
            mask_image = self.prepare_image(mask_image, (height, width))[:, :1, 0]
        else:
            mask_image = torch.ones([1, 1, height, width]) * 255

        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width
        )
        mask_condition = torch.tile(mask_condition, [1, 3, 1, 1]).to(
            dtype=weight_dtype, device=device
        )

        if inpaint_image is not None:
            inpaint_image = self.prepare_image(inpaint_image, (height, width))[:, :, 0]
        else:
            inpaint_image = torch.zeros([1, 3, height, width])
        init_image = self.diffusers_image_processor.preprocess(
            inpaint_image, height=height, width=width
        )
        init_image = init_image.to(dtype=weight_dtype, device=device) * (
            mask_condition < 0.5
        )
        inpaint_latent = self.vae.encode(init_image)[0].mode()

        if control_image is not None:
            control_image = self.prepare_image(control_image, (height, width))[:, :, 0]
            control_image = self.diffusers_image_processor.preprocess(
                control_image, height=height, width=width
            )
            control_image = control_image.to(dtype=weight_dtype, device=device)
            control_latents = self.vae.encode(control_image)[0].mode()
        else:
            control_latents = torch.zeros_like(inpaint_latent)

        mask_condition = F.interpolate(
            1 - mask_condition[:, :1], size=control_latents.size()[-2:], mode="nearest"
        ).to(device, weight_dtype)
        mask_condition = self._patchify_latents(mask_condition)
        mask_condition = self._pack_latents(mask_condition)

        if inpaint_image is not None:
            inpaint_latent = self._patchify_latents(inpaint_latent)
            inpaint_latent = self.vae.normalize_latents(inpaint_latent)
            inpaint_latent = self._pack_latents(inpaint_latent)
        else:
            inpaint_latent = self._patchify_latents(inpaint_latent)
            inpaint_latent = self._pack_latents(inpaint_latent)

        if control_image is not None:
            control_latents = self._patchify_latents(control_latents)
            control_latents = self.vae.normalize_latents(control_latents)
            control_latents = self._pack_latents(control_latents)
        else:
            control_latents = self._patchify_latents(control_latents)
            control_latents = self._pack_latents(control_latents)
        control_context = torch.concat(
            [control_latents, mask_condition, inpaint_latent], dim=2
        )

        if offload:
            self._offload("vae")

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if offload:
            self._offload("text_encoder")

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(
                    img, height=image_height, width=image_width, resize_mode="crop"
                )
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        # 5. prepare latent variables
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = self.compute_empirical_mu(
            image_seq_len=image_seq_len, num_steps=num_inference_steps
        )
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                control_context_input = control_context.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(
                        self.transformer.dtype
                    )
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                    local_bs, local_length, local_c = control_context.size()
                    control_context_input = torch.cat(
                        [
                            control_context,
                            torch.zeros(
                                [local_bs, image_latents.size()[1], local_c]
                            ).to(control_context.device, control_context.dtype),
                        ],
                        dim=1,
                    ).to(self.transformer.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # (B, image_seq_len, C)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=latent_image_ids,  # B, image_seq_len, 4
                    joint_attention_kwargs=self._attention_kwargs,
                    control_context=control_context_input,
                    control_context_scale=control_context_scale,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if offload:
            self._offload("transformer")

        if return_latents:
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            if not self.vae:
                self.load_component_by_type("vae")
            self.to_device(self.vae)

            latents = self.vae.denormalize_latents(latents)

            latents = self._unpatchify_latents(latents)

            image = self.vae_decode(latents, offload=offload, denormalize_latents=False)
            image = self._tensor_to_frame(image)

        return image
