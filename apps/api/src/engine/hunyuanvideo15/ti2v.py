import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from PIL import Image
from diffusers.video_processor import VideoProcessor
from loguru import logger
from src.engine.hunyuanvideo15.shared import HunyuanVideo15Shared
from src.types.media import InputImage
from src.helpers.hunyuanvideo15.cache import CacheHelper
from src.utils.progress import safe_emit_progress, make_mapped_progress


class HunyuanVideo15TI2VEngine(HunyuanVideo15Shared):
    """HunyuanVideo 1.5 Text/Image-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        # Default configuration mirrors the upstream pipeline
        self.target_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.autocast_enabled = True
        self.vae_autocast_enabled = True
        self.vision_encoder = None
        self.text_encoder_2 = None
        self.prompt_format = None
        self.byt5_max_length = 256
        self.byt5_encoder = None
        self._transformer_config = None
        self.glyph_byT5_v2 = bool(self.config.get("glyph_byT5_v2", True))
        self.vision_num_semantic_tokens = self.config.get(
            "vision_num_semantic_tokens", 729
        )
        self.vision_states_dim = self.config.get("vision_states_dim", 1152)
        # Compression defaults fall back to model values once the VAE is loaded
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16
        self.num_channels_latents = 16
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }
        # Guidance state placeholders
        self._guidance_scale = 6.0
        self._guidance_rescale = 0.0
        self._clip_skip = None

    # ----------------------------
    # Helper methods from pipeline
    # ----------------------------
    def _refresh_vae_factors(self):
        """Update cached VAE scale factors when the VAE is available."""
        if getattr(self, "vae", None) is None:
            return
        self.vae_scale_factor_spatial = getattr(
            self.vae.config, "ffactor_spatial", self.vae_scale_factor_spatial
        )
        self.vae_scale_factor_temporal = getattr(
            self.vae.config, "ffactor_temporal", self.vae_scale_factor_temporal
        )
        self.num_channels_latents = getattr(
            self.vae.config, "latent_channels", self.num_channels_latents
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        vae_inference_config = self.get_vae_inference_config()
        self.vae.set_tile_sample_min_size(
            vae_inference_config["sample_size"],
            vae_inference_config["tile_overlap_factor"],
        )

    def _get_byt5_encoder(self):
        if self.byt5_encoder is None:
            self.load_component_by_name("byt5_encoder")
            if getattr(self, "byt5_encoder", None) is not None:
                self.to_device(self.byt5_encoder)
        return self.byt5_encoder

    def _extract_glyph_texts(self, prompt: str) -> List[str]:
        import re

        pattern = r"\"(.*?)\"|“(.*?)”"
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        return list(dict.fromkeys(result)) if len(result) > 1 else result

    def _process_single_byt5_prompt(
        self, prompt_text: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros(
            (1, self.byt5_max_length), device=device, dtype=torch.int64
        )
        self._get_prompt_format()

        glyph_texts = self._extract_glyph_texts(prompt_text)
        if len(glyph_texts) > 0 and self.prompt_format is not None:
            text_styles = [
                {"color": None, "font-family": None} for _ in range(len(glyph_texts))
            ]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            self.load_component_by_name("byt5_encoder")
            if getattr(self, "byt5_encoder", None) is not None:
                byt5_encoder = self.byt5_encoder
                self._add_special_token(
                    byt5_encoder,
                    add_color=True,
                    add_font=True,
                    color_ann_path=self.prompt_format.color_path,
                    font_ann_path=self.prompt_format.font_path,
                    multilingual=True,
                )
                byt5_outputs = byt5_encoder.encode(
                    formatted_text,
                    clean_text=False,
                    return_attention_mask=True,
                    output_type="hidden_states",
                    use_attention_mask=True,
                    max_sequence_length=self.byt5_max_length,
                    pad_with_zero=False,
                )
                byt5_embeddings = byt5_outputs[0].to(device=device)
                byt5_mask = byt5_outputs[1].to(device=device)

        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(
        self, prompts: Union[str, List[str]], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        if not self.glyph_byT5_v2:
            return {}

        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings: List[torch.Tensor] = []
        positive_masks: List[torch.Tensor] = []
        negative_embeddings: List[torch.Tensor] = []
        negative_masks: List[torch.Tensor] = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)

        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)

            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {"byt5_text_states": byt5_embeddings, "byt5_text_mask": byt5_masks}

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Any = None,
        data_type: str = "image",
    ):
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(
                prompt, data_type=data_type, max_length=text_encoder.max_length
            )
            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype or self.component_dtypes.get(
                "text_encoder", None
            )
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype or self.component_dtypes.get(
                "transformer", None
            )
        else:
            prompt_embeds_dtype = prompt_embeds.dtype or self.component_dtypes.get(
                "text_encoder", None
            )

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(
                uncond_tokens, data_type=data_type, max_length=text_encoder.max_length
            )
            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    @staticmethod
    def prepare_extra_func_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_step_kwargs: Dict[str, Any] = {}
        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        video_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _prepare_vision_states(
        self,
        reference_image: InputImage | None,
        target_resolution: str,
        latents: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        self._get_vision_encoder()
        if reference_image is None:
            vision_states = torch.zeros(
                latents.shape[0],
                self.vision_num_semantic_tokens,
                self.vision_states_dim,
                device=latents.device,
            )
        else:
            reference_image = (
                np.array(reference_image)
                if isinstance(reference_image, Image.Image)
                else reference_image
            )
            if len(reference_image.shape) == 4:
                reference_image = reference_image[0]

            height, width = self.get_closest_resolution_given_reference_image(
                reference_image, target_resolution
            )
            if self.vision_encoder is not None:
                input_image_np = self._resize_and_center_crop(
                    reference_image, target_width=width, target_height=height
                )
                vision_states = self.vision_encoder.encode_images(input_image_np)
                vision_states = vision_states.last_hidden_state.to(
                    device=device, dtype=self.target_dtype
                )
            else:
                vision_states = None

        if self.do_classifier_free_guidance and vision_states is not None:
            vision_states = vision_states.repeat(2, 1, 1)
        return vision_states

    def _prepare_cond_latents(
        self,
        task_type: str,
        cond_latents: Optional[torch.Tensor],
        latents: torch.Tensor,
        multitask_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cond_latents is not None and task_type == "i2v":
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros_like(latents)

        mask_zeros = torch.zeros(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_ones = torch.ones(
            latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
        )
        mask_concat = self._merge_tensor_by_mask(
            mask_zeros, mask_ones, mask=multitask_mask.cpu(), dim=2
        ).to(device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)
        return cond_latents

    def get_task_mask(self, task_type: str, latent_target_length: int) -> torch.Tensor:
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def get_closest_resolution_given_reference_image(
        self, reference_image: Union[Image.Image, np.ndarray], target_resolution: str
    ) -> Tuple[int, int]:
        if isinstance(reference_image, Image.Image):
            origin_size = reference_image.size
        elif isinstance(reference_image, np.ndarray):
            height, width, _ = reference_image.shape
            origin_size = (width, height)
        else:
            raise ValueError(
                f"Unsupported reference_image type: {type(reference_image)}. Must be PIL Image or numpy array"
            )

        return self.get_closest_resolution_given_original_size(
            origin_size, target_resolution
        )

    def get_closest_resolution_given_original_size(
        self, origin_size: Tuple[int, int], target_size: str
    ) -> Tuple[int, int]:
        bucket_hw_base_size = self.target_size_config[target_size][
            "bucket_hw_base_size"
        ]
        bucket_hw_bucket_stride = self.target_size_config[target_size][
            "bucket_hw_bucket_stride"
        ]

        crop_size_list = self._generate_crop_size_list(
            bucket_hw_base_size, bucket_hw_bucket_stride
        )
        aspect_ratios = np.array(
            [round(float(h) / float(w), 5) for h, w in crop_size_list]
        )
        closest_size, _ = self._get_closest_ratio(
            origin_size[1], origin_size[0], aspect_ratios, crop_size_list
        )
        height = closest_size[0]
        width = closest_size[1]
        return height, width

    def get_latent_size(
        self, video_length: int, height: int, width: int
    ) -> Tuple[int, int, int]:
        spatial_ratio = self.vae_scale_factor_spatial
        temporal_ratio = self.vae_scale_factor_temporal
        video_length = (video_length - 1) // temporal_ratio + 1
        height, width = height // spatial_ratio, width // spatial_ratio

        if height <= 0 or width <= 0 or video_length <= 0:
            raise ValueError(
                f"height: {height}, width: {width}, video_length: {video_length}"
            )

        return video_length, height, width

    def get_image_condition_latents(
        self,
        task_type: str,
        reference_image: InputImage,
        height: int,
        width: int,
        offload: bool = True,
    ) -> Optional[torch.Tensor]:
        if task_type == "t2v":
            return None

        if reference_image is None:
            raise ValueError("reference_image must be provided for i2v task.")

        import torchvision.transforms as transforms

        origin_size = reference_image.size
        target_height, target_width = height, width
        original_width, original_height = origin_size

        scale_factor = max(
            target_width / original_width, target_height / original_height
        )
        resize_width = int(round(original_width * scale_factor))
        resize_height = int(round(original_height * scale_factor))

        ref_image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (resize_height, resize_width),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        ref_images_pixel_values = (
            ref_image_transform(reference_image)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(self.device)
        )
        cond_latents = self.vae_encode(ref_images_pixel_values, offload=offload)
        return cond_latents

    def _get_vision_encoder(self):
        if self.vision_encoder is None:
            helper = self.helpers["vision_encoder"]
            if helper is not None:
                self.vision_encoder = helper
                self.to_device(self.vision_encoder)
            else:
                logger.warning(
                    "Vision encoder helper 'hunyuan.vision_encoder' not configured."
                )
        return self.vision_encoder

    def _get_text_encoder(self):
        if self.text_encoder is None:
            helper = self.helpers["text_encoder"]
            if helper is not None:
                self.text_encoder = helper
                self.to_device(self.text_encoder)
            else:
                logger.warning(
                    "Text encoder helper 'hunyuanvideo15.text_encoder' not configured."
                )
        return self.text_encoder

    def _get_text_encoder_2(self):
        if getattr(self, "text_encoder_2", None) is None:
            helper = self.helpers["text_encoder_2"]
            if helper is not None:
                self.text_encoder_2 = helper
                self.to_device(self.text_encoder_2)
            else:
                logger.warning(
                    "Text encoder helper 'hunyuanvideo15.text_encoder_2' not configured."
                )
        return getattr(self, "text_encoder_2", None)

    def _get_prompt_format(self):
        if self.prompt_format is None:
            helper = self.helpers["prompt_format"]
            if helper is not None:
                self.prompt_format = helper
                self.to_device(self.prompt_format)
            else:
                logger.warning(
                    "Prompt format helper 'hunyuanvideo15.prompt_format' not configured."
                )
        return self.prompt_format

    # -------------------------
    # Properties matching pipeline
    # -------------------------
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def transformer_config(self):
        if self._transformer_config is not None:
            return self._transformer_config
        config = self.load_config_by_type("transformer")
        self._transformer_config = config
        return self._transformer_config

    @property
    def ideal_resolution(self):
        return getattr(self.transformer_config, "ideal_resolution", None)

    @property
    def ideal_task(self):
        return getattr(self.transformer_config, "ideal_task", None)

    @property
    def use_meanflow(self):
        return getattr(self.transformer_config, "use_meanflow", False)

    def run(
        self,
        prompt: Union[str, List[str]],
        aspect_ratio: str = "16:9",
        duration: str | int = 121,
        fps: int = 24,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        embedded_guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        num_videos: int = 1,
        resolution: str = "720p",  # or 480p or #1080p
        sparse_attn: bool = False,
        enable_cache: bool = False,
        cache_start_step: int = 11,
        cache_end_step: int = 45,
        total_steps: int = 50,
        cache_step_interval: int = 4,
        seed: Optional[int] = None,
        flow_shift: Optional[float] = None,
        reference_image: InputImage = None,
        output_type: str = "pil",
        return_latents: bool = False,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        eta: float = 0.0,
        offload: bool = True,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ):
        # Accept progress callback via kwargs as well (older orchestrators).
        resolution = str(resolution).lower()
        if not resolution.endswith("p"):
            resolution = f"{resolution}p"
        if resolution == "custom":
            resolution = self.ideal_resolution or "720p"
        progress_callback = kwargs.pop("progress_callback", progress_callback)

        safe_emit_progress(
            progress_callback,
            0.0,
            "Starting HunyuanVideo 1.5 text/image-to-video pipeline",
        )
        device = self.device

        if generator is None and seed is not None:
            safe_emit_progress(progress_callback, 0.01, "Setting random seed")
            generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        transformer_dtype = self.component_dtypes.get("transformer", self.target_dtype)
        self.target_dtype = transformer_dtype

        safe_emit_progress(progress_callback, 0.03, "Preparing configuration")
        if guidance_scale is None:
            guidance_scale = self.config.get("guidance_scale", self._guidance_scale)
        if embedded_guidance_scale is None:
            embedded_guidance_scale = self.config.get("embedded_guidance_scale", None)
        if flow_shift is None:
            flow_shift = self.config.get(
                "flow_shift", getattr(self.transformer_config, "flow_shift", 7.0)
            )

        target_resolution = resolution or self.ideal_resolution or "720p"

        # Determine task type and reference image
        if reference_image is not None:
            safe_emit_progress(progress_callback, 0.05, "Loading reference image")
            task_type = "i2v"
            reference_image = self._load_image(reference_image)
        else:
            task_type = "t2v"

        if self.ideal_task is not None and self.ideal_task != task_type:
            raise ValueError(
                f"The loaded transformer is trained for '{self.ideal_task}' but received '{task_type}'."
            )

        if reference_image is not None:
            height, width = self.get_closest_resolution_given_reference_image(
                reference_image, target_resolution
            )
        elif height is not None and width is not None:
            height, width = height, width
        else:
            if ":" not in aspect_ratio:
                raise ValueError(
                    "aspect_ratio must be separated by a colon, e.g. '16:9'"
                )
            width_str, height_str = aspect_ratio.split(":")
            if not width_str.isdigit() or not height_str.isdigit():
                raise ValueError("aspect_ratio must contain integer width and height")
            height, width = self.get_closest_resolution_given_original_size(
                (int(width_str), int(height_str)), target_resolution
            )

        safe_emit_progress(progress_callback, 0.08, "Computing latent sizes")
        video_length = self._parse_num_frames(duration=duration, fps=fps)
        latent_target_length, latent_height, latent_width = self.get_latent_size(
            video_length, height, width
        )
        n_tokens = latent_target_length * latent_height * latent_width
        multitask_mask = self.get_task_mask(task_type, latent_target_length)

        # Guidance bookkeeping
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip

        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        logger.info(f"{getattr(self, 'text_encoder', None)} text encoder")

        if getattr(self, "text_encoder", None) is None:
            safe_emit_progress(progress_callback, 0.10, "Loading text encoder")
            self._get_text_encoder()
        safe_emit_progress(progress_callback, 0.11, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        self.text_len = getattr(self.text_encoder, "max_length", 1000)

        safe_emit_progress(progress_callback, 0.12, "Encoding prompt (text encoder)")
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos,
            self.do_classifier_free_guidance,
            negative_prompt,
            clip_skip=self.clip_skip,
            data_type="video",
        )

        prompt_embeds_2 = None
        negative_prompt_embeds_2 = None
        prompt_mask_2 = None
        negative_prompt_mask_2 = None

        if getattr(self, "text_encoder_2", None) is None:
            try:
                safe_emit_progress(progress_callback, 0.14, "Loading text encoder 2")
                self.load_component_by_name("text_encoder_2")
                if getattr(self, "text_encoder_2", None) is not None:
                    safe_emit_progress(
                        progress_callback, 0.15, "Moving text encoder 2 to device"
                    )
                    self.to_device(self.text_encoder_2)
            except Exception:
                self.text_encoder_2 = None

        if getattr(self, "text_encoder_2", None) is not None:
            safe_emit_progress(
                progress_callback, 0.16, "Encoding prompt (text encoder 2)"
            )
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos,
                self.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type="video",
            )

        if offload:
            safe_emit_progress(progress_callback, 0.18, "Offloading text encoders")
            self._offload("text_encoder")
            if getattr(self, "text_encoder_2", None) is not None:
                self._offload("text_encoder_2")
            safe_emit_progress(progress_callback, 0.19, "Text encoders offloaded")

        safe_emit_progress(progress_callback, 0.20, "Preparing auxiliary embeddings")
        extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # Scheduler setup
        if getattr(self, "scheduler", None) is None:
            safe_emit_progress(progress_callback, 0.22, "Loading scheduler")
            self.load_component_by_type("scheduler")
        else:
            try:
                # Refresh scheduler if flow_shift changes
                scheduler_config = dict(getattr(self.scheduler, "config", {}))
                scheduler_class = self.scheduler.__class__
                scheduler_config["shift"] = flow_shift
                self.scheduler = scheduler_class(**scheduler_config)
            except Exception:
                pass

        safe_emit_progress(progress_callback, 0.23, "Moving scheduler to device")
        self.to_device(self.scheduler)
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )

        safe_emit_progress(progress_callback, 0.25, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            **extra_set_timesteps_kwargs,
        )
        safe_emit_progress(progress_callback, 0.28, "Timesteps prepared")

        num_channels_latents = getattr(
            self.transformer_config, "in_channels", self.num_channels_latents
        )
        safe_emit_progress(progress_callback, 0.30, "Initializing latents")
        latents = self.prepare_latents(
            batch_size * num_videos,
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )
        safe_emit_progress(progress_callback, 0.34, "Latents initialized")

        safe_emit_progress(
            progress_callback, 0.35, "Preparing conditioning (VAE / vision)"
        )
        self.load_component_by_type("vae")
        self.to_device(self.vae)
        image_cond = self.get_image_condition_latents(
            task_type, reference_image, height, width, offload=offload
        )
        cond_latents = self._prepare_cond_latents(
            task_type, image_cond, latents, multitask_mask
        )
        vision_states = self._prepare_vision_states(
            reference_image, target_resolution, latents, device
        )
        safe_emit_progress(progress_callback, 0.40, "Conditioning prepared")

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta}
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if getattr(self, "transformer", None) is None:
            safe_emit_progress(progress_callback, 0.42, "Loading transformer")
            self.load_component_by_type("transformer")
        safe_emit_progress(progress_callback, 0.44, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.45, "Transformer ready")

        if enable_cache:
            safe_emit_progress(progress_callback, 0.46, "Enabling transformer cache")
            no_cache_steps = (
                list(range(0, cache_start_step))
                + list(range(cache_start_step, cache_end_step, cache_step_interval))
                + list(range(cache_end_step, total_steps))
            )
            cache_helper = CacheHelper(
                model=self.transformer,
                no_cache_steps=no_cache_steps,
                no_cache_block_id={
                    "double": [53]
                },  # Added single block to skip caching
            )
            cache_helper.enable()
        else:
            cache_helper = None

        if sparse_attn:
            safe_emit_progress(progress_callback, 0.47, "Enabling sparse attention")
            if not self.is_sparse_attn_supported():
                raise RuntimeError(
                    f"Current GPU is {torch.cuda.get_device_properties(0).name}, which does not support sparse attention."
                )
            if self.transformer.config.attn_mode != "flex-block-attn":
                self.logger.warning(
                    f"The transformer loaded is not trained with sparse attention. Forcing to use sparse attention may lead to artifacts in the generated video."
                    f"To enable sparse attention, we recommend loading `{self.transformer_version}_distilled_sparse` instead."
                )
            self.transformer.set_attn_mode("flex-block-attn")

        if cache_helper is not None:
            cache_helper.clear_cache()
            assert num_inference_steps == total_steps

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(
            progress_callback,
            0.50,
            f"Starting denoise (CFG: {'on' if self.do_classifier_free_guidance else 'off'})",
        )

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising HunyuanVideo1.5"
        ) as progress_bar:
            total_ts = len(timesteps)
            for i, t in enumerate(timesteps):
                if cache_helper is not None:
                    cache_helper.cur_timestep = i
                latents_concat = torch.concat([latents, cond_latents], dim=1)

                latent_model_input = (
                    torch.cat([latents_concat] * 2)
                    if self.do_classifier_free_guidance
                    else latents_concat
                )

                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                t_expand = t.repeat(latent_model_input.shape[0])
                if self.use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(self.target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=transformer_dtype,
                    enabled=self.autocast_enabled and self.device.type == "cuda",
                ):
                    output = self.transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        prompt_embeds_2,
                        prompt_mask,
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=task_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if self.guidance_rescale > 0.0:
                        noise_pred = self._rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(max(total_ts, 1)),
                    f"Denoising step {i + 1}/{total_ts}",
                )

        if offload:
            safe_emit_progress(progress_callback, 0.91, "Offloading transformer")
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.94, "Decoding video")
        video_latents = latents if latents.ndim == 5 else latents.unsqueeze(2)
        video_frames = self.vae_decode(video_latents, offload=offload)
        safe_emit_progress(progress_callback, 0.97, "Postprocessing video")

        if output_type == "np":
            video_frames = video_frames.numpy()
        elif output_type == "pil":
            video_frames = self.video_processor.postprocess_video(
                video_frames, output_type="pil"
            )

        safe_emit_progress(
            progress_callback, 1.0, "Completed HunyuanVideo 1.5 TI2V pipeline"
        )
        return video_frames

    def vae_decode(self, latents, offload=True):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.component_dtypes.get("vae", self.vae_dtype),
            enabled=True,
        ):
            self.load_component_by_type("vae")
            self.to_device(self.vae)
            self._refresh_vae_factors()
            self.vae.enable_tiling()
            out = super().vae_decode(latents, offload=offload)
            self.vae.disable_tiling()
        if offload:
            self._offload("vae")
        return out

    def vae_encode(
        self, frames, sample_mode="mode", sample_generator=None, offload=True
    ):

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.component_dtypes.get("vae", self.vae_dtype),
            enabled=True,
        ):
            self.load_component_by_type("vae")
            self.to_device(self.vae)
            self._refresh_vae_factors()
            self.vae.enable_tiling()
            out = super().vae_encode(
                frames,
                sample_mode=sample_mode,
                sample_generator=sample_generator,
                offload=offload,
            )
        return out
