from src.engine.base_engine import BaseEngine
from typing import Union, List, Optional
import torch
import numpy as np
from src.utils.cache import empty_cache
from diffusers.video_processor import VideoProcessor
from src.utils.defaults import get_lora_path
import re

ASPECT_RATIO_627 = {
    "0.26": ([320, 1216], 1),
    "0.31": ([352, 1120], 1),
    "0.38": ([384, 1024], 1),
    "0.43": ([416, 960], 1),
    "0.52": ([448, 864], 1),
    "0.58": ([480, 832], 1),
    "0.67": ([512, 768], 1),
    "0.74": ([544, 736], 1),
    "0.86": ([576, 672], 1),
    "0.95": ([608, 640], 1),
    "1.05": ([640, 608], 1),
    "1.17": ([672, 576], 1),
    "1.29": ([704, 544], 1),
    "1.35": ([736, 544], 1),
    "1.50": ([768, 512], 1),
    "1.67": ([800, 480], 1),
    "1.73": ([832, 480], 1),
    "2.00": ([896, 448], 1),
    "2.31": ([960, 416], 1),
    "2.58": ([992, 384], 1),
    "2.75": ([1056, 384], 1),
    "3.09": ([1088, 352], 1),
    "3.70": ([1184, 320], 1),
    "3.80": ([1216, 320], 1),
    "3.90": ([1248, 320], 1),
    "4.00": ([1280, 320], 1),
}

ASPECT_RATIO_627_F64 = {
    "0.26": ([320, 1216], 1),
    "0.38": ([384, 1024], 1),
    "0.50": ([448, 896], 1),
    "0.67": ([512, 768], 1),
    "0.82": ([576, 704], 1),
    "1.00": ([640, 640], 1),
    "1.22": ([704, 576], 1),
    "1.50": ([768, 512], 1),
    "1.86": ([832, 448], 1),
    "2.00": ([896, 448], 1),
    "2.50": ([960, 384], 1),
    "2.83": ([1088, 384], 1),
    "3.60": ([1152, 320], 1),
    "3.80": ([1216, 320], 1),
    "4.00": ([1280, 320], 1),
}

ASPECT_RATIO_627_F128 = {
    "0.25": ([256, 1024], 1),
    "0.38": ([384, 1024], 1),
    "0.43": ([384, 896], 1),
    "0.57": ([512, 896], 1),
    "0.67": ([512, 768], 1),
    "1.00": ([640, 640], 1),
    "1.50": ([768, 512], 1),
    "1.75": ([896, 512], 1),
    "2.33": ([896, 384], 1),
    "2.67": ([1024, 384], 1),
    "4.00": ([1024, 256], 1),
}

ASPECT_RATIO_627_F256 = {
    "0.25": ([256, 1024], 1),
    "0.33": ([256, 768], 1),
    "0.50": ([256, 512], 1),
    "0.67": ([512, 768], 1),
    "1.00": ([512, 512], 1),
    "1.50": ([768, 512], 1),
    "2.00": ([512, 256], 1),
    "3.00": ([768, 256], 1),
    "4.00": ([1024, 256], 1),
}

ASPECT_RATIO_960 = {
    "0.25": ([480, 1920], 1),
    "0.29": ([512, 1792], 1),
    "0.32": ([544, 1696], 1),
    "0.36": ([576, 1600], 1),
    "0.40": ([608, 1504], 1),
    "0.49": ([672, 1376], 1),
    "0.54": ([704, 1312], 1),
    "0.59": ([736, 1248], 1),
    "0.69": ([800, 1152], 1),
    "0.74": ([832, 1120], 1),
    "0.82": ([864, 1056], 1),
    "0.88": ([896, 1024], 1),
    "0.94": ([928, 992], 1),
    "1.00": ([960, 960], 1),
    "1.07": ([992, 928], 1),
    "1.14": ([1024, 896], 1),
    "1.22": ([1056, 864], 1),
    "1.31": ([1088, 832], 1),
    "1.35": ([1120, 832], 1),
    "1.44": ([1152, 800], 1),
    "1.70": ([1248, 736], 1),
    "2.00": ([1344, 672], 1),
    "2.05": ([1376, 672], 1),
    "2.47": ([1504, 608], 1),
    "2.53": ([1536, 608], 1),
    "2.83": ([1632, 576], 1),
    "3.06": ([1664, 544], 1),
    "3.12": ([1696, 544], 1),
    "3.62": ([1856, 512], 1),
    "3.93": ([1888, 480], 1),
    "4.00": ([1920, 480], 1),
}

ASPECT_RATIO_960_F64 = {
    "0.22": ([448, 2048], 1),
    "0.29": ([512, 1792], 1),
    "0.36": ([576, 1600], 1),
    "0.45": ([640, 1408], 1),
    "0.55": ([704, 1280], 1),
    "0.63": ([768, 1216], 1),
    "0.76": ([832, 1088], 1),
    "0.88": ([896, 1024], 1),
    "1.00": ([960, 960], 1),
    "1.14": ([1024, 896], 1),
    "1.31": ([1088, 832], 1),
    "1.50": ([1152, 768], 1),
    "1.58": ([1216, 768], 1),
    "1.82": ([1280, 704], 1),
    "1.91": ([1344, 704], 1),
    "2.20": ([1408, 640], 1),
    "2.30": ([1472, 640], 1),
    "2.67": ([1536, 576], 1),
    "2.89": ([1664, 576], 1),
    "3.62": ([1856, 512], 1),
    "3.75": ([1920, 512], 1),
}

ASPECT_RATIO_960_F128 = {
    "0.20": ([384, 1920], 1),
    "0.27": ([512, 1920], 1),
    "0.33": ([512, 1536], 1),
    "0.42": ([640, 1536], 1),
    "0.50": ([640, 1280], 1),
    "0.60": ([768, 1280], 1),
    "0.67": ([768, 1152], 1),
    "0.78": ([896, 1152], 1),
    "1.00": ([1024, 1024], 1),
    "1.29": ([1152, 896], 1),
    "1.50": ([1152, 768], 1),
    "1.67": ([1280, 768], 1),
    "2.00": ([1280, 640], 1),
    "2.40": ([1536, 640], 1),
    "3.00": ([1536, 512], 1),
    "3.75": ([1920, 512], 1),
    "5.00": ([1920, 384], 1),
}

ASPECT_RATIO_960_F256 = {
    "0.33": ([512, 1536], 1),
    "0.60": ([768, 1280], 1),
    "1.00": ([1024, 1024], 1),
    "1.67": ([1280, 768], 1),
    "3.00": ([1536, 512], 1),
}


def get_bucket_config(resolution, scale_factor_spatial):
    if resolution == "480p":
        if scale_factor_spatial == 16 or scale_factor_spatial == 32:
            return ASPECT_RATIO_627
        elif scale_factor_spatial == 64:
            return ASPECT_RATIO_627_F64
        elif scale_factor_spatial == 128:
            return ASPECT_RATIO_627_F128
        elif scale_factor_spatial == 256:
            return ASPECT_RATIO_627_F256
    elif resolution == "720p":
        if scale_factor_spatial == 16 or scale_factor_spatial == 32:
            return ASPECT_RATIO_960
        elif scale_factor_spatial == 64:
            return ASPECT_RATIO_960_F64
        elif scale_factor_spatial == 128:
            return ASPECT_RATIO_960_F128
        elif scale_factor_spatial == 256:
            return ASPECT_RATIO_960_F256

    raise ValueError(
        f"Unsupported resolution '{resolution}' or scale_factor_spatial '{scale_factor_spatial}'"
    )


class LongCatShared(BaseEngine):
    """Base class for LongCat engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, auto_apply_loras=False, **kwargs)

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )

        self.vae_scale_factor_temporal = (
            self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self._num_timesteps = 1000
        self._num_distill_sample_steps = 50

    def apply_distill_lora(self):
        self._init_lora_manager(get_lora_path())
        if not self.transformer:
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)

        cfg_step_lora = self.preloaded_loras.get("cfg_step_lora", None)
        self.logger.info(f"Applying cfg_step_lora: {cfg_step_lora}")
        if cfg_step_lora:
            self.apply_loras(
                [cfg_step_lora], adapter_names=["cfg_step_lora"]
            )

    def apply_refinement_lora(self):
        refinement_lora = self.preloaded_loras.get("refinement_lora", None)
        self.logger.info(f"Applying refinement lora: {refinement_lora}")
        if refinement_lora:
            self.apply_loras(
                [refinement_lora], adapter_names=["refinement_lora"]
            )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype
        device = device or self.device

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embed, mask = self.text_encoder.encode(
            prompt,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            return_attention_mask=True,
            pad_with_zero=False,
            use_attention_mask=True,
            output_type="hidden_states",
        )

        prompt_embed = prompt_embed.unsqueeze(1).to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        return prompt_embed, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
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

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        device = device or self.device

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance:
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

            negative_prompt_embeds, negative_prompt_attention_mask = (
                self._get_t5_prompt_embeds(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_cond_frames_added: int = 0,
        offload: bool = True,
    ) -> torch.Tensor:
        if (image is not None) and (video is not None):
            raise ValueError(
                "Cannot provide both `image and video` at the same time. Please provide only one."
            )
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Generate random noise with shape latent_shape
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )

        if image is not None or video is not None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

            condition_data = image if image is not None else video
            is_image = image is not None
            cond_latents = []
            for i in range(batch_size):
                gen = generator[i] if isinstance(generator, list) else generator
                if is_image:
                    encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                else:
                    encoded_input = condition_data[i][
                        :, -(num_cond_frames - num_cond_frames_added) :
                    ].unsqueeze(0)
                if num_cond_frames_added > 0:
                    pad_front = encoded_input[:, :, 0:1].repeat(
                        1, 1, num_cond_frames_added, 1, 1
                    )
                    encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                assert encoded_input.shape[2] == num_cond_frames
                latent = self.vae_encode(
                    encoded_input,
                    sample_mode="sample",
                    sample_generator=gen,
                    offload=offload,
                )
                cond_latents.append(latent)

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
            num_cond_latents = (
                1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
            )
            latents[:, :, :num_cond_latents] = cond_latents

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def num_distill_sample_steps(self):
        return self._num_distill_sample_steps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool = False):
        if use_distill:
            distill_indices = torch.arange(
                1, self.num_distill_sample_steps + 1, dtype=torch.float32
            )
            distill_indices = (
                (
                    distill_indices
                    * (self.num_timesteps // self.num_distill_sample_steps)
                )
                .round()
                .long()
            )

            inference_indices = np.linspace(
                0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False
            )
            inference_indices = np.floor(inference_indices).astype(np.int64)

            sigmas = (
                torch.flip(distill_indices, [0])[inference_indices].float()
                / self.num_timesteps
            )
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        sigmas = sigmas.to(torch.float32)
        return sigmas

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _cache_clean_latents(
        self, cond_latents, model_max_length, offload_kv_cache, device, dtype
    ):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(
            device=device, dtype=dtype
        )
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        text_encoder_config = self.load_config_by_type("text_encoder")
        empty_embeds = torch.zeros(
            [cond_latents.shape[0], 1, model_max_length, text_encoder_config.d_model],
            device=device,
            dtype=dtype,
        )
        _, kv_cache_dict = self.transformer(
            hidden_states=cond_latents,
            timestep=timestep,
            encoder_hidden_states=empty_embeds,
            return_kv=True,
            skip_crs_attn=True,
            offload_kv_cache=offload_kv_cache,
        )

        self._update_kv_cache_dict(kv_cache_dict)

    def _get_kv_cache_dict(self):
        return self.kv_cache_dict

    def _clear_cache(self):
        self.kv_cache_dict = None
        empty_cache()

    def get_condition_shape(self, condition, resolution, scale_factor_spatial=32):
        bucket_config = get_bucket_config(
            resolution, scale_factor_spatial=scale_factor_spatial
        )

        obj = condition[0] if isinstance(condition, list) and condition else condition
        try:
            height = getattr(obj, "height")
            width = getattr(obj, "width")
        except AttributeError:
            raise ValueError("Unsupported condition type")

        ratio = height / width
        # Find the closest bucket
        closest_bucket = sorted(
            list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio)
        )[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w

    def optimized_scale(self, positive_flat, negative_flat):
        """from CFG-zero paper"""
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
        st_star = dot_product / squared_norm
        return st_star

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:

        alphabets = "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov|edu|me)"
        digits = "([0-9])"
        multiple_dots = r"\.{2,}"
        """
        Split the text into sentences.
    
        If the text contains substrings "<prd>" or "<stop>", they would lead 
        to incorrect splitting because they are used as markers for splitting.
    
        :param text: text to be split into sentences
        :type text: str
    
        :return: list of sentences
        :rtype: list[str]
        """
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
        text = re.sub(
            multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text
        )
        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(
            alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
            "\\1<prd>\\2<prd>\\3<prd>",
            text,
        )
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if '"' in text:
            text = text.replace('."', '".')
        if "!" in text:
            text = text.replace('!"', '"!')
        if "?" in text:
            text = text.replace('?"', '"?')
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]:
            sentences = sentences[:-1]
        return sentences
