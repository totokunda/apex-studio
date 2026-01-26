from src.engine.base_engine import BaseEngine
import torch
from .shared.tokenizer_wrapper import TokenizerWrapper, JointImageInfo
from src.transformer.hunyuanimage3.base.model import (
    HunyuanStaticCache,
    build_batch_2d_rope,
)
from .shared.system_prompt import get_system_prompt
from .shared.image_processor import HunyuanImage3ImageProcessor
from src.transformer.hunyuanimage3.base.config import HunyuanImage3Config
from typing import Union, List, Dict, Any, Callable, Optional
from diffusers.utils.torch_utils import randn_tensor
from transformers.utils.generic import ModelOutput
from transformers.generation.utils import ALL_CACHE_NAMES
import inspect
import os
import random
from diffusers.image_processor import VaeImageProcessor
from src.utils.progress import safe_emit_progress, make_mapped_progress


def default(val, d):
    return val if val is not None else d


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor],
        guidance_scale: float,
        step: int,
    ) -> torch.Tensor:

        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


class HunyuanImage3T2IEngine(BaseEngine):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.tokenizer_wrapper = TokenizerWrapper(self.helpers["tokenizer"])
        # We need to load the config
        transformer_component, transformer_index = None, None
        vae_component, vae_index = None, None
        self._config = {}
        components = self.config["components"]
        for index, component in enumerate(components):
            if component["type"] == "transformer":
                transformer_component = component
                transformer_index = index
            elif component["type"] == "vae":
                vae_component = component
                vae_index = index
        if transformer_component is None:
            raise ValueError("Transformer component not found")
        if vae_component is None:
            raise ValueError("VAE component not found")

        model_path = transformer_component.get("model_path", "")
        config_path = transformer_component.get("config_path", "")
        if not config_path:
            self.check_weights = False
            config_path = os.path.join(model_path, "config.json")

        conf = self.fetch_config(config_path)
        if conf is not None:
            self._config = conf

        vae_config = self._config.get("vae", {})
        if not vae_component.get("config") or not vae_component.get("config_path"):
            vae_component["config"] = vae_config

        transformer_component["config"] = self._config
        # Back to official config
        self.config["components"][transformer_index] = transformer_component
        self.config["components"][vae_index] = vae_component

        self.pretrained_config = HunyuanImage3Config(**self._config)
        self.hy_image_processor = HunyuanImage3ImageProcessor(self.pretrained_config)
        self.latent_scale_factor = self.pretrained_config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.latent_scale_factor
        )
        self.cfg_operator = ClassifierFreeGuidance()
        self.num_channels_latents = self._config.get("vae", {}).get(
            "latent_channels", 32
        )

    def post_init(self):
        """
        Run BaseEngine's post-init logic, then customize memory management
        for this engine (we don't need memory management for the VAE).
        """
        super().post_init()
        if self._memory_management_map is not None:
            self._memory_management_map.pop("vae", None)

    def vae_encode(self, image, cfg_factor=1, offload=True):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        config = self.vae.config
        device_type = self.device.type

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=True):
            vae_encode_result = self.vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(config, "shift_factor") and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, "scaling_factor") and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        if hasattr(self.vae, "ffactor_temporal"):
            assert (
                latents.shape[2] == 1
            ), "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        # Here we always use t=0 to declare it is a clean conditional image
        t = torch.zeros((latents.shape[0],))

        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        if offload:
            self._offload("vae")

        return t, latents

    def vae_decode(self, latents, generator=None, offload=True):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        if (
            hasattr(self.vae.config, "scaling_factor")
            and self.vae.config.scaling_factor
        ):
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=True
        ):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        if hasattr(self.vae, "ffactor_temporal"):
            assert (
                image.shape[2] == 1
            ), "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        if offload:
            self._offload("vae")

        return image

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(
                    f"Length of seed must be equal to the batch_size({batch_size}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        return seeds

    def _get_latents(
        self,
        batch_size,
        latent_channel,
        image_size,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(
            self.latent_scale_factor, list
        ):
            assert len(self.latent_scale_factor) == len(
                image_size
            ), "len(latent_scale_factor) shoudl be the same as len(image_size)"
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {self.latent_scale_factor}"
            )

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                latents_shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _prepare_attention_mask_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_image_slices = [
            tokenizer_output.joint_image_slices[i]
            + tokenizer_output.gen_image_slices[i]
            for i in range(bsz)
        ]
        attention_mask = (
            torch.ones(seq_len, seq_len, dtype=torch.bool)
            .tril(diagonal=0)
            .repeat(bsz, 1, 1)
        )
        for i in range(bsz):
            for j, image_slice in enumerate(batch_image_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        tokenizer_output=None,
        batch_gen_image_info=None,
        generator=None,
        **kwargs,
    ):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            if input_ids.shape[1] != kwargs["position_ids"].shape[1]:  # in decode steps
                input_ids = torch.gather(input_ids, dim=1, index=kwargs["position_ids"])
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": kwargs["position_ids"],
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "custom_pos_emb": kwargs["custom_pos_emb"],
                "mode": kwargs["mode"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timestep": kwargs.get("timestep"),
                "gen_timestep_scatter_index": kwargs.get("gen_timestep_scatter_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_timestep": kwargs.get("cond_timestep"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "vit_kwargs": kwargs.get("vit_kwargs"),
                "cond_timestep_scatter_index": kwargs.get(
                    "cond_timestep_scatter_index"
                ),
            }
        )
        return model_inputs

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_kwargs[k] = v
        return extra_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "custom_pos_emb": model_kwargs["custom_pos_emb"],
        }

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                updated_model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "tokenizer_output" in model_kwargs:
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a real_pos to index the valid
                # end position of the sequence. If tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                real_pos = to_device(
                    model_kwargs["tokenizer_output"].real_pos, self.device
                )
                updated_model_kwargs["position_ids"] = real_pos
            else:
                # position ids
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = (
                    torch.arange(seq_len, device=image_mask.device)
                    .unsqueeze(0)
                    .repeat(bsz, 1)
                )
                position_ids = index.masked_select(image_mask.bool()).reshape(bsz, -1)
                timestep_position_ids = index[
                    torch.arange(bsz), model_kwargs["gen_timestep_scatter_index"][:, -1]
                ].unsqueeze(-1)
                updated_model_kwargs["position_ids"] = torch.cat(
                    [timestep_position_ids, position_ids], dim=1
                )

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                    model_kwargs["attention_mask"], updated_model_kwargs["position_ids"]
                ):
                    mask_list.append(
                        torch.index_select(
                            attention_mask_i, dim=1, index=position_ids_i.reshape(-1)
                        )
                    )
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs[
                    "gen_timestep_scatter_index"
                ]

        else:
            if mode == "gen_text":
                # Now we are in the decode steps.
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs[
                    "gen_timestep_scatter_index"
                ]

        return updated_model_kwargs

    def _encode_cond_image(
        self,
        batch_cond_image_info_list: List[List[JointImageInfo]],
        cfg_factor: int = 1,
    ):
        # VAE encode one by one, as we assume cond images have different sizes
        batch_cond_vae_images, batch_cond_t, batch_cond_vit_images = [], [], []
        for cond_image_info_list in batch_cond_image_info_list:
            cond_vae_image_list, cond_t_list, cond_vit_image_list = [], [], []
            for image_info in cond_image_info_list:
                cond_t_, cond_vae_image_ = self.vae_encode(
                    image_info.vae_image_info.image_tensor.to(self.device),
                )
                cond_vit_image_list.append(image_info.vision_image_info.image_tensor)
                cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                cond_t_list.append(cond_t_)
            batch_cond_vae_images.append(cond_vae_image_list)
            batch_cond_t.append(cond_t_list)
            batch_cond_vit_images.append(torch.cat(cond_vit_image_list, dim=0))

        # If only one cond image for each sample and all have the same size, we can batch them together
        # In this case, cond_vae_images is a 4-D tensor.
        if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
            items[0].shape == batch_cond_vae_images[0][0].shape
            for items in batch_cond_vae_images
        ):
            cond_vae_images = torch.stack(
                [items[0] for items in batch_cond_vae_images], dim=0
            )
            cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
            if cfg_factor > 1:
                cond_t = cond_t.repeat(cfg_factor)
                cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
        else:
            # In this case, cond_vae_images is a list of 4-D tensors or a list of lists of 3-D tensors.
            cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
            cond_vae_images = []
            for items in batch_cond_vae_images:
                if all(items[0].shape == item.shape for item in items):
                    cond_vae_images.append(torch.stack(items, dim=0))
                else:
                    cond_vae_images.append(items)
            if cfg_factor > 1:
                cond_t = cond_t * cfg_factor
                cond_vae_images = cond_vae_images * cfg_factor

        if cfg_factor > 1:
            batch_cond_vit_images = batch_cond_vit_images * cfg_factor

        return cond_vae_images, cond_t, batch_cond_vit_images

    @staticmethod
    def build_batch_rope_image_info(output, sections):
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            image_shapes = []
            for section in sections_i:
                if "image" in section["type"]:
                    if isinstance(section["token_height"], list):
                        assert len(section["token_height"]) == len(
                            section["token_height"]
                        ), (
                            f"token_height and token_width should have the same length, "
                            f"but got {len(section['token_height'])} and {len(section['token_width'])}"
                        )
                        image_shapes.extend(
                            list(zip(section["token_height"], section["token_width"]))
                        )
                    else:
                        image_shapes.append(
                            (section["token_height"], section["token_width"])
                        )
            assert len(image_slices) == len(
                image_shapes
            ), f"Size miss matching: Image slices({len(image_slices)}) != image shapes({len(image_shapes)})"
            rope_image_info.append(list(zip(image_slices, image_shapes)))
        return rope_image_info

    def prepare_model_inputs(
        self,
        prompt=None,
        mode="gen_text",
        system_prompt=None,
        cot_text=None,
        image_size="auto",
        message_list=None,
        device=None,
        max_new_tokens=None,
        drop_think=False,
        sequence_template="pretrain",
        max_sequence_length=12800,
        **kwargs,
    ):
        # 1. Sanity check
        device = default(device, self.device)
        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt
        batch_gen_image_info = None
        # TODO: construct with user input images
        batch_cond_image_info = None

        #   -- 2.1 message_list
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            batch_gen_image_info = [
                [
                    message["content"]
                    for message in message_list_
                    if message["type"] == "gen_image"
                ]
                for message_list_ in batch_message_list
            ]
            # At most one gen_image is allowed for each message_list
            batch_gen_image_info = [
                info[-1] if len(info) > 0 else None for info in batch_gen_image_info
            ]
            # Multiple cond images are allowed.
            batch_cond_image_info = [
                [
                    message["content"]
                    for message in message_list_
                    if message["type"] == "joint_image"
                ]
                for message_list_ in batch_message_list
            ]

        #   -- 2.2 Prompt, cot text, system prompt
        else:
            if isinstance(batch_prompt, str):
                batch_prompt = [batch_prompt]
            batch_size = len(batch_prompt)

            if batch_cot_text is not None:
                if isinstance(batch_cot_text, str):
                    batch_cot_text = [batch_cot_text]
                else:
                    assert (
                        isinstance(batch_cot_text, list)
                        and len(batch_cot_text) == batch_size
                    ), "`cot_text` should be a string or a list of strings with the same length as `prompt`."

            if batch_system_prompt is not None:
                if isinstance(batch_system_prompt, str):
                    batch_system_prompt = [batch_system_prompt]
                else:
                    assert (
                        isinstance(batch_system_prompt, list)
                        and len(batch_system_prompt) == batch_size
                    ), "`system_prompts` should be a string or a list of strings with the same length as `prompt`."

            if mode == "gen_image":
                batch_gen_image_info = [
                    self.hy_image_processor.build_image_info(image_size)
                    for _ in range(batch_size)
                ]

        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get("seed"), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 3. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        bot_task = kwargs.pop("bot_task", "auto")
        # If `drop_think` enabled, always drop <think> parts in the context.
        # Apply batched prompt or batched message_list to build input sequence with associated info.
        out = self.tokenizer_wrapper.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_image_info=batch_cond_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get("max_length"),
            bot_task=bot_task,
            image_base_size=self.pretrained_config.image_base_size,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor[mode],
            drop_think=drop_think,
        )
        output, sections = out["output"], out["sections"]

        # 4. Encode conditional images
        if batch_cond_image_info is not None and len(batch_cond_image_info[0]) > 0:
            cond_vae_images, cond_timestep, cond_vit_images = self._encode_cond_image(
                batch_cond_image_info, cfg_factor[mode]
            )
            # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
            vit_kwargs = {"spatial_shapes": [], "attention_mask": []}
            for cond_image_info in batch_cond_image_info:
                vit_kwargs["spatial_shapes"].append(
                    torch.stack(
                        [
                            item.vision_encoder_kwargs["spatial_shapes"]
                            for item in cond_image_info
                        ]
                    )
                )
                vit_kwargs["attention_mask"].append(
                    torch.stack(
                        [
                            item.vision_encoder_kwargs["pixel_attention_mask"]
                            for item in cond_image_info
                        ]
                    )
                )
            if cfg_factor[mode] > 1:
                vit_kwargs["spatial_shapes"] = (
                    vit_kwargs["spatial_shapes"] * cfg_factor[mode]
                )
                vit_kwargs["attention_mask"] = (
                    vit_kwargs["attention_mask"] * cfg_factor[mode]
                )
        else:
            cond_vae_images, cond_timestep, cond_vit_images = None, None, None
            vit_kwargs = None

        # 5. Build position embeddings
        rope_image_info = self.build_batch_rope_image_info(output, sections)

        seq_len = output.tokens.shape[1]
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.pretrained_config.attention_head_dim,
            device=device,
            base=self.pretrained_config.rope_theta,
        )

        # 6. Build kv cache
        if bot_task == "img_ratio":
            max_new_tokens = 1
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + default(
                max_new_tokens, max_sequence_length
            )
        cache = HunyuanStaticCache(
            config=self.pretrained_config,
            batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=torch.bfloat16,
            dynamic=False,
        )

        # 7. Build position ids
        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device
        )[None].expand(
            batch_size * cfg_factor[mode], -1
        )  # use expand to share indices to save memory

        # 8. Build model input kwargs
        tkw = self.tokenizer_wrapper
        if image_size == "auto":
            extra_auto_stops = [
                tkw.special_token_map[f"<img_ratio_{i}>"] for i in range(33)
            ]
        else:
            extra_auto_stops = [tkw.boi_token_id]
        stop_token_id = dict(
            auto=[tkw.eos_token_id] + extra_auto_stops,
            image=[tkw.eos_token_id],
            recaption=[
                tkw.end_recaption_token_id,
                tkw.end_answer_token_id,
                tkw.eos_token_id,
            ],
            think=[
                tkw.end_recaption_token_id,
                tkw.end_answer_token_id,
                tkw.eos_token_id,
            ],
            img_ratio=extra_auto_stops,
        )
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            position_ids=batch_input_pos,
            past_key_values=cache,
            custom_pos_emb=(cos, sin),
            mode=mode,
            image_mask=to_device(output.gen_image_mask, device),
            gen_timestep_scatter_index=to_device(
                output.gen_timestep_scatter_index, device
            ),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_timestep=to_device(cond_timestep, device),
            cond_vae_image_mask=to_device(output.cond_vae_image_mask, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.cond_vit_image_mask, device),
            vit_kwargs=(
                {k: to_device(v, self.device) for k, v in vit_kwargs.items()}
                if vit_kwargs is not None
                else None
            ),
            cond_timestep_scatter_index=to_device(
                output.cond_timestep_scatter_index, device
            ),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            # generation config
            eos_token_id=stop_token_id[bot_task],
            max_new_tokens=max_new_tokens,
        )

        return model_input_kwargs

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    def run(
        self,
        prompt: Union[List[str], str],
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
        use_system_prompt: bool | None = None,
        system_prompt: str | None = None,
        bot_task: str | None = "image",
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        guidance_rescale: float | None = 0.0,
        progress_callback: Callable | None = None,
        render_on_step_callback: Callable | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        sigmas: List[float] | None = None,
        timesteps_as_indices: bool = True,
        attention_kwargs: Dict[str, Any] = {},
        cot_text: str | None = None,
        verbose: int = 0,
        sequence_template: str = "pretrain",
        drop_think: bool = False,
        max_sequence_length: int = 12800,
        render_on_step_interval: int = 3,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)
        image_size = f"{height}x{width}"
        # Generate image
        model_kwargs = self.prepare_model_inputs(
            prompt=prompt,
            cot_text=cot_text,
            system_prompt=system_prompt,
            mode="gen_image",
            seed=seed,
            image_size=image_size,
            sequence_template=sequence_template,
            drop_think=drop_think,
            max_sequence_length=max_sequence_length,
        )
        safe_emit_progress(progress_callback, 0.10, "Prepared model inputs")

        batch_gen_image_info = model_kwargs["batch_gen_image_info"]
        batch_size = len(batch_gen_image_info)
        image_size = [
            batch_gen_image_info[0].image_height,
            batch_gen_image_info[0].image_width,
        ]
        generator = model_kwargs["generator"]
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        safe_emit_progress(progress_callback, 0.15, "Resolved generation metadata")

        cfg_factor = 1 + self.do_classifier_free_guidance

        # Define call parameters
        device = self.device

        if self.scheduler is None:
            self.load_component_by_type("scheduler")
        safe_emit_progress(progress_callback, 0.20, "Scheduler ready")

        # Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        safe_emit_progress(progress_callback, 0.25, "Timesteps computed")

        # Prepare latent variables
        latents = self._get_latents(
            batch_size=batch_size,
            latent_channel=self.num_channels_latents,
            image_size=image_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            latents=latents,
        )
        safe_emit_progress(progress_callback, 0.32, "Initialized latent noise")
        # Prepare extra step kwargs.
        _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator}
        )

        if self.transformer is None:
            safe_emit_progress(progress_callback, 0.34, "Loading transformer")
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.36, "Transformer ready")

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self._prepare_attention_mask_for_generation(  # noqa
            input_ids,
            model_kwargs=model_kwargs,
        )

        model_kwargs["attention_mask"] = attention_mask.to(latents.device)
        safe_emit_progress(
            progress_callback, 0.40, "Prepared attention mask and model kwargs"
        )

        # Sampling loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        safe_emit_progress(progress_callback, 0.45, "Starting denoising loop")

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * cfg_factor)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0])

                model_inputs = self.prepare_inputs_for_generation(
                    input_ids,
                    images=latent_model_input,
                    timestep=t_expand,
                    **model_kwargs,
                )

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=True
                ):
                    model_output = self.transformer(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]

                # perform guidance
                if self.do_classifier_free_guidance:
                    pred_cond, pred_uncond = pred.chunk(2)
                    pred = self.cfg_operator(
                        pred_cond, pred_uncond, guidance_scale, step=i
                    )

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    pred = rescale_noise_cfg(
                        pred, pred_cond, guidance_rescale=guidance_rescale
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False
                )[0]

                # Map inner denoising progress into [0.50, 0.90]
                if denoise_progress_callback is not None and self._num_timesteps > 0:
                    local_p = float(i + 1) / float(self._num_timesteps)
                    denoise_progress_callback(
                        local_p, f"Denoising step {i + 1}/{self._num_timesteps}"
                    )

                if i != len(timesteps) - 1:
                    model_kwargs = self._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                        input_ids = torch.gather(
                            input_ids, 1, index=model_kwargs["position_ids"]
                        )

                if (
                    render_on_step_callback is not None
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        image = self.vae_decode(latents, generator=generator, offload=offload)
        # b c t h w

        do_denormalize = [True] * image.shape[0]
        image = self._tensor_to_frame(
            image, output_type="pil", do_denormalize=do_denormalize
        )
        safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
        return image

    def _render_step(self, latents, render_on_step_callback):
        image = self.vae_decode(latents, offload=True)
        do_denormalize = [True] * image.shape[0]
        image = self._tensor_to_frame(
            image, output_type="pil", do_denormalize=do_denormalize
        )
        render_on_step_callback(image[0])
