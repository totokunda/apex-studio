from .shared import WanShared
from src.types import InputImage, InputAudio
from typing import List, Dict, Any, Callable
from torch import Tensor
from PIL import Image
import torch
from src.utils.progress import safe_emit_progress, make_mapped_progress
import numpy as np
import torch.nn.functional as F
import math
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
import torch.amp as amp


class HuMoEngine(WanShared):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

    @staticmethod
    def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
        features = features.transpose(1, 2)  # [1, C, T]
        seq_len = features.shape[2] / float(input_fps)
        if output_len is None:
            output_len = int(seq_len * output_fps)
        output_features = F.interpolate(
            features, size=output_len, align_corners=True, mode="linear"
        )
        return output_features.transpose(1, 2)

    def load_image_latents(
        self, image: InputImage | List[InputImage], size, device=None
    ):
        # Load size.
        h, w = size[1], size[0]
        height, width = None, None
        device = device or self.device

        if self.vae is None:
            self.load_component_by_type("vae")
            self.to_device(self.vae)

        # Load image.
        if isinstance(image, list):
            ref_vae_latents = []
            for _image in image:
                img = self._load_image(_image)
                img, height, width = self._aspect_ratio_resize(img, max_area=h * w)
                new_img = self.video_processor.preprocess(img).unsqueeze(2)
                img_vae_latent = self.vae_encode(new_img)[0]
                ref_vae_latents.append(img_vae_latent)

            return [torch.cat(ref_vae_latents, dim=1)], height, width
        else:
            img = self._load_image(image)
            # Calculate the required size to keep aspect ratio and fill the rest with padding.
            img, height, width = self._aspect_ratio_resize(img, max_area=h * w)
            new_img = self.video_processor.preprocess(img).unsqueeze(2)
            img_vae_latent = self.vae_encode(new_img)[0]
            return [img_vae_latent], height, width

    def get_audio_emb_window(self, audio_emb, frame_num, frame0_idx, audio_shift=2):
        zero_audio_embed = torch.zeros(
            (audio_emb.shape[1], audio_emb.shape[2]),
            dtype=audio_emb.dtype,
            device=audio_emb.device,
        )
        zero_audio_embed_3 = torch.zeros(
            (3, audio_emb.shape[1], audio_emb.shape[2]),
            dtype=audio_emb.dtype,
            device=audio_emb.device,
        )  # device=audio_emb.device
        iter_ = 1 + (frame_num - 1) // 4
        audio_emb_wind = []
        for lt_i in range(iter_):
            if lt_i == 0:
                st = frame0_idx + lt_i - 2
                ed = frame0_idx + lt_i + 3
                wind_feat = torch.stack(
                    [
                        (
                            audio_emb[i]
                            if (0 <= i < audio_emb.shape[0])
                            else zero_audio_embed
                        )
                        for i in range(st, ed)
                    ],
                    dim=0,
                )
                wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
            else:
                st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
                ed = frame0_idx + 1 + 4 * lt_i + audio_shift
                wind_feat = torch.stack(
                    [
                        (
                            audio_emb[i]
                            if (0 <= i < audio_emb.shape[0])
                            else zero_audio_embed
                        )
                        for i in range(st, ed)
                    ],
                    dim=0,
                )
            audio_emb_wind.append(wind_feat)
        audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

        return audio_emb_wind, ed - audio_shift

    def audio_emb_enc(self, audio_emb, wav_enc_type="whisper"):
        if wav_enc_type == "wav2vec":
            feat_merge = audio_emb
        elif wav_enc_type == "whisper":
            feat0 = self.linear_interpolation_fps(
                audio_emb[:, :, 0:8].mean(dim=2), 50, 25
            )
            feat1 = self.linear_interpolation_fps(
                audio_emb[:, :, 8:16].mean(dim=2), 50, 25
            )
            feat2 = self.linear_interpolation_fps(
                audio_emb[:, :, 16:24].mean(dim=2), 50, 25
            )
            feat3 = self.linear_interpolation_fps(
                audio_emb[:, :, 24:32].mean(dim=2), 50, 25
            )
            feat4 = self.linear_interpolation_fps(audio_emb[:, :, 32], 50, 25)
            feat_merge = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
        else:
            raise ValueError(f"Unsupported wav_enc_type: {wav_enc_type}")

        return feat_merge

    def parse_output(self, output):
        latent = output[0][0]
        mask = None
        return latent, mask

    def forward_tia(
        self,
        latents,
        timestep,
        t,
        step_change,
        arg_tia,
        arg_ti,
        arg_i,
        arg_null,
        scale_a,
        scale_t,
        cfg_guidance: bool = True,
    ):
        pos_tia, _ = self.parse_output(self.transformer(latents, t=timestep, **arg_tia))
        torch.cuda.empty_cache()

        pos_ti, _ = self.parse_output(self.transformer(latents, t=timestep, **arg_ti))
        torch.cuda.empty_cache()

        # If CFG is disabled, ignore negative/uncond prompt by skipping the neg forward pass.
        # This reduces to a 2-branch blend between the two positive predictions.
        if not cfg_guidance:
            neg = pos_ti
        elif t > step_change:
            neg, _ = self.parse_output(
                self.transformer(latents, t=timestep, **arg_i)
            )  # img included in null, same with official Wan-2.1
            torch.cuda.empty_cache()
        else:
            neg, _ = self.parse_output(
                self.transformer(latents, t=timestep, **arg_null)
            )  # img not included in null
            torch.cuda.empty_cache()

        if t > step_change or not cfg_guidance:
            noise_pred = scale_a * (pos_tia - pos_ti) + scale_t * (pos_ti - neg) + neg
        else:
            noise_pred = (
                scale_a * (pos_tia - pos_ti) + (scale_t - 2.0) * (pos_ti - neg) + neg
            )
        return noise_pred

    def forward_ta(
        self,
        latents,
        timestep,
        arg_ta,
        arg_t,
        arg_null,
        scale_a,
        scale_t,
        cfg_guidance: bool = True,
    ):
        pos_ta, _ = self.parse_output(self.transformer(latents, t=timestep, **arg_ta))
        torch.cuda.empty_cache()

        pos_t, _ = self.parse_output(self.transformer(latents, t=timestep, **arg_t))
        torch.cuda.empty_cache()

        # If CFG is disabled, ignore negative/uncond prompt by skipping the neg forward pass.
        # This reduces to a 2-branch blend between the two positive predictions.
        if not cfg_guidance:
            neg = pos_t
        else:
            neg, _ = self.parse_output(self.transformer(latents, t=timestep, **arg_null))
            torch.cuda.empty_cache()

        noise_pred = scale_a * (pos_ta - pos_t) + scale_t * (pos_t - neg) + neg
        return noise_pred

    def forward_tia_small(
        self,
        latents,
        latents_ref,
        latents_ref_neg,
        timestep,
        arg_t,
        arg_ta,
        arg_null,
        scale_a,
        scale_t,
        cfg_guidance: bool = True,
    ):
        pos_t = self.transformer(
            [
                torch.cat(
                    [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                )
                for latent, latent_ref_neg in zip(latents, latents_ref_neg)
            ],
            t=timestep,
            **arg_t,
        )[0][0]

        # If CFG is disabled, ignore negative/uncond prompt by skipping the neg forward pass.
        # This reduces to a 2-branch blend between the two positive predictions.
        if not cfg_guidance:
            neg = pos_t
        else:
            neg = self.transformer(
                [
                    torch.cat(
                        [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                    )
                    for latent, latent_ref_neg in zip(latents, latents_ref_neg)
                ],
                t=timestep,
                **arg_null,
            )[0][0]
        pos_ta = self.transformer(
            [
                torch.cat(
                    [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                )
                for latent, latent_ref_neg in zip(latents, latents_ref_neg)
            ],
            t=timestep,
            **arg_ta,
        )[0][0]
        pos_tia = self.transformer(
            [
                torch.cat([latent[:, : -latent_ref.shape[1]], latent_ref], dim=1)
                for latent, latent_ref in zip(latents, latents_ref)
            ],
            t=timestep,
            **arg_ta,
        )[0][0]

        noise_pred = (
            scale_a * (pos_tia - pos_ta)
            + scale_a * (pos_ta - pos_t)
            + scale_t * (pos_t - neg)
            + neg
        )

        return noise_pred

    def forward_ta_small(
        self,
        latents,
        latents_ref_neg,
        timestep,
        arg_t,
        arg_ta,
        arg_null,
        scale_a,
        scale_t,
        cfg_guidance: bool = True,
    ):
        pos_t = self.transformer(
            [
                torch.cat(
                    [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                )
                for latent, latent_ref_neg in zip(latents, latents_ref_neg)
            ],
            t=timestep,
            **arg_t,
        )[0][0]

        # If CFG is disabled, ignore negative/uncond prompt by skipping the neg forward pass.
        # This reduces to a 2-branch blend between the two positive predictions.
        if not cfg_guidance:
            neg = pos_t
        else:
            neg = self.transformer(
                [
                    torch.cat(
                        [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                    )
                    for latent, latent_ref_neg in zip(latents, latents_ref_neg)
                ],
                t=timestep,
                **arg_null,
            )[0][0]
        pos_ta = self.transformer(
            [
                torch.cat(
                    [latent[:, : -latent_ref_neg.shape[1]], latent_ref_neg], dim=1
                )
                for latent, latent_ref_neg in zip(latents, latents_ref_neg)
            ],
            t=timestep,
            **arg_ta,
        )[0][0]

        noise_pred = scale_a * (pos_ta - pos_t) + scale_t * (pos_t - neg) + neg

        return noise_pred

    def get_zero_vae(self, height, width):
        if width * height < 720 * 1280:
            return self.zero_vae_480
        else:
            return self.zero_vae_720

    def get_num_weights(self) -> int:
        """
        Return the total number of parameters of the HuMo diffusion backbone.
        Can be used to distinguish between the 1.7B and 17B checkpoints.
        """
        # Prefer the HuMo transformer if it is present, otherwise fall back to DiT.
        model = None
        if getattr(self, "transformer", None) is not None:
            model = self.transformer
        else:
            # Lazily load the transformer component if nothing is initialized yet.
            self.load_component_by_type("transformer")
            model = self.transformer

        return sum(p.numel() for p in model.parameters())

    def run(
        self,
        prompt: List[str] | str,
        audio: InputAudio | None,
        image: InputImage | List[InputImage] | None = None,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 97,
        fps: int = 25,
        num_inference_steps: int = 50,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        progress_callback: Callable | None = None,
        denoise_progress_callback: Callable | None = None,
        render_on_step_callback: Callable | None = None,
        render_on_step: bool = False,
        render_on_step_interval: int = 3,
        text_encoder_kwargs: Dict[str, Any] = {},
        offload: bool = True,
        guidance_scale_a: float = 5.5,
        guidance_scale_t: float = 5.0,
        use_audio_length: bool = False,
        step_change: int = 980,
        return_latents: bool = False,
        resolution: int | None = None,
        aspect_ratio: str | None = None,
        **kwargs,
    ):

        use_cfg_guidance = (
            guidance_scale_t > 1.0
            and negative_prompt is not None
        )
        
        safe_emit_progress(progress_callback, 0.0, "Starting HuMo pipeline")

        safe_emit_progress(
            progress_callback, 0.01, f"Resolved target size ({width}x{height})"
        )

        transformer_dtype = self.component_dtypes["transformer"]
        if image is not None:
            generation_mode = "TIA"
            safe_emit_progress(progress_callback, 0.02, "Loading image condition (TIA)")
            latents_ref, height, width = self.load_image_latents(image, (width, height))
        else:
            generation_mode = "TA"
            safe_emit_progress(progress_callback, 0.02, "No image provided (TA)")
            latents_ref = [
                torch.zeros(16, 1, height // 8, width // 8).to(
                    self.device, dtype=transformer_dtype
                )
            ]
            height, width = self.get_height_width(height, width, resolution, aspect_ratio)

        latents_ref_neg = [torch.zeros_like(latent_ref) for latent_ref in latents_ref]
        latents_ref = [latent_ref for latent_ref in latents_ref]

        if offload and self.vae is not None:
            self._offload("vae")
            safe_emit_progress(progress_callback, 0.03, "VAE offloaded")

        audio_processor = None

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            safe_emit_progress(progress_callback, 0.04, f"Seeded generator ({seed})")

        frame_num = self._parse_num_frames(duration, fps)
        safe_emit_progress(progress_callback, 0.05, f"Resolved frames: {frame_num} @ {fps}fps")

        if audio is not None:
            audio_processor = self.helpers["wan.humo_audio_processor"]
            audio_processor.update_sample_rate(16000)
            audio_processor.update_fps(fps)
            safe_emit_progress(progress_callback, 0.06, "Preprocessing audio")
            audio_emb, audio_length = audio_processor.preprocess(audio)
            safe_emit_progress(
                progress_callback, 0.10, f"Audio preprocessed (length={audio_length})"
            )
        else:
            audio_emb = torch.zeros(frame_num, 5, 1280).to(self.device)
            audio_length = frame_num
            safe_emit_progress(progress_callback, 0.10, "No audio provided; using zeros")

        if offload and audio_processor is not None:
            self._offload("audio_processor")
            safe_emit_progress(progress_callback, 0.11, "Audio processor offloaded")

        if use_audio_length:
            frame_num = audio_length
        else:
            frame_num = frame_num if frame_num != -1 else audio_length
        frame_num = 4 * ((frame_num - 1) // 4) + 1
        safe_emit_progress(progress_callback, 0.12, f"Clamped/rounded frame_num -> {frame_num}")
        audio_emb, _ = self.get_audio_emb_window(audio_emb, frame_num, frame0_idx=0)
        zero_audio_pad = torch.zeros(latents_ref[0].shape[1], *audio_emb.shape[1:]).to(
            audio_emb.device
        )
        audio_emb = torch.cat([audio_emb, zero_audio_pad], dim=0)
        audio_emb = [audio_emb.to(self.device)]
        audio_emb_neg = [torch.zeros_like(audio_emb[0])]

        safe_emit_progress(progress_callback, 0.13, "Preparing initial noise latents")
        noise = self._get_latents(
            batch_size=1,
            num_channels_latents=self.num_channels_latents,
            duration=frame_num,
            fps=fps,
            height=height,
            width=width,
            dtype=torch.float32,
            seed=seed,
        )
        safe_emit_progress(progress_callback, 0.16, "Initialized noise latents")

        zero_video_tensor = torch.zeros(
            1, 3, frame_num, noise.shape[3] * 8, noise.shape[4] * 8
        ).to(self.device)
        safe_emit_progress(progress_callback, 0.17, "Encoding zero VAE latent (reference)")
        zero_vae = self.vae_encode(
            zero_video_tensor, offload=offload, sample_mode="mode"
        )[0].squeeze(0)
        transformer_config = self.load_config_by_type("transformer")
        noise_shape = noise.shape[2:]
        noise_shape = [
            noise_shape[i] // transformer_config["patch_size"][i]
            for i in range(len(noise_shape))
        ]
        seq_len = math.prod(noise_shape)
        noise = [noise.squeeze(0)]
        target_shape = tuple(noise[0].shape)
        safe_emit_progress(progress_callback, 0.19, f"Prepared seq_len={seq_len}")

        safe_emit_progress(
            progress_callback,
            0.19,
            f"Encoding prompts (CFG: {'on' if use_cfg_guidance else 'off'})",
        )
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            num_videos=1,
            progress_callback=progress_callback,
            text_encoder_kwargs=text_encoder_kwargs,
            offload=offload,
        )

        safe_emit_progress(progress_callback, 0.20, "Encoded prompts")

        context = [prompt_embeds[0]]
        if use_cfg_guidance:
            context_null = [negative_prompt_embeds[0]]
        else:
            context_null = [torch.zeros_like(prompt_embeds[0])]

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.30, "Loading scheduler")
            self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
        )

        safe_emit_progress(
            progress_callback, 0.40, "Scheduler ready and timesteps prepared"
        )

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.42, "Loading transformer")
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.45, "Transformer ready")

        # HuMo has two variants: ~1.7B params and ~17B params.
        # Use a threshold safely between them so this is True only for the 1.7B model.
        num_params = self.get_num_weights()
        small_model = num_params < 10_000_000_000
        safe_emit_progress(
            progress_callback,
            0.46,
            f"Model params={num_params} (variant={'small' if small_model else 'large'})",
        )

        latents = noise

        # Reserve a progress span for denoising [0.50, 0.90]
        if denoise_progress_callback is None:
            denoise_progress_callback = make_mapped_progress(
                progress_callback, 0.50, 0.90
            )
        total_steps = len(timesteps)
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        self.logger.info(f"Using HuMo model with {num_params} parameters")

        with amp.autocast(device_type=self.device.type, dtype=transformer_dtype):
            if not small_model:
                msk = torch.ones(
                    4,
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    device=self.device,
                )
                msk[:, : -latents_ref[0].shape[1]] = 0

                y_c = torch.cat(
                    [
                        zero_vae[:, : target_shape[1] - latents_ref[0].shape[1]],
                        latents_ref[0],
                    ],
                    dim=1,
                )

                y_c = [torch.concat([msk, y_c])]
                y_null = zero_vae[:, : target_shape[1]].to(
                    device=self.device, dtype=transformer_dtype
                )
                y_null = [torch.concat([msk, y_null])]
                arg_null = {
                    "seq_len": seq_len,
                    "audio": audio_emb_neg,
                    "y": y_null,
                    "context": context_null,
                }
                arg_t = {
                    "seq_len": seq_len,
                    "audio": audio_emb_neg,
                    "y": y_null,
                    "context": context,
                }
                arg_i = {
                    "seq_len": seq_len,
                    "audio": audio_emb_neg,
                    "y": y_c,
                    "context": context_null,
                }
                arg_ti = {
                    "seq_len": seq_len,
                    "audio": audio_emb_neg,
                    "y": y_c,
                    "context": context,
                }
                arg_ta = {
                    "seq_len": seq_len,
                    "audio": audio_emb,
                    "y": y_null,
                    "context": context,
                }
                arg_tia = {
                    "seq_len": seq_len,
                    "audio": audio_emb,
                    "y": y_c,
                    "context": context,
                }

                for i, t in enumerate(
                    tqdm(timesteps, desc="Sampling", total=len(timesteps))
                ):
                    timestep = [t]
                    timestep = torch.stack(timestep)
                    if generation_mode == "TIA":
                        noise_pred = self.forward_tia(
                            latents,
                            timestep,
                            t,
                            step_change,
                            arg_tia,
                            arg_ti,
                            arg_i,
                            arg_null,
                            guidance_scale_a,
                            guidance_scale_t,
                            cfg_guidance=use_cfg_guidance,
                        )
                    elif generation_mode == "TA":
                        noise_pred = self.forward_ta(
                            latents,
                            timestep,
                            arg_ta,
                            arg_t,
                            arg_null,
                            guidance_scale_a,
                            guidance_scale_t,
                            cfg_guidance=use_cfg_guidance,
                        )

                    temp_x0 = self.scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latents = [temp_x0.squeeze(0)]

                    if (
                        render_on_step
                        and render_on_step_callback
                        and ((i + 1) % render_on_step_interval == 0 or i == 0)
                        and i != len(timesteps) - 1
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i + 1) / float(total_steps) if total_steps else 1.0,
                            "Rendering preview",
                        )
                        self._render_step(
                            torch.stack(
                                [x0_[:, : -latents_ref[0].shape[1]] for x0_ in latents]
                            ),
                            render_on_step_callback,
                        )

                    safe_emit_progress(
                        denoise_progress_callback,
                        float(i + 1) / float(total_steps) if total_steps else 1.0,
                        f"Denoising step {i + 1}/{total_steps}",
                    )
            else:
                arg_ta = {"context": context, "seq_len": seq_len, "audio": audio_emb}
                arg_t = {"context": context, "seq_len": seq_len, "audio": audio_emb_neg}
                arg_null = {
                    "context": context_null,
                    "seq_len": seq_len,
                    "audio": audio_emb_neg,
                }

                for i, t in enumerate(
                    tqdm(timesteps, desc="Sampling", total=num_inference_steps)
                ):
                    timestep = [t]
                    timestep = torch.stack(timestep)
                    if generation_mode == "TIA":
                        noise_pred = self.forward_tia_small(
                            latents,
                            latents_ref,
                            latents_ref_neg,
                            timestep,
                            arg_t,
                            arg_ta,
                            arg_null,
                            guidance_scale_a,
                            guidance_scale_t,
                            cfg_guidance=use_cfg_guidance,
                        )
                    elif generation_mode == "TA":
                        noise_pred = self.forward_ta_small(
                            latents,
                            latents_ref_neg,
                            timestep,
                            arg_t,
                            arg_ta,
                            arg_null,
                            guidance_scale_a,
                            guidance_scale_t,
                            cfg_guidance=use_cfg_guidance,
                        )

                    temp_x0 = self.scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latents = [temp_x0.squeeze(0)]

                    if (
                        render_on_step
                        and render_on_step_callback
                        and ((i + 1) % render_on_step_interval == 0 or i == 0)
                        and i != len(timesteps) - 1
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i + 1) / float(total_steps) if total_steps else 1.0,
                            "Rendering preview",
                        )
                        self._render_step(
                            torch.stack(
                                [x0_[:, : -latents_ref[0].shape[1]] for x0_ in latents]
                            ),
                            render_on_step_callback,
                        )

                    safe_emit_progress(
                        denoise_progress_callback,
                        float(i + 1) / float(total_steps) if total_steps else 1.0,
                        f"Denoising step {i + 1}/{total_steps}",
                    )

        x0 = latents
        x0 = torch.stack([x0_[:, : -latents_ref[0].shape[1]] for x0_ in x0])

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return x0

        safe_emit_progress(progress_callback, 0.96, "Decoding video")
        video = self.vae_decode(x0, offload=offload)
        video = self._tensor_to_frames(video)
        safe_emit_progress(progress_callback, 1.0, "Completed HuMo pipeline")
        return video
