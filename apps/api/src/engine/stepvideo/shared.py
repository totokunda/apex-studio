import torch
import os
import packaging.version as pver
import numpy as np
from torchvision import transforms
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor

OPTIMUS_LOAD_LIBRARIES = {
    "torch2.2": "https://huggingface.co/stepfun-ai/stepvideo-t2v/resolve/main/lib/liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so",
    "torch2.3": "https://huggingface.co/stepfun-ai/stepvideo-t2v/resolve/main/lib/liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so",
    "torch2.5": "https://huggingface.co/stepfun-ai/stepvideo-t2v/resolve/main/lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so",
}


class StepVideoShared(BaseEngine):
    """Base class for StepVideo engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):

        super().__init__(yaml_path, **kwargs)

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temporal_compression_ratio)
            if getattr(self.vae, "temporal_compression_ratio", None)
            else 8
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.spatial_compression_ratio)
            if getattr(self.vae, "spatial_compression_ratio", None)
            else 16
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "z_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        self._load_optimus()

    def _load_optimus(self):
        if hasattr(torch.ops, "Optimus") and hasattr(torch.ops.Optimus, "fwd"):
            self.logger.info("Optimus library already loaded")
            return True
        else:
            # Check torch version
            torch_version = torch.__version__.split("+")[0]  # Remove any +cu121 suffix

            # Check if torch version is >= 2.2 and CUDA is available
            if (
                pver.parse(torch_version) >= pver.parse("2.2.0")
                and torch.cuda.is_available()
            ):
                # Check CUDA version
                cuda_version = torch.version.cuda
                if cuda_version and pver.parse(cuda_version) >= pver.parse("12.0"):
                    # Determine which library to download based on torch version
                    if pver.parse(torch_version) >= pver.parse("2.5.0"):
                        library_key = "torch2.5"
                    elif pver.parse(torch_version) >= pver.parse("2.3.0"):
                        library_key = "torch2.3"
                    else:  # >= 2.2.0
                        library_key = "torch2.2"

                    library_url = OPTIMUS_LOAD_LIBRARIES.get(library_key)
                    if library_url:
                        # Construct save path - avoid /dev/shm for shared libraries
                        save_path = getattr(self, "save_path", None)
                        if save_path is None:
                            from src.utils.defaults import DEFAULT_SAVE_PATH

                            save_path = DEFAULT_SAVE_PATH

                        # Use a specific optimus directory instead of preprocessors
                        optimus_dir = os.path.join(save_path, "optimus")
                        os.makedirs(optimus_dir, exist_ok=True)

                        # Create a simple filename instead of preserving URL structure
                        filename = f"liboptimus_ths-{library_key}.so"
                        final_lib_path = os.path.join(optimus_dir, filename)

                        try:
                            # Check if library already exists
                            if os.path.exists(final_lib_path):
                                self.logger.info(
                                    f"Optimus library already exists at {final_lib_path}"
                                )
                            else:
                                # Download the library
                                self.logger.info(
                                    f"Downloading Optimus library {library_key} from {library_url}"
                                )
                                downloaded_path = self._download(
                                    library_url, optimus_dir
                                )

                                if downloaded_path and os.path.exists(downloaded_path):
                                    # If the downloaded path is different from our target, copy it
                                    if downloaded_path != final_lib_path:
                                        import shutil

                                        self.logger.info(
                                            f"Moving library from {downloaded_path} to {final_lib_path}"
                                        )
                                        shutil.move(downloaded_path, final_lib_path)
                                        # Clean up any empty directories created by the URL structure
                                        try:
                                            parent_dir = os.path.dirname(
                                                downloaded_path
                                            )
                                            while (
                                                parent_dir != optimus_dir
                                                and os.path.exists(parent_dir)
                                            ):
                                                if not os.listdir(
                                                    parent_dir
                                                ):  # Only remove if empty
                                                    os.rmdir(parent_dir)
                                                    parent_dir = os.path.dirname(
                                                        parent_dir
                                                    )
                                                else:
                                                    break
                                        except:
                                            pass  # Ignore cleanup errors
                                else:
                                    self.logger.error(
                                        f"Failed to download Optimus library"
                                    )
                                    return False

                            # Ensure proper permissions for shared library
                            if os.path.exists(final_lib_path):
                                os.chmod(final_lib_path, 0o755)  # rwxr-xr-x

                                # Load the library
                                self.logger.info(
                                    f"Loading Optimus library from {final_lib_path}"
                                )
                                torch.ops.load_library(final_lib_path)

                                # Verify it's loaded correctly
                                if hasattr(torch.ops, "Optimus") and hasattr(
                                    torch.ops.Optimus, "fwd"
                                ):
                                    self.logger.info(
                                        "Optimus library loaded successfully"
                                    )
                                    return True
                                else:
                                    self.logger.warning(
                                        "Optimus library downloaded but not loaded properly"
                                    )
                                    return False
                            else:
                                self.logger.error(
                                    f"Library file not found at {final_lib_path}"
                                )
                                return False

                        except Exception as e:
                            self.logger.error(f"Failed to load Optimus library: {e}")
                            # Try to provide more specific error information
                            if "failed to map segment" in str(e):
                                self.logger.error(
                                    "This error often indicates permission issues or incompatible library. Try:"
                                )
                                self.logger.error(
                                    "1. Check if /tmp or target directory has execute permissions"
                                )
                                self.logger.error("2. Ensure sufficient disk space")
                                self.logger.error(
                                    "3. Verify CUDA and torch versions match the library"
                                )
                            return False
                    else:
                        self.logger.warning(
                            f"No Optimus library available for torch version {torch_version}"
                        )
                        return False
                else:
                    cuda_ver_str = cuda_version if cuda_version else "unknown"
                    self.logger.info(
                        f"CUDA version {cuda_ver_str} is too old, need CUDA 12+ for Optimus"
                    )
                    return False
            else:
                self.logger.info(
                    f"Torch version {torch_version} is too old or CUDA not available, need torch 2.2+ with CUDA for Optimus"
                )
                return False

    def _parse_num_frames(self, duration: int | str, fps: int = 16):
        """Accepts a duration in seconds or a string like "16" or "16s" and returns the number of frames.

        Args:
            duration (int | str): duration in seconds or a string like "16" or "16s"

        Returns:
            int: number of frames
        """

        if isinstance(duration, str):
            if duration.endswith("s"):
                duration = int(duration[:-1]) * fps + 1
            elif duration.endswith("f"):
                duration = int(duration[:-1])
            else:
                duration = int(duration)

        if duration % 17 != 0:
            duration = duration // 17 * 17
        duration = max(duration, 1)
        return duration

    @torch.no_grad()
    def vae_decode(
        self,
        latents: torch.Tensor,
        offload: bool = False,
        dtype: torch.dtype | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (16, 16),
        smooth_scale: float = 0.6,
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        denormalized_latents = self.vae.denormalize_latents(latents).to(
            dtype=self.vae.dtype, device=self.device
        )
        video = self.vae.decode(
            denormalized_latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            smooth_scale=smooth_scale,
            return_dict=False,
        )[0]
        if offload:
            self._offload("vae")
        return video.to(dtype=dtype)

    def resize_to_desired_aspect_ratio(self, video, aspect_size):
        ## video is in shape [f, c, h, w]
        height, width = video.shape[-2:]

        aspect_ratio = [w / h for h, w in aspect_size]
        # # resize
        aspect_ratio_fact = width / height
        bucket_idx = np.argmin(np.abs(aspect_ratio_fact - np.array(aspect_ratio)))
        aspect_ratio = aspect_ratio[bucket_idx]
        target_size_height, target_size_width = aspect_size[bucket_idx]

        if aspect_ratio_fact < aspect_ratio:
            scale = target_size_width / width
        else:
            scale = target_size_height / height

        width_scale = int(round(width * scale))
        height_scale = int(round(height * scale))

        # # crop
        delta_h = height_scale - target_size_height
        delta_w = width_scale - target_size_width
        assert delta_w >= 0
        assert delta_h >= 0
        assert not all([delta_h, delta_w])
        top = delta_h // 2
        left = delta_w // 2

        ## resize image and crop
        resize_crop_transform = transforms.Compose(
            [
                transforms.Resize((height_scale, width_scale)),
                lambda x: transforms.functional.crop(
                    x, top, left, target_size_height, target_size_width
                ),
            ]
        )

        video = torch.stack(
            [resize_crop_transform(frame.contiguous()) for frame in video], dim=0
        )
        return video

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        with self._progress_bar(
            len(timesteps), desc=f"Sampling {self.model_type}"
        ) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                latent_model_input = latent_model_input.to(transformer_dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(transformer_dtype)

                # Forward pass with both text and image conditioning
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                if use_cfg_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)

            self.logger.info("Denoising completed.")

        return latents
