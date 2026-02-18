from src.engine import UniversalEngine
from safetensors.torch import save_model
import os
import torch


def save_model_no_shared_storage(model: torch.nn.Module, filename: str) -> None:
    """
    `safetensors.torch.save_model()` refuses to save tensors that are views into a
    larger storage (no "complete" tensor covers the whole storage). Some models
    (e.g. vision towers / sharded params) can contain such parameters.

    For exporting weights, we can safely materialize each tensor as its own
    contiguous storage and write via `save_file`.
    """
    def _is_complete_storage(t: torch.Tensor) -> bool:
        # Mirrors safetensors' internal "complete storage" check: the tensor must
        # cover the whole underlying storage (not be a view into a larger buffer).
        if t.device.type == "meta" or t.numel() == 0:
            return True
        try:
            st = t.untyped_storage()
            return (t.data_ptr() == st.data_ptr()) and (
                t.numel() * t.element_size() == st.nbytes()
            )
        except Exception:
            # If storage introspection fails (older torch / weird storage),
            # fall back to cloning to be safe.
            return False

    state_dict = model.state_dict()
    sanitized: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue

        # Keep on CPU to avoid GPU RAM spikes while exporting.
        t = v.detach()
        if t.device.type != "cpu":
            t = t.cpu()

        # If the tensor is a view into a larger storage (the cause of your crash),
        # materialize it into its own storage. `contiguous()` is enough and avoids
        # a second allocation vs `clone().contiguous()`.
        if (not _is_complete_storage(t)) or (not t.is_contiguous()):
            t = t.contiguous()

        sanitized[k] = t

    save_file(sanitized, filename)

engine = UniversalEngine(
    yaml_path="/home/tosin_coverquick_co/apex-studio/apps/api/new_manifest/image/flux2-dev-text-to-image-edit-1.0.0.v1.yml",
    selected_components={
        "vae": {
            "variant": "default"
        },
        "text_encoder": {
            "variant": "default"
        },
        "transformer": {
            "variant": "default"
        }
    },
    attention_type="sage",
    debug_component_vram=True,
    disable_text_encoder_cache=True,
    download_weights=False,
    components_to_load=["vae"]
)

vae = engine.vae

os.makedirs("/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/vae", exist_ok=True)

save_model(vae, "/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/vae/vae-bf16.safetensors")