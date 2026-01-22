import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers.models.attention_processor import Attention
from einops.layers.torch import Rearrange
from safetensors.torch import load_file
from skimage import transform as trans
from accelerate import init_empty_weights
from src.helpers.base import BaseHelper
from torch.nn import RMSNorm
from src.attention import attention_register

try:
    from flash_attn_interface import flash_attn_varlen_func  # flash attn 3
except Exception:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
        )  # flash attn 2
    except Exception:  # pragma: no cover
        flash_attn_varlen_func = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None
try:
    from facexlib.recognition import init_recognition_model
except Exception:  # pragma: no cover
    init_recognition_model = None


# ------------------------------- Math helpers ------------------------------- #
def masked_mean(x: torch.Tensor, dim: int, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = torch.clamp(mask.sum(dim=dim, keepdim=True), min=1e-6)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom


def flash_attention(query, key, value, q_lens, kv_lens, causal=False):

    B, H, T_q, D_h = query.shape
    T_k = key.shape[2]
    device = query.device

    q = query.permute(0, 2, 1, 3).reshape(B * T_q, H, D_h)
    k = key.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)
    v = value.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)

    q_lens_tensor = torch.tensor(q_lens, device=device, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    cu_seqlens_q = torch.zeros(len(q_lens_tensor) + 1, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(
        len(kv_lens_tensor) + 1, device=device, dtype=torch.int32
    )
    cu_seqlens_q[1:] = torch.cumsum(q_lens_tensor, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens_tensor, dim=0)

    max_seqlen_q = int(q_lens_tensor.max().item())
    max_seqlen_k = int(kv_lens_tensor.max().item())

    if attention_register.is_available("flash_varlen"):
        key = "flash_varlen"
    elif attention_register.is_available("metal_flash_varlen"):
        key = "metal_flash_varlen"
    else:
        key = "sdpa_varlen"

    out = attention_register.call(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_causal=causal,
        key=key,
    )
    out = out[0] if not torch.is_tensor(out) else out
    out = out.view(B, T_q, H, D_h).permute(0, 2, 1, 3).contiguous()
    return out


def vector_to_list(tensor, lens, dim):
    return list(torch.split(tensor, lens, dim=dim))


def list_to_vector(tensor_list, dim):
    lens = [tensor.shape[dim] for tensor in tensor_list]
    tensor = torch.cat(tensor_list, dim)
    return tensor, lens


def merge_token_lists(list1, list2, dim):
    assert len(list1) == len(list2)
    return [torch.cat((t1, t2), dim) for t1, t2 in zip(list1, list2)]


# ------------------------------- Vision utils ------------------------------ #
detector = None


def get_landmarks_from_image(image: Image.Image):
    global detector
    if detector is None:
        if FaceAnalysis is None:
            raise ImportError("insightface is required for landmark detection")
        detector = FaceAnalysis()
        detector.prepare(ctx_id=0, det_size=(640, 640))

    in_image = np.array(image).copy()
    faces = detector.get(in_image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return face.kps


def get_arcface_dst(extend_face_crop=False, extend_ratio=0.8):
    arcface_dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    if extend_face_crop:
        arcface_dst[:, 1] = arcface_dst[:, 1] + 10
        arcface_dst = (arcface_dst - 112 / 2) * extend_ratio + 112 / 2
    return arcface_dst


def estimate_norm(lmk, image_size=112, arcface_dst=None):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def align_face(
    image_pil: Image.Image,
    face_kpts,
    extend_face_crop=False,
    extend_ratio=0.8,
    face_size=112,
):
    arcface_dst = get_arcface_dst(extend_face_crop, extend_ratio)
    M = estimate_norm(face_kpts, face_size, arcface_dst)
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    face_image_cv2 = cv2.warpAffine(
        image_cv2, M, (face_size, face_size), borderValue=0.0
    )
    face_image = Image.fromarray(cv2.cvtColor(face_image_cv2, cv2.COLOR_BGR2RGB))
    return face_image


class FaceEncoderArcFace:
    """ArcFace encoder wrapper (no training)."""

    def __init__(self):
        self.encoder_model = None
        self.device = None

    def init_encoder_model(self, device, eval_mode=True):
        if init_recognition_model is None:
            raise ImportError("facexlib is required for ArcFace encoding")
        self.device = device
        self.encoder_model = init_recognition_model("arcface", device=device)
        if eval_mode:
            self.encoder_model.eval()

    @torch.no_grad()
    def input_preprocessing(self, in_image, landmarks, image_size=112):
        assert landmarks is not None, "landmarks are not provided!"
        in_image = np.array(in_image)
        landmark = np.array(landmarks)
        face_aligned = align_face(
            Image.fromarray(in_image),
            landmark,
            extend_face_crop=False,
            face_size=image_size,
        )
        return face_aligned

    @torch.no_grad()
    def __call__(self, in_image, need_proc=False, landmarks=None, image_size=112):
        if need_proc:
            preprocessed = self.input_preprocessing(in_image, landmarks, image_size)
            import torchvision.transforms as T

            image_transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
            in_image = image_transform(preprocessed).unsqueeze(0).to(self.device)
        else:
            assert isinstance(in_image, torch.Tensor)

        in_image = in_image[:, [2, 1, 0], :, :].contiguous()
        image_embeds = self.encoder_model(in_image)
        return image_embeds


# ------------------------------- Resampler ------------------------------- #
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.pos_emb = (
            nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None
        )
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(
                x,
                dim=1,
                mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


# ----------------------------- Adapter helpers ----------------------------- #
class WanIPAttnProcessor(nn.Module):
    def __init__(
        self, cross_attention_dim: int, dim: int, n_registers: int, bias: bool
    ):
        super().__init__()
        self.to_k_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        self.to_v_ip = nn.Linear(cross_attention_dim, dim, bias=bias)
        self.registers = (
            nn.Parameter(torch.randn(1, n_registers, cross_attention_dim) / dim**0.5)
            if n_registers > 0
            else None
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ip_lens: List[int] = None,
        ip_scale: float = 1.0,
        **kwargs,
    ):
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            ip_hidden_states=ip_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            q_lens=q_lens,
            kv_lens=kv_lens,
            ip_lens=ip_lens,
            ip_scale=ip_scale,
            **kwargs,
        )

    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ip_lens: List[int] = None,
        ip_scale: float = 1.0,
        **kwargs,
    ):

        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if ip_hidden_states is not None:
            if self.registers is not None:
                ip_hidden_states_list = vector_to_list(ip_hidden_states, ip_lens, 1)
                ip_hidden_states_list = merge_token_lists(
                    ip_hidden_states_list,
                    [self.registers] * len(ip_hidden_states_list),
                    1,
                )
                ip_hidden_states, ip_lens = list_to_vector(ip_hidden_states_list, 1)

            ip_query = query
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            if attn.norm_q is not None:
                ip_query = attn.norm_q(ip_query)
            if attn.norm_k is not None:
                ip_key = attn.norm_k(ip_key)
            ip_query = ip_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_key = ip_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_value = ip_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ip_hidden_states = flash_attention(
                ip_query, ip_key, ip_value, q_lens=q_lens, kv_lens=ip_lens
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).flatten(2, 3)
            ip_hidden_states = ip_hidden_states.type_as(ip_query)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = flash_attention(
            query, key, value, q_lens=q_lens, kv_lens=kv_lens
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if ip_hidden_states is not None:
            hidden_states = hidden_states + ip_scale * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanIPAttnProcessorLight(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        scale=1.0,
        bias=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=bias
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=bias
        )

        torch.nn.init.zeros_(self.to_k_ip.weight)
        torch.nn.init.zeros_(self.to_v_ip.weight)
        if bias:
            torch.nn.init.zeros_(self.to_k_ip.bias)
            torch.nn.init.zeros_(self.to_v_ip.bias)

        self.norm_rms_k = RMSNorm(hidden_size, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        image_embed: torch.Tensor = None,
        ip_scale: float = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # =============================================================
        batch_size = image_embed.size(0)

        ip_hidden_states = image_embed
        ip_query = query  # attn.to_q(hidden_states.clone())
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        if attn.norm_q is not None:
            ip_query = attn.norm_q(ip_query)
        ip_key = self.norm_rms_k(ip_key)

        ip_inner_dim = ip_key.shape[-1]
        ip_head_dim = ip_inner_dim // attn.heads

        ip_query = ip_query.view(batch_size, -1, attn.heads, ip_head_dim).transpose(
            1, 2
        )
        ip_key = ip_key.view(batch_size, -1, attn.heads, ip_head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, ip_head_dim).transpose(
            1, 2
        )

        ip_hidden_states = attention_register.call(
            ip_query,
            ip_key,
            ip_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * ip_head_dim
        )
        ip_hidden_states = ip_hidden_states.to(ip_query.dtype)
        # ===========================================================================

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb_inner(
                hidden_states: torch.Tensor, freqs: torch.Tensor
            ):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb_inner(query, rotary_emb)
            key = apply_rotary_emb_inner(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Add IPA residual
        ip_scale = ip_scale or self.scale
        hidden_states = hidden_states + ip_scale * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class WanRefAttnProcessor(nn.Module):
    def __init__(self, dim: int, bias: bool):
        super().__init__()
        self.to_k_ref = nn.Linear(dim, dim, bias=bias)
        self.to_v_ref = nn.Linear(dim, dim, bias=bias)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ref_feature: Optional[tuple] = None,
        ref_scale: float = 1.0,
        **kwargs,
    ):
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            q_lens=q_lens,
            kv_lens=kv_lens,
            ref_feature=ref_feature,
            ref_scale=ref_scale,
            **kwargs,
        )

    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        ref_feature: Optional[tuple] = None,
        ref_scale: float = 1.0,
        **kwargs,
    ):

        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if ref_feature is None:
            ref_hidden_states = None
        else:
            ref_hidden_states, ref_lens = ref_feature
            ref_query = query
            ref_key = self.to_k_ref(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states)
            if attn.norm_q is not None:
                ref_query = attn.norm_q(ref_query)
            if attn.norm_k is not None:
                ref_key = attn.norm_k(ref_key)
            ref_query = ref_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_key = ref_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_value = ref_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            ref_hidden_states = flash_attention(
                ref_query, ref_key, ref_value, q_lens=q_lens, kv_lens=ref_lens
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).flatten(2, 3)
            ref_hidden_states = ref_hidden_states.type_as(ref_query)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = flash_attention(
            query, key, value, q_lens=q_lens, kv_lens=kv_lens
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if ref_hidden_states is not None:
            hidden_states = hidden_states + ref_scale * ref_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _logical_tensor_shape(t: torch.Tensor) -> torch.Size:
    """
    Return the *logical* shape for a tensor.

    For GGUF-loaded weights we may have a `GGMLTensor` where `.shape` reflects the
    quantized byte-storage layout, while the original (dequantized) shape is
    stored on `tensor_shape`.
    """
    logical = getattr(t, "tensor_shape", None)
    if logical is not None:
        try:
            return torch.Size(logical)
        except Exception:
            pass
    return t.shape


def register_ip_adapter_full(
    model,
    cross_attention_dim=None,
    n_registers=0,
    init_method="zero",
    dtype=torch.float32,
    layers=None,
):
    attn_procs = {}
    transformer_sd = model.state_dict()
    for layer_idx, block in enumerate(model.blocks):
        name = f"blocks.{layer_idx}.attn2.processor"
        layer_name = name.split(".processor")[0]

        # Prefer module metadata (robust to GGUF quantized storage shapes).
        dim = None
        inferred_cross_attention_dim = None
        try:
            to_k = block.attn2.to_k
            dim = int(getattr(to_k, "out_features"))
            inferred_cross_attention_dim = int(getattr(to_k, "in_features"))
        except Exception:
            w = transformer_sd[layer_name + ".to_k.weight"]
            w_shape = _logical_tensor_shape(w)  # (out_features, in_features)
            dim = int(w_shape[0])
            inferred_cross_attention_dim = int(w_shape[1])

        attn_procs[name] = WanIPAttnProcessor(
            cross_attention_dim=(
                inferred_cross_attention_dim
                if cross_attention_dim is None
                else cross_attention_dim
            ),
            dim=dim,
            n_registers=n_registers,
            bias=True,
        )
        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.bias)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.bias)
        elif init_method == "clone":
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_k_ip.bias": transformer_sd[layer_name + ".to_k.bias"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
                "to_v_ip.bias": transformer_sd[layer_name + ".to_v.bias"],
            }
            attn_procs[name].load_state_dict(weights)
        block.attn2.processor = attn_procs[name]
    ip_layers = torch.nn.ModuleList(attn_procs.values())
    ip_layers.to(device=model.device, dtype=dtype)
    return model, ip_layers


def register_ip_adapter_light(
    model,
    hidden_size=5120,
    cross_attention_dim=2048,
    dtype=torch.float32,
    init_method="zero",
    layers=None,
    **kwargs,
):
    attn_procs = {}
    transformer_sd = model.state_dict()

    if layers is None:
        layers = list(range(0, len(model.blocks)))
    elif isinstance(layers, int):  # Only interval provided
        layers = list(range(0, len(model.blocks), layers))

    for i, block in enumerate(model.blocks):
        if i not in layers:
            continue

        name = f"blocks.{i}.attn2.processor"
        attn_procs[name] = WanIPAttnProcessorLight(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ip.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ip.weight)
        elif init_method == "clone":
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": transformer_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:
            raise ValueError(f"{init_method} is not supported.")

        block.attn2.processor = attn_procs[name]

    ip_layers = torch.nn.ModuleList(attn_procs.values())
    ip_layers.to(model.device, dtype=dtype)

    return model, ip_layers


def register_ref_adapter(model, init_method="zero", dtype=torch.float32):
    attn_procs = {}
    transformer_sd = model.state_dict()
    for layer_idx, block in enumerate(model.blocks):
        name = f"blocks.{layer_idx}.attn1.processor"
        layer_name = name.split(".processor")[0]
        # Prefer module metadata (robust to GGUF quantized storage shapes).
        try:
            to_k = block.attn1.to_k
            dim = int(getattr(to_k, "out_features"))
        except Exception:
            w = transformer_sd[layer_name + ".to_k.weight"]
            dim = int(_logical_tensor_shape(w)[0])  # (out_features, in_features)
        attn_procs[name] = WanRefAttnProcessor(dim=dim, bias=True)
        if init_method == "zero":
            torch.nn.init.zeros_(attn_procs[name].to_k_ref.weight)
            torch.nn.init.zeros_(attn_procs[name].to_k_ref.bias)
            torch.nn.init.zeros_(attn_procs[name].to_v_ref.weight)
            torch.nn.init.zeros_(attn_procs[name].to_v_ref.bias)
        elif init_method == "clone":
            weights = {
                "to_k_ref.weight": transformer_sd[layer_name + ".to_k.weight"],
                "to_k_ref.bias": transformer_sd[layer_name + ".to_k.bias"],
                "to_v_ref.weight": transformer_sd[layer_name + ".to_v.weight"],
                "to_v_ref.bias": transformer_sd[layer_name + ".to_v.bias"],
            }
            attn_procs[name].load_state_dict(weights)
        block.attn1.processor = attn_procs[name]
    ref_layers = torch.nn.ModuleList(attn_procs.values())
    ref_layers.to(device=model.device, dtype=dtype)
    return model, ref_layers


# ------------------------------- Main helper ------------------------------- #
class WanLynxHelper(BaseHelper):
    """Utility helper for Lynx personalization (face encode, adapters, ref buffers)."""

    def __init__(self, adapter_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.adapter_path = adapter_path
        self.resampler: Resampler | None = None
        self.ip_loaded = False
        self.ref_loaded = False
        self.face_encoder: FaceEncoderArcFace | None = None

    def resolve_adapter_path(
        self, config: Dict[str, any] | None = None, override: Optional[str] = None
    ) -> Optional[str]:
        if override:
            return override
        env_path = os.getenv("LYNX_ADAPTER_PATH")
        if env_path:
            return env_path
        cfg = config or {}
        return (
            cfg.get("adapter_path")
            or cfg.get("lynx_adapter_path")
            or cfg.get("adapter_dir")
            or self.adapter_path
        )

    def get_face_encoder(self, device: str | torch.device):
        if self.face_encoder is None:
            encoder = FaceEncoderArcFace()
            encoder.init_encoder_model(device)
            self.face_encoder = encoder
        return self.face_encoder

    def prepare_face_data(
        self,
        image: Image.Image,
        face_embeds: Optional[np.ndarray | torch.Tensor],
        landmarks: Optional[np.ndarray | torch.Tensor],
        device: str | torch.device,
        load_image_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
        image_pil = (
            load_image_fn(image)
            if load_image_fn is not None
            else (
                image
                if isinstance(image, Image.Image)
                else Image.open(image).convert("RGB")
            )
        )
        if landmarks is None:
            landmarks = get_landmarks_from_image(image_pil)

        if face_embeds is None:
            encoder = self.get_face_encoder(device)
            embeds = encoder(image_pil, need_proc=True, landmarks=landmarks)
            embeds = embeds.squeeze(0).detach().cpu().numpy()
        else:
            embeds = (
                face_embeds.detach().cpu().numpy()
                if isinstance(face_embeds, torch.Tensor)
                else np.asarray(face_embeds)
            )
        return embeds, np.asarray(landmarks), image_pil

    def load_adapters(
        self, transformer, adapter_path: str, device: torch.device, dtype: torch.dtype
    ):
        if not adapter_path:
            raise ValueError(
                "Lynx adapter path is required (adapter_path or LYNX_ADAPTER_PATH)."
            )
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

        if self.resampler is None:
            resampler_path = os.path.join(adapter_path, "resampler.safetensors")
            if not os.path.exists(resampler_path):
                raise FileNotFoundError(
                    f"Missing resampler weights at {resampler_path}"
                )

            state_dicts = load_file(resampler_path, device="cpu")
            out_dim = state_dicts["norm_out.weight"].shape[0]
            with init_empty_weights():
                resampler = Resampler(
                    depth=4,
                    dim=1280,
                    dim_head=64,
                    embedding_dim=512,
                    ff_mult=4,
                    heads=20,
                    num_queries=16,
                    output_dim=out_dim,
                )

            resampler.load_state_dict(state_dicts, assign=True)
            resampler.to(device=device, dtype=dtype)
            resampler.eval()
            self.resampler = resampler

        if not self.ip_loaded:
            ip_path = os.path.join(adapter_path, "ip_layers.safetensors")
            if os.path.exists(ip_path):
                ip_sd = load_file(ip_path, device="cpu")
                cross_attention_dim = ip_sd["0.to_k_ip.weight"].shape[1]
                n_registers = (
                    ip_sd["0.registers"].shape[1] if "0.registers" in ip_sd else 0
                )
                register_fn = (
                    register_ip_adapter_full
                    if cross_attention_dim == 5120
                    else register_ip_adapter_light
                )
                transformer, ip_layers = register_fn(
                    transformer,
                    cross_attention_dim=cross_attention_dim,
                    n_registers=n_registers,
                    layers=2 if cross_attention_dim != 5120 else None,
                    dtype=dtype,
                )
                ip_layers.load_state_dict(ip_sd)
                self.ip_loaded = True
            else:
                raise FileNotFoundError(f"IP adapter weights not found at {ip_path}")

        if not self.ref_loaded:
            ref_path = os.path.join(adapter_path, "ref_layers.safetensors")
            if os.path.exists(ref_path):
                ref_sd = load_file(ref_path, device="cpu")
                transformer, ref_layers = register_ref_adapter(transformer, dtype=dtype)
                ref_layers.load_state_dict(ref_sd)
                self.ref_loaded = True
            else:
                # Ref is optional for lite models; keep a flag for availability.
                self.ref_loaded = False
        return transformer

    def build_ip_states(
        self, face_embeds: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.resampler is None:
            raise RuntimeError("Resampler not initialized before building IP states.")

        face_tensor = torch.tensor(face_embeds, device=device, dtype=dtype).view(
            1, 1, -1
        )
        ip_hidden_states = [self.resampler(face_tensor)]
        ip_hidden_states_uncond = [self.resampler(torch.zeros_like(face_tensor))]
        
        return ip_hidden_states, ip_hidden_states_uncond

    def cal_mean_and_std(self, vae, device, dtype):
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)
        return latents_mean, latents_std

    def encode_reference_buffer(
        self,
        engine,
        face_image: Image.Image,
        device: torch.device,
        dtype: torch.dtype,
        drop: bool = False,
        generator: Optional[torch.Generator] = None,
        offload: bool = True,
    ):
        if not self.ref_loaded:
            return None

        ref_array = np.array(face_image)
        batch_ref_image = torch.tensor(ref_array, device=device, dtype=dtype)
        batch_ref_image = batch_ref_image / 255.0 * 2 - 1
        batch_ref_image = batch_ref_image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        # offload transformer to cpu
        if offload:
            engine._offload("transformer", offload_type="cpu")

        if drop:
            batch_ref_image = batch_ref_image * 0
        if not engine.vae:
            engine.load_component_by_type("vae")
        engine.to_device(engine.vae)
        vae_feat = engine.vae_encode(
            batch_ref_image,
            offload=offload,
            sample_mode="sample",
            sample_generator=generator,
        )
        vae_feat_list = [vae_feat]

        if offload:
            engine._offload("vae")

        ref_prompt_list = ["image of a face"]
        if not engine.text_encoder:
            engine.load_component_by_type("text_encoder")

        engine.to_device(engine.text_encoder)
        ref_text_embeds = engine.text_encoder.encode(
            ref_prompt_list,
            device=device,
            num_videos_per_prompt=1,
            max_sequence_length=512,
        ).to(device=device, dtype=dtype)
        ref_text_embeds_list = [ref_text_embeds]
        if offload:
            engine._offload("text_encoder")

        if not engine.transformer:
            engine.load_component_by_type("transformer")
        engine.to_device(engine.transformer)

        ref_buffer = engine.transformer(
            hidden_states=vae_feat_list,
            timestep=torch.LongTensor([0]).to(device),
            encoder_hidden_states=ref_text_embeds_list,
            attention_kwargs={"ref_feature_extractor": True},
            return_dict=False,
        )
        return ref_buffer

    def align_face(self, face_image: Image.Image, face_landmarks, face_size: int = 256):
        return align_face(
            face_image, face_landmarks, extend_face_crop=True, face_size=face_size
        )

    def encode_face_embedding(
        self, face_embeds, do_classifier_free_guidance, device, dtype
    ):
        num_images_per_prompt = 1

        if isinstance(face_embeds, torch.Tensor):
            face_embeds = face_embeds.clone().detach()
        else:
            face_embeds = torch.from_numpy(face_embeds)

        face_embeds = face_embeds.reshape([1, -1, 512])

        if do_classifier_free_guidance:
            face_embeds = torch.cat([torch.zeros_like(face_embeds), face_embeds], dim=0)
        else:
            face_embeds = torch.cat([face_embeds], dim=0)

        face_embeds = face_embeds.to(device=device, dtype=dtype)
        face_embeds = self.resampler(face_embeds)

        bs_embed, seq_len, _ = face_embeds.shape
        face_embeds = face_embeds.repeat(1, num_images_per_prompt, 1)
        face_embeds = face_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return face_embeds.to(device=device, dtype=dtype)
