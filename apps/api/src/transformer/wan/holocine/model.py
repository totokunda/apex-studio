try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from src.transformer.wan.holocine.camera import SimpleAdapter
from einops import rearrange
from typing import List, Sequence, Optional
from src.attention import attention_register
from src.transformer.efficiency.mod import InplaceRMSNorm
from diffusers import ModelMixin, ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.configuration_utils import register_to_config


def build_global_reps_from_shots(
    K_local_shots: List[torch.Tensor],
    V_local_shots: List[torch.Tensor],
    g_per: int,
    mode: str = "firstk",  # "mean" | "firstk" | "linspace"
):
    """
    简单的代表池构造：从每个 shot 的本地 K/V 生成若干代表 token，并拼成共享池。
    K_local_shots[i]: [Ni, H, D]
    返回:
      K_global: [G_total, H, D], V_global: [G_total, H, D]
    """
    reps_k, reps_v = [], []
    S = len(K_local_shots)
    if S == 0:
        return (torch.empty(0), torch.empty(0))

    # g_per = max(1, G // S) if G > 0 else 0
    G = g_per * S

    for Ki, Vi in zip(K_local_shots, V_local_shots):
        Ni = Ki.size(0)
        if Ni == 0 or g_per == 0:
            continue
        if mode == "mean":

            idx = torch.linspace(0, Ni - 1, steps=g_per, device=Ki.device).long()

            reps_k.append(Ki.index_select(0, idx))
            reps_v.append(Vi.index_select(0, idx))
        elif mode == "firstk":
            take = min(g_per, Ni)
            reps_k.append(Ki[:take])
            reps_v.append(Vi[:take])
        elif mode == "linspace":
            idx = torch.linspace(0, Ni - 1, steps=g_per, device=Ki.device).long()
            reps_k.append(Ki.index_select(0, idx))
            reps_v.append(Vi.index_select(0, idx))
        else:
            raise ValueError(f"unknown mode {mode}")
    if len(reps_k) == 0:
        return (
            torch.empty(
                0,
                *K_local_shots[0].shape[1:],
                device=K_local_shots[0].device,
                dtype=K_local_shots[0].dtype,
            ),
            torch.empty(
                0,
                *V_local_shots[0].shape[1:],
                device=V_local_shots[0].device,
                dtype=V_local_shots[0].dtype,
            ),
        )
    K_global = torch.cat(reps_k, dim=0)
    V_global = torch.cat(reps_v, dim=0)

    # if G > 0:
    #     if K_global.size(0) < G:
    #         need = G - K_global.size(0)
    #         K_global = torch.cat([K_global, K_global[:need]], dim=0)
    #         V_global = torch.cat([V_global, V_global[:need]], dim=0)
    #     elif K_global.size(0) > G:
    #         K_global = K_global[:G]
    #         V_global = V_global[:G]
    return K_global, V_global


def attention_per_batch_with_shots(
    q: torch.Tensor,  # [b, s, n_heads*head_dim]
    k: torch.Tensor,  # [b, s, n_heads*head_dim]
    v: torch.Tensor,  # [b, s, n_heads*head_dim]
    shot_latent_indices: Sequence[Sequence[int]],
    num_heads: int,
    # use_shared_global: bool = True,
    per_g: int = 64,
    # G_per_shot: int = 0,
    dropout_p: float = 0.0,
    causal: bool = False,
):

    assert q.shape == k.shape == v.shape
    b, s_tot, hd = q.shape
    assert hd % num_heads == 0
    d = hd // num_heads
    dtype = q.dtype
    device = q.device

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads).contiguous()
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads).contiguous()
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads).contiguous()

    outputs = []

    if not attention_register.is_available("flash_varlen"):
        attn_func = "sdpa_varlen"
    else:
        attn_func = "flash_varlen"

    for bi in range(b):

        cuts = list(shot_latent_indices[bi])
        assert (
            cuts[0] == 0 and cuts[-1] == s_tot
        ), "shot_latent_indices must start with 0 and end with s_tot"

        Q_shots, K_shots, V_shots = [], [], []
        N_list = []
        for a, bnd in zip(cuts[:-1], cuts[1:]):
            Q_shots.append(q[bi, :, a:bnd, :])  # [n, Ni, d]
            K_shots.append(k[bi, :, a:bnd, :])
            V_shots.append(v[bi, :, a:bnd, :])
            N_list.append(bnd - a)

        Q_locals = [rearrange(Qi, "n s d -> s n d") for Qi in Q_shots]
        K_locals = [rearrange(Ki, "n s d -> s n d") for Ki in K_shots]
        V_locals = [rearrange(Vi, "n s d -> s n d") for Vi in V_shots]

        K_global, V_global = build_global_reps_from_shots(
            K_locals, V_locals, per_g, mode="firstk"
        )

        K_list = [
            torch.cat([K_locals[i], K_global], dim=0) for i in range(len(K_locals))
        ]
        V_list = [
            torch.cat([V_locals[i], V_global], dim=0) for i in range(len(V_locals))
        ]
        kv_lengths = [Ni + K_global.size(0) for Ni in N_list]

        Q_packed = torch.cat(Q_locals, dim=0)  # [sum_N, n, d]
        K_packed = torch.cat(K_list, dim=0)  # [sum_(N+G), n, d]
        V_packed = torch.cat(V_list, dim=0)  # [sum_(N+G), n, d]

        Sshots = len(N_list)
        q_seqlens = torch.tensor(
            [0] + [sum(N_list[: i + 1]) for i in range(Sshots)],
            device=device,
            dtype=torch.int32,
        )
        kv_seqlens = torch.tensor(
            [0] + [sum(kv_lengths[: i + 1]) for i in range(Sshots)],
            device=device,
            dtype=torch.int32,
        )
        max_q_seqlen = max(N_list) if len(N_list) > 0 else 0
        max_kv_seqlen = max(kv_lengths) if len(kv_lengths) > 0 else 0

        if attention_register.is_available("flash_varlen"):
            O_packed = flash_attn_varlen_func(
                Q_packed,
                K_packed,
                V_packed,
                q_seqlens,
                kv_seqlens,
                max_q_seqlen,
                max_kv_seqlen,
                softmax_scale=None,
                causal=causal,
            )  # [sum_N, n, d]
        else:
            O_packed = attention_register.call(
                Q_packed,
                K_packed,
                V_packed,
                q_seqlens,
                kv_seqlens,
                softmax_scale=None,
                is_causal=causal,
                key="sdpa_varlen",
            )  # [sum_N, n, d]

        O_list = []
        for i in range(Sshots):
            st = q_seqlens[i].item()
            ed = q_seqlens[i + 1].item()
            Oi = O_packed[st:ed]  # [Ni, n, d]
            O_list.append(Oi)
        O_local = torch.cat(O_list, dim=0)  # [s_tot, n, d]
        O_local = rearrange(O_local, "s n d -> n s d").contiguous()  # [n, s, d]
        outputs.append(O_local)

    x = torch.stack(outputs, dim=0)  # [b, n, s, d]
    x = rearrange(x, "b n s d -> b s (n d)")
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs_cis_4d(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
    shot_id_dim_ratio: float = 0.5,
    max_shots: int = 256,
):

    print(f"Precomputing 4D RoPE frequencies (Shot, Time, H, W)...")

    dim_h = dim // 3
    dim_w = dim // 3

    dim_t_total = dim - dim_h - dim_w

    dim_shot = int(dim_t_total * shot_id_dim_ratio)
    dim_shot = (dim_shot // 2) * 2

    dim_time = dim_t_total - dim_shot

    print(f"  - Total RoPE dim: {dim}")
    print(
        f"  - Allocating dims -> Shot ID: {dim_shot}, Time: {dim_time}, H: {dim_h}, W: {dim_w}"
    )
    assert dim_shot + dim_time + dim_h + dim_w == dim

    shot_freqs_cis = precompute_freqs_cis(dim_shot, max_shots, theta)
    time_freqs_cis = precompute_freqs_cis(dim_time, end, theta)
    h_freqs_cis = precompute_freqs_cis(dim_h, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim_w, end, theta)

    return shot_freqs_cis, time_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None, shot_latent_indices=None, per_g=0):
        if attn_mask is not None:
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            x = attention_register.call(
                q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask, key="sdpa"
            )
            x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        elif shot_latent_indices is not None:
            x = attention_per_batch_with_shots(
                q=q,
                k=k,
                v=v,
                shot_latent_indices=shot_latent_indices,
                num_heads=self.num_heads,
                per_g=per_g,
            )
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            x = attention_register.call(q=q, k=k, v=v, num_heads=self.num_heads)
            x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, shot_latent_indices=None, per_g=0):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v, shot_latent_indices=shot_latent_indices, per_g=per_g)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        if attn_mask is not None:
            x = self.attn(q, k, v, attn_mask=attn_mask)

        else:
            x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k_img = rearrange(k_img, "b s (n d) -> b n s d", n=self.num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b n s d", n=self.num_heads)
            y = attention_register.call(q=q, k=k_img, v=v_img)
            y = rearrange(y, "b n s d -> b s (n d)", n=self.num_heads)
            x = x + y

        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_mask=None,
        shot_latent_indices=None,
        per_g=0,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(
                input_x, freqs, shot_latent_indices=shot_latent_indices, per_g=per_g
            ),
        )
        x = x + self.cross_attn(self.norm3(x), context, attn_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        # if self.max_shots > 1 and use_shot_embedding:
        #     self.shot_embedding = nn.Embedding(self.max_shots, self.dim)
        # else:
        #     self.shot_embedding = None

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        # self.freqs_4d=precompute_freqs_cis_4d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

        self.shot_embedding = None

    def patchify(
        self, x: torch.Tensor, control_camera_latents_input: torch.Tensor = None
    ):
        x = self.patch_embedding(x)
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        shot_indices: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        if self.shot_embedding is not None and shot_indices is not None:
            assert shot_indices.shape == (
                x.shape[0],
                f,
            ), f"Shot indices shape mismatch. Expected (batch, {f}), got {shot_indices.shape}"
            shot_ids = shot_indices.repeat_interleave(h * w, dim=1)
            shot_embs = self.shot_embedding(shot_ids)
            x = x + shot_embs

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
