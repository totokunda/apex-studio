# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import os
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from einops import rearrange
from src.helpers.helpers import helpers
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
import torch.nn as nn
from src.mixins.cache_mixin import CacheMixin
import warnings
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from src.quantize.ggml_layer import patch_model
from src.quantize.load import load_gguf


def safediv(n, d):
    q, r = divmod(n, d)
    assert r == 0
    return q


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class LLaMaEmbedding(nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.params_dtype = cfg.params_dtype
        self.fp32_residual_connection = cfg.fp32_residual_connection
        self.embedding_weights_in_fp32 = cfg.embedding_weights_in_fp32
        self.word_embeddings = torch.nn.Embedding(
            cfg.padded_vocab_size,
            self.hidden_size,
        )
        self.embedding_dropout = torch.nn.Dropout(cfg.hidden_dropout)

    def forward(self, input_ids):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            embeddings = embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class StepChatTokenizer:
    """Step Chat Tokenizer"""

    def __init__(
        self,
        model_file,
        name="StepChatTokenizer",
        bot_token="<|BOT|>",  # Begin of Turn
        eot_token="<|EOT|>",  # End of Turn
        call_start_token="<|CALL_START|>",  # Call Start
        call_end_token="<|CALL_END|>",  # Call End
        think_start_token="<|THINK_START|>",  # Think Start
        think_end_token="<|THINK_END|>",  # Think End
        mask_start_token="<|MASK_1e69f|>",  # Mask start
        mask_end_token="<|UNMASK_1e69f|>",  # Mask end
    ):
        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        for token in [
            bot_token,
            eot_token,
            call_start_token,
            call_end_token,
            think_start_token,
            think_end_token,
        ]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"
            assert (
                token in self._special_tokens
            ), f"Token '{token}' is not a special token"

        for token in [mask_start_token, mask_end_token]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"

        self._bot_id = self._tokenizer.piece_to_id(bot_token)
        self._eot_id = self._tokenizer.piece_to_id(eot_token)
        self._call_start_id = self._tokenizer.piece_to_id(call_start_token)
        self._call_end_id = self._tokenizer.piece_to_id(call_end_token)
        self._think_start_id = self._tokenizer.piece_to_id(think_start_token)
        self._think_end_id = self._tokenizer.piece_to_id(think_end_token)
        self._mask_start_id = self._tokenizer.piece_to_id(mask_start_token)
        self._mask_end_id = self._tokenizer.piece_to_id(mask_end_token)

        self._underline_id = self._tokenizer.piece_to_id("\u2581")

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode_as_ids(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode_ids(token_ids)


class Tokens:
    def __init__(
        self, input_ids, cu_input_ids, attention_mask, cu_seqlens, max_seq_len
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.cu_input_ids = cu_input_ids
        self.cu_seqlens = cu_seqlens
        self.max_seq_len = max_seq_len

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.cu_input_ids = self.cu_input_ids.to(device)
        self.cu_seqlens = self.cu_seqlens.to(device)
        return self


class Wrapped_StepChatTokenizer(StepChatTokenizer):
    def __call__(
        self,
        text,
        max_length=320,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ):
        # [bos, ..., eos, pad, pad, ..., pad]
        self.BOS = 1
        self.EOS = 2
        self.PAD = 2
        out_tokens = []
        attn_mask = []
        if len(text) == 0:
            part_tokens = [self.BOS] + [self.EOS]
            valid_size = len(part_tokens)
            if len(part_tokens) < max_length:
                part_tokens += [self.PAD] * (max_length - valid_size)
            out_tokens.append(part_tokens)
            attn_mask.append([1] * valid_size + [0] * (max_length - valid_size))
        else:
            for part in text:
                part_tokens = self.tokenize(part)
                part_tokens = part_tokens[
                    : (max_length - 2)
                ]  # leave 2 space for bos and eos
                part_tokens = [self.BOS] + part_tokens + [self.EOS]
                valid_size = len(part_tokens)
                if len(part_tokens) < max_length:
                    part_tokens += [self.PAD] * (max_length - valid_size)
                out_tokens.append(part_tokens)
                attn_mask.append([1] * valid_size + [0] * (max_length - valid_size))

        out_tokens = torch.tensor(out_tokens, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)

        # padding y based on tp size
        padded_len = 0
        padded_flag = True if padded_len > 0 else False
        if padded_flag:
            pad_tokens = torch.tensor(
                [[self.PAD] * max_length], device=out_tokens.device
            )
            pad_attn_mask = torch.tensor(
                [[1] * padded_len + [0] * (max_length - padded_len)],
                device=attn_mask.device,
            )
            out_tokens = torch.cat([out_tokens, pad_tokens], dim=0)
            attn_mask = torch.cat([attn_mask, pad_attn_mask], dim=0)

        # cu_seqlens
        cu_out_tokens = out_tokens.masked_select(attn_mask != 0).unsqueeze(0)
        seqlen = attn_mask.sum(dim=1).tolist()
        cu_seqlens = torch.cumsum(torch.tensor([0] + seqlen), 0).to(
            device=out_tokens.device, dtype=torch.int32
        )
        max_seq_len = max(seqlen)
        return Tokens(out_tokens, cu_out_tokens, attn_mask, cu_seqlens, max_seq_len)


def attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    return_attn_probs=False,
    tp_group_rank=0,
    tp_group_size=1,
):
    softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale
    if hasattr(torch.ops.Optimus, "fwd"):
        results = torch.ops.Optimus.fwd(
            q,
            k,
            v,
            None,
            dropout_p,
            softmax_scale,
            causal,
            return_attn_probs,
            None,
            tp_group_rank,
            tp_group_size,
        )[0]
    else:
        warnings.warn(
            "Cannot load `torch.ops.Optimus.fwd`. Using `torch.nn.functional.scaled_dot_product_attention` instead."
        )
        results = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            scale=softmax_scale,
        ).transpose(1, 2)
    return results


class FlashSelfAttention(torch.nn.Module):
    def __init__(
        self,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, cu_seqlens=None, max_seq_len=None):
        if cu_seqlens is None:
            output = attn_func(q, k, v, dropout_p=self.dropout_p)
        else:
            raise ValueError("cu_seqlens is not supported!")

        return output


class MultiQueryAttention(nn.Module):
    def __init__(self, cfg, layer_id=None):
        super().__init__()

        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn
        assert self.use_flash_attention, "FlashAttention is required!"

        self.n_groups = cfg.num_attention_groups
        self.tp_size = 1
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_groups = self.n_groups

        # GGUF unfused path flag
        self.is_gguf = getattr(cfg, "is_gguf", False)

        if self.is_gguf:
            # Unfused projections to match loader mapping
            self.wq = nn.Linear(
                cfg.hidden_size,
                cfg.hidden_size,
                bias=False,
            )
            self.wk = nn.Linear(
                cfg.hidden_size,
                self.head_dim * self.n_local_groups,
                bias=False,
            )
            self.wv = nn.Linear(
                cfg.hidden_size,
                self.head_dim * self.n_local_groups,
                bias=False,
            )
            self.wo = nn.Linear(
                cfg.hidden_size,
                cfg.hidden_size,
                bias=False,
            )
        else:
            # Fused wqkv path
            self.wqkv = nn.Linear(
                cfg.hidden_size,
                cfg.hidden_size + self.head_dim * 2 * self.n_groups,
                bias=False,
            )
            self.wo = nn.Linear(
                cfg.hidden_size,
                cfg.hidden_size,
                bias=False,
            )

        assert self.use_flash_attention, "non-Flash attention not supported yet."
        self.core_attention = FlashSelfAttention(
            attention_dropout=cfg.attention_dropout
        )

        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seq_len: Optional[torch.Tensor],
    ):
        seqlen, bsz, dim = x.shape

        if self.is_gguf:
            xq = self.wq(x)  # [s, b, h*d]
            xk = self.wk(x)  # [s, b, g*d]
            xv = self.wv(x)  # [s, b, g*d]

            xq = xq.view(seqlen, bsz, self.n_local_heads, self.head_dim)
            xk = xk.view(seqlen, bsz, self.n_local_groups, self.head_dim)
            xv = xv.view(seqlen, bsz, self.n_local_groups, self.head_dim)
        else:
            xqkv = self.wqkv(x)

            xq, xkv = torch.split(
                xqkv,
                (
                    dim // self.tp_size,
                    self.head_dim * 2 * self.n_groups // self.tp_size,
                ),
                dim=-1,
            )

            # gather on 1st dimension
            xq = xq.view(seqlen, bsz, self.n_local_heads, self.head_dim)
            xkv = xkv.view(seqlen, bsz, self.n_local_groups, 2 * self.head_dim)
            xk, xv = xkv.chunk(2, -1)

        # rotary embedding + flash attn
        xq = rearrange(xq, "s b h d -> b s h d")
        xk = rearrange(xk, "s b h d -> b s h d")
        xv = rearrange(xv, "s b h d -> b s h d")

        q_per_kv = self.n_local_heads // self.n_local_groups
        if q_per_kv > 1:
            b, s, h, d = xk.size()
            if h == 1:
                xk = xk.expand(b, s, q_per_kv, d)
                xv = xv.expand(b, s, q_per_kv, d)
            else:
                """To cover the cases where h > 1, we have
                the following implementation, which is equivalent to:
                    xk = xk.repeat_interleave(q_per_kv, dim=-2)
                    xv = xv.repeat_interleave(q_per_kv, dim=-2)
                but can avoid calling aten::item() that involves cpu.
                """
                idx = (
                    torch.arange(q_per_kv * h, device=xk.device)
                    .reshape(q_per_kv, -1)
                    .permute(1, 0)
                    .flatten()
                )
                xk = torch.index_select(
                    xk.repeat(1, 1, q_per_kv, 1), 2, idx
                ).contiguous()
                xv = torch.index_select(
                    xv.repeat(1, 1, q_per_kv, 1), 2, idx
                ).contiguous()

        if self.use_flash_attention:
            output = self.core_attention(
                xq, xk, xv, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
            )
            # reduce-scatter only support first dimension now
            output = rearrange(output, "b s h d -> s b (h d)").contiguous()
        else:
            xq, xk, xv = [
                rearrange(x, "b s ... -> s b ...").contiguous() for x in (xq, xk, xv)
            ]
            output = self.core_attention(xq, xk, xv, mask)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
        layer_id: int,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.swiglu = swiglu

        # GGUF unfused path flag
        self.is_gguf = getattr(cfg, "is_gguf", False)

        if self.is_gguf:
            # Unfused SWIGLU: gate and up are separate
            self.ffn_gate = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )
            self.ffn_up = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )
            self.ffn_down = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
            )
        else:
            self.w1 = nn.Linear(
                dim,
                2 * hidden_dim,
                bias=False,
            )
            self.w2 = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
            )

    def forward(self, x):
        if self.is_gguf:
            gate = self.ffn_gate(x)
            up = self.ffn_up(x)
            x = F.silu(gate) * up
            output = self.ffn_down(x)
            return output
        else:
            x = self.swiglu(self.w1(x))
            output = self.w2(x)
            return output


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention = MultiQueryAttention(
            cfg,
            layer_id=layer_id,
        )

        self.feed_forward = FeedForward(
            cfg,
            dim=cfg.hidden_size,
            hidden_dim=cfg.ffn_hidden_size,
            layer_id=layer_id,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )
        self.ffn_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seq_len: Optional[torch.Tensor],
    ):

        residual = self.attention.forward(
            self.attention_norm(x), mask, cu_seqlens, max_seq_len
        )
        h = x + residual
        ffn_res = self.feed_forward.forward(self.ffn_norm(h))
        out = h + ffn_res
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        config,
        max_seq_size=8192,
    ):
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = self._build_layers(config)

    def _build_layers(self, config):
        layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            layers.append(
                TransformerBlock(
                    config,
                    layer_id=layer_id + 1,
                )
            )
        return layers

    def forward(
        self,
        hidden_states,
        attention_mask,
        cu_seqlens=None,
        max_seq_len=None,
    ):

        if max_seq_len is not None and not isinstance(max_seq_len, torch.Tensor):
            max_seq_len = torch.tensor(max_seq_len, dtype=torch.int32, device="cpu")

        for lid, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                cu_seqlens,
                max_seq_len,
            )
        return hidden_states


class Step1Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        if getattr(config, "is_gguf", False):
            self.tok_embeddings = LLaMaEmbedding(config)
            self.transformer = Transformer(config)
        else:
            self.tok_embeddings = LLaMaEmbedding(config)
            self.transformer = Transformer(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):

        hidden_states = self.tok_embeddings(input_ids)

        hidden_states = self.transformer(
            hidden_states,
            attention_mask,
        )
        return hidden_states


@helpers("stepvideo.text_encoder")
class StepVideoTextEncoder(nn.Module, LoaderMixin, OffloadMixin, CacheMixin):
    def __init__(
        self,
        model_path,
        tokenizer_path=None,
        save_path=DEFAULT_COMPONENTS_PATH,
        dtype=torch.bfloat16,
        max_length=320,
        gguf_path=None,
        **kwargs,
    ):
        super(StepVideoTextEncoder, self).__init__(model_path, save_path)

        self.max_length = max_length
        self.save_path = save_path
        self.dtype = dtype
        tokenizer_path = self._download(tokenizer_path, save_path)
        self.text_tokenizer = Wrapped_StepChatTokenizer(tokenizer_path)
        if gguf_path is not None:
            model_weights, _ = load_gguf(gguf_path, type="text_encoder", key_map="step")
            # Ensure model is constructed with GGUF unfused path
            cfg = PretrainedConfig.from_pretrained(model_path)
            cfg.is_gguf = True
            with init_empty_weights():
                text_encoder = Step1Model(cfg)
            patch_model(text_encoder)
            text_encoder.load_state_dict(model_weights, assign=True)
        else:
            text_encoder = Step1Model.from_pretrained(model_path)
        self.text_encoder = text_encoder.eval().to(dtype)
        self.dtype = dtype

    @torch.no_grad
    def __call__(self, prompts, with_mask=True, max_length=None, **kwargs):
        self.device = next(self.text_encoder.parameters()).device
        if type(prompts) is str:
            prompts = [prompts]

        prompt_hash = self.hash(prompts)

        if self.enable_cache:
            cached = self.load_cached(prompt_hash)
            if cached is not None:
                cached_embeds, cached_mask = cached
                cached_embeds = cached_embeds.to(device=self.device)
                cached_mask = cached_mask.to(device=self.device)
                return cached_embeds, cached_mask

        txt_tokens = self.text_tokenizer(
            prompts,
            max_length=max_length or self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        y = self.text_encoder(
            txt_tokens.input_ids.to(self.device),
            attention_mask=(
                txt_tokens.attention_mask.to(self.device) if with_mask else None
            ),
        )

        y_mask = txt_tokens.attention_mask
        y_out = y.transpose(0, 1)
        if self.enable_cache:
            self.cache(prompt_hash, y_out, y_mask)
        return y_out, y_mask
