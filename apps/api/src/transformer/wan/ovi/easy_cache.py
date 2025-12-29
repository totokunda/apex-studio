import torch
from torch.amp import autocast


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def easycache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
):
    """
    Args:
        x (List[Tensor]): List of input video tensors with shape [C_in, F, H, W]
        t (Tensor): Diffusion timesteps tensor of shape [B]
        context (List[Tensor]): List of text embeddings each with shape [L, C]
        seq_len (int): Maximum sequence length for positional encoding
        clip_fea (Tensor, optional): CLIP image features for image-to-video mode
        y (List[Tensor], optional): Conditional video inputs for image-to-video mode
    Returns:
        List[Tensor]: List of denoised video tensors with original input shapes
    """
    if self.model_type == 'i2v':
        assert y is not None

    # Store original raw input for end-to-end caching
    raw_input = [u.clone() for u in x]

    # params
    device = next(self.patch_embedding.parameters()).device

    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Track which type of step (even=condition, odd=uncondition)
    self.is_even = (self.cnt % 2 == 0)
    
    # Only make decision on even (condition) steps
    if self.is_even:
        # Always compute first ret_steps and last steps
        if self.cnt < self.ret_steps or self.cnt >= (
                ((getattr(self, "low_start_step", None) is not None and getattr(self, "is_high_noise", False)) and (
                        self.low_start_step - 1) * 2 - 2) or
                ((getattr(self, "low_start_step", None) is not None and not getattr(self, "is_high_noise", False)) and (
                        self.num_steps - self.low_start_step) * 2 - 2) or
                (self.num_steps * 2 - 2)
        ):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0
        else:
            # Check if we have previous step data for comparison
            if hasattr(self, 'previous_raw_input_even') and hasattr(self, 'previous_raw_output_even') and \
                    self.previous_raw_input_even is not None and self.previous_raw_output_even is not None:
                # Calculate input changes
                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, self.previous_raw_input_even)
                ]).abs().mean()

                # Compute predicted change if we have k factors
                if hasattr(self, 'k') and self.k is not None:
                    # Calculate output norm for relative comparison
                    output_norm = torch.cat([
                        u.flatten() for u in self.previous_raw_output_even
                    ]).abs().mean()
                    pred_change = self.k * (raw_input_change / output_norm)
                    combined_pred_change = pred_change
                    # Accumulate predicted error
                    if not hasattr(self, 'accumulated_error_even'):
                        self.accumulated_error_even = 0
                    self.accumulated_error_even += combined_pred_change
                    # Decide if we need full calculation
                    if self.accumulated_error_even < self.thresh:
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0
                else:
                    # First time after ret_steps or missing k factors, need to calculate
                    self.should_calc_current_pair = True
            else:
                # No previous data yet, must calculate
                self.should_calc_current_pair = True

        # Store current input state
        self.previous_raw_input_even = [u.clone() for u in raw_input]
    

    # Check if we can use cached output and return early
    if self.is_even and not self.should_calc_current_pair and \
            hasattr(self, 'previous_raw_output_even') and self.previous_raw_output_even is not None:
        # Use cached output directly
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_even)]

    elif not self.is_even and not self.should_calc_current_pair and \
            hasattr(self, 'previous_raw_output_odd') and self.previous_raw_output_odd is not None:
        # Use cached output directly
        self.cnt += 1
        # return [u.float() for u in self.previous_raw_output_odd]
        return [(u + v).float() for u, v in zip(raw_input, self.cache_odd)]

    # Continue with normal processing since we need to calculate
    # embeddings
    

    x = [
        self.patch_embedding(u.unsqueeze(0)) for u in x
    ]  ## x is list of [B L D] or [B C F H W]
    if self.is_audio_type:
        # [B, 1]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x]
        )
    else:
        # [B, 3]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [
            u.flatten(2).transpose(1, 2) for u in x
        ]  # [B C F H W] -> [B (F H W) C] -> [B L C]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert (
        seq_lens.max() <= seq_len
    ), f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
    x = torch.cat(
        [
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ]
    )  # single [B, L, C]
    # time embeddings
    if t.dim() == 1:
        if first_frame_is_clean:
            t = torch.ones(
                (t.size(0), seq_len), device=t.device, dtype=t.dtype
            ) * t.unsqueeze(1)
            _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
            for i in range(t.size(0)):
                t[i, : _first_images_seq_len[i]] = 0
            # print(f"zeroing out first {_first_images_seq_len} from t: {t.shape}, {t}")
        else:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
    with autocast(device.type, dtype=torch.bfloat16):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t)
            .unflatten(0, (bt, seq_len))
            .bfloat16()
        )
        e0 = self.time_projection(e).unflatten(
            2, (6, self.dim)
        )  # [1, 26784, 6, 3072] - B, seq_len, 6, dim
        assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16
    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]
        )
    )
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)
    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    # Apply transformer blocks
    for block in self.blocks:
        x = block(x, **kwargs)

    # Apply head
    # Unpatchify
    output = self.post_transformer_block_out(x, kwargs["grid_sizes"], e)

    # Update cache and calculate change rates if needed
    if self.is_even:  # Condition path
        # If we have previous output, calculate k factors for future predictions
        if hasattr(self, 'previous_raw_output_even') and self.previous_raw_output_even is not None:
            # Calculate output change at the raw level
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.previous_raw_output_even)
            ]).abs().mean()

            # Check if we have previous input state for comparison
            if hasattr(self, 'prev_prev_raw_input_even') and self.prev_prev_raw_input_even is not None:
                # Calculate input change
                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        self.previous_raw_input_even, self.prev_prev_raw_input_even
                    )
                ]).abs().mean()

                self.k = output_change / input_change

                # Update history
        self.prev_prev_raw_input_even = getattr(self, 'previous_raw_input_even', None)
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [u - v for u, v in zip(output, raw_input)]

    else:  # Uncondition path
        # Store output for unconditional path
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [u - v for u, v in zip(output, raw_input)]

    # Update counter
    self.cnt += 1
    return [u.float() for u in output]