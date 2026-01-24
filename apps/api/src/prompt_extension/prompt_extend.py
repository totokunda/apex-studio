from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer
from src.utils.defaults import (
    DEFAULT_PREPROCESSOR_SAVE_PATH,
    DEFAULT_CONFIG_SAVE_PATH,
    get_torch_device,
)
from src.helpers.base import (
    PreprocessorType,
    BaseOutput,
    BaseHelper,
)


class PromptExtendOutput(BaseOutput):
    """Container for prompt extension outputs."""

    prompt: str
    full_text: Optional[str]


DEFAULT_SYSTEM_PROMPT = """You are an expert prompt engineer for diffusion VIDEO generation.
Expand the user's prompt into a single, vivid, production-ready prompt that preserves their intent while adding concise, concrete cinematic detail.

Include, in natural prose: subject and action; setting/environment (time/era/weather); mood; composition; camera (shot size, lens mm, aperture, movement/path); lighting (quality, direction); color palette; texture/material cues; background details; motion/physics (parallax, cloth, water, particles); and any style/genre the user names. Keep the main subject consistent across frames with continuity hints.

If the user mentions multiple beats, merge them into one continuous shot unless they explicitly ask for a shot list. Resolve ambiguity with sensible defaults. Prefer specific nouns/verbs over vague adjectives. Avoid contradictions and cluttered style mixes.

ONLY output the extended prompt—no preface, no headings, no quotes, no JSON. Keep it concise by default (≈60–120 words) unless the user asks for longer.

Do not include commentary or instructions—return ONLY the final extended prompt."""


class PromptExtendHelper(BaseHelper):
    """
    Generic prompt extension preprocessor backed by Hugging Face Transformers models.

    - Works with most causal/seq2seq text generation models (no pipelines).
    - Supports chat-style models via tokenizer.apply_chat_template when available.
    - Unified interface: configure with model base, tokenizer, and generation args.
    """

    def __init__(
        self,
        model_path: str,
        *,
        model_config_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        base: str = "AutoModelForCausalLM",
        tokenizer_name: Optional[str] = None,
        tokenizer_class: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_save_path: Optional[str] = DEFAULT_CONFIG_SAVE_PATH,
        save_path: Optional[str] = DEFAULT_PREPROCESSOR_SAVE_PATH,
        dtype: torch.dtype | None = None,
        device: Optional[str] | torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            config_path=model_config_path,
            save_path=save_path,
            **kwargs,
        )

        # Store config paths
        self.config_save_path = config_save_path

        # Resolve precision and device
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.load_dtype = dtype
        if device is None:
            self.device = get_torch_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load tokenizer (prefer explicit tokenizer_name)
        tokenizer_kwargs = tokenizer_kwargs or {}
        tokenizer_kwargs.setdefault("trust_remote_code", True)
        tokenizer_kwargs.setdefault("use_fast", True)

        # Try explicit name → HF repo id → local snapshot path
        tokenizer_loaded = False
        self.original_model_path = model_path
        try:
            if tokenizer_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, **tokenizer_kwargs
                )
                tokenizer_loaded = True
        except Exception:
            pass
        if not tokenizer_loaded:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, **tokenizer_kwargs
                )
                tokenizer_loaded = True
            except Exception:
                pass
        if not tokenizer_loaded:
            # Fallback to the locally downloaded snapshot path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, **tokenizer_kwargs
            )

        # Ensure pad/eos tokens are set for generation
        if self.tokenizer.pad_token_id is None:
            # Prefer eos token as pad when missing
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # As a last resort, create a pad token
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Load the model weights using the shared loader (no transformers pipelines)
        self.model = self._load_model(
            {
                "base": base,
                "model_path": self.model_path,
                "config_path": model_config_path,
                "config": model_config or {},
            },
            module_name="transformers",
            load_dtype=self.load_dtype,
        )

        # If we added new special tokens (e.g., pad), resize embeddings
        if hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Move to device
        self.model.to(self.device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        length_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        **generate_kwargs: Any,
    ) -> PromptExtendOutput:
        """Generate an extended version of the input prompt."""

        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build model input – prefer chat templates when supported

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Here is the prompt to extend: {prompt}\n\n",
                },
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Generic fallback format
            input_text = f"System: {sys_prompt}\n\n" f"User: {prompt}\n\n" f"Assistant:"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True,
        ).to(self.device)

        # Generate
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else 0,
            num_beams=num_beams,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **generate_kwargs,
        )

        # Decode and strip the prompt portion
        generated_ids = gen_out[0]
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[input_len:]
        extended = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Optionally apply stop strings
        if stop_strings:
            for s in stop_strings:
                if s in extended:
                    extended = extended.split(s, 1)[0]

        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return PromptExtendOutput(prompt=extended.strip(), full_text=full_text.strip())

    def __str__(self) -> str:
        base = type(self.model).__name__
        tok = type(self.tokenizer).__name__
        return f"PromptExtendHelper(model={base}, tokenizer={tok}, device={self.device}, dtype={self.load_dtype})"

    def __repr__(self) -> str:
        return self.__str__()
