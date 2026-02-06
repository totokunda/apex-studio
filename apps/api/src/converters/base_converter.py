from src.converters.utils import update_state_dict_
import re
from typing import Dict, Any, Iterable, List


class BaseConverter:
    def __init__(self):
        self.rename_dict = {}
        self.special_keys_map = {}
        self.pre_special_keys_map = {}

    @staticmethod
    def _iter_lora_stripped_variants(key: str) -> Iterable[str]:
        """
        Yield match variants for LoRA-shaped keys.

        Many LoRA state dicts use intermediate module names like `lora_A` / `lora_B`
        (PEFT) or `lora_up` / `lora_down` (base/Kohya). When comparing against
        `model_keys` from a *base* model, these segments are not present, so we
        produce variants with the LoRA segment removed:

          - `...to_q.lora_A.weight`   -> `...to_q.weight`
          - `...to_q.lora_down.weight` -> `...to_q.weight`
        """
        if not key:
            return
        yield key

        # Case-insensitive handling for common LoRA segment spellings.
        # We only remove the segment when it appears as its own dotted token.
        patterns = (
            ".lora_A.",
            ".lora_B.",
            ".lora_up.",
            ".lora_down.",
            ".Lora_A.",
            ".Lora_B.",
            ".Lora_up.",
            ".Lora_down.",
        )
        for p in patterns:
            if p in key:
                yield key.replace(p, ".", 1)

    @staticmethod
    def _matches_any_model_prefix(candidate: str, model_key_set: set) -> bool:
        """
        Return True if `candidate` matches `model_keys` either exactly OR as a
        parameter key that lives under a module key in `model_keys`.

        This supports callers that pass module names (e.g. from `named_modules()`)
        instead of full parameter keys (from `state_dict().keys()`).
        """
        if candidate in model_key_set:
            return True
        if "." not in candidate:
            return False
        parts = candidate.split(".")
        # Check progressively longer dotted prefixes: "a", "a.b", "a.b.c", ...
        prefix = parts[0]
        if prefix in model_key_set:
            return True
        for i in range(1, len(parts)):
            prefix = f"{prefix}.{parts[i]}"
            if prefix in model_key_set:
                return True
        return False

    @classmethod
    def _model_key_overlap_score(
        cls, keys: Iterable[str], model_keys: List[str]
    ) -> int:
        """
        Count how many keys appear to correspond to `model_keys`, allowing:
        - exact matches
        - module-prefix matches
        - LoRA-segment-stripped matches (lora_A/B, lora_up/down)
        """
        if not keys or not model_keys:
            return 0
        model_set = set(model_keys)
        score = 0
        for k in keys:
            matched = False
            for v in cls._iter_lora_stripped_variants(k):
                if cls._matches_any_model_prefix(v, model_set):
                    matched = True
                    break
            if matched:
                score += 1
        return score

    @staticmethod
    def _common_prefix_token_candidates(
        keys: List[str], max_tokens: int = 8
    ) -> List[str]:
        """
        Generate dotted-prefix candidates from the unanimous common token prefix of `keys`.

        Example:
          keys start with `base_model.model.blocks...` -> candidates include
          `base_model.` and `base_model.model.`
        """
        if not keys:
            return []
        split = [k.split(".") for k in keys if k]
        if not split:
            return []
        # Compute the unanimous common token prefix length.
        first = split[0]
        common_len = 0
        for i in range(min(len(first), max_tokens)):
            tok = first[i]
            if not tok:
                break
            if all(len(s) > i and s[i] == tok for s in split[1:]):
                common_len += 1
            else:
                break
        return [".".join(first[:i]) + "." for i in range(1, common_len + 1)]

    @staticmethod
    def _strip_prefixes_for_overlap(
        keys: List[str], reference_set: set, candidate_prefixes: Iterable[str]
    ) -> List[str]:
        """
        Return a *new* list of keys where we strip unanimous wrapper prefixes only when
        doing so increases overlap with `reference_set`.

        This is used to align `model_keys` and checkpoint keys that may differ only by
        common wrappers like "model." / "module.".
        """
        if not keys:
            return keys
        out = list(keys)

        changed = True
        while changed:
            changed = False
            for p in candidate_prefixes:
                if not p:
                    continue
                if not all(k.startswith(p) for k in out):
                    continue
                stripped = [k[len(p) :] for k in out]
                # Guard against empty keys and collisions.
                if any(not k for k in stripped):
                    continue
                if len(set(stripped)) != len(stripped):
                    continue

                before = sum(1 for k in out if k in reference_set)
                after = sum(1 for k in stripped if k in reference_set)
                if after > before:
                    out = stripped
                    changed = True
                    break
        return out

    @classmethod
    def _model_keys_indicate_already_converted(
        cls, state_keys: List[str], model_keys: List[str]
    ) -> bool:
        """
        If `model_keys` is provided, use it as strong evidence that `state_keys` are
        already in the target format.

        This is conservative: we only return True when *most* checkpoint keys are
        present in `model_keys` (allowing a small number of extras).
        """
        if not state_keys or not model_keys:
            return False

        state_set = set(state_keys)
        model_set = set(model_keys)

        # Try to align wrapper prefixes in either direction.
        candidate_prefixes = (
            "model.diffusion_model.",
            "diffusion_model.model.",
            "model.",
            "diffusion_model.",
            "module.",
            "unet.",
        )
        # Normalize model keys against state, then state against model (symmetrically).
        model_keys_norm = cls._strip_prefixes_for_overlap(
            list(model_set), state_set, candidate_prefixes
        )
        model_set_norm = set(model_keys_norm)
        state_keys_norm = cls._strip_prefixes_for_overlap(
            list(state_set), model_set_norm, candidate_prefixes
        )
        state_set_norm = set(state_keys_norm)

        matched = len(state_set_norm & model_set_norm)
        total = len(state_set_norm)
        if total == 0:
            return False

        # Require a meaningful number of matched keys to avoid tiny-dict false positives.
        if matched < min(10, total):
            return False

        # Allow a small number of extra keys (e.g. EMA/aux tensors), but require that
        # almost all checkpoint keys are recognized by the model.
        unmatched = total - matched
        return (matched / total) >= 0.98 and unmatched <= max(2, int(0.02 * total))

    @staticmethod
    def _looks_like_regex(pattern: str) -> bool:
        """
        Heuristic: treat `pattern` as a regex only when it contains explicit regex
        constructs (anchors, groups, character classes, escapes, quantifiers).

        We intentionally do NOT treat '.' as a regex indicator because most rename
        keys are literal dotted paths and should remain substring replacements.
        """
        if not pattern:
            return False
        # Fast-path: most keys are plain substrings.
        # Any of these characters strongly suggests an intentional regex.
        #
        # NOTE: we intentionally do NOT treat '*' as a regex indicator because we use
        # it as a glob-style wildcard placeholder in rename_dict (see `_apply_rename_dict`).
        return any(
            ch in pattern
            for ch in ("^", "$", "(", ")", "[", "]", "{", "}", "|", "?", "+", "\\")
        )

    @staticmethod
    def _looks_like_glob_star(pattern: str) -> bool:
        """
        Return True if `pattern` should be interpreted as a glob-style mapping pattern
        using '*' as a placeholder.

        This is used for patterns like:
          - "single_blocks.*.linear1"
        where '*' is meant to capture some literal substring (often an integer index)
        and the same captured substring should be substituted into the target mapping.
        """
        if not pattern or "*" not in pattern:
            return False
        # If the user is using explicit regex constructs, keep regex semantics.
        return not BaseConverter._looks_like_regex(pattern)

    @staticmethod
    def _apply_glob_star_rule(key: str, src: str, tgt: str) -> str:
        """
        Apply a single glob-style '*' rule:
          - src contains one or more '*' wildcards
          - tgt contains the same number of '*' wildcards
        and each '*' capture from src is substituted into tgt in order.
        """
        src_stars = src.count("*")
        tgt_stars = tgt.count("*")
        if src_stars != tgt_stars:
            raise ValueError(
                f"Glob-star rename rule must have the same number of '*' in src and tgt; "
                f"got src={src!r} ({src_stars}) tgt={tgt!r} ({tgt_stars})"
            )

        # Build a safe regex where all literal parts are escaped and each '*' becomes
        # a non-greedy capture group.
        parts = src.split("*")
        pattern = "".join(
            re.escape(p) + ("(.*?)" if i < len(parts) - 1 else "")
            for i, p in enumerate(parts)
        )
        compiled = re.compile(pattern)

        def _repl(m: re.Match) -> str:
            repl = tgt
            for g in m.groups():
                repl = repl.replace("*", g, 1)
            return repl

        return compiled.sub(_repl, key)

    def _apply_rename_dict(self, key: str) -> str:
        """
        Apply `rename_dict` to a key using substring replacements by default, with
        opt-in support for regex patterns (see `_looks_like_regex`).
        """
        new_key = key
        for src, tgt in self.rename_dict.items():
            if self._looks_like_glob_star(src):
                new_key = self._apply_glob_star_rule(new_key, src, tgt)
            elif self._looks_like_regex(src):
                try:
                    new_key = re.sub(src, tgt, new_key)
                except re.error as e:
                    raise ValueError(f"Invalid regex rename pattern: {src!r}") from e
            else:
                # Default: literal substring replacement.
                #
                # IMPORTANT: When `src` ends with a numeric index token (e.g. "down_blocks.1"),
                # naive `.replace()` can accidentally match prefixes of larger indices:
                #   - "down_blocks.10" contains "down_blocks.1"
                # This silently corrupts keys and commonly leads to partially-loaded models
                # where many parameters remain on `meta`.
                #
                # To avoid this, for any `src` that ends with a digit we only replace it
                # when the next character is a dotted-path boundary or end-of-string.
                if src and src[-1].isdigit():
                    # Treat `src` literally; `tgt` is also literal here.
                    new_key = re.sub(re.escape(src) + r"(?=\.|$)", tgt, new_key)
                else:
                    new_key = new_key.replace(src, tgt)
        return new_key

    @staticmethod
    def _is_specific_marker(s: str) -> bool:
        """
        Return True if `s` is a "specific" key fragment that can be used as a reliable
        signal when determining whether a checkpoint has already been converted.

        We intentionally ignore very generic fragments (e.g. "norm2") that may appear
        in *both* source and target key formats.
        """
        if not s:
            return False
        # Common ambiguous fragments that can exist in both source & target layouts.
        if s in {"norm", "norm1", "norm2", "norm3", "weight", "bias"}:
            return False
        # Prefer dotted/underscored fragments; otherwise require sufficient length.
        return ("." in s) or ("_" in s) or (len(s) >= 8)

    def _already_converted(
        self, state_dict: Dict[str, Any], model_keys: List[str] = None
    ) -> bool:
        """
        Best-effort heuristic to detect whether `state_dict` appears to already be in
        the *target* key format for this converter.

        This is intentionally conservative:
        - Requires *positive* evidence of target keys (target markers present)
        - Requires *absence* of source markers that strongly suggest an unconverted ckpt
        - Refuses to early-exit if we'd otherwise drop keys via pre/special handlers
        """

        if not state_dict:
            return True

        keys = list(state_dict.keys())

        # Strongest evidence: if the checkpoint keys already match the instantiated model's
        # keys (modulo common wrapper prefixes), then we should treat this as converted.
        # This helps especially for converters without strong target markers, and avoids
        # running special/pre-special handlers on already-converted checkpoints.
        if model_keys and self._model_keys_indicate_already_converted(keys, model_keys):
            return True

        # Regex-based renames are hard to reason about via marker matching (source markers
        # won't appear verbatim in real keys, and some targets are intentionally generic,
        # e.g. prefixing with "model."). For converters that include any regex rules,
        # prefer a direct, conservative check: if applying the rename rules would change
        # any key, then we are NOT already converted.
        #
        # This keeps conversion correct for mixed/partially-prefixed checkpoints.
        if any(
            self._looks_like_regex(k) or self._looks_like_glob_star(k)
            for k in self.rename_dict.keys()
        ):
            # If a model-key check was provided and didn't match, don't early-exit just
            # because the regex rules are a no-op; keep it conservative.
            if model_keys:
                return False
            return not any(self._apply_rename_dict(k) != k for k in keys)

        # Guard against partially-converted states introduced by placeholder hacks.
        if any("norm__placeholder" in k for k in keys):
            return False

        # If we'd drop or synthesize keys, we are not "fully matching" yet.
        if self.pre_special_keys_map:
            for pre_special_key in self.pre_special_keys_map.keys():
                if any(pre_special_key in k for k in keys):
                    return False
        if self.special_keys_map:
            for special_key in self.special_keys_map.keys():
                if any(special_key in k for k in keys):
                    return False

        # Build conservative marker sets from the rename map.
        # NOTE: source-side markers must be literal fragments; regex patterns won't appear
        # verbatim in real keys and would create false negatives/positives here.
        source_markers = [
            k
            for k in self.rename_dict.keys()
            if (not self._looks_like_regex(k)) and self._is_specific_marker(k)
        ]
        target_markers = [
            v for v in self.rename_dict.values() if self._is_specific_marker(v)
        ]
        # Without target markers we cannot safely assert the dict is already converted.
        if not target_markers:
            return False

        has_target = any(any(m in k for m in target_markers) for k in keys)
        if not has_target:
            return False

        has_source = any(any(m in k for m in source_markers) for k in keys)
        return not has_source

    @staticmethod
    def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
        state_dict.pop(key, None)

    def _sort_rename_dict(self):
        """
        Sort `rename_dict` to ensure proper replacement order.

        Default strategy:
        - Keep the WAN norm swap hack keys first (in their original insertion order),
          because they are intended to operate on *source* keys only. If we allow
          length-based sorting to move them later, they can accidentally rewrite
          *target* keys introduced by other renames (e.g. `img_emb.proj.4` -> `...norm2`).
        - For all remaining keys, sort by source-key length (longest to shortest)
          to avoid partial-substring collisions (e.g. `cross_attn.k_img` vs `cross_attn.k`).
        """
        priority_keys = ("norm2", "norm3", "norm__placeholder")
        priority_set = set(priority_keys)

        priority_items = []
        for k in priority_keys:
            if k in self.rename_dict:
                priority_items.append((k, self.rename_dict[k]))

        other_items = [
            (k, v) for k, v in self.rename_dict.items() if k not in priority_set
        ]
        other_items.sort(key=lambda item: len(item[0]), reverse=True)

        self.rename_dict = dict(priority_items + other_items)

    def _conversion_match_score(self, keys: Iterable[str]) -> int:
        """
        Roughly estimate how "applicable" this converter is to a given set of keys.

        We use this to decide whether stripping a common wrapper prefix (e.g. "model.")
        would *help* conversion (i.e. reveal patterns that start matching).
        """
        score = 0
        for k in keys:
            try:
                if self._apply_rename_dict(k) != k:
                    score += 1
            except Exception:
                # If rename application fails (e.g. invalid regex in a subclass), don't
                # let prefix stripping be the thing that breaks conversion.
                pass
            if self.pre_special_keys_map and any(
                p in k for p in self.pre_special_keys_map.keys()
            ):
                score += 1
            if self.special_keys_map and any(
                p in k for p in self.special_keys_map.keys()
            ):
                score += 1
        return score

    def _strip_prefix_inplace_if_better(
        self, state_dict: Dict[str, Any], prefix: str
    ) -> bool:
        """
        If *all* keys start with `prefix`, and stripping it increases this converter's
        applicability (match score), then strip it in-place and return True.
        """
        if not state_dict or not prefix:
            return False

        keys = list(state_dict.keys())
        if not keys or not all(k.startswith(prefix) for k in keys):
            return False

        stripped = [k[len(prefix) :] for k in keys]

        # Guard against pathological prefixes and key collisions.
        if any(not k for k in stripped):
            return False
        if len(set(stripped)) != len(stripped):
            return False

        for old_key, new_key in zip(keys, stripped):
            update_state_dict_(state_dict, old_key, new_key)
        return True

    def _strip_prefix_inplace_if_better_with_model_keys(
        self, state_dict: Dict[str, Any], prefix: str, model_keys: List[str]
    ) -> bool:
        """
        Strip `prefix` from keys that start with it (even if not unanimous), but only
        when doing so *strictly increases* overlap with `model_keys`.

        This is a safe extension of `_strip_prefix_inplace_if_better` for cases where
        a checkpoint mixes wrapped and unwrapped keys (common for LoRA exports that
        partially include `diffusion_model.` / `model.`).
        """
        if not state_dict or not prefix or not model_keys:
            return False

        keys = list(state_dict.keys())
        if not keys:
            return False

        prefixed = [k for k in keys if k.startswith(prefix)]
        if not prefixed:
            return False

        current_score = self._model_key_overlap_score(keys, model_keys)

        # Build hypothetical key list and guard against collisions/empties.
        new_keys = []
        for k in keys:
            if k.startswith(prefix):
                nk = k[len(prefix) :]
                if not nk:
                    return False
                new_keys.append(nk)
            else:
                new_keys.append(k)

        # Collision guard: would stripping create duplicate keys?
        if len(set(new_keys)) != len(new_keys):
            return False

        new_score = self._model_key_overlap_score(new_keys, model_keys)
        if new_score <= current_score:
            return False

        # Apply in-place for the affected subset only.
        for old_key in list(state_dict.keys()):
            if old_key.startswith(prefix):
                update_state_dict_(state_dict, old_key, old_key[len(prefix) :])
        return True

    def _strip_known_prefixes_inplace(
        self, state_dict: Dict[str, Any], model_keys: List[str] = None
    ) -> None:
        """
        Strip common wrapper prefixes (only when unanimous and helpful).

        This is intentionally conservative: we only strip when it increases the
        likelihood that this converter's rename/special rules will match.
        """
        if not state_dict:
            return

        keys = list(state_dict.keys())
        if not keys:
            return

        # If `model_keys` are available, use them as the ground truth to pick the
        # best unanimous strip prefix (including partial prefixes like `base_model.`).
        if model_keys:
            # Seed with common wrappers but also add token-derived candidates so we can
            # strip only part of a compound wrapper (e.g. `base_model.` but keep `model.`).
            base_candidates = (
                "model.diffusion_model.",
                "diffusion_model.model.",
                "model.",
                "diffusion_model.",
                "module.",
                "unet.",
                "base_model.model.",
                "base_model.",
            )

            changed = True
            while changed:
                changed = False
                keys = list(state_dict.keys())
                if not keys:
                    return

                current_score = self._model_key_overlap_score(keys, model_keys)

                token_candidates = self._common_prefix_token_candidates(keys)
                candidate_prefixes = []
                seen = set()
                for p in list(base_candidates) + token_candidates:
                    if not p or p in seen:
                        continue
                    seen.add(p)
                    candidate_prefixes.append(p)

                best_prefix = None
                best_score = current_score

                for p in candidate_prefixes:
                    # Consider both unanimous and *subset* stripping when model_keys are
                    # available. Subset stripping is common for LoRA exports where only
                    # some tensors are wrapped with e.g. `diffusion_model.`.
                    if not any(k.startswith(p) for k in keys):
                        continue
                    stripped = [k[len(p) :] if k.startswith(p) else k for k in keys]
                    if any(not k for k in stripped):
                        continue
                    if len(set(stripped)) != len(stripped):
                        continue
                    s = self._model_key_overlap_score(stripped, model_keys)
                    if s > best_score:
                        best_score = s
                        best_prefix = p
                    elif s == best_score and best_prefix is not None:
                        # Tie-break: prefer *less* stripping to avoid overstripping.
                        if len(p) < len(best_prefix):
                            best_prefix = p

                if best_prefix and best_score > current_score:
                    # Apply only to keys that actually have the prefix.
                    for old_key in list(state_dict.keys()):
                        if old_key.startswith(best_prefix):
                            update_state_dict_(
                                state_dict, old_key, old_key[len(best_prefix) :]
                            )
                    changed = True
            # If we successfully used model_keys, stop here.
            if self._model_key_overlap_score(list(state_dict.keys()), model_keys) > 0:
                return

        # Legacy behaviour (no model_keys or model_keys not helpful): prefer longer
        # compound prefixes first, then single-token wrappers.
        candidate_prefixes = (
            "model.diffusion_model.",
            "diffusion_model.model.",
            "diffusion_model.",
            "unet.",
            "base_model.model.",
        )

        changed = True
        while changed:
            changed = False
            for p in candidate_prefixes:
                if self._strip_prefix_inplace_if_better(state_dict, p):
                    changed = True
                    break

    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        self._sort_rename_dict()

        if self._already_converted(state_dict, model_keys):
            return state_dict
        # Apply pre-special keys map
        for key in list(state_dict.keys()):
            for (
                pre_special_key,
                handler_fn_inplace,
            ) in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)

        for key in list(state_dict.keys()):
            new_key = self._apply_rename_dict(key)
            update_state_dict_(state_dict, key, new_key)

        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key not in key:
                    continue

                handler_fn_inplace(key, state_dict)

        self._strip_known_prefixes_inplace(state_dict, model_keys=model_keys)

        return state_dict
