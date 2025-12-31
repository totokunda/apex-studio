from src.converters.utils import update_state_dict_
import re
from typing import Dict, Any, Iterable


class BaseConverter:
    def __init__(self):
        self.rename_dict = {}
        self.special_keys_map = {}
        self.pre_special_keys_map = {}

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

    def _already_converted(self, state_dict: Dict[str, Any]) -> bool:
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

    def _strip_known_prefixes_inplace(self, state_dict: Dict[str, Any]) -> None:
        """
        Strip common wrapper prefixes (only when unanimous and helpful).

        This is intentionally conservative: we only strip when it increases the
        likelihood that this converter's rename/special rules will match.
        """
        # Prefer longer compound prefixes first, then single-token wrappers.
        candidate_prefixes = (
            "model.diffusion_model.",
            "diffusion_model.model.",
            "model.",
            "diffusion_model.",
            "module.",
            "unet.",
        )

        changed = True
        while changed:
            changed = False
            for p in candidate_prefixes:
                if self._strip_prefix_inplace_if_better(state_dict, p):
                    changed = True
                    break

    def convert(self, state_dict: Dict[str, Any]):
        self._sort_rename_dict()
        # Some checkpoints are stored under a wrapper prefix (e.g. "model." or
        # "diffusion_model."). If *every* key has the prefix and stripping it makes
        # our conversion rules match better, strip it before any conversion passes.
        self._strip_known_prefixes_inplace(state_dict)
        # If this looks like a checkpoint that already matches the target key layout,
        # exit early to keep conversion idempotent.

        if self._already_converted(state_dict):
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
        return state_dict
