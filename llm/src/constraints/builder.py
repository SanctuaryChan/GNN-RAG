from __future__ import annotations

from typing import List, Optional

from .entity_vocab import surface_forms
from .spec import ConstraintInputs, ConstraintSpec
from .trie import TokenTrie


class ConstraintBuilder:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def build(
        self,
        inputs: ConstraintInputs,
        mode: str = "entity",
        strength: str = "hard",
        max_candidates: Optional[int] = None,
        penalty_lambda: Optional[float] = None,
    ) -> ConstraintSpec:
        if mode == "none":
            return ConstraintSpec(mode="none", strength=strength)
        if mode != "entity":
            # MVP only supports entity mode.
            return ConstraintSpec(mode=mode, strength=strength)

        candidates: List[str] = list(inputs.candidates or [])
        if inputs.extra_candidates:
            candidates.extend(inputs.extra_candidates)
        if max_candidates is not None and max_candidates > 0:
            candidates = candidates[:max_candidates]

        sequences: List[List[int]] = []
        for name in candidates:
            for form in surface_forms(name):
                token_ids = self.tokenizer.encode(form, add_special_tokens=False)
                if token_ids:
                    sequences.append(token_ids)
                # Also add a leading-space variant to improve matching.
                spaced = " " + form
                spaced_ids = self.tokenizer.encode(spaced, add_special_tokens=False)
                if spaced_ids and spaced_ids != token_ids:
                    sequences.append(spaced_ids)

        if not sequences:
            return ConstraintSpec(
                mode="none",
                strength=strength,
                debug_info={"reason": "empty_candidates"},
            )

        trie = TokenTrie()
        for seq in sequences:
            trie.add(seq)

        return ConstraintSpec(
            mode=mode,
            strength=strength,
            allowed_sequences=sequences,
            trie=trie,
            penalty_lambda=penalty_lambda,
        )
