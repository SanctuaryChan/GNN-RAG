from __future__ import annotations

from typing import List, Optional, Union

import torch
from transformers import LogitsProcessor

from .trie import TokenTrie


class HardConstraintProcessor(LogitsProcessor):
    """Hard mask logits using a prefix trie."""

    def __init__(
        self,
        trie: TokenTrie,
        prompt_len: Union[int, List[int]],
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.trie = trie
        self.prompt_len = prompt_len
        self.eos_token_id = eos_token_id

    def _prompt_len(self, batch_id: int) -> int:
        if isinstance(self.prompt_len, list):
            return self.prompt_len[batch_id]
        return int(self.prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        vocab_size = scores.size(-1)
        neg_inf = torch.finfo(scores.dtype).min

        for i in range(batch_size):
            prompt_len = self._prompt_len(i)
            prefix = input_ids[i, prompt_len:].tolist()
            allowed = self.trie.next_tokens(prefix, eos_token_id=self.eos_token_id, allow_eos=True)
            if not allowed:
                # Fallback: allow EOS to terminate if prefix falls off trie.
                if self.eos_token_id is not None:
                    allowed = [self.eos_token_id]
                else:
                    continue
            mask = torch.full((vocab_size,), neg_inf, device=scores.device, dtype=scores.dtype)
            mask[allowed] = scores[i, allowed]
            scores[i] = mask
        return scores
