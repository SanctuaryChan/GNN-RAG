from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class TokenTrie:
    """Simple token trie for prefix-allowed decoding."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def add(self, token_ids: Iterable[int]) -> None:
        node = self.root
        for tok in token_ids:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.is_end = True

    def _traverse(self, prefix: List[int]) -> Optional[TrieNode]:
        node = self.root
        for tok in prefix:
            if tok not in node.children:
                return None
            node = node.children[tok]
        return node

    def next_tokens(
        self,
        prefix: List[int],
        eos_token_id: Optional[int] = None,
        allow_eos: bool = True,
    ) -> List[int]:
        node = self._traverse(prefix)
        if node is None:
            return []
        allowed: Set[int] = set(node.children.keys())
        if allow_eos and node.is_end and eos_token_id is not None:
            allowed.add(eos_token_id)
        return list(allowed)
