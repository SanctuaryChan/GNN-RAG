from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Triplet = Tuple[str, str, str]


@dataclass
class ConstraintInputs:
    """Unified input protocol for constraint construction.

    All fields are optional except candidates; providers should normalize into
    this structure so GCD can remain decoupled from upstream retrievers.
    """

    candidates: List[str] = field(default_factory=list)
    evidence_paths: Optional[List[List[Triplet]]] = None
    evidence_relations: Optional[List[str]] = None
    extra_candidates: Optional[List[str]] = None


@dataclass
class ConstraintSpec:
    """Specification passed to the decoder for constrained generation."""

    mode: str = "none"  # none|entity|relation|path
    strength: str = "hard"  # hard|soft
    allowed_sequences: List[List[int]] = field(default_factory=list)
    allowed_tokens: Optional[Set[int]] = None
    penalty_lambda: Optional[float] = None
    trie: Optional[object] = None
    debug_info: Optional[Dict] = None
