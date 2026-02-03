from __future__ import annotations

from typing import List


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def surface_forms(name: str) -> List[str]:
    """Return surface forms for an entity name (MVP: canonical only)."""
    if not name:
        return []
    norm = normalize_whitespace(name)
    if not norm:
        return []
    return [norm]
