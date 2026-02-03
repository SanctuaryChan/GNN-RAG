from __future__ import annotations

from .base import ConstraintProvider
from ..spec import ConstraintInputs


class GnnRagProvider(ConstraintProvider):
    """Project adapter: map sample fields to ConstraintInputs."""

    def build(self, sample: dict) -> ConstraintInputs:
        candidates = sample.get("cand", []) or []
        evidence_paths = sample.get("predicted_paths", None)
        return ConstraintInputs(candidates=candidates, evidence_paths=evidence_paths)
