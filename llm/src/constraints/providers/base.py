from __future__ import annotations

from ..spec import ConstraintInputs


class ConstraintProvider:
    def build(self, sample: dict) -> ConstraintInputs:
        raise NotImplementedError
