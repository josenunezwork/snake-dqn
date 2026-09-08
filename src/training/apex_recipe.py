"""Explicit, comparable Apex training recipes and continuation validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.core.runtime_contract import canonical_digest


@dataclass(frozen=True)
class ApexRecipe:
    """Numerical and replay semantics that must agree before a continuation."""

    mode: str
    optimizer: str
    grad_clip_norm: float
    beta_clock: str
    gamma: float
    n_step: int
    priority_alpha: float
    priority_eps: float

    def __post_init__(self) -> None:
        expected = {
            "local": ("AdamW", 50.0, "successful_local_sample"),
            "distributed": ("Adam", 100.0, "successful_distributed_sample"),
        }
        if self.mode not in expected:
            raise ValueError(f"Unsupported Apex recipe mode {self.mode!r}")
        if (self.optimizer, self.grad_clip_norm, self.beta_clock) != expected[self.mode]:
            raise ValueError(f"{self.mode} Apex recipe numerical semantics are fixed")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def to_metadata(self) -> dict[str, Any]:
        return {"apex_recipe": asdict(self), "apex_recipe_digest": self.digest}


def validate_recipe_continuation(
    checkpoint: Mapping[str, Any], recipe: ApexRecipe, *, weights_only: bool
) -> None:
    """Reject incompatible resume state before optimizer or child construction."""
    if weights_only:
        return
    stored = checkpoint.get("apex_recipe")
    stored_digest = checkpoint.get("apex_recipe_digest")
    if not isinstance(stored, dict) or not isinstance(stored_digest, str):
        raise ValueError("optimizer continuation requires a verified Apex recipe")
    if canonical_digest(stored) != stored_digest:
        raise ValueError("checkpoint Apex recipe digest does not match its descriptor")
    if stored_digest != recipe.digest:
        raise ValueError("checkpoint Apex recipe conflicts with requested continuation")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("optimizer continuation requires optimizer_state_dict")
