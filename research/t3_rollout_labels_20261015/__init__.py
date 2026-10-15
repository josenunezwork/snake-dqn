"""Batched, fidelity-checked counterfactual rollouts for the T3 label study."""

from research.t3_rollout_labels_20261015.engine import (
    batched_rollout,
    extract_prepared_env,
)

__all__ = ("batched_rollout", "extract_prepared_env")
