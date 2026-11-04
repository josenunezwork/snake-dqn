"""Stage 0 fresh probe world allocation and bare B8 death-cause audit (T34 ExIt-B8)."""

from __future__ import annotations

from research.hero_stage0.protocol import (
    COUNT,
    HORIZON,
    LOOKBACK_FRAMES,
    NAMESPACE,
    STAGE1A_ARM_SPECS,
    STAGE1A_ARMS,
    STAGE1A_BRANCH_HORIZON,
    STAGE1A_CONTROLLER,
    STAGE1A_INTERVAL,
    STAGE1A_MIN_LENGTH,
    STAGE1A_SCORING_SUFFIXES,
    STAGE1A_SELECTION_SUFFIXES,
    allocation,
    seed_for,
    training_exclusions,
)
from research.hero_stage0.stage1a import (
    aggregate_stage1a,
    branch_suffix_seed,
    dense_batch_h48_nb,
    evaluation_order_nb,
    reseed_branch_suffix,
    run_paired_world_worker,
    run_stage1a,
    run_stage1a_arm,
    score_override_independent_suffixes,
)

__all__ = [
    "COUNT",
    "HORIZON",
    "LOOKBACK_FRAMES",
    "NAMESPACE",
    "STAGE1A_ARM_SPECS",
    "STAGE1A_ARMS",
    "STAGE1A_BRANCH_HORIZON",
    "STAGE1A_CONTROLLER",
    "STAGE1A_INTERVAL",
    "STAGE1A_MIN_LENGTH",
    "STAGE1A_SCORING_SUFFIXES",
    "STAGE1A_SELECTION_SUFFIXES",
    "aggregate_stage1a",
    "allocation",
    "branch_suffix_seed",
    "dense_batch_h48_nb",
    "evaluation_order_nb",
    "reseed_branch_suffix",
    "run_paired_world_worker",
    "run_stage1a",
    "run_stage1a_arm",
    "score_override_independent_suffixes",
    "seed_for",
    "training_exclusions",
]
