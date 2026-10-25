"""The prospective sampler must reproduce the independently checked runtime."""

import numpy as np
import torch

from research.bl_round30.batch_plan import sampler_indices
from research.bl_round30.objective import ObjectiveRecipe, reproduce_batch_plan


def test_sampler_only_plan_matches_runtime_without_feature_construction():
    n = 4
    table = {
        "life_id": np.asarray(["a/parent/0", "b/parent/0", "c/parent/0", "d/parent/0"]),
        "ordinary": np.asarray([False, False, True, True]),
        "legal": np.ones((n, 6), dtype=bool),
        "action_mask": np.zeros((n, 6), dtype=bool),
        "status": np.full((n, 6), "EXACT"),
        "died": np.asarray([[False, True, True, True, True, True]] * n),
    }
    cache = {
        key: torch.as_tensor(value.copy())
        for key, value in table.items()
        if key not in ("life_id", "status")
    }
    cache.update(
        life_id=torch.arange(n),
        status=torch.ones(n, 6, dtype=torch.bool),
        features=torch.zeros(n, 256),
        q=torch.zeros(n, 6),
        mass=torch.ones(n, 6),
    )
    for seed in (3403, 3404):
        expected = reproduce_batch_plan(cache, ObjectiveRecipe(updates=3), seed=seed)
        torch.testing.assert_close(sampler_indices(table, seed=seed, updates=3), expected.indices)
