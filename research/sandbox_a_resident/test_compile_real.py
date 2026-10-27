"""Real CPU compiler coverage for every resident A-3 fullgraph boundary."""

from __future__ import annotations

import torch

from research.gpu_sim_spike.v8_device import v8_decide
from research.sandbox_a_resident import run
from research.sandbox_a_resident.core import Recipe, ResidentAssay
from research.sandbox_a_resident.tensor_blocks import TensorBlocks
from src.model.ego2s_network import Ego2sNet


def test_real_aot_eager_compiles_and_retains_all_resident_blocks() -> None:
    """Exercise each block across ticks and an in-place reset without fallback.

    ``aot_eager`` is an actual ``torch.compile`` backend selected to keep this
    CPU admission test quick; deployment retains the requested Inductor mode.
    It is intentionally not a fake compiler or a replacement eager callable.
    """
    torch.set_num_threads(1)
    runner = ResidentAssay(
        Recipe(worlds=1, horizon=2, minibatch=2, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    runner.enable_compile("default", backend="aot_eager")
    first = runner.collect()
    runner.update(first)
    runner._reset(torch.ones(1, dtype=torch.bool))
    second = runner.collect()
    runner.update(second)

    expected = {
        "ActorCritic_collect",
        "ActorCritic_train",
        "v8_decide",
        "build_observation",
        "anchors",
    }
    assert set(runner.compile_report) == expected
    for name in expected:
        entry = runner.compile_report[name]
        assert entry["backend"] == "aot_eager"
        assert entry["status"] == "compiled", (name, entry.get("fallback_reason"))
        assert entry["validation_calls"] >= 1


def test_tensor_v8_feature_boundary_matches_source_before_and_after_reset() -> None:
    """R2 inputs stay outside the graph without changing the B8-v8 planes."""
    runner = ResidentAssay(
        Recipe(worlds=2, horizon=1, minibatch=2, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    blocks = TensorBlocks(runner.sim)
    for _ in range(2):
        runner.sim.prepare()
        rows = runner._rows
        mask = runner.sim.advisory_mask()[:, 0]
        expected = (
            v8_decide(runner.sim, rows, runner._v8_q, mask, 8.0, features=True)
            .feature_planes()
            .reshape(runner.recipe.worlds, 30)
        )
        all_occ, other_occ, own_cells, own_valid = blocks.v8_inputs(runner.sim, rows)
        actual = blocks.v8_features(
            *blocks.state(runner.sim),
            rows=rows,
            mask=mask,
            all_occ=all_occ,
            other_occ=other_occ,
            own_cells=own_cells,
            own_valid=own_valid,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        runner._reset(torch.ones(runner.recipe.worlds, dtype=torch.bool))


def test_shared_compile_cache_handles_four_world_shapes_under_tf32_timing_mode() -> None:
    """Bound the known shape matrix without mutable-cache recompilation/fallback."""
    torch.set_num_threads(1)
    cache = run.CellCompileCache()
    reused = []
    with (
        torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=256),
        run.numerical_mode(deterministic=False, tf32=True),
    ):
        for worlds in (1, 2, 3, 4):
            runner = ResidentAssay(
                Recipe(worlds=worlds, horizon=1, minibatch=worlds, episode_limit=2, tiny=True),
                Ego2sNet(local_channels=7, n_scalars=60),
            )
            reused.append(cache.attach(runner, "default", backend="aot_eager"))
            runner.update(runner.collect())
            for entry in runner.compile_report.values():
                assert entry["status"] == "compiled", entry.get("fallback_reason")
    assert reused == [False, True, True, True]
