"""Real CPU compiler coverage for every resident A-3 fullgraph boundary."""

from __future__ import annotations

import gc
import weakref

import torch

from research.gpu_sim_spike.v8_device import v8_decide
from research.sandbox_a_resident import run
from research.sandbox_a_resident.core import Recipe, ResidentAssay
from research.sandbox_a_resident.runtime_tools import CompiledBlock
from research.sandbox_a_resident.tensor_blocks import TensorBlocks
from src.model.ego2s_network import Ego2sNet


def test_real_aot_eager_compiles_and_retains_all_resident_blocks() -> None:
    """Exercise each block across ticks and an in-place reset without fallback.

    ``aot_eager`` is an actual ``torch.compile`` backend selected to keep this
    CPU admission test quick; deployment retains the requested Inductor mode.
    It is intentionally not a fake compiler or a replacement eager callable.
    """
    torch.compiler.reset()
    torch.set_num_threads(1)
    with run.numerical_mode(deterministic=False, tf32=True):
        runner = ResidentAssay(
            Recipe(worlds=1, horizon=2, minibatch=2, episode_limit=2, tiny=True),
            Ego2sNet(local_channels=7, n_scalars=60),
        )
        runner.enable_compile("default", backend="aot_eager")
        assert runner.warm_compile_buckets() == [32]
        # Compile forward and backward under the production numerical globals
        # before the separately measured steady-state cycles.
        runner.update(runner.collect())
        validations_after_warmup = {
            name: entry["validation_calls"] for name, entry in runner.compile_report.items()
        }
        for _ in range(3):
            before = run.compiler_metrics()
            runner.update(runner.collect())
            assert run.compiler_delta(before, run.compiler_metrics()) == {}
        assert {
            name: entry["validation_calls"] for name, entry in runner.compile_report.items()
        } == validations_after_warmup

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


def test_production_v8_buckets_are_pure_and_match_each_live_window() -> None:
    """Every cap-1024 R2 shape is valid before it is prewarmed on CUDA."""
    torch.compiler.reset()
    torch.set_num_threads(1)
    runner = ResidentAssay(
        Recipe(worlds=1, horizon=1, minibatch=1, episode_limit=2),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    blocks, rows = runner.tensor_blocks, runner._rows
    assert blocks.v8_bucket_spans(runner.sim) == (256, 512, 768, 1024)
    runner.sim.prepare()
    before = {
        name: value.detach().clone()
        for name, value in runner.sim.__dict__.items()
        if isinstance(value, torch.Tensor)
    }
    generator_before = runner.generator.get_state().clone()
    actor_before = {
        name: value.detach().clone() for name, value in runner.actor.state_dict().items()
    }
    optimizer_before = len(runner.optimizer.state)
    with run.numerical_mode(deterministic=False, tf32=True):
        runner.enable_compile("default", backend="aot_eager")
        assert runner.warm_compile_buckets() == [256, 512, 768, 1024]
        assert runner.compile_report["v8_decide"]["validation_calls"] == 4
        with torch.no_grad():
            compiler_warm = run.compiler_metrics()
            mask = runner.sim.advisory_mask()[:, 0]
            expected = (
                v8_decide(runner.sim, rows, runner._v8_q, mask, 8.0, features=True)
                .feature_planes()
                .reshape(1, 30)
            )
            for span in (1024, 256, 768, 512, 256):
                all_occ, other_occ, own_cells, own_valid = blocks.v8_inputs(
                    runner.sim, rows, span=span
                )
                actual = runner._v8(
                    *blocks.state(runner.sim),
                    rows=rows,
                    mask=mask,
                    all_occ=all_occ,
                    other_occ=other_occ,
                    own_cells=own_cells,
                    own_valid=own_valid,
                )
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            assert run.compiler_delta(compiler_warm, run.compiler_metrics()) == {}
        assert runner.compile_report["v8_decide"]["validation_calls"] == 4
    assert torch.equal(runner.generator.get_state(), generator_before)
    assert len(runner.optimizer.state) == optimizer_before
    for name, expected_tensor in before.items():
        assert torch.equal(getattr(runner.sim, name), expected_tensor), name
    for name, expected_tensor in actor_before.items():
        assert torch.equal(runner.actor.state_dict()[name], expected_tensor), name


def test_real_aot_eager_specializes_once_per_tf32_global_mode() -> None:
    """The wrapper validates each real compiler global-state specialization once."""
    torch.compiler.reset()
    torch.set_num_threads(1)
    report: dict[str, object] = {}
    block = CompiledBlock(
        lambda value: value @ value.T, "default", report, "linear", backend="aot_eager"
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            torch.backends.cuda.matmul.allow_tf32 = False
            block(torch.ones(2, 2))
            false_warm = run.compiler_metrics()
            block(torch.ones(2, 2))
            assert run.compiler_delta(false_warm, run.compiler_metrics()) == {}
            torch.backends.cuda.matmul.allow_tf32 = True
            block(torch.ones(2, 2))
            true_warm = run.compiler_metrics()
            block(torch.ones(2, 2))
            assert run.compiler_delta(true_warm, run.compiler_metrics()) == {}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    entry = report["linear"]
    assert isinstance(entry, dict)
    assert entry["status"] == "compiled", entry.get("fallback_reason")
    assert entry["compile_specialization_calls"] == 2
    assert entry["validation_calls"] == 2


def test_real_compile_cache_does_not_retain_first_assay_or_its_adam_state() -> None:
    """Cached pure functions may retain the actor, never a resident assay."""
    torch.compiler.reset()
    torch.set_num_threads(1)
    cache = run.CellCompileCache()
    first = ResidentAssay(
        Recipe(worlds=1, horizon=2, minibatch=2, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    assert cache.attach(first, "default", backend="aot_eager") is False
    first.warm_compile_buckets()
    first.update(first.collect())
    assert first.optimizer.state
    assert all(entry["status"] == "compiled" for entry in first.compile_report.values())
    first_ref = weakref.ref(first)
    sim_ref = weakref.ref(first.sim)
    optimizer_ref = weakref.ref(first.optimizer)

    second = ResidentAssay(
        Recipe(worlds=2, horizon=2, minibatch=2, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    assert cache.attach(second, "default", backend="aot_eager") is True
    second.update(second.collect())
    assert second.optimizer.state
    assert all(entry["status"] == "compiled" for entry in second.compile_report.values())
    del first
    gc.collect()
    assert first_ref() is None
    assert sim_ref() is None
    assert optimizer_ref() is None
