"""CUDA-recipe vector61 dispatch tests without requiring a CUDA runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from research.t34_exit_b8.policy import B8Policy


class _Sim:
    E, S = 2, 3

    def __init__(self) -> None:
        self._alive = np.ones((self.E, self.S), dtype=bool)
        self._resolved = np.ones((self.E, self.S, 6), dtype=bool)

    def get_alive(self) -> np.ndarray:
        return self._alive

    def get_resolved_action_mask(self) -> np.ndarray:
        return self._resolved


class _Runtime:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls: list[np.ndarray] = []

    def selection(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slots = np.asarray(slots, dtype=np.int64)
        self.calls.append(slots.copy())
        states = np.zeros((len(slots), 61), dtype=np.float32)
        states[:, 0] = self.offset + slots[:, 0] * 10 + slots[:, 1]
        return states, np.ones((len(slots), 6), dtype=bool)


class _NativeVectorPolicy:
    """Minimal native hook whose carry must remain isolated per policy object."""

    input_size = 61

    def __init__(self, network: torch.nn.Module, runtime: _Runtime) -> None:
        self.network = network
        self.runtime = runtime
        self.veto_slots = frozenset({1})
        self.carry: list[np.ndarray] = []
        self.original_q_calls = 0

    def q_values(self, states: np.ndarray) -> torch.Tensor:
        self.original_q_calls += 1
        return self.network(torch.from_numpy(states))

    def actions(self, masks: np.ndarray, sim: _Sim, slots: np.ndarray) -> np.ndarray:
        del masks, sim
        states, _ = self.runtime.selection(slots)
        q = self.q_values(states)
        self.carry.append(q.detach().cpu().numpy().copy())
        # Stand in for the v8 hook: action choice uses the supplied native Qs,
        # while this policy-local carry records the exact slice it saw.
        return q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)


def test_cuda_recipe_batches_shared_network_and_restores_native_q_method(monkeypatch) -> None:
    """Independent v8-shaped policy hooks receive their own cached Q slices."""
    network = torch.nn.Linear(61, 6, bias=False)
    with torch.no_grad():
        network.weight.zero_()
        network.weight[:, 0] = torch.arange(6, dtype=torch.float32)
    first = _NativeVectorPolicy(network, _Runtime(1.0))
    second = _NativeVectorPolicy(network, _Runtime(100.0))
    runner = B8Policy("cpu", batch_device_vectors=True)
    runner.device = "cuda"  # Exercise the CUDA-only recipe while its cache stays on CPU.
    forwards: list[int] = []

    def cached_model(source: torch.nn.Module) -> torch.nn.Module:
        assert source is network
        forwards.append(1)
        return source

    monkeypatch.setattr(runner._models, "model_for", cached_model)
    sim = _Sim()
    grid = np.zeros((2, 3), dtype=np.int64)
    runner.dispatch(sim, object(), [[first, second], [first, second]], grid)

    assert forwards == [1]
    assert grid[:, 1:].tolist() == [[5, 5], [5, 5]]
    assert [len(policy.carry) for policy in (first, second)] == [1, 1]
    assert first.carry[0].shape == second.carry[0].shape == (2, 6)
    assert not np.array_equal(first.carry[0], second.carry[0])
    assert first.original_q_calls == second.original_q_calls == 0
    assert "q_values" not in vars(first)
    assert "q_values" not in vars(second)


def test_cuda_recipe_restores_instance_q_override_after_native_hook_failure(monkeypatch) -> None:
    """A failed hook cannot leave a shared policy with a stale cached Q source."""
    network = torch.nn.Linear(61, 6, bias=False)
    policy = _NativeVectorPolicy(network, _Runtime(1.0))

    def original(states):
        return torch.full((len(states), 6), 7.0)

    policy.q_values = original

    def fail_actions(masks, sim, slots):
        del masks, sim, slots
        raise RuntimeError("native veto failure")

    policy.actions = fail_actions
    runner = B8Policy("cpu", batch_device_vectors=True)
    runner.device = "cuda"
    monkeypatch.setattr(runner._models, "model_for", lambda source: source)
    with np.testing.assert_raises_regex(RuntimeError, "native veto failure"):
        runner.dispatch(
            _Sim(), SimpleNamespace(), [[policy, policy], [policy, policy]], np.zeros((2, 3))
        )
    assert policy.q_values is original


def test_cuda_device_path_is_native_unless_t34_batching_opted_in(monkeypatch) -> None:
    """The shared B8 adapter keeps v1's CUDA vector61 recipe unchanged by default."""
    network = torch.nn.Linear(61, 6, bias=False)
    policy = _NativeVectorPolicy(network, _Runtime(1.0))
    runner = B8Policy("cpu")
    runner.device = "cuda"
    monkeypatch.setattr(
        runner._models,
        "model_for",
        lambda source: (_ for _ in ()).throw(AssertionError("unexpected cached batch forward")),
    )
    runner.dispatch(
        _Sim(), SimpleNamespace(), [[policy, policy], [policy, policy]], np.zeros((2, 3))
    )
    assert policy.original_q_calls == 1


class _V8Runtime:
    """Prepared vector rows plus fixed decision-time free-space counts."""

    featurizer = SimpleNamespace(gw=100, gh=80)

    def selection(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slots = np.asarray(slots, dtype=np.int64)
        states = np.zeros((len(slots), 61), dtype=np.float32)
        states[:, 0] = slots[:, 0] * 10 + slots[:, 1]
        return states, np.ones((len(slots), 6), dtype=bool)

    def decision_free_space(self, sim, slots: np.ndarray) -> np.ndarray:
        del sim
        return np.tile(np.asarray([[200, 120, 180]], dtype=np.int64), (len(slots), 1))


def _real_v8_policy(network: torch.nn.Module, slot: int):
    """Build a real native Vector61 v8 hook without loading a checkpoint."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policy = Vector61SimdPolicy.__new__(Vector61SimdPolicy)
    policy._torch = torch
    policy.network = network
    policy.device = torch.device("cpu")
    policy.input_size = 61
    policy.runtime = _V8Runtime()
    policy.forward = "rowwise"
    policy._configure_veto((slot,), "v8", 4.0, None)
    return policy


def _real_v8_sim():
    """Return two prepared full-grid worlds with independently carried v8 rows."""
    from src.core.game_config import GameConfig
    from src.simd_env.batch_sim import CARDINAL, BatchSim, BatchSimConfig

    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=3,
        game_width=1000,
        game_height=800,
        initial_food=0,
        max_food=0,
        min_boost_length=int(GameConfig.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(GameConfig.BOOST_LENGTH_COST_FRAMES),
        mechanics_version=int(GameConfig.MECHANICS_VERSION),
    )
    sim = BatchSim(cfg, seeds=[17, 23], train_mode=False)
    sim.alive[:, :] = False
    cardinal = [tuple(int(value) for value in row) for row in CARDINAL.tolist()]
    for env in range(2):
        for slot, (head_x, heading) in enumerate(((50 + env, (1, 0)), (40, (1, 0)), (70, (-1, 0)))):
            cells = [(head_x - index, 40 + slot * 8) for index in range(8)]
            sim.head_ptr[env, slot] = len(cells) - 1
            for index, cell in enumerate(cells):
                sim.bodies[env, slot, len(cells) - 1 - index] = cell
            sim.seg_count[env, slot] = sim.length[env, slot] = len(cells)
            sim.direction[env, slot] = cardinal.index(heading)
            sim.alive[env, slot] = True
    return sim


def _v8_snapshot(policy) -> dict[str, object]:
    """Capture the hook-owned state whose cross-policy isolation is contractual."""
    last = {}
    for key, value in policy.last_veto.items():
        last[key] = value.copy() if isinstance(value, np.ndarray) else value
    return {
        "last": last,
        "records": [policy.veto_record(env) for env in range(2)],
        "diagnostics": [policy.veto_diagnostics(env) for env in range(2)],
    }


def _without_timing(value):
    """V8 diagnostic timing is intentionally not part of action/carry parity."""
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items() if "seconds" not in key}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def test_cuda_recipe_matches_real_vector61_v8_actions_and_carry_on_cpu_cache(
    setup_config, monkeypatch
) -> None:
    """Cached device Qs retain native v8 masks, hooks, counters, and last-veto bytes."""
    del setup_config
    network = torch.nn.Linear(61, 6)
    known_q = torch.tensor([0.5, 2.0, 9.0, -1.0, -2.0, -3.0], dtype=torch.float32)
    with torch.no_grad():
        network.weight.zero_()
        network.bias.copy_(known_q)
    native = [_real_v8_policy(network, 1), _real_v8_policy(network, 2)]
    native_grid = np.zeros((2, 3), dtype=np.int64)
    B8Policy("cpu").dispatch(
        _real_v8_sim(), object(), [[native[0], native[1]], [native[0], native[1]]], native_grid
    )
    expected = [_v8_snapshot(policy) for policy in native]

    batched = [_real_v8_policy(network, 1), _real_v8_policy(network, 2)]
    runner = B8Policy("cpu", batch_device_vectors=True)
    runner.device = "cuda"  # The cache remains CPU so this test needs no GPU.
    calls: list[torch.nn.Module] = []
    original_model_for = runner._models.model_for

    def counted_model_for(source: torch.nn.Module) -> torch.nn.Module:
        calls.append(source)
        return original_model_for(source)

    monkeypatch.setattr(runner._models, "model_for", counted_model_for)
    batched_grid = np.zeros((2, 3), dtype=np.int64)
    runner.dispatch(
        _real_v8_sim(), object(), [[batched[0], batched[1]], [batched[0], batched[1]]], batched_grid
    )

    assert calls == [network]
    assert batched_grid.tobytes() == native_grid.tobytes()
    actual = [_v8_snapshot(policy) for policy in batched]
    for before, after in zip(expected, actual):
        assert before["records"] == after["records"]
        assert _without_timing(before["diagnostics"]) == _without_timing(after["diagnostics"])
        assert before["last"].keys() == after["last"].keys()
        for key in before["last"]:
            left, right = before["last"][key], after["last"][key]
            if isinstance(left, np.ndarray):
                assert left.tobytes() == right.tobytes()
            else:
                assert left == right
