"""Independent parity contracts for the optimized Phase-0 search teacher.

The oracle is vendored from commit 45069720, before the speed work.  Do not
replace it with an import from ``research.phase0_search_20261019.teacher``:
that would make a shared regression invisible to this suite.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from research.phase0_search_20261019 import teacher

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests/fixtures/phase0_teacher_speed"
ORACLE_PATH = FIXTURE_DIR / "teacher_pre_45069720.py"
ORACLE_SHA256 = "d15960f0dab45b3e08d49549990620e341a0e6b2c665654ebf669b43d9d14f17"
B8_CHECKPOINT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
)


def _load_oracle() -> ModuleType:
    """Load the frozen scorer under a private name, separate from the candidate."""
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
    spec = importlib.util.spec_from_file_location("phase0_teacher_pre_45069720", ORACLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle() -> ModuleType:
    return _load_oracle()


@pytest.fixture(scope="module")
def timing_rows() -> dict[str, dict[str, Any]]:
    bank = json.loads(
        (ROOT / "research/phase0_search_20261019/WORLD_BANK.json").read_text(encoding="utf-8")
    )
    return {str(row["mix"]): row for row in bank["fixtures"]["timing"]}


def _source_digest(world: Any) -> bytes:
    """Digest source simulator and Vector61 carry arrays without candidate helpers."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    digest = hashlib.sha256(gid.env_digest(world.sim, 0).encode("ascii"))

    def add_arrays(prefix: str, value: Any) -> None:
        for name, item in sorted(vars(value).items()):
            if isinstance(item, np.ndarray):
                digest.update(prefix.encode("utf-8"))
                digest.update(name.encode("utf-8"))
                digest.update(str(item.dtype).encode("ascii"))
                digest.update(repr(item.shape).encode("ascii"))
                digest.update(np.ascontiguousarray(item).tobytes())

    add_arrays("runtime.", world.runtime)
    if hasattr(world.runtime, "memory"):
        add_arrays("runtime.memory.", world.runtime.memory)
    return digest.digest()


def _world(row: dict[str, Any], length: int, module: ModuleType) -> Any:
    if not B8_CHECKPOINT.is_file():
        pytest.skip("the local, hash-pinned B8 checkpoint inventory is unavailable")
    from research.phase0_search_20261019.preflight import fixture

    # Use the candidate's registered fixture constructor only to make the
    # strict source state.  The score oracle itself is entirely vendored.
    return fixture(row, B8_CHECKPOINT, length)


def _search(module: ModuleType, world: Any, *, device: str = "cpu") -> tuple[int, dict[str, Any]]:
    """Search at the evaluator's prepared Watch boundary without moving it."""
    config = module.SearchConfig(rollouts=8, horizon=128, epsilon=0.02, length_gate=0)
    captured: dict[str, Any] = {}

    class _BoundaryRead(Exception):
        pass

    def select(prepared: Any) -> np.ndarray:
        world.runtime.prepare(prepared)
        before = _source_digest(world)
        searcher = (
            module.SearchTeacher(config, device=device)
            if module is teacher
            else module.SearchTeacher(config)
        )
        action, diagnostic = searcher.choose(world)
        captured.update(
            action=action, diagnostic=diagnostic, before=before, after=_source_digest(world)
        )
        raise _BoundaryRead

    try:
        world.sim.step_with_policy(select)
    except _BoundaryRead:
        pass
    assert captured["before"] == captured["after"]
    return int(captured["action"]), dict(captured["diagnostic"])


@pytest.mark.parametrize("mix", ("scripted", "frozen", "mixed"))
@pytest.mark.parametrize("length", (200, 800, 1500))
@pytest.mark.parametrize("device", ("cpu", pytest.param("mps", marks=pytest.mark.requires_gpu)))
def test_registered_timing_fixture_matches_frozen_teacher(
    oracle: ModuleType,
    timing_rows: dict[str, dict[str, Any]],
    mix: str,
    length: int,
    device: str,
) -> None:
    """N=8/H=128 choices and scores agree for every registered timing regime."""
    teacher.configure_cpu()
    oracle.configure_cpu()
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable on this host")
    reference_world = _world(timing_rows[mix], length, oracle)
    candidate_world = _world(timing_rows[mix], length, teacher)
    reference_action, reference_diag = _search(oracle, reference_world)
    candidate_action, candidate_diag = _search(teacher, candidate_world, device=device)

    assert reference_diag["searched"] and candidate_diag["searched"]
    assert candidate_action == reference_action
    assert tuple(candidate_diag["scores"]) == tuple(reference_diag["scores"])
    # Row-batched Torch forwards may reassociate float32 arithmetic.  The
    # contract permits that numerical difference but never a changed action.
    np.testing.assert_allclose(
        [candidate_diag["scores"][action] for action in sorted(candidate_diag["scores"])],
        [reference_diag["scores"][action] for action in sorted(reference_diag["scores"])],
        rtol=0.0,
        atol=1e-3,
    )
    np.testing.assert_allclose(candidate_diag["q"], reference_diag["q"], rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("mix", ("scripted", "frozen", "mixed"))
def test_optimized_teacher_is_exactly_deterministic_on_registered_fixture(
    timing_rows: dict[str, dict[str, Any]], mix: str
) -> None:
    """Repeated CPU scoring has no scheduling-dependent action or score drift."""
    teacher.configure_cpu()
    first_action, first = _search(teacher, _world(timing_rows[mix], 800, teacher))
    second_action, second = _search(teacher, _world(timing_rows[mix], 800, teacher))
    assert first_action == second_action
    assert first["scores"] == second["scores"]
    np.testing.assert_array_equal(first["q"], second["q"])


def test_exact_score_tie_keeps_first_legal_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimized selector retains the registered first-index tie rule."""
    world = SimpleNamespace(
        sim=SimpleNamespace(frame=np.array([17]), length=np.array([[800]])),
        seed=919,
        mix="scripted",
    )
    monkeypatch.setattr(
        teacher,
        "_q_and_legal",
        lambda _world: (
            np.arange(6, dtype=np.float32),
            np.array([False, True, False, True, False, False]),
            np.array([False, True, False, True, False, False]),
        ),
    )
    monkeypatch.setattr(
        teacher,
        "_batched_scores",
        lambda _world, actions, _config, _decision, **_kwargs: {action: 4.0 for action in actions},
    )
    action, diagnostic = teacher.SearchTeacher(
        teacher.SearchConfig(length_gate=0), implementation="batched"
    ).choose(world)
    assert action == 1
    assert diagnostic["scores"] == {1: 4.0, 3: 4.0}


def test_near_q_tie_reruns_only_close_rows_on_reference_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched forward roundoff cannot change a near-tied action decision."""
    import torch

    from src.simd_env import ego2s_policy, ego_raster

    sim = SimpleNamespace(
        get_legal_action_mask=lambda: np.array(
            [
                [[True, True, False, False, False, False]],
                [[False, False, True, False, False, False]],
            ]
        ),
        get_advisory_action_mask=lambda: np.array(
            [
                [[True, True, False, False, False, False]],
                [[False, False, True, False, False, False]],
            ]
        ),
    )
    calls: list[tuple[str, str, tuple[int, ...]]] = []

    def raster(_sim: Any, _config: Any, backend: str, rows: np.ndarray) -> dict[str, np.ndarray]:
        assert backend == "numba"
        return {
            "rows": rows.copy(),
            "local": rows[:, :1].astype(np.float32),
            "global": rows[:, :1].astype(np.float32),
            "scalars": rows[:, :1].astype(np.float32),
        }

    def forward(net: str, obs: dict[str, np.ndarray], mode: str) -> torch.Tensor:
        calls.append((net, mode, tuple(obs["rows"][:, 0])))
        if net == "batch":
            return torch.tensor(
                [[10.0, 9.9995, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 7.0, 0.0, 0.0, 0.0]]
            )
        assert mode == "rowwise"
        return torch.tensor([[10.0, 11.0, 0.0, 0.0, 0.0, 0.0]])

    monkeypatch.setattr(ego_raster, "build_ego_raster", raster)
    monkeypatch.setattr(ego2s_policy, "forward_q", forward)
    q, legal, select = teacher._rollout_q(sim, "batch", "reference", np.array([0, 1]))
    np.testing.assert_array_equal(legal, sim.get_legal_action_mask()[:, 0])
    np.testing.assert_array_equal(select, sim.get_advisory_action_mask()[:, 0])
    np.testing.assert_array_equal(q[0], np.array([10.0, 11.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(q[1], np.array([0.0, 0.0, 7.0, 0.0, 0.0, 0.0]))
    assert calls == [("batch", "batched", (0, 1)), ("reference", "rowwise", (0,))]


def test_observation_deduplication_reconstructs_exact_input_bytes() -> None:
    """Only byte-identical local, global, and scalar inputs may share a forward."""
    obs = {
        "local": np.array([[[1]], [[1]], [[1]], [[2]], [[1]]], dtype=np.int8),
        "global": np.array([[[3]], [[3]], [[3]], [[3]], [[4]]], dtype=np.int8),
        # Row 2 differs solely by signed zero, which must not be silently merged.
        "scalars": np.array([[0.0], [0.0], [-0.0], [0.0], [0.0]], dtype=np.float32),
    }
    first, inverse = teacher._unique_observations(obs)
    assert len(first) == 4
    assert inverse[0] == inverse[1]
    assert inverse[0] != inverse[2]
    reconstructed = first[inverse]
    for key, value in obs.items():
        for original, selected in enumerate(reconstructed):
            assert value[original].tobytes() == value[selected].tobytes()


def test_near_score_tie_reruns_frozen_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leaf-score margin inside the float32 bound returns the CPU reference result."""
    from src.simd_env import eval_engine

    sim = SimpleNamespace(
        E=2,
        S=1,
        alive=np.zeros((2, 1), dtype=bool),
        length=np.zeros((2, 1), dtype=np.int64),
        get_resolved_action_mask=lambda: np.ones((2, 1, 6), dtype=bool),
        step=lambda _actions: None,
    )
    replica = SimpleNamespace(
        sim=sim,
        runtime=SimpleNamespace(observe_step=lambda _sim: None),
        opponents=[[], []],
    )
    source = SimpleNamespace(seed=7, hero=SimpleNamespace(net=object()))
    observed: dict[str, Any] = {}

    monkeypatch.setattr(teacher, "_replicate", lambda _source, _count: replica)
    monkeypatch.setattr(eval_engine, "_dispatch_actions", lambda *_args: None)

    def reference(_source: Any, actions: list[int], config: Any, decision: int) -> dict[int, float]:
        observed.update(actions=actions, config=config, decision=decision)
        return {0: 12.0, 1: 13.0}

    monkeypatch.setattr(teacher, "_reference_scores", reference)
    config = teacher.SearchConfig(rollouts=1, horizon=1, length_gate=0)
    assert teacher._batched_scores(source, [0, 1], config, 31) == {0: 12.0, 1: 13.0}
    assert observed == {"actions": [0, 1], "config": config, "decision": 31}
