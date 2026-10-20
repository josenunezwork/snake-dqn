"""Real-fixture checks for the T33 teacher at the prepared Watch seam.

These tests use the audited timing fixtures and the independently frozen
pre-45069720 oracle.  The oracle is loaded by path and its source digest is
checked before every comparison so a shared import cannot hide a regression.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from research.phase0_search_20261019 import teacher as phase0_teacher
from research.phase0_search_20261019.preflight import fixture
from research.t33_batched_teacher.teacher import BatchedTeacher, merge_worlds

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests/fixtures/phase0_teacher_speed/teacher_pre_45069720.py"
ORACLE_SHA256 = "d15960f0dab45b3e08d49549990620e341a0e6b2c665654ebf669b43d9d14f17"
B8_CHECKPOINT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
)
WORLD_BANK = ROOT / "research/phase0_search_20261019/WORLD_BANK.json"


def _load_oracle() -> ModuleType:
    """Load the frozen Phase 0 scorer without importing the candidate module."""
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256
    spec = importlib.util.spec_from_file_location("t33_phase0_oracle", ORACLE_PATH)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = oracle
    spec.loader.exec_module(oracle)
    oracle.configure_cpu()
    return oracle


@pytest.fixture(scope="module")
def oracle() -> ModuleType:
    """Return the independently frozen pre-speedup Phase 0 implementation."""
    if not B8_CHECKPOINT.is_file():
        pytest.skip("the hash-pinned local B8 checkpoint inventory is unavailable")
    phase0_teacher.configure_cpu()
    return _load_oracle()


@pytest.fixture(scope="module")
def timing_rows() -> dict[str, dict[str, Any]]:
    """Return the three audited timing rows, keyed by mix."""
    bank = json.loads(WORLD_BANK.read_text(encoding="utf-8"))
    return {str(row["mix"]): row for row in bank["fixtures"]["timing"]}


class _PreparedBoundary(Exception):
    """Stop a real simulator exactly after Watch preparation."""


def _prepared_world(row: dict[str, Any], length: int) -> Any:
    """Build and freeze a real fixture at the next prepared Watch boundary."""
    world = fixture(row, B8_CHECKPOINT, length)

    def capture(prepared: Any) -> np.ndarray:
        world.runtime.prepare(prepared)
        raise _PreparedBoundary

    try:
        world.sim.step_with_policy(capture)
    except _PreparedBoundary:
        pass
    world.sim._policy_phase_prepared = True
    world.sim._policy_step_in_progress = False
    assert getattr(world.sim, "_policy_phase_prepared", False)
    return world


def _source_digest(world: Any) -> str:
    """Digest simulator, RNG-backed arrays, and Vector61 carry state."""
    from research.redesign_scope_20261007 import grid_h5000_identity as identity

    digest = hashlib.sha256(identity.env_digest(world.sim, 0).encode("ascii"))

    def add_arrays(prefix: str, value: Any) -> None:
        for name, item in sorted(vars(value).items()):
            if isinstance(item, np.ndarray):
                digest.update(prefix.encode("utf-8"))
                digest.update(name.encode("utf-8"))
                digest.update(str(item.dtype).encode("ascii"))
                digest.update(repr(item.shape).encode("ascii"))
                digest.update(np.ascontiguousarray(item).tobytes())

    add_arrays("runtime.", world.runtime)
    if getattr(world.runtime, "memory", None) is not None:
        add_arrays("runtime.memory.", world.runtime.memory)
    return digest.hexdigest()


def _oracle_diagnostic(oracle: ModuleType, world: Any, config: Any) -> dict[int, float]:
    """Score all domain-legal first actions with the frozen rowwise oracle."""
    legal = list(map(int, np.flatnonzero(world.sim.get_legal_action_mask()[0, 0])))
    decision = int(world.sim.frame[0]) - 1
    return oracle._batched_scores(world, legal, config, decision)


def _assert_scores_match(candidate: dict[int, float], reference: dict[int, float]) -> None:
    assert tuple(candidate) == tuple(reference)
    np.testing.assert_allclose(
        [candidate[action] for action in candidate],
        [reference[action] for action in reference],
        rtol=0.0,
        atol=1e-3,
    )


def test_two_distinct_real_sources_match_oracle_and_preserve_source_state(
    oracle: ModuleType, timing_rows: dict[str, dict[str, Any]]
) -> None:
    """A two-source call agrees with individually scored Phase 0 fixtures."""
    config = phase0_teacher.SearchConfig(rollouts=1, horizon=4, epsilon=0.0, length_gate=0)
    worlds = [
        _prepared_world(timing_rows["scripted"], 800),
        _prepared_world(timing_rows["frozen"], 800),
    ]
    before = [_source_digest(world) for world in worlds]
    actions, diagnostics = BatchedTeacher(config, common_random=False).choose(worlds)

    for world, action, diagnostic, initial_digest in zip(worlds, actions, diagnostics, before):
        reference_scores = _oracle_diagnostic(oracle, world, config)
        _assert_scores_match(diagnostic["scores"], reference_scores)
        assert int(action) == max(reference_scores, key=reference_scores.__getitem__)
        assert diagnostic["searched"]
        assert diagnostic["common_random"] is False
        assert _source_digest(world) == initial_digest

    # Actions 2 and 5 die on this real fixture; zero totals prove dead rows are
    # retained in the fixed resident batch but do not contribute mass.
    assert any(diagnostic["scores"][2] == 0.0 for diagnostic in diagnostics)
    assert any(diagnostic["scores"][5] == 0.0 for diagnostic in diagnostics)


@pytest.mark.parametrize("mix", ["scripted", "frozen", "mixed"])
def test_full_real_n8_h128_source_matches_frozen_oracle(
    oracle: ModuleType, timing_rows: dict[str, dict[str, Any]], mix: str
) -> None:
    """The registered full search agrees on one real fixture, including deaths."""
    config = phase0_teacher.SearchConfig(rollouts=8, horizon=128, epsilon=0.0, length_gate=0)
    world = _prepared_world(timing_rows[mix], 800)
    before = _source_digest(world)
    candidate_teacher = BatchedTeacher(config, common_random=False)
    actions, diagnostics = candidate_teacher.choose([world])
    reference_scores = _oracle_diagnostic(oracle, world, config)

    _assert_scores_match(diagnostics[0]["scores"], reference_scores)
    assert int(actions[0]) == max(reference_scores, key=reference_scores.__getitem__)
    assert _source_digest(world) == before
    assert diagnostics[0]["resident_rows"] == 6 * config.rollouts
    assert candidate_teacher.profile()["steps"] == config.horizon


def test_two_source_concat_isolation_and_repeat_determinism(
    timing_rows: dict[str, dict[str, Any]],
) -> None:
    """Grouping two distinct policy rosters cannot change either source result."""
    config = phase0_teacher.SearchConfig(rollouts=2, horizon=16, epsilon=0.0, length_gate=0)
    combined_worlds = [
        _prepared_world(timing_rows["mixed"], 800),
        _prepared_world(timing_rows["frozen"], 800),
    ]
    combined_teacher = BatchedTeacher(config, common_random=False)
    combined_actions, combined_diags = combined_teacher.choose(combined_worlds)
    repeated_actions, repeated_diags = combined_teacher.choose(combined_worlds)

    np.testing.assert_array_equal(combined_actions, repeated_actions)
    for first, second in zip(combined_diags, repeated_diags):
        assert first["scores"] == second["scores"]
        np.testing.assert_array_equal(first["q"], second["q"])

    for index, mix in enumerate(("mixed", "frozen")):
        solo = _prepared_world(timing_rows[mix], 800)
        solo_teacher = BatchedTeacher(config, common_random=False)
        solo_actions, solo_diags = solo_teacher.choose([solo])
        assert int(combined_actions[index]) == int(solo_actions[0])
        assert tuple(combined_diags[index]["scores"]) == tuple(solo_diags[0]["scores"])
        np.testing.assert_allclose(
            list(combined_diags[index]["scores"].values()),
            list(solo_diags[0]["scores"].values()),
            rtol=0.0,
            atol=1e-3,
        )


def test_prepared_seam_and_illegal_rows_are_explicitly_enforced(
    timing_rows: dict[str, dict[str, Any]],
) -> None:
    """The merge contract rejects raw worlds and records inactive illegal rows."""
    raw = fixture(timing_rows["scripted"], B8_CHECKPOINT, 800)
    with pytest.raises(ValueError, match="prepared Watch boundary"):
        merge_worlds([raw], rollouts=1)

    prepared = _prepared_world(timing_rows["scripted"], 800)
    legal = np.array([True, False, True, False, True, False], dtype=bool)
    merged = merge_worlds([prepared], rollouts=1, legal=[legal])
    np.testing.assert_array_equal(
        merged.layout.active,
        np.array([True, False, True, False, True, False], dtype=bool),
    )
    assert merged.sim.E == 6
    assert merged.layout.source.tolist() == [0, 0, 0, 0, 0, 0]
    assert merged.layout.action.tolist() == list(range(6))


@pytest.mark.parametrize("leaf", ["max", "value"])
def test_crn_teacher_matches_rowwise_calibration_reference(timing_rows, leaf) -> None:
    from research.t33_batched_teacher import reference

    config = phase0_teacher.SearchConfig(rollouts=2, horizon=8, epsilon=0.0, length_gate=0)
    worlds = [_prepared_world(timing_rows[mix], 800) for mix in ("scripted", "frozen")]
    search = BatchedTeacher(config, common_random=True, leaf=leaf)
    actions, diagnostics = search.choose(worlds)
    repeated, repeated_diagnostics = search.choose(worlds)
    assert actions.tobytes() == repeated.tobytes()
    for world, action, diag, again in zip(worlds, actions, diagnostics, repeated_diagnostics):
        legal = list(map(int, np.flatnonzero(world.sim.get_legal_action_mask()[0, 0])))
        expected = reference.scores(world, legal, config, int(world.sim.frame[0]) - 1, leaf=leaf)
        _assert_scores_match(diag["scores"], expected)
        assert action == max(expected, key=expected.__getitem__)
        assert diag["scores"] == again["scores"]
