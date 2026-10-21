"""Narrow spike harness checks: no hidden acceptance of unknown divergence."""

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.parity import compare_decision
from research.gpu_sim_spike.rollout import select_dense


def _label(action: int) -> dict:
    return {
        "action": action,
        "q": [1.0] * 6,
        "outcomes": {a: {"alive_steps": 2, "mass_integral": 10.0} for a in range(6)},
    }


def test_unknown_disagreement_is_bug_even_when_rng_differs() -> None:
    result = compare_decision(_label(0), _label(1), evidence={"counter_rng": True})
    assert result["classification"] == "bug"


def test_tie_requires_identical_branch_evidence() -> None:
    result = compare_decision(_label(0), _label(1), evidence={"identical_branch_outcomes": True})
    assert result["classification"] == "float tie"
    result = compare_decision(
        _label(0), _label(1), evidence={"matched_event_replay_recovers_label": True}
    )
    assert result["classification"] == "RNG-stream difference"


def test_dense_survival_beats_q_but_bare_wins_survival_tie() -> None:
    q = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    legal = torch.ones((1, 6), dtype=torch.bool)
    assert select_dense(q, legal, torch.tensor([[1, 2, 2, 0, 0, 0]])).item() == 1
    assert select_dense(q, legal, torch.tensor([[2, 2, 2, 0, 0, 0]])).item() == 0
    legal[:, 0] = False
    assert select_dense(q, legal, torch.tensor([[3, 2, 2, 0, 0, 0]])).item() == 1


def test_all_illegal_fallback_excludes_boost() -> None:
    q = torch.tensor([[1.0, 3.0, 2.0, 99.0, 99.0, 99.0]])
    legal = torch.zeros((1, 6), dtype=torch.bool)
    assert select_dense(q, legal, torch.zeros((1, 6), dtype=torch.int64)).item() == 1


def test_missing_branch_fails_instead_of_inflating_agreement() -> None:
    candidate = _label(0)
    del candidate["outcomes"][1]
    with pytest.raises(ValueError, match="missing legal branch"):
        compare_decision(_label(0), candidate)


def test_value_error_uses_full_horizon_mass() -> None:
    reference = _label(0)
    candidate = _label(0)
    candidate["outcomes"][2]["mass_integral"] = 0.0
    result = compare_decision(reference, candidate)
    assert np.isclose(result["mean_absolute_mass_integral_error"], 10 / 6)


def test_sparse_source_commit_uses_manifest_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from research.gpu_sim_spike import benchmark

    pinned = "a" * 40
    monkeypatch.setenv("GPU_SIM_SPIKE_COMMIT", pinned)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=128, stdout="", stderr="not a git repository"),
    )
    assert benchmark.source_commit() == pinned


def test_source_commit_rejects_pin_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from research.gpu_sim_spike import benchmark

    monkeypatch.setenv("GPU_SIM_SPIKE_COMMIT", "a" * 40)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="b" * 40, stderr=""),
    )
    with pytest.raises(ValueError, match="differs from manifest pin"):
        benchmark.source_commit()


def test_candidate_nonfinite_q_cannot_count_as_agreement() -> None:
    from research.gpu_sim_spike.benchmark import _candidate_json

    result = {
        "action": torch.zeros(1, dtype=torch.long),
        "q": torch.tensor([[float("nan"), 0, 0, 0, 0, 0]]),
        "legal": torch.ones((1, 6), dtype=torch.bool),
        "alive_steps": torch.ones((1, 6), dtype=torch.long),
        "mass_integral": torch.ones((1, 6)),
        "action_trace": torch.ones((1, 6, 2, 6), dtype=torch.long),
    }
    with pytest.raises(ValueError, match="nonfinite"):
        _candidate_json(result, depth=2, snakes=6)
