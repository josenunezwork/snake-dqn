"""Focused unit tests for the frozen E3 fitting boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.bl_info_value import fit


def _arm(regret: float, worlds: dict[str, float], switch: float = 0.0) -> dict[str, object]:
    return {
        "decisive_mean_regret": regret,
        "decisive_world_regret": worlds,
        "unnecessary_switch_rate": switch,
    }


def _tiny_arrays() -> dict[str, np.ndarray]:
    rows = 6
    legal = np.zeros((rows, 6), dtype=bool)
    legal[:, :2] = True
    returns = np.zeros((rows, 6), dtype=np.float32)
    returns[:, 0] = 2.0
    returns[:, 1] = 1.0
    return {
        "local": np.zeros((rows, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((rows, 4, 37, 37), dtype=np.uint8),
        "scalars": np.zeros((rows, 60), dtype=np.float32),
        "geometry": np.zeros((rows, 7, 83, 145), dtype=np.float32),
        "history": np.zeros((rows, 128, 43), dtype=np.float32),
        "legal": legal,
        "returns": returns,
        "action": np.zeros(rows, dtype=np.int64),
        "decisive": np.array([1, 0, 1, 0, 1, 0], dtype=bool),
        "world": np.array(["w0", "w0", "w1", "w1", "w2", "w2"]),
        "split": np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        "decision": np.arange(rows, dtype=np.int64),
    }


def test_decision_requires_whole_world_reduction_and_counterweight_guard() -> None:
    a = _arm(10.0, {"w0": 10.0, "w1": 10.0, "w2": 10.0})
    b = _arm(7.0, {"w0": 7.0, "w1": 7.0, "w2": 7.0}, switch=0.01)
    c = _arm(6.0, {"w0": 6.0, "w1": 6.0, "w2": 6.0}, switch=0.0100001)
    decision = fit.decide_outcome({"A": a, "B": b, "C": c})
    assert decision["outcome"] == "GO"
    assert decision["qualifying_arms"] == ["B"]
    assert decision["selected_architecture"] == "B"


def test_decision_uses_world_equal_not_root_weighted_metrics() -> None:
    # B looks excellent if roots are pooled, but its three world means only
    # reduce A by 19%, below the frozen 20% threshold.
    a = _arm(10.0, {"w0": 10.0, "w1": 10.0, "w2": 10.0})
    b = _arm(8.1, {"w0": 8.1, "w1": 8.1, "w2": 8.1})
    c = _arm(10.0, {"w0": 10.0, "w1": 10.0, "w2": 10.0})
    assert fit.decide_outcome({"A": a, "B": b, "C": c})["outcome"] == "KILL"


def test_exact_twenty_and_ten_percent_boundaries_are_inclusive() -> None:
    a = _arm(10.0, {"w0": 10.0, "w1": 10.0})
    b = _arm(8.0, {"w0": 8.0, "w1": 8.0})
    c = _arm(7.2, {"w0": 7.2, "w1": 7.2})
    decision = fit.decide_outcome({"A": a, "B": b, "C": c})
    assert decision["qualifying_arms"] == ["B", "C"]
    assert decision["history_retained"]
    assert decision["selected_architecture"] == "C"


def test_validation_rejects_missing_leakage_and_duplicate_roots() -> None:
    arrays = _tiny_arrays()
    del arrays["history"]
    with pytest.raises(fit.IncompleteInput, match="missing arrays"):
        fit.validate_arrays(arrays, smoke=True)
    arrays = _tiny_arrays()
    arrays["decision"][1] = 0
    with pytest.raises(fit.IncompleteInput, match="duplicate"):
        fit.validate_arrays(arrays, smoke=True)
    arrays = _tiny_arrays()
    arrays["legal"][0] = False
    with pytest.raises(fit.IncompleteInput, match="resolved legal"):
        fit.validate_arrays(arrays, smoke=True)


def test_masked_actions_never_win() -> None:
    score = np.array([[0.0, 99.0, 2.0, 1.0, 0.0, 0.0]])
    legal = np.array([[True, False, True, False, False, False]])
    assert fit._masked_choice(score, legal).tolist() == [2]


def test_smoke_fit_exports_supervisor_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(6))

        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.weight.unsqueeze(0).expand(batch["local"].shape[0], -1)

    def make_students(seed: int = 0) -> dict[str, Tiny]:
        del seed
        return {name: Tiny() for name in ("A", "B", "C")}

    def loss(scores: torch.Tensor, returns: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        del returns, legal
        return scores.square().mean()

    import research.bl_info_value.students as students

    monkeypatch.setattr(students, "make_students", make_students)
    monkeypatch.setattr(students, "ranking_loss", loss)
    report = fit.fit_students(_tiny_arrays(), tmp_path, epochs=1, smoke=True)
    assert report["decision"]["outcome"] == "SMOKE_NOT_SCIENTIFIC"
    for name in ("fit_report.json", "student_A_best.pt", "student_B_best.pt", "student_C_best.pt"):
        assert (tmp_path / name).is_file()
        assert (tmp_path / (name + ".sha256")).is_file()
        assert (tmp_path / (name + ".complete")).is_file()


def test_fit_can_export_raw_claim_inputs_without_opening_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(6))

        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.weight.unsqueeze(0).expand(batch["local"].shape[0], -1)

    import research.bl_info_value.students as students

    monkeypatch.setattr(
        students, "make_students", lambda seed=0: {name: Tiny() for name in ("A", "B", "C")}
    )
    monkeypatch.setattr(
        students, "ranking_loss", lambda scores, _returns, _legal: scores.square().mean()
    )
    report = fit.fit_students(
        _tiny_arrays(), tmp_path, epochs=1, smoke=True, compute_decision=False
    )
    assert report["analysis_status"] == "UNOPENED_PENDING_PUBLISHED_LEDGER"
    assert "decision" not in report
