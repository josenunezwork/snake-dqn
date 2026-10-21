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


def test_decisive_audit_uses_cpu_survival_gain_not_candidate_choice() -> None:
    from research.gpu_sim_spike.parity import summarize

    reference = _label(1)
    reference["greedy"] = 0
    reference["outcomes"][1]["alive_steps"] = 3
    mismatch = compare_decision(reference, _label(0))
    assert mismatch["decisive_root"]
    assert mismatch["cpu_survival_gain_over_greedy"] == 1
    other = compare_decision(_label(0), _label(0))
    audit = summarize([mismatch, other])
    assert audit["first_action_agreement"] == 0.5
    assert audit["decisive_roots"]["decisions"] == 1
    assert audit["decisive_roots"]["first_action_agreement"] == 0
    assert audit["decisive_roots"]["classification_counts"]["bug"] == 1
    assert audit["other_roots"]["first_action_agreement"] == 1
    absent = summarize([other])["decisive_roots"]
    assert absent["coverage"] == "NO_COVERAGE"
    assert absent["first_action_agreement"] is None


def test_receipt_sidecars_are_discovered_and_verified_by_supervisor(tmp_path) -> None:
    import hashlib

    from research.colab_cli.runtime import _exports
    from research.gpu_sim_spike.benchmark import write_receipt

    target = tmp_path / "output" / "benchmark.json"
    write_receipt(target, {"status": "PREFLIGHT"})
    assert target.with_name("benchmark.json.complete").read_text() == "complete\n"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.with_name("benchmark.json.sha256").read_text() == f"{digest}  benchmark.json\n"
    assert _exports(tmp_path, 1024) == [
        {"path": "benchmark.json", "sha256": digest, "size": target.stat().st_size}
    ]
    with pytest.raises(FileExistsError):
        write_receipt(target, {"status": "OVERWRITE"})
    target.with_name("benchmark.json.sha256").unlink()
    with pytest.raises(RuntimeError, match="invalid completed export marker"):
        _exports(tmp_path, 1024)


def test_full_world_throughput_uses_six_branches_not_six_snake_rows(monkeypatch) -> None:
    from research.gpu_sim_spike.benchmark import throughput_report

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA L4")
    rows = [
        {"case": name, "full_world_steps": 1800, "device_loop_seconds": 0.09}
        for name in ("long-scripted", "long-frozen", "long-mixed")
    ]
    report = throughput_report(rows, device="cuda", pipeline_seconds=1)
    assert report["world_steps"] == 5400
    assert report["full_world_steps_per_second"] == pytest.approx(20000)
    assert report["pipeline_world_steps_per_second"] == 5400
    assert report["l4_floor_status"] == "DIAGNOSTIC_ONLY_REQUIRES_REALISTIC_PRODUCTION_SHAPE"
    rows[0]["device_loop_seconds"] = 0.1
    assert (
        throughput_report(rows, device="cuda", pipeline_seconds=1)["l4_floor_status"]
        == "DIAGNOSTIC_ONLY_REQUIRES_REALISTIC_PRODUCTION_SHAPE"
    )
    assert (
        throughput_report(rows, device="cpu", pipeline_seconds=1)["l4_floor_status"]
        == "NOT_L4_DATA"
    )
