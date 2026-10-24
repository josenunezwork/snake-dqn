"""Focused I0 parity-helper checks; the benchmark invokes ``prove`` separately."""

from types import SimpleNamespace

import pytest
import torch

from research.gpu_sim_spike import integration_parity as parity


def test_frozen_move_clone_is_independent_and_preserves_padding_bytes():
    source = SimpleNamespace(
        bodies=torch.tensor([[[[1, 2], [-7, -7]]]], dtype=torch.long),
        move_backend="reference",
        cfg=SimpleNamespace(name="fixture"),
    )
    frozen_type = type("Frozen", (), {})
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(parity, "_frozen_move_type", lambda: frozen_type)
        clone = parity.frozen_move_clone(source)
    assert type(clone) is frozen_type
    assert clone.bodies.numpy().tobytes() == source.bodies.numpy().tobytes()
    clone.bodies[0, 0, 1] = 99
    assert source.bodies[0, 0, 1].tolist() == [-7, -7]


def test_state_match_requires_complete_frozen_comparator(monkeypatch):
    monkeypatch.setattr(
        parity.move2_parity, "compare", lambda left, right: {"match": True, "fields": 3}
    )
    assert parity._state_match(object(), object(), "I")["fields"] == 3
    monkeypatch.setattr(
        parity.move2_parity, "compare", lambda left, right: {"match": False, "stage": "padding"}
    )
    with pytest.raises(parity.IntegrationParityError, match="padding"):
        parity._state_match(object(), object(), "I")


def test_prove_requires_exact_two_fresh_arms():
    with pytest.raises(ValueError, match="exactly R and I"):
        parity.prove({}, "cpu")


def test_cpu_discrepancy_register_requires_the_established_frozen_report(monkeypatch, tmp_path):
    from research.gpu_sim_spike import move2_bench

    report = {"status": "DIAGNOSTIC", "rows": [{"mix": "frozen"}]}
    monkeypatch.setattr(move2_bench, "_cpu_discrepancies", lambda checkpoint, device: report)
    assert parity.cpu_discrepancy_register(tmp_path / "b8.pth") is report
    monkeypatch.setattr(
        move2_bench, "_cpu_discrepancies", lambda checkpoint, device: {"status": "INVALID"}
    )
    with pytest.raises(parity.IntegrationParityError, match="invalid"):
        parity.cpu_discrepancy_register(tmp_path / "b8.pth")
