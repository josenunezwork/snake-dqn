"""Excluded software proof of amended P1 replay and all execution boundaries.

The positive path seeds synthetic complete natural-labelled bank rows solely
inside pytest because a 16-frame fixture cannot naturally grow to L800. This
substitution exercises admission and durable state; it supplies no scientific
natural-support, critic-accuracy, live Y3 or qualification evidence. The actual
main/run_train/collector/model/update/snapshot/recovery/flat-transport paths run.
The separate unseeded source tests prove fixture-only data refuse admission.
"""

from __future__ import annotations

import json

import pytest
import torch

from research.bl_ppo import p1_runtime, run
from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.learner import PPOLearner
from research.bl_ppo.recovery import read_snapshot
from research.bl_ppo.refit import CompleteLifeRefitBank
from research.bl_ppo.test_p1_entrypoints import (
    _copy_export,
    _entrypoint,
    _snapshot_path,
    _status,
)
from research.bl_ppo.test_p1_entrypoints import tiny_p1 as _tiny_p1_fixture

tiny_p1 = _tiny_p1_fixture
_REGISTERED_QUALITY = PPOLearner.critic_quality


@pytest.fixture(autouse=True)
def synthetic_natural_support(monkeypatch, tiny_p1):
    original = CompleteLifeRefitBank.__init__
    constructed = 0

    def seed_primary(self, *args, **kwargs):
        nonlocal constructed
        original(self, *args, **kwargs)
        primary = constructed % 2 == 0
        constructed += 1
        if not primary:
            return
        for split, count, offset in (("fit", 160, 10001), ("holdout", 40, 10000)):
            for index in range(count):
                world = offset + 5 * index
                features = torch.zeros(4, 257)
                features[:, -1] = torch.tensor([0.95, 0.75, 0.5, 0.25])
                self.add_batch(
                    torch.full((4,), world, dtype=torch.int64),
                    features,
                    torch.tensor([900.0, 1000.0, 1100.0, 1200.0]),
                    features[:, -1],
                    torch.full((4,), float(1 + index % 7)),
                    torch.tensor([False, False, False, True]),
                    mixes=[("scripted", "frozen", "mixed")[index % 3]] * 4,
                )

    monkeypatch.setattr(CompleteLifeRefitBank, "__init__", seed_primary)


def test_amended_y3_replays_full_state_and_every_pause(tmp_path, monkeypatch, tiny_p1):
    """This test must fail if main, run_train or recovery is replaced by a stub."""
    shape = tiny_p1
    _, source, argv = _entrypoint(tmp_path / "source", monkeypatch, shape, "y3-source")
    assert p1_runtime.main(argv) == 0
    origin = _snapshot_path(source, "origin")
    origin_state = read_snapshot(origin)
    assert origin_state["counters"]["actor_transitions"] == 0
    assert origin_state["counters"]["actor_updates"] == 0
    assert origin_state["counters"]["actor_enabled"] is True
    origin_status = _status(source, "origin")
    assert origin_status["critic_qualified"] is True
    support = origin_status["critic_refit"]["late_body_support"]
    assert support["fixture_late_body"]["worlds_complete"] == 2
    assert support["fixture_diagnostic"]["diagnostic_only"] is True
    assert support["refit_support_supplement"]["used"] is False
    assert support["support_continuation"]["used"] is False
    assert origin_status["critic_refit"]["holdout_support"]["meets_minimum"] is True
    assert "hero_length_histogram_by_frame" in origin_status["e8_diagnostics"]
    assert "fixture_late_body_bank" in origin_state["adapter"]
    assert origin_status["critic_refit"]["heldout_late_body"]["count"] > 0
    # The primary worlds continue naturally after support-only collection.
    assert all((shard["sim"].length[:, 0] < 800).all() for shard in origin_state["collector"])
    assert origin_state["boundary"] == {
        "requested_pause_at": 0,
        "pause_at": 0,
        "actual_pause_at": 0,
        "kind": "exact",
    }
    assert not list(source.rglob("endpoint.pt"))

    # Exercise the actual flat, transitive transport seam. The replacement has
    # no dependency on source directory layout or in-memory source objects.
    transport = tmp_path / "transport"
    transport.mkdir()
    receipt = json.loads((source / "receipt.json").read_text())
    for row in receipt["transitive_snapshot_catalog"]:
        for candidate in source.rglob(row["name"]):
            _copy_export(candidate, transport / row["name"])
    _copy_export(source / "source-continuation/action-trace.json", transport / "action-trace.json")
    _copy_export(source / "receipt.json", transport / "receipt.json")
    _, replacement, argv = _entrypoint(
        tmp_path / "replacement",
        monkeypatch,
        shape,
        "y3-replacement",
        resume=transport / origin.name,
        proof=transport / "receipt.json",
    )
    assert p1_runtime.main(argv) == 0
    y3 = json.loads((replacement / "receipt.json").read_text())
    assert y3["status"] == "PASS"  # TEST ONLY, never a live L4 Y3 PASS.
    assert y3["actions_compared"] == 1000
    assert y3["actor_updates"] >= 3 and y3["episode_boundaries"] >= 1
    assert y3["full_tape_actions_compared"] >= 1000
    assert y3["source_continuation_state_fingerprint"] == y3["replacement_state_fingerprint"]
    assert not list(replacement.rglob("endpoint.pt"))

    inputs, qualification, argv = _entrypoint(
        tmp_path / "qual",
        monkeypatch,
        shape,
        "qual-1m",
        resume=transport / origin.name,
        **{
            "pause_at": shape.qualification_pause,
            "requested_pause_at": shape.qualification_requested,
            "prerequisites": {"y3": {"sha256": digest(replacement / "receipt.json")}},
        },
    )
    # Renaming creates new transport-name sidecars exactly as bundle_inputs does.
    export_json(inputs / "admission.json", y3)
    assert p1_runtime.main(argv) == 0
    status = _status(qualification, "qual-1m")
    assert status["requested_pause_at"] == 129 and status["actual_pause_at"] == 130
    assert (
        read_snapshot(_snapshot_path(qualification, "qual-1m"))["boundary"]["actual_pause_at"]
        == 130
    )
    assert not list(qualification.rglob("endpoint.pt"))

    from research.bl_ppo.test_p1_manifests import qualified

    previous_snapshot = _snapshot_path(qualification, "qual-1m")
    previous_receipt = json.loads((qualification / "receipt.json").read_text())
    admission = qualified() | {
        "source_commit": "a" * 40,
        "snapshot": {"sha256": digest(previous_snapshot)},
    }
    for chunk in range(1, 6):
        start = shape.qualification_pause if chunk == 1 else (chunk - 1) * shape.tranche_size
        pause = chunk * shape.tranche_size
        # export_json uses sorted JSON + newline; bind actual payload bytes.
        temp = tmp_path / f"admission-{chunk}.json"
        export_json(temp, admission)
        previous_temp = tmp_path / f"previous-{chunk}.json"
        export_json(previous_temp, previous_receipt)
        inputs, output, argv = _entrypoint(
            tmp_path / f"chunk-{chunk}",
            monkeypatch,
            shape,
            "tranche",
            resume=previous_snapshot,
            **{
                "start_at": start,
                "pause_at": pause,
                "requested_pause_at": pause,
                "chunk_index": chunk,
                "previous_receipt_sha256": digest(previous_temp),
                "prerequisites": {"qualification": {"sha256": digest(temp)}},
            },
        )
        export_json(inputs / "admission.json", admission)
        export_json(inputs / "previous.json", previous_receipt)
        assert p1_runtime.main(argv) == 0
        state = _status(output, "tranche")
        assert state["counters"]["actor_transitions"] == pause
        assert (
            state["counters"]["critic_refit_transitions"]
            == origin_state["counters"]["critic_refit_transitions"]
        )
        assert state["status"] == ("COMPLETE" if chunk == 5 else "PAUSED")
        assert (output / "tranche/endpoint.pt").exists() == (chunk == 5)
        previous_snapshot = _snapshot_path(output, "tranche")
        previous_receipt = json.loads((output / "receipt.json").read_text())


def test_required_row_export_failure_invalidates_qualified_pause(tmp_path, monkeypatch, tiny_p1):
    def failed_export(*_args, **_kwargs):
        raise OSError("test-only required rows export failure")

    monkeypatch.setattr(run, "_export_late_body_rows", failed_export)
    _, output, argv = _entrypoint(tmp_path / "rows-failure", monkeypatch, tiny_p1, "y3-source")
    with pytest.raises(ValueError):
        p1_runtime.main(argv)
    value = _status(output, "origin")
    assert value["status"] == "INCOMPLETE"
    assert value["late_body_rows_sidecar"] is None
    assert "test-only required rows export failure" in value["late_body_rows_sidecar_error"]
    assert not (output / "receipt.json").exists()


def test_sufficient_natural_support_accuracy_failure_never_extends(tmp_path, monkeypatch, tiny_p1):
    def failed_quality(*args, **inputs):
        if args:
            inputs.update(zip(("predictions", "returns", "lengths", "remaining_horizon"), args))
        return _REGISTERED_QUALITY(**(inputs | {"predictions": -inputs["returns"] - 1000.0}))

    monkeypatch.setattr(PPOLearner, "critic_quality", staticmethod(failed_quality))
    _, output, argv = _entrypoint(tmp_path / "accuracy-failure", monkeypatch, tiny_p1, "y3-source")
    with pytest.raises(ValueError, match="NOT_READY: E8 held-out critic gate"):
        p1_runtime.main(argv)
    value = _status(output, "origin")
    critic = value["critic_refit"]
    assert critic["status"] == "CRITIC_QUALITY_REFUSAL"
    assert critic["heldout_late_body"]["metrics_available"] is True
    assert critic["heldout_late_body"]["explained_variance"] < 0.1
    assert critic["late_body_support"]["support_continuation"]["used"] is False
    assert critic["late_body_support"]["refit_support_supplement"]["used"] is False
    assert value["counters"]["actor_transitions"] == 0
