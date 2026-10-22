"""Mac preflight of the exact P1 phase entrypoints, never scientific evidence.

Only fixture dimensions, calibration cost and critic predictions at the gate are
substituted. Real serpentine states and complete-life targets enter the bank.
The phase parser/executor, control admission, production training
loop, real B8 CNN/PPO updates, spawned native-v8 worlds, E8 threshold function,
transitive snapshots/restore, action tapes and fingerprints all execute.
There is no production tiny-mode switch or environment-based gate bypass.
"""

from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path

import pytest
import torch

from research.bl_ppo import p1_contract, p1_runtime, run
from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.learner import PPOLearner
from research.bl_ppo.recovery import read_snapshot
from research.bl_ppo.test_p1_runtime import _control

CRITIC_QUALITY = PPOLearner.critic_quality


@pytest.fixture
def tiny_p1(monkeypatch):
    """Scale execution only inside pytest; preserve every production boundary."""
    from research.bl_ppo import collector, diagnostics, late_body, learner, monitor

    shape = p1_contract.ExecutionShape(
        target=1280,
        worlds=2,
        processes=1,
        rollout_steps=16,
        qualification_requested=129,
        tranche_size=256,
    )
    monkeypatch.setattr(p1_contract, "SHAPE", shape)
    monkeypatch.setattr(run, "HORIZON", 16)
    monkeypatch.setattr(run, "CRITIC_REFIT_TRANSITIONS", 64)
    monkeypatch.setattr(late_body, "FIXED_LATE_BODY_WORLDS", 2)
    monkeypatch.setattr(run, "_calibrated_temperature", lambda args: 0.7)
    # Keep class identities intact for actual spawn and snapshot pickling.
    config_post_init = collector.CollectorConfig.__post_init__

    def tiny_collector_validation(self):
        object.__setattr__(self, "preflight", True)
        config_post_init(self)

    monkeypatch.setattr(collector.CollectorConfig, "__post_init__", tiny_collector_validation)
    monkeypatch.setattr(
        learner, "PPOLearner", functools.partial(learner.PPOLearner, critic_refit_transitions=64)
    )
    bank_init = monitor.FixedOrdinaryBank.__init__

    def tiny_bank_init(self, **kwargs):
        bank_init(self, capacity=16, **kwargs)

    monkeypatch.setattr(monitor.FixedOrdinaryBank, "__init__", tiny_bank_init)
    configure_deciles = diagnostics.E8Diagnostics.configure_length_deciles

    def tiny_deciles(self, lengths):
        # Scale the fixed-bank cardinality witness while retaining real decile
        # construction and diagnostic updates on the collected ordinary rows.
        return configure_deciles(self, (lengths * 625)[:10_000])

    monkeypatch.setattr(diagnostics.E8Diagnostics, "configure_length_deciles", tiny_deciles)

    def controlled_predictions(**inputs):
        # Test-only perfect predictions exercise PASS through the real bank and
        # unchanged EV/MSE rule. Lengths, worlds, suffix returns and baseline all
        # come from actual collection; none are replaced by fabricated vectors.
        assert (inputs["lengths"] >= 800).any()
        return CRITIC_QUALITY(**(inputs | {"predictions": inputs["returns"].clone()}))

    monkeypatch.setattr(
        learner.PPOLearner.func, "critic_quality", staticmethod(controlled_predictions)
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    monkeypatch.setenv("BL_PPO_DEVICE", "cpu")
    torch.set_num_threads(1)
    yield shape


def _entrypoint(tmp_path, monkeypatch, shape, phase, *, resume=None, proof=None, **changes):
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir(parents=True)
    control = _control(
        inputs,
        monkeypatch,
        **(
            dict(
                phase=phase,
                target=shape.target,
                worlds=shape.worlds,
                processes=shape.processes,
                rollout_steps=shape.rollout_steps,
                pause_at=0,
                requested_pause_at=0,
            )
            | changes
        ),
    )
    # Local stand-in for the supervisor's non-secret endpoint/generation file.
    # No provider is imported or called; these are expressly test identities.
    root = tmp_path / "supervisor"
    root.mkdir()
    (root / "spec.json").write_text(
        json.dumps(
            {
                "release_endpoint": f"TEST-ONLY-{tmp_path.name}",
                "generation": "test-generation",
            }
        )
    )
    monkeypatch.setenv("COLAB_CLI_ROOT", str(root))
    checkpoint = (
        Path(__file__).resolve().parents[3] / "snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"
    )
    assert checkpoint.is_file(), "exact-entrypoint preflight requires the released local B8 fixture"
    argv = [
        phase,
        "--control",
        str(control),
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
    ]
    if resume:
        argv += ["--resume", str(resume)]
    if proof:
        argv += ["--expected-trace", str(proof)]
    return inputs, output, argv


def _status(output, phase):
    return json.loads((output / phase / "training-status.json").read_text())


def _snapshot_path(output, phase):
    value = _status(output, phase)
    return output / phase / value["snapshot"]["path"]


def _copy_export(source, target):
    verify_export(source)
    if target.exists():
        assert digest(target) == digest(source), "transitive flat-name collision"
        return
    for suffix in ("", ".sha256", ".complete"):
        shutil.copyfile(str(source) + suffix, str(target) + suffix)
    verify_export(target)


def test_exact_y3_entrypoints_and_every_pause_resume(tmp_path, monkeypatch, tiny_p1):
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
    assert support["fixed_holdout"]["worlds_complete"] == 2
    assert support["holdout_late_by_source"]["fixed_injected_long_body"] >= 2
    assert support["refit_support_supplement"]["used"] is True
    assert support["fit_late_by_source"]["refit_long_body_support_supplement"] > 0
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


def test_exact_source_refuses_failed_e8_at_zero(tmp_path, monkeypatch, tiny_p1):
    from research.bl_ppo import learner

    def refused_predictions(**inputs):
        assert (inputs["lengths"] >= 800).any()
        error = 1000.0 * torch.arange(len(inputs["returns"]), dtype=torch.float32)
        return CRITIC_QUALITY(**(inputs | {"predictions": -inputs["returns"] - 100.0 - error}))

    monkeypatch.setattr(
        learner.PPOLearner.func, "critic_quality", staticmethod(refused_predictions)
    )
    _, output, argv = _entrypoint(tmp_path / "failure", monkeypatch, tiny_p1, "y3-source")
    with pytest.raises(ValueError, match="NOT_READY: E8 held-out critic gate"):
        p1_runtime.main(argv)
    value = _status(output, "origin")
    assert value["status"] == "INCOMPLETE" and value["pause_at"] == 0
    assert value["counters"]["actor_transitions"] == 0
    assert value["counters"]["actor_updates"] == 0
    heldout = value["critic_refit"]["heldout_late_body"]
    assert heldout["count"] > 0
    assert heldout["explained_variance"] < 0.1 and heldout["mse"] >= heldout["baseline_mse"]
    assert "snapshot" not in value and not (output / "receipt.json").exists()
