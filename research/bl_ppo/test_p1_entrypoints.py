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
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.bl_ppo import p1_contract, p1_runtime, run
from research.bl_ppo.exports import digest, verify_export
from research.bl_ppo.learner import PPOLearner
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
    monkeypatch.setattr(run, "MAX_CRITIC_REFIT_FIT_TRANSITIONS", 128)
    monkeypatch.setattr(late_body, "FIXED_LATE_BODY_WORLDS", 2)
    fixed_config = late_body.fixed_holdout_config

    def one_process_fixture(*args, **kwargs):
        return replace(fixed_config(*args, **kwargs), processes=1)

    monkeypatch.setattr(late_body, "fixed_holdout_config", one_process_fixture)
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

    def controlled_predictions(*args, **inputs):
        # Test-only perfect predictions exercise PASS through the real bank and
        # unchanged EV/MSE rule. Lengths, worlds, suffix returns and baseline all
        # come from actual collection; none are replaced by fabricated vectors.
        if args:
            predictions, returns, lengths, remaining_horizon = args
            inputs = inputs | {
                "predictions": returns.clone(),
                "returns": returns,
                "lengths": lengths,
                "remaining_horizon": remaining_horizon,
            }
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


def test_exact_source_quarantines_fixture_support(tmp_path, monkeypatch, tiny_p1):
    """The exact runner refuses when only constructed fixture support exists."""
    shape = tiny_p1
    _, source, argv = _entrypoint(tmp_path / "source", monkeypatch, shape, "y3-source")
    with pytest.raises(ValueError, match="NOT_READY: E8 held-out critic gate"):
        p1_runtime.main(argv)
    origin_status = _status(source, "origin")
    support = origin_status["critic_refit"]["late_body_support"]
    assert origin_status["critic_qualified"] is False
    assert origin_status["counters"]["actor_transitions"] == 0
    assert support["fixture_late_body"]["worlds_complete"] == 2
    assert all(
        row["provenance"] == "natural" and row["length_bucket"] == "<800"
        for row in support["holdout_by_provenance_mix_length_horizon"]
    )
    assert support["fixture_by_provenance_mix_length_horizon"]
    assert origin_status["critic_refit"]["status"] == "INSUFFICIENT_LATE_BODY_SUPPORT"
    assert origin_status["late_body_rows_sidecar"] is not None
    assert not (source / "receipt.json").exists()


def test_exact_source_records_one_support_continuation_on_refusal(tmp_path, monkeypatch, tiny_p1):
    _, output, argv = _entrypoint(tmp_path / "failure", monkeypatch, tiny_p1, "y3-source")
    with pytest.raises(ValueError, match="NOT_READY: E8 held-out critic gate"):
        p1_runtime.main(argv)
    value = _status(output, "origin")
    assert value["status"] == "INCOMPLETE" and value["pause_at"] == 0
    assert value["counters"]["actor_transitions"] == 0
    assert value["counters"]["actor_updates"] == 0
    heldout = value["critic_refit"]["heldout_late_body"]
    assert heldout["metrics_available"] is False
    assert set(("explained_variance", "mse", "baseline_mse", "bias")) <= set(heldout)
    continuation = value["critic_refit"]["late_body_support"]["support_continuation"]
    assert continuation["used"] is True
    assert continuation["status"] == "NOT_READY(P1 support)"
    assert value["counters"]["critic_refit_fit_transitions"] <= 128
    assert "snapshot" not in value and not (output / "receipt.json").exists()


def test_exact_source_nonfinite_refit_keeps_kill_status_and_rows(tmp_path, monkeypatch, tiny_p1):
    def nonfinite_fit(self, features, returns, *, batch_size=512):
        raise FloatingPointError("non-finite critic refit")

    monkeypatch.setattr(PPOLearner, "fit_completed_critic", nonfinite_fit)
    _, output, argv = _entrypoint(tmp_path / "nonfinite", monkeypatch, tiny_p1, "y3-source")
    with pytest.raises(ValueError, match="NOT_READY: E8 held-out critic gate"):
        p1_runtime.main(argv)
    value = _status(output, "origin")
    assert value["status"] == "KILL"
    assert value["critic_qualified"] is False
    assert value["late_body_rows_sidecar"] is not None
    assert any(row["kind"] == "nonfinite" for row in value["tripwires"])
