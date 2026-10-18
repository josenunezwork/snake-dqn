"""Deterministic unit tests for the Q0 profiled-evaluation metric adapter."""

from __future__ import annotations

import numpy as np
import pytest

from research.q0q1_20261017.audit_episode_metrics import FrameObserver


def _snapshot(frame: int, *, valid=(True, True), alive=(True, False), done=(False, True)):
    return {
        "frame": frame,
        "seeds": (11, 22),
        "pre": {
            "logical_mass": np.array([650, 810]),
            "resolved_masks": np.array(
                [[True, True, False, True, False, False], [True, False, False, False, False, False]]
            ),
            "actions": np.array([3, 1]),
        },
        "post": {
            "alive": np.array(alive),
            "logical_mass": np.array([651, 811]),
            "transition_valid": np.array(valid),
            "food_ate": np.array([True, True]),
            "boosted": np.array([True, False]),
            "done": np.array(done),
            "death_cause": np.array([0, 4]),
            "reward": np.array([2.5, -3.0]),
            "kills": np.array([1, 0]),
        },
    }


def test_frame_observer_uses_exact_post_events_and_strata() -> None:
    observer = FrameObserver(horizon=3)
    observer(_snapshot(0))
    observer(_snapshot(1, valid=(True, False), alive=(True, False), done=(False, False)))
    result = observer.result()

    first, second = result["episodes"]["11"], result["episodes"]["22"]
    assert result["complete_horizon"] is False
    assert first["decision_frames"] == 2
    assert first["food_mass_collected"] == 2
    assert first["boost_eligible_decisions"] == 2
    assert first["boost_executed_frames"] == 2
    assert first["reward_on_surviving_frames"] == 5.0
    assert first["mass_sum"] == 1302.0
    assert first["by_length"]["short_lt700"]["food_event_frames"] == 2
    assert first["by_space"]["open_3plus_legal_actions"]["decision_frames"] == 2
    assert second["decision_frames"] == 1
    assert second["terminal_padding_frames"] == 1
    assert second["food_mass_collected"] == 1  # death-step event remains visible
    assert second["mass_sum"] == 0.0  # terminal death step has zero mass
    assert second["reward_on_death_frames"] == -3.0
    assert second["death_causes"] == {"4": 1}
    assert second["by_length"]["long_ge800"]["death_events"] == 1
    assert observer.alive_frames == {11: 2, 22: 0}
    assert observer.death_frame == {11: None, 22: 1}


def test_frame_observer_fails_closed_on_bad_sequence_or_shape() -> None:
    observer = FrameObserver(horizon=2)
    with pytest.raises(ValueError, match="non-contiguous"):
        observer(_snapshot(1))
    broken = _snapshot(0)
    broken["pre"]["resolved_masks"] = np.ones((2, 5), dtype=bool)
    with pytest.raises(ValueError, match="resolved_masks"):
        observer(broken)


def test_evaluate_world_accepts_real_frame_observer_contract(monkeypatch, tmp_path):
    from research.t23_r0_screen_20261017 import evaluate
    from src.simd_env import eval_engine

    monkeypatch.setattr(evaluate, "HORIZON", 2)
    monkeypatch.setattr(evaluate, "_opponents", lambda *_: [])
    observer = FrameObserver(horizon=2)

    def injected(*args, **kwargs):
        for frame in range(2):
            snap = _snapshot(frame, done=(False, False))
            snap["seeds"] = (11,)
            for section in ("pre", "post"):
                snap[section] = {key: value[:1] for key, value in snap[section].items()}
            kwargs["frame_observer"](snap)
        return [
            {
                "denominators": {"alive_frames": 2, "scored_frames": 2},
                "deaths": 0,
                "mass_integral": 651.0,
                "survival_fraction": 1.0,
            }
        ]

    monkeypatch.setattr(eval_engine, "run_simd_eval", injected)
    world = {
        "world_seed": 11,
        "mix": "scripted",
        "world_semantic_sha256": "world",
        "roster_semantic_sha256": "roster",
    }
    row = evaluate.evaluate_world(
        world,
        tmp_path / "unused.pth",
        ("hero", "hero"),
        {"lookup": {}, "profile": object()},
        observer_factory=lambda _: observer,
    )
    assert row["mass_integral"] == observer.result()["episodes"]["11"]["mass_integral"]
    assert row["death_frame"] is None


def test_cumulative_mass_carries_terminal_zero_padding_and_policy_metrics():
    from types import SimpleNamespace

    probe = SimpleNamespace(
        pending={
            0: {
                "probe_decisions": 1,
                "greedy_boost_decisions": 1,
                "reward_gamma": 0.99,
                "reward_death_value": -3.0,
            }
        }
    )
    observer = FrameObserver(horizon=500, policy_probe=probe)
    for frame in range(500):
        snap = _snapshot(
            frame, valid=(frame == 0, False), alive=(False, False), done=(frame == 0, False)
        )
        observer(snap)
    row = observer.result()["episodes"]["11"]
    assert row["cumulative_mass"]["500"] == 0
    assert row["policy_diagnostics"]["greedy_boost_decisions"] == 1
    assert row["reward_decomposition"]["potential"] == -65.0
    assert row["counterfactual_training_return"]["0.99"] == -68.0


def test_policy_probe_keeps_actual_actions_and_frozen_b8_inference(monkeypatch):
    from types import SimpleNamespace

    import torch

    from research.q0q1_20261017.audit_episode_metrics import PolicyProbe
    from src.model import ego2s_network
    from src.simd_env import ego2s_policy

    net = torch.nn.Linear(1, 6)
    net.q_fixture = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    base = torch.nn.Linear(1, 6)
    base.q_fixture = np.array([[0.0, 5.0, 2.0, 3.0, 4.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(ego2s_network, "load_ego2s_checkpoint", lambda *_: base)
    monkeypatch.setattr(
        ego2s_policy, "student_q", lambda model, *_: torch.from_numpy(model.q_fixture)
    )
    policy = object.__new__(ego2s_policy.Ego2sSimdPolicy)
    policy.net, policy.forward, policy.mask_mode = net, "rowwise", "advisory"
    mask = np.ones((1, 1, 6), dtype=bool)
    sim = SimpleNamespace(
        get_advisory_action_mask=lambda: mask,
        cfg=SimpleNamespace(
            gamma=0.99, death_value=-3.0, kill_scale=0.0, boost_length_cost_frames=3
        ),
    )
    before = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    original = ego2s_policy.Ego2sSimdPolicy.actions
    with PolicyProbe("unused.pth") as probe:
        actions = policy.actions(mask[:, 0], sim, np.array([[0, 0]]))
        assert actions.tolist() == [5]
        assert probe.pending[0]["action_disagreements_b8"] == 1
        assert probe.pending[0]["greedy_boost_when_eligible"] == 1
        assert probe.pending[0]["q_legal_gap_sum"] == 1.0
    assert ego2s_policy.Ego2sSimdPolicy.actions is original
    assert all(torch.equal(before[name], tensor) for name, tensor in base.state_dict().items())
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in base.parameters()
    )
