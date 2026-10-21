"""Focused streaming contracts for E8's bounded training diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_ppo.diagnostics import E8Diagnostics


def _raw(**overrides):
    values = {
        "world_id": np.array([0]),
        "episode": np.array([0]),
        "mix": np.array(["scripted"]),
        "frame": np.array([0]),
        "length": np.array([850]),
        "reward": np.array([1.0], dtype=np.float32),
        "done": np.array([False]),
        "survived_horizon": np.array([False]),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rollout_cuts_preserve_open_run_and_do_not_double_count_snapshot() -> None:
    diagnostics = E8Diagnostics()
    diagnostics.observe(_raw(), actions=[2], b8_actions=[1])
    first = diagnostics.snapshot()
    second = diagnostics.snapshot()
    assert first == second
    assert first["non_greedy_runs_at_l_ge_800"]["open_runs_ge2"] == 0
    diagnostics.observe(_raw(frame=np.array([1])), actions=[2], b8_actions=[1])
    diagnostics.observe(
        _raw(frame=np.array([2]), done=np.array([True]), reward=np.array([0.0])),
        actions=[1],
        b8_actions=[1],
    )
    report = diagnostics.snapshot()
    runs = report["non_greedy_runs_at_l_ge_800"]
    assert runs["runs_ge2"] == runs["lives_with_run_ge2"] == 1
    assert runs["runs_ge3"] == 0
    assert runs["terminated_lives"] == 1 and runs["open_lives"] == 0
    assert runs["mass_ledger_ge2_sum"] == 0.4


def test_occupancy_departures_drift_and_recovery_are_bounded() -> None:
    diagnostics = E8Diagnostics()
    raw = _raw(
        world_id=np.array([0, 1, 2]),
        episode=np.array([0, 0, 0]),
        mix=np.array(["scripted", "frozen", "mixed"]),
        frame=np.array([0, 2000, 3500]),
        length=np.array([399, 800, 1200]),
        reward=np.ones(3),
        done=np.zeros(3, dtype=bool),
        survived_horizon=np.zeros(3, dtype=bool),
    )
    diagnostics.observe(raw, actions=[1, 2, 3], b8_actions=[1, 1, 3])
    diagnostics.observe_fixed_bank(
        policy_logits=[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
        b8_logits=[[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        lengths=[50, 850],
    )
    report = diagnostics.snapshot()
    assert report["occupancy"]["scripted"]["lt400"]["lt2000"] == 1
    assert report["occupancy"]["frozen"]["800_1199"]["2000_3499"] == 1
    assert report["long_body_exposure"] == {"scripted": 0, "frozen": 1, "mixed": 1}
    assert report["realised_departures_by_length_decile"]["d8"]["departures"] == 1
    assert report["fixed_ordinary_bank_drift_by_length_decile"]["d8"]["greedy_disagreements"] == 1
    restored = E8Diagnostics()
    restored.load_state_dict(diagnostics.state_dict())
    assert restored.snapshot() == report


def test_endpoint_ledger_has_b8_death_frame_length_rescue_and_redeath_rows() -> None:
    diagnostics = E8Diagnostics()
    diagnostics.record_paired_endpoint(
        mix="mixed", b8_death_frame=2400, b8_length=810, candidate_death_frame=2600
    )
    diagnostics.record_paired_endpoint(
        mix="mixed", b8_death_frame=2400, b8_length=810, candidate_death_frame=None
    )
    row = diagnostics.snapshot()["endpoint_ledger_by_b8_death_frame_x_length"]["mixed"]
    row = row["b8_frame_2400__length_800_1199"]
    assert row == {
        "pairs": 2,
        "both_alive": 0,
        "candidate_only": 1,
        "b8_only": 0,
        "both_dead": 1,
        "rescues": 2,
        "re_deaths": 1,
    }


def test_fixed_bank_deciles_are_immutable_tie_explicit_and_shared() -> None:
    diagnostics = E8Diagnostics(require_fixed_bank=True)
    diagnostics.observe(_raw(), actions=[1], b8_actions=[1])
    assert diagnostics.snapshot()["departures_pending_fixed_deciles"] == 1
    bank = np.repeat(np.arange(10), 1000)
    assert diagnostics.configure_length_deciles(bank) == tuple(range(0, 9))
    diagnostics.observe(_raw(length=np.array([0])), actions=[2], b8_actions=[1])
    diagnostics.observe_fixed_bank(
        policy_logits=[[0, 1, 0, 0, 0, 0]],
        b8_logits=[[0, 0, 1, 0, 0, 0]],
        lengths=[0],
    )
    report = diagnostics.snapshot()
    # side='right': a length tied at the first cut moves to d1.
    assert report["realised_departures_by_length_decile"]["d1"]["departures"] == 1
    assert report["fixed_ordinary_bank_drift_by_length_decile"]["d1"]["observations"] == 1
    assert report["length_decile_definition"]["boundaries"] == list(range(0, 9))
    restored = E8Diagnostics(require_fixed_bank=True)
    restored.load_state_dict(diagnostics.state_dict())
    assert restored.snapshot() == report
    with pytest.raises(RuntimeError, match="immutable"):
        diagnostics.configure_length_deciles(bank)


def test_censored_closed_runs_and_final_frame_death_are_not_misreported() -> None:
    diagnostics = E8Diagnostics()
    diagnostics.observe(_raw(frame=np.array([0])), actions=[2], b8_actions=[1])
    diagnostics.observe(_raw(frame=np.array([1])), actions=[1], b8_actions=[1])
    interim = diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]
    assert interim["runs_ge2"] == 0
    # Make a closed >=2 run in an unfinished life: it is reported as censored.
    diagnostics = E8Diagnostics()
    diagnostics.observe(_raw(frame=np.array([0])), actions=[2], b8_actions=[1])
    diagnostics.observe(_raw(frame=np.array([1])), actions=[2], b8_actions=[1])
    diagnostics.observe(_raw(frame=np.array([2])), actions=[1], b8_actions=[1])
    interim = diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]
    assert interim["runs_ge2"] == 1
    assert interim["censored_lives_with_run_ge2"] == 1
    assert interim["censored_mass_ledger_ge2_sum"] == pytest.approx(0.6)
    diagnostics.observe(
        _raw(
            frame=np.array([4999]),
            done=np.array([True]),
            survived_horizon=np.array([False]),
            reward=np.array([0.0]),
        ),
        actions=[1],
        b8_actions=[1],
    )
    terminal = diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]
    assert terminal["terminated_lives"] == 1
    assert terminal["surviving_lives"] == 0


def test_prebank_counts_and_censored_runs_survive_recovery_once() -> None:
    diagnostics = E8Diagnostics(require_fixed_bank=True)
    for frame, action in enumerate((2, 2, 1)):
        diagnostics.observe(_raw(frame=np.array([frame])), actions=[action], b8_actions=[1])
    restored = E8Diagnostics(require_fixed_bank=True)
    restored.load_state_dict(diagnostics.state_dict())
    assert restored.snapshot() == diagnostics.snapshot()
    assert restored.snapshot()["non_greedy_runs_at_l_ge_800"]["runs_ge2"] == 1
    restored.configure_length_deciles(np.repeat(np.arange(10) * 100, 1000))
    report = restored.snapshot()
    assert report["departures_pending_fixed_deciles"] == 0
    assert sum(x["decisions"] for x in report["realised_departures_by_length_decile"].values()) == 3
    assert (
        sum(x["departures"] for x in report["realised_departures_by_length_decile"].values()) == 2
    )


def test_active_qualifying_run_reports_unique_life_and_accrued_mass_once() -> None:
    diagnostics = E8Diagnostics()
    for frame in (0, 1):
        diagnostics.observe(_raw(frame=np.array([frame])), actions=[2], b8_actions=[1])
    report = diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]
    assert report["runs_ge2"] == report["open_runs_ge2"] == 1
    assert report["lives_with_run_ge2"] == report["censored_lives_with_run_ge2"] == 1
    assert report["mass_ledger_ge2_sum"] == pytest.approx(0.4)
    restored = E8Diagnostics()
    restored.load_state_dict(diagnostics.state_dict())
    assert restored.snapshot() == diagnostics.snapshot()
    diagnostics.observe(_raw(frame=np.array([2])), actions=[1], b8_actions=[1])
    assert diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]["runs_ge2"] == 1


def test_run_counts_per_h5000_surviving_life_have_separate_denominator() -> None:
    diagnostics = E8Diagnostics()
    for frame in (0, 1):
        diagnostics.observe(_raw(frame=np.array([frame])), actions=[2], b8_actions=[1])
    diagnostics.observe(
        _raw(frame=np.array([4999]), done=np.array([True]), survived_horizon=np.array([True])),
        actions=[1],
        b8_actions=[1],
    )
    rows = diagnostics.snapshot()["non_greedy_runs_at_l_ge_800"]["by_completed_life_outcome"]
    assert rows["surviving"]["lives"] == 1
    assert rows["surviving"]["runs_ge2_per_life"] == 1
    assert rows["terminated"]["lives"] == 0
    restored = E8Diagnostics()
    restored.load_state_dict(diagnostics.state_dict())
    assert restored.snapshot() == diagnostics.snapshot()
