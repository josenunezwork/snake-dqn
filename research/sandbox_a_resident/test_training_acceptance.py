from __future__ import annotations

import numpy as np
import pytest

from research.sandbox_a_resident.training_acceptance import (
    distribution_screen_descriptor,
    forced_action_common_state,
    vector_carry_common_state,
    vector_v8_common_state,
)
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def test_forced_actions_match_cpu_discrete_common_state_fixture() -> None:
    assert forced_action_common_state(((1, 1), (4, 1), (1, 0)))["status"] == "PASS"


def test_v8_mask_and_action_match_cpu_on_explicit_state() -> None:
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, game_width=120, game_height=100, max_capacity=32)
    sim = GridBatchSim(cfg, seeds=[9], train_mode=True, allow_respawn=False, jit=False)
    q = np.array(((0.0, 2.0, 1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 2.0, 0.0, 0.0, 0.0)), dtype=np.float32)
    assert vector_v8_common_state(sim, q)["status"] == "PASS"
    assert vector_carry_common_state(sim)["status"] == "PASS"


def test_distribution_screen_is_explicitly_nongating() -> None:
    report = distribution_screen_descriptor()
    assert report["worlds_per_mix"] == 32
    assert "not a gate" in report["status_meaning"]


def test_cpu_reference_uses_real_frozen_roster_short_horizon() -> None:
    import torch

    from research.sandbox_a_resident.training_cpu_reference import CpuRosterTask

    task = CpuRosterTask("frozen", 3, 5000, seed=19)
    assert task.observe()[2].shape == (3, 60)
    assert task.step(torch.ones(3, dtype=torch.long)).mass.shape == (3,)


def test_distribution_screen_records_first_lives_and_gross_support(monkeypatch) -> None:
    """Exercise both complete screen arms with a tiny explicit event tape."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident import training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "distribution_screen_descriptor", lambda **kw: kw)

    class Task:
        def __init__(self, mix, worlds, horizon):
            self.tick = 0
            bodies = torch.zeros((worlds, 2, 1, 2), dtype=torch.long)
            bodies[:, 0, 0] = torch.tensor([1, 1])
            bodies[:, 1, 0] = torch.tensor([4, 1])
            self.sim = SimpleNamespace(
                bodies=bodies,
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (
                torch.zeros(3, 1),
                torch.zeros(3, 1),
                torch.zeros(3, 1),
                torch.ones(3, 6, dtype=torch.bool),
            )

        def step(self, actions):
            self.tick += 1
            terminal = torch.tensor([self.tick == 1, False, False])
            self.sim.alive[0, 0] = False
            return SimpleNamespace(
                mass=torch.tensor([0.0, 10.0, 10.0]),
                terminal=terminal,
                reset=torch.zeros(3, dtype=torch.bool),
            )

    report = acceptance.run_distribution_screen(
        cpu_factory=Task,
        device_factory=Task,
        cpu_actor=lambda *args: torch.zeros(3, 6),
        device_actor=lambda *args: torch.zeros(3, 6),
        worlds_per_mix=3,
        horizon=4,
    )
    assert report["material_gaps"] == []
    for mix in ("scripted", "frozen", "mixed"):
        row = report["results"][mix]["cpu"]
        assert row["death_events"] == 1
        assert row["no_death_fraction"] == pytest.approx(2 / 3)
        assert row["mean_mass_integral"] == pytest.approx(20 / 3)
        assert row["nearest_spawn_mean_cells"] == 3


def test_distribution_screen_does_not_count_reset_frame_opponent_respawns(monkeypatch) -> None:
    """A terminal hero reset cannot contribute old-world opponent events."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident import training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "distribution_screen_descriptor", lambda **kw: kw)

    class ResetTask:
        def __init__(self, _mix, worlds, _horizon):
            self.calls = 0
            self.sim = SimpleNamespace(
                bodies=torch.tensor([[[[1, 1]], [[4, 1]]]] * worlds),
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.tensor([[True, False]] * worlds),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (torch.zeros(1, 1),) * 3 + (torch.ones(1, 6, dtype=torch.bool),)

        def step(self, _actions):
            self.calls += 1
            # Simulate a device task that resets its world in the terminal frame.
            self.sim.alive[:] = True
            self.sim._last_food_ate[:] = True
            return SimpleNamespace(
                mass=torch.zeros(1),
                terminal=torch.ones(1, dtype=torch.bool),
                reset=torch.ones(1, dtype=torch.bool),
            )

    report = acceptance.run_distribution_screen(
        cpu_factory=ResetTask,
        device_factory=ResetTask,
        cpu_actor=lambda *args: torch.zeros(1, 6),
        device_actor=lambda *args: torch.zeros(1, 6),
        worlds_per_mix=1,
        horizon=4,
    )
    for mix in ("scripted", "frozen", "mixed"):
        for arm in ("cpu", "device"):
            row = report["results"][mix][arm]
            assert row["opponent_respawns"] == 0
            assert row["food_consumption_events"] == 0


def test_distribution_screen_stops_when_every_first_life_is_terminal(monkeypatch) -> None:
    """No dead screen world is advanced through the unused horizon tail."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident import training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "distribution_screen_descriptor", lambda **kw: kw)

    class Task:
        instances = []

        def __init__(self, _mix, worlds, _horizon):
            self.steps = 0
            Task.instances.append(self)
            self.sim = SimpleNamespace(
                bodies=torch.tensor([[[[1, 1]], [[4, 1]]]] * worlds),
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (torch.zeros(1, 1),) * 3 + (torch.ones(1, 6, dtype=torch.bool),)

        def step(self, _actions):
            self.steps += 1
            self.sim.alive[:, 0] = False
            return SimpleNamespace(
                mass=torch.zeros(1),
                terminal=torch.ones(1, dtype=torch.bool),
                reset=torch.zeros(1, dtype=torch.bool),
            )

    acceptance.run_distribution_screen(
        cpu_factory=Task,
        device_factory=Task,
        cpu_actor=lambda *args: torch.zeros(1, 6),
        device_actor=lambda *args: torch.zeros(1, 6),
        worlds_per_mix=1,
        horizon=7,
    )
    assert [task.steps for task in Task.instances] == [1] * 6


def test_prefix_screen_finishes_all_arms_without_shortening_natural_tasks():
    """The cheap startup check is labelled a prefix; factories retain H5000."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident.training_acceptance import run_distribution_screen

    tasks = []
    progress = []

    class Task:
        def __init__(self, mix, worlds, horizon):
            assert horizon == 5000
            self.steps = 0
            self.worlds = worlds
            tasks.append(self)
            self.sim = SimpleNamespace(
                bodies=torch.tensor([[[[1, 1]], [[4, 1]]]] * worlds),
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (torch.zeros(self.worlds, 1),) * 3 + (
                torch.ones(self.worlds, 6, dtype=torch.bool),
            )

        def step(self, actions):
            self.steps += 1
            return SimpleNamespace(
                mass=torch.full((self.worlds,), 10.0),
                terminal=torch.zeros(self.worlds, dtype=torch.bool),
                reset=torch.zeros(self.worlds, dtype=torch.bool),
            )

    report = run_distribution_screen(
        cpu_factory=Task,
        device_factory=Task,
        cpu_actor=lambda local, *_: torch.zeros(len(local), 6),
        device_actor=lambda local, *_: torch.zeros(len(local), 6),
        on_progress=lambda stage, values: progress.append((stage, values)),
    )
    assert [task.steps for task in tasks] == [256] * 6
    assert report["coverage"] == "early-life prefix only"
    assert report["task_horizon"] == 5000
    assert report["horizon"] == 256
    assert report["test_only"] is False
    assert report["material_gaps"] == []
    assert report["results"]["mixed"]["device"]["mean_mass_integral"] == 10.0
    assert len([stage for stage, _ in progress if stage == "distribution-screen-complete"]) == 6
    assert progress[-1][0] == "distribution-screen-compared"


def test_small_or_partial_screen_requires_explicit_test_only():
    from research.sandbox_a_resident.training_acceptance import run_distribution_screen

    with pytest.raises(ValueError, match="E32"):
        distribution_screen_descriptor(worlds_per_mix=3)
    assert distribution_screen_descriptor(horizon=5000)["coverage"] == "complete first lives"
    with pytest.raises(ValueError, match="all three mixes"):
        run_distribution_screen(
            cpu_factory=None,
            device_factory=None,
            cpu_actor=None,
            device_actor=None,
            mixes=("mixed",),
        )


def _terminal_screen_task_factory(
    deaths: int, *, mass: float = 1.0, cause: int = 4, terminal_tick: int = 1
):
    """Build a one-tick E32 screen task with an explicit terminal-event tape."""
    from types import SimpleNamespace

    import torch

    class TerminalTask:
        def __init__(self, _mix, worlds, _horizon):
            assert worlds == 32
            self.worlds = worlds
            self.tick = 0
            self.sim = SimpleNamespace(
                bodies=torch.tensor([[[[1, 1]], [[4, 1]]]] * worlds),
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (torch.zeros(self.worlds, 1),) * 3 + (
                torch.ones(self.worlds, 6, dtype=torch.bool),
            )

        def step(self, _actions):
            self.tick += 1
            terminal = torch.zeros(self.worlds, dtype=torch.bool)
            if self.tick == terminal_tick:
                terminal[:deaths] = True
                self.sim.alive[:deaths, 0] = False
                self.sim._last_death_cause[:deaths, 0] = cause
            return SimpleNamespace(
                mass=torch.full((self.worlds,), mass),
                terminal=terminal,
                reset=torch.zeros(self.worlds, dtype=torch.bool),
            )

    return TerminalTask


def _e32_terminal_screen(
    monkeypatch, cpu_deaths: int, device_deaths: int, *, horizon: int = 1, **device_kwargs
):
    """Run the terminal-support comparison against compact, explicit E32 fixtures."""
    import torch

    from research.sandbox_a_resident import training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "distribution_screen_descriptor", lambda **kw: kw)
    return acceptance.run_distribution_screen(
        cpu_factory=_terminal_screen_task_factory(cpu_deaths),
        device_factory=_terminal_screen_task_factory(device_deaths, **device_kwargs),
        cpu_actor=lambda *_: torch.zeros(32, 6),
        device_actor=lambda *_: torch.zeros(32, 6),
        worlds_per_mix=32,
        horizon=horizon,
        mixes=("mixed",),
        test_only=True,
    )


def test_terminal_metrics_with_small_nonzero_death_counts_are_insufficient_support(monkeypatch):
    report = _e32_terminal_screen(monkeypatch, 1, 2)
    terminal = [
        row
        for row in report["comparisons"]
        if row["metric"]
        in {"terminal_rate_per_1k_alive_frames", "starvation_share", "collision_share"}
    ]

    assert len(terminal) == 3
    assert {row["status"] for row in terminal} == {"insufficient_support"}
    assert report["material_gaps"] == []


@pytest.mark.parametrize("cpu_deaths, device_deaths", ((16, 15), (15, 16)))
def test_terminal_metrics_require_support_in_both_e32_arms(
    monkeypatch, cpu_deaths: int, device_deaths: int
):
    report = _e32_terminal_screen(monkeypatch, cpu_deaths, device_deaths)
    terminal = [row for row in report["comparisons"] if row["metric"] == "collision_share"]

    assert terminal == [
        {
            "mix": "mixed",
            "metric": "collision_share",
            "cpu": 1.0,
            "device": 1.0,
            "status": "insufficient_support",
        }
    ]


def test_terminal_metrics_with_e32_support_still_report_material_gap(monkeypatch):
    report = _e32_terminal_screen(monkeypatch, 16, 16, cause=0)
    collision = next(row for row in report["comparisons"] if row["metric"] == "collision_share")

    assert collision["status"] == "material_gap"
    assert {key: collision[key] for key in ("mix", "metric", "cpu", "device")} in report[
        "material_gaps"
    ]


def test_supported_terminal_rate_exposure_difference_still_reports_material_gap(monkeypatch):
    report = _e32_terminal_screen(monkeypatch, 16, 16, horizon=4, terminal_tick=3)
    terminal_rate = next(
        row for row in report["comparisons"] if row["metric"] == "terminal_rate_per_1k_alive_frames"
    )

    assert terminal_rate["status"] == "material_gap"
    assert {key: terminal_rate[key] for key in ("mix", "metric", "cpu", "device")} in report[
        "material_gaps"
    ]


def test_nonterminal_metrics_keep_25_percent_rule_when_terminal_support_is_low(monkeypatch):
    report = _e32_terminal_screen(monkeypatch, 1, 2, mass=2.0)
    mass = next(row for row in report["comparisons"] if row["metric"] == "mean_mass_integral")

    assert mass["status"] == "material_gap"
    assert {key: mass[key] for key in ("mix", "metric", "cpu", "device")} in report["material_gaps"]


def test_partial_test_only_screen_reports_requested_mix_and_stable_engine_labels():
    """Progress telemetry must not serialize factories or unrequested mix names."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident.training_acceptance import run_distribution_screen

    progress = []

    class Task:
        def __init__(self, _mix, worlds, _horizon):
            self.worlds = worlds
            self.sim = SimpleNamespace(
                bodies=torch.tensor([[[[1, 1]], [[4, 1]]]] * worlds),
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (torch.zeros(self.worlds, 1),) * 3 + (
                torch.ones(self.worlds, 6, dtype=torch.bool),
            )

        def step(self, _actions):
            return SimpleNamespace(
                mass=torch.ones(self.worlds),
                terminal=torch.zeros(self.worlds, dtype=torch.bool),
                reset=torch.zeros(self.worlds, dtype=torch.bool),
            )

    report = run_distribution_screen(
        cpu_factory=Task,
        device_factory=Task,
        cpu_actor=lambda local, *_: torch.zeros(len(local), 6),
        device_actor=lambda local, *_: torch.zeros(len(local), 6),
        worlds_per_mix=1,
        mixes=("mixed",),
        test_only=True,
        on_progress=lambda stage, values: progress.append((stage, values)),
    )

    assert report["mixes"] == ["mixed"]
    for stage in (
        "distribution-screen-start",
        "distribution-screen-advance",
        "distribution-screen-complete",
    ):
        values = [values for actual_stage, values in progress if actual_stage == stage]
        assert values
        assert {value["engine"] for value in values} == {"Task"}
        assert all("0x" not in str(value) for value in values)
