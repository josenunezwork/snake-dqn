"""E8 trace-derived screen-ledger contract checks."""

from __future__ import annotations

import copy

import pytest

from research.bl_ppo import screen


def _controller(masses: list[float]) -> dict:
    trace = [{"frame": frame, "alive": mass > 0, "mass": mass} for frame, mass in enumerate(masses)]
    return {"trace": trace, "mass_integral": sum(masses) / 5}


def _retained_controller(masses: list[float]) -> dict:
    controller = _controller(masses)
    trace = controller["trace"]
    controller.update(
        {
            "trace_sha256": screen.banks.canonical_sha(trace),
            "active_frames": len(trace),
            "alive_frame_fraction": sum(step["alive"] for step in trace) / 5,
            "no_death_episode": False,
        }
    )
    return controller


def test_three_term_ledger_reconciles_endpoint_and_counts_rescue_redeth() -> None:
    controllers = {
        "B8": _controller([900, 910, 0]),
        "R": _controller([900, 920, 930, 0]),
        "F": _controller([900, 910, 0]),
    }
    ledger = screen._world_ledger(controllers, 5)

    mass = ledger["R"]["mass"]
    assert mass == {
        "both_alive_growth": 10.0,
        "candidate_only": 930.0,
        "b8_only": 0.0,
        "total": 940.0,
        "per_frame": 188.0,
    }
    assert ledger["R"]["endpoint"] == {
        "b8_death_frame_bin": "lt2000",
        "b8_death_length_bin": "800_1199",
        "rescued": 1,
        "full_horizon_rescues": 0,
        "re_death": 1,
    }
    assert ledger["R"]["occupancy_by_frame_length"] == {"lt2000:800_1199": 3}
    assert ledger["R"]["candidate_growth_slope_l_ge_800"] == 15.0


def test_aggregate_preserves_endpoint_mass_terms_occupancy_and_exposure() -> None:
    world = screen._world_ledger(
        {
            "B8": _controller([900, 910, 0]),
            "R": _controller([900, 920, 930, 0]),
            "F": _controller([900, 910, 0]),
        },
        5,
    )
    aggregate = screen._empty_e8_ledgers()
    screen._add_e8_ledger(aggregate, "scripted", world)
    endpoint = aggregate["endpoint_by_arm_mix"]["R"]["scripted"]["lt2000:800_1199"]
    assert endpoint == {
        "exposure": 1,
        "rescues": 1,
        "full_horizon_rescues": 0,
        "re_deaths": 1,
        "both_alive_growth": 10.0,
        "candidate_only": 930.0,
        "b8_only": 0.0,
    }
    assert aggregate["occupancy_by_arm_mix"]["R"]["scripted"] == {"lt2000:800_1199": 3}


def test_trace_corruption_cannot_produce_a_finite_ledger() -> None:
    controllers = {
        "B8": _controller([900, 0]),
        "R": _controller([900, 0]),
        "F": _controller([900, 0]),
    }
    corrupted = copy.deepcopy(controllers)
    corrupted["R"]["trace"][1]["alive"] = True
    with pytest.raises(ValueError, match="terminal death"):
        screen._world_ledger(corrupted, 5)


def test_retained_cached_trace_cannot_be_reused_with_prefilled_ledger() -> None:
    controllers = {
        "B8": _retained_controller([900, 0]),
        "R": _retained_controller([900, 0]),
        "F": _retained_controller([900, 0]),
    }
    ledger = screen._world_ledger(controllers, 5)
    cached = copy.deepcopy(controllers)
    cached["R"]["trace"][0]["mass"] = 901
    with pytest.raises(ValueError, match="trace hash"):
        screen._validate_trace_metrics(cached, 5)
    assert cached != controllers
    assert ledger == screen._world_ledger(controllers, 5)
