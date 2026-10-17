"""Contracts for T10 TTL measurement and censored trajectory summaries."""

import numpy as np

from research.t1_commitment_20261015.topology import component_geometry
from research.t9_divergence_20261017.metrics import _arrival_reference, _race_counts
from research.t10_certificate_horizon_20261017.certificate import measure, occupancy, summarize
from tests.test_t9_metrics import _sim


def test_compiled_geometry_and_arrivals_match_t9_reference() -> None:
    for length in (3, 5, 130):
        sim = _sim(7, 3, [[(0, 1), (1, 1), (2, 1)], [(6, 1), (5, 1)]], length=[length, 2])
        g = component_geometry(sim)
        owners, ttl = occupancy(
            sim.bodies[0], sim.head_ptr[0], sim.seg_count[0], sim.length[0], sim.alive[0], 7, 3
        )
        np.testing.assert_array_equal(owners, g["owners"])
        np.testing.assert_array_equal(ttl, g["ttl"])
        arrival = _arrival_reference(
            [g["head"]], np.zeros_like(owners, bool), ttl, horizon=max(128, length)
        )
        expected = int(arrival[g["tail"][1], g["tail"][0]])
        got = measure(sim)
        assert got["tail_arrival"] == expected
        assert got["tail_L"] == (expected <= length)
        assert got["tail_128"] == (expected <= 128)
        assert got["race_128"] == _race_counts(g["head"], [g["heads"][1]], owners, ttl)["race_128"]


def test_actual_tail_target_and_inclusive_deadline() -> None:
    sim = _sim(5, 1, [[(0, 0), (1, 0), (2, 0)]])
    got = measure(sim)
    assert got["tail_arrival"] == 3  # wait at neck TTL2, then one step to tail
    assert got["tail_L"] is True
    sim = _sim(5, 1, [[(0, 0), (1, 0), (2, 0)]], length=[130])
    got = measure(sim)
    assert got["tail_arrival"] == 130
    assert got["tail_L"] is True and got["tail_128"] is False


def test_length_deadline_is_not_tautologically_reachable() -> None:
    # A second row permits a real detour, but neither it nor waiting at the
    # neck (TTL3) reaches the tail by L4: both require five cell steps.
    sim = _sim(4, 2, [[(0, 0), (1, 0), (2, 0), (3, 0)]])
    got = measure(sim)
    assert got["tail_arrival"] == 5
    assert got["tail_L"] is False and got["tail_128"] is True


def test_optimistic_positive_does_not_certify_a_feasible_next_move() -> None:
    # Full board: every neighbor of the central head remains occupied next
    # step. The permissive flood nevertheless waits at the head for two units.
    body = [(1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    sim = _sim(3, 3, [body])
    got = measure(sim)
    assert got["tail_arrival"] == 3
    assert got["tail_L"] is True and got["tail_128"] is True
    geometry = component_geometry(sim)
    assert all(geometry["ttl"][y, x] > 1 for x, y in ((1, 0), (0, 1), (1, 2), (2, 1)))


def _episode(policy, env, decisions, flags, survived=False):
    return {
        "policy": policy,
        "arm": "b8_s0",
        "env": env,
        "world_seed": env,
        "survived": survived,
        "death_decision": None if survived else 10,
        "death_length": None if survived else 900,
        "trace": [
            {"decision": d, "length": 800, "tail_L": f, "tail_128": f, "race_128": 400}
            for d, f in zip(decisions, flags)
        ],
    }


def test_summary_strict_area_threshold_censoring_and_off_by_one() -> None:
    rows = [
        _episode("b8", 0, [8, 9, 10], [True, False, False]),
        _episode("b8", 1, [9, 10], [False, False]),
        _episode("b8", 2, [], []),
        _episode("b8", 3, [10], [True]),
        _episode("dense", 0, list(range(10)), [True] * 10, True),
    ]
    got = summarize(rows)
    assert got["b8"]["counts"]["race_above_L2"] == 0
    assert got["b8"]["deaths"][0]["tail_L"]["decisions_before_death"] == 2
    assert got["b8"]["deaths"][3]["tail_L"]["decisions_before_death"] == 0
    assert got["b8"]["deaths"][1]["tail_L"]["status"] == "never_held_in_eligible_states"
    assert got["b8"]["deaths"][2]["tail_L"]["status"] == "no_L800_states"
    assert got["b8"]["loss_duration"]["tail_L"]["observed"] == 2
    assert got["dense"]["conditional_survival"]["tail_L"]["10"]["survival"] == 1
    assert got["dense"]["conditional_survival"]["tail_L"]["25"]["survival"] is None


def test_consecutive_condition_breaks_on_ineligible_gap_or_false() -> None:
    rows = [
        _episode("b8", 0, [0, 1, 3, 4, 5], [True] * 5, True),
        _episode("b8", 1, [0, 1, 2, 3, 4], [True, True, False, True, True]),
    ]
    got = summarize(rows)["b8"]["conditional_survival"]["tail_L"]
    assert got["1"]["episodes"] == 2
    assert got["5"]["episodes"] == 0


def test_measure_does_not_mutate_sim() -> None:
    sim = _sim(5, 3, [[(0, 0), (1, 0), (2, 0)]])
    arrays = {k: v.copy() for k, v in vars(sim).items() if isinstance(v, np.ndarray)}
    measure(sim)
    for key, value in arrays.items():
        np.testing.assert_array_equal(getattr(sim, key), value)
