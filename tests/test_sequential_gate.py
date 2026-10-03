"""Tests for src/evaluation/sequential_gate.py (group-sequential strict gate, opt-in)."""

import json
import math
from statistics import NormalDist

import numpy as np
import pytest

from src.evaluation.sequential_gate import (
    SequentialGatePlan,
    conditional_power,
    look_sizes_for,
    round_robin_plan,
    run_sequential_gate,
    sequential_decision,
    sequential_gate_plan,
    worker_look_counts,
)
from src.scripts.eval_stats import paired_delta_test, scripted_noninferiority

N = NormalDist()
MIXES = ("frozen", "scripted", "mixed")


@pytest.fixture(scope="module")
def plan() -> SequentialGatePlan:
    return sequential_gate_plan(243, mde=30.0)


def _draw(means, n, sd=140.0, seed=0):
    rng = np.random.default_rng(seed)
    return {m: (rng.standard_normal(n) * sd + mu).tolist() for m, mu in zip(MIXES, means)}


def _prefix(deltas, n):
    return {m: v[:n] for m, v in deltas.items()}


# ----------------------------------------------------------------- plan and boundaries


def test_look_sizes_round_up_and_validate():
    assert look_sizes_for(243) == (61, 122, 183, 243)
    assert look_sizes_for(100) == (25, 50, 75, 100)
    with pytest.raises(ValueError):
        look_sizes_for(243, (0.5, 0.9))
    with pytest.raises(ValueError):
        look_sizes_for(243, (0.5, 0.5, 1.0))
    with pytest.raises(ValueError):
        look_sizes_for(4, (0.1, 1.0))
    with pytest.raises(ValueError):
        look_sizes_for(True)


def test_boundaries_match_published_obf_values():
    # Per-mix alpha 0.075 / 3 = 0.025 one-sided, four equally spaced looks: gsDesign
    # sfLDOF upper bounds 4.3326, 2.9631, 2.3590, 2.0141.
    published = [4.3326, 2.9631, 2.3590, 2.0141]
    p = sequential_gate_plan(100, mde=10.0, family_alpha=0.075, ni_alpha=0.025)
    assert p.efficacy_alpha_per_mix == pytest.approx(0.025)
    assert list(p.efficacy_boundaries) == pytest.approx(published, abs=2e-4)
    assert list(p.ni_boundaries) == pytest.approx(published, abs=2e-4)
    assert p.efficacy_alpha_spent[-1] == pytest.approx(0.025, abs=1e-6)


def test_default_plan_values(plan):
    assert plan.look_sizes == (61, 122, 183, 243)
    assert plan.fractions == pytest.approx((61 / 243, 122 / 243, 183 / 243, 1.0))
    assert list(plan.efficacy_boundaries) == pytest.approx(
        [4.6368, 3.1834, 2.5385, 2.1740], abs=2e-4
    )
    assert list(plan.ni_boundaries) == pytest.approx([3.7412, 2.5339, 2.0112, 1.7211], abs=2e-4)
    assert plan.efficacy_alpha_spent[-1] == pytest.approx(0.05 / 3, abs=1e-6)
    assert plan.ni_alpha_spent[-1] == pytest.approx(0.05, abs=1e-6)
    for c, p in zip(plan.efficacy_boundaries, plan.efficacy_nominal_p):
        assert p == pytest.approx(1.0 - N.cdf(c), rel=1e-9)
    # OBF spends almost nothing early; the final nominal level is below alpha/3.
    assert plan.efficacy_nominal_p[0] < 1e-5 < plan.efficacy_nominal_p[-1] < 0.05 / 3
    out = plan.as_dict()
    assert out["multiplicity"] == "bonferroni_across_mixes"
    assert json.loads(json.dumps(out)) == out


def test_boundaries_hold_per_mix_alpha_by_brownian_simulation(plan):
    rng = np.random.default_rng(7)
    sims = 200_000
    steps = np.diff((0.0,) + plan.fractions)
    brownian = np.cumsum(rng.standard_normal((sims, 4)) * np.sqrt(steps), axis=1)
    z = brownian / np.sqrt(np.array(plan.fractions))
    crossed = np.any(z >= np.array(plan.efficacy_boundaries), axis=1).mean()
    alpha = 0.05 / 3
    assert crossed == pytest.approx(alpha, abs=4 * math.sqrt(alpha * (1 - alpha) / sims))


def test_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=0.0)
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, scripted_mix="other")
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, family_alpha=0.6)
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, required_successes=4)


def test_single_look_reduces_to_fixed_bonferroni_and_fixed_ni():
    p = sequential_gate_plan(40, mde=30.0, fractions=(1.0,))
    assert p.efficacy_boundaries[0] == pytest.approx(N.inv_cdf(1 - 0.05 / 3), abs=1e-6)
    deltas = _draw((40.0, 25.0, -5.0), 40, seed=3)
    out = sequential_decision(p, 0, deltas, 3.5, [True])
    for mix in MIXES:
        fixed = paired_delta_test(deltas[mix], alpha=p.efficacy_nominal_p[0])
        assert out["per_mix"][mix]["crossed_at_this_look"] is fixed["superior"]
        assert out["per_mix"][mix]["conditional_power"] is None
    ni = scripted_noninferiority(deltas["scripted"], 3.5, alpha=0.05)
    assert out["scripted_noninferiority"]["lower_bound"] == pytest.approx(
        ni["lower_bound"], abs=1e-6
    )
    assert out["decision"] in ("FINAL_PASS", "FINAL_FAIL")


# ----------------------------------------------------------------- conditional power


def test_conditional_power_oracles():
    # B(0.5) = 0, drift 2: B(1) ~ N(1, 0.5); P(B(1) >= 2) = 1 - Phi(1 / sqrt(0.5)).
    assert conditional_power(0.0, 0.5, 2.0, 2.0) == pytest.approx(
        1.0 - N.cdf(1.0 / math.sqrt(0.5)), abs=1e-12
    )
    assert conditional_power(2.5, 1.0, 2.0, 0.0) == 1.0
    assert conditional_power(1.5, 1.0, 2.0, 0.0) == 0.0
    assert conditional_power(math.inf, 0.5, 2.0, 0.0) == 1.0
    assert conditional_power(-math.inf, 0.5, 2.0, 9.0) == 0.0
    low = conditional_power(0.0, 0.25, 2.17, 3.0)
    assert low < conditional_power(1.0, 0.25, 2.17, 3.0)
    assert low < conditional_power(0.0, 0.25, 2.17, 4.0)
    with pytest.raises(ValueError):
        conditional_power(0.0, 0.0, 2.0, 1.0)


# ----------------------------------------------------------------- decision logic


def test_strong_effect_stops_for_efficacy_at_first_look(plan):
    out = sequential_decision(plan, 0, _draw((300, 300, 300), 61, seed=1), 3.5, [True])
    assert out["decision"] == "STOP_PASS"
    assert out["passes"] and out["valid"] and out["stopped"]
    assert out["successful_mixes"] == list(MIXES)
    assert out["crossing_looks"] == {m: 0 for m in MIXES}
    assert out["ni_crossing_look"] == 0


def test_band_failure_blocks_stop_and_crossings_persist(plan):
    deltas = _draw((300, 300, 300), 122, seed=1)
    first = sequential_decision(plan, 0, _prefix(deltas, 61), 3.5, [False])
    assert first["decision"] == "CONTINUE" and not first["passes"]
    second = sequential_decision(plan, 1, deltas, 3.5, [False, True])
    assert second["decision"] == "STOP_PASS" and second["valid"]
    assert second["crossing_looks"] == {m: 0 for m in MIXES}


def test_noninferiority_failure_continues_then_fails_at_final_look(plan):
    deltas = _draw((300, -300, 300), 243, seed=2)
    interim = sequential_decision(plan, 0, _prefix(deltas, 61), 3.5, [True])
    assert interim["decision"] == "CONTINUE"
    assert interim["successful_mixes"] == ["frozen", "mixed"]
    assert interim["futile_mixes"] == ["scripted"] and not interim["ni_established"]
    final = sequential_decision(plan, 3, deltas, 3.5, [True] * 4)
    assert final["decision"] == "FINAL_FAIL" and final["final_look"]
    assert final["valid"] and not final["passes"]
    assert all(r["conditional_power"] is None for r in final["per_mix"].values())


def test_clear_loser_stops_for_futility(plan):
    out = sequential_decision(plan, 0, _draw((-100, -100, -100), 61, seed=4), 3.5, [True])
    assert out["decision"] == "STOP_FUTILE" and not out["passes"]
    assert out["futile_mixes"] == list(MIXES)
    assert all(r["conditional_power"] < 0.10 for r in out["per_mix"].values())


def test_futility_is_non_binding_but_efficacy_stop_is_final(plan):
    losers = _draw((-100, -100, -100), 122, seed=4)
    overridden = sequential_decision(plan, 1, losers, 3.5, [True, True])
    assert overridden["valid"] and overridden["futility_overrides"] == [0]
    winners = _draw((300, 300, 300), 122, seed=1)
    late = sequential_decision(plan, 1, winners, 3.5, [True, True])
    assert not late["valid"] and not late["passes"]
    assert late["invalid_reason"] == "already_stopped_for_efficacy_at_look_0"


def test_crossing_is_kept_when_later_data_turn(plan):
    rng = np.random.default_rng(5)
    frozen = (400 + 50 * rng.standard_normal(61)).tolist()
    frozen += (-600 + 50 * rng.standard_normal(61)).tolist()
    deltas = {"frozen": frozen, **{m: (140 * rng.standard_normal(122)).tolist() for m in MIXES[1:]}}
    first = sequential_decision(plan, 0, _prefix(deltas, 61), 3.5, [True])
    assert first["decision"] == "CONTINUE" and first["successful_mixes"] == ["frozen"]
    out = sequential_decision(plan, 1, deltas, 3.5, [True, True])
    assert out["per_mix"]["frozen"]["mean_delta"] < 0
    assert not out["per_mix"]["frozen"]["crossed_at_this_look"]
    assert out["per_mix"]["frozen"]["crossed_so_far"]
    assert "frozen" in out["successful_mixes"] and out["crossing_looks"]["frozen"] == 0


def test_decision_input_validation(plan):
    good = _draw((0, 0, 0), 61)
    with pytest.raises(ValueError):
        sequential_decision(plan, 0, _draw((0, 0, 0), 60), 3.5, [True])
    with pytest.raises(ValueError):
        sequential_decision(plan, 0, {"frozen": good["frozen"]}, 3.5, [True])
    with pytest.raises(ValueError):
        sequential_decision(plan, 0, good, 3.5, [True, True])
    with pytest.raises(ValueError):
        sequential_decision(plan, 0, good, 0.0, [True])
    with pytest.raises(ValueError):
        sequential_decision(plan, 4, good, 3.5, [True])
    bad = dict(good, scripted=good["scripted"][:-1] + [float("nan")])
    with pytest.raises(ValueError):
        sequential_decision(plan, 0, bad, 3.5, [True])


def test_determinism(plan):
    deltas = _draw((30, 30, 30), 183, seed=9)
    first = sequential_decision(plan, 2, deltas, 3.5, [True] * 3)
    again = sequential_decision(sequential_gate_plan(243, mde=30.0), 2, deltas, 3.5, [True] * 3)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_run_sequential_gate_walks_looks(plan):
    strong = run_sequential_gate(plan, _draw((300, 300, 300), 243, seed=1), 3.5)
    assert strong["decision"] == "STOP_PASS" and strong["worlds_per_mix_used"] == 61
    losers = _draw((-100, -100, -100), 243, seed=4)
    stop = run_sequential_gate(plan, losers, 3.5)
    assert stop["decision"] == "STOP_FUTILE" and stop["look"] == 0
    full = run_sequential_gate(plan, losers, 3.5, honor_futility=False)
    assert full["decision"] == "FINAL_FAIL" and full["futility_overrides"] == [0, 1, 2]
    partial = run_sequential_gate(plan, _draw((0, 0, 0), 130, seed=6), 3.5, honor_futility=False)
    assert partial["look"] == 1
    with pytest.raises(ValueError):
        run_sequential_gate(plan, _draw((0, 0, 0), 60), 3.5)


# ----------------------------------------------------------------- interleaving


def test_round_robin_plan_balances_every_look(plan):
    units = round_robin_plan(243)
    assert len(units) == 3 * 243
    assert units == round_robin_plan(243)
    assert len({(u["mix"], u["world_index"]) for u in units}) == len(units)
    counts = worker_look_counts(plan.look_sizes)
    for look, size in enumerate(plan.look_sizes):
        prefix = units[: 3 * size]
        assert {m: sum(u["mix"] == m for u in prefix) for m in MIXES} == {m: size for m in MIXES}
        for worker in (0, 1):
            shard = [u for u in units if u["worker"] == worker]
            in_prefix = [u for u in prefix if u["worker"] == worker]
            assert in_prefix == shard[: len(in_prefix)]  # look prefix = shard prefix
            assert len(in_prefix) == counts[look]["per_worker"][worker]
            per_mix = [sum(u["mix"] == m for u in in_prefix) for m in MIXES]
            assert max(per_mix) - min(per_mix) <= 1
        assert sum(counts[look]["per_worker"]) == 3 * size


def test_round_robin_plan_validation():
    with pytest.raises(ValueError):
        round_robin_plan(10, mixes=("a", "a"))
    with pytest.raises(ValueError):
        round_robin_plan(10, workers=0)
    assert {u["worker"] for u in round_robin_plan(4, workers=1)} == {0}


# ----------------------------------------------------------------- small simulation sanity


def test_small_simulation_sanity(plan):
    strong = [
        run_sequential_gate(plan, _draw((130, 130, 130), 243, seed=s), 3.5) for s in range(20)
    ]
    assert all(r["passes"] for r in strong)
    assert np.mean([r["worlds_per_mix_used"] for r in strong]) <= 122
    null = [run_sequential_gate(plan, _draw((0, 0, 0), 243, seed=100 + s), 3.5) for s in range(20)]
    assert not any(r["passes"] for r in null)
    assert all(r["decision"] in ("STOP_FUTILE", "FINAL_FAIL") for r in null)
