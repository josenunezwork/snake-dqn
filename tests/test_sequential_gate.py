"""Tests for src/evaluation/sequential_gate.py (group-sequential strict gate, opt-in)."""

import hashlib
import json
import math
from statistics import NormalDist

import numpy as np
import pytest

from src.evaluation.sequential_gate import (
    SequentialGatePlan,
    band_check,
    conditional_power,
    look_sizes_for,
    paired_band_check,
    plan_paired_band_check,
    plan_pooled_band_check,
    pooled_band_check,
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
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, futility_policy="sometimes")
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, band_policy="recheck_every_look")
    with pytest.raises(ValueError):
        sequential_gate_plan(243, mde=30.0, band_margin_z=-1.0)


def test_policies_are_part_of_the_hashed_plan(plan):
    frozen = plan.as_dict()
    assert frozen["futility_policy"] == "followed"
    assert frozen["band_policy"] == "block_at_stop"
    assert frozen["band_margin_z"] == pytest.approx(1.645)
    other = sequential_gate_plan(243, mde=30.0, futility_policy="overridable").as_dict()
    assert json.dumps(other, sort_keys=True) != json.dumps(frozen, sort_keys=True)


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


def test_band_failure_at_qualifying_look_fails_and_does_not_delay(plan):
    deltas = _draw((300, 300, 300), 122, seed=1)
    first = sequential_decision(plan, 0, _prefix(deltas, 61), 3.5, [False])
    assert first["decision"] == "STOP_FAIL_BANDS" and first["stopped"]
    assert first["valid"] and not first["passes"]
    assert first["history"][0]["bands_judged"]
    second = sequential_decision(plan, 1, deltas, 3.5, [False, True])
    assert not second["valid"] and not second["passes"]
    assert second["invalid_reason"] == "already_stopped_for_bands_at_look_0"
    walked = run_sequential_gate(plan, deltas, 3.5, [False, True, True, True])
    assert walked["decision"] == "STOP_FAIL_BANDS" and walked["look"] == 0


def test_bands_are_ignored_before_the_qualifying_look(plan):
    deltas = _draw((300, -300, 300), 122, seed=2)  # NI fails, so no look qualifies
    out = sequential_decision(plan, 1, deltas, 3.5, [False, False])
    assert out["decision"] == "CONTINUE" and out["valid"]
    assert not any(h["bands_judged"] for h in out["history"])


def test_band_check_margin_shrinks_to_point_rule_at_final_n():
    rng = np.random.default_rng(3)
    values = (0.8 + 0.1 * rng.standard_normal(243)).tolist()
    final = band_check(values, 0.75, 1.8, n_final=243)
    assert final["margin"] == 0.0 and final["passes"]
    assert final["lower_effective"] == 0.75
    early = band_check(values[:61], 0.75, 1.8, n_final=243)
    sd = early["sample_std"]
    expected = 1.645 * sd * (1 / math.sqrt(61) - 1 / math.sqrt(243))
    assert early["margin"] == pytest.approx(expected)
    assert early["lower_effective"] == pytest.approx(0.75 + expected)
    edge = [0.75 + early["margin"] / 2 + v - early["mean"] for v in values[:61]]
    assert not band_check(edge, 0.75, 1.8, n_final=243)["passes"]
    assert band_check(edge, 0.75, 1.8, n_final=243, margin_z=0.0)["passes"]
    with pytest.raises(ValueError):
        band_check([0.5], 0.0, 1.0, n_final=243)
    with pytest.raises(ValueError):
        band_check([0.5, float("nan")], 0.0, 1.0, n_final=243)


@pytest.mark.parametrize("d", [0.0, 0.5, 1.0, 1.645, 2.0, 3.0])
@pytest.mark.parametrize("ratio", [61 / 243, 0.5, 0.75])
def test_interim_band_never_looser_than_fixed_at_its_95_percent_catch_point(d, ratio):
    # Interim pass probability at true violation D (final-N SE units), known sd.
    z = 1.645
    interim = N.cdf(-(d * math.sqrt(ratio) + z * (1 - math.sqrt(ratio))))
    fixed = N.cdf(-d)
    assert interim <= max(fixed, N.cdf(-z)) + 1e-12
    if d <= z:
        assert interim <= fixed + 1e-12


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
    assert overridden["futility_overrides"] == [0]
    assert not overridden["valid"] and overridden["unregistered_futility_override"]
    assert (
        overridden["invalid_reason"] == "futility_stop_at_look_0_overridden_under_policy_followed"
    )
    open_plan = sequential_gate_plan(243, mde=30.0, futility_policy="overridable")
    allowed = sequential_decision(open_plan, 1, losers, 3.5, [True, True])
    assert allowed["valid"] and allowed["futility_overrides"] == [0]
    assert not allowed["unregistered_futility_override"]
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
    assert not full["valid"] and full["unregistered_futility_override"]
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


def _skew_probe():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / (
        "research/sequential_gate_validation_20261002/skew_probe.py"
    )
    spec = importlib.util.spec_from_file_location("seq_skew_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skew_probe_rates_and_resample_check(plan, tmp_path):
    probe = _skew_probe()
    normal = probe.crossing_rates(
        probe.gamma_sampler(0.0), plan, reps=20_000, seed=3, centre=0.0, ni=False
    )
    assert 0.010 < normal["any_look"] < 0.024 and normal["look_1"] < 1e-3
    skewed = probe.crossing_rates(
        probe.gamma_sampler(-2.83), plan, reps=20_000, seed=3, centre=0.0, ni=False
    )
    assert skewed["any_look"] > normal["any_look"]
    saved = tmp_path / "deltas.json"
    saved.write_text(json.dumps((np.random.default_rng(4).standard_normal(300) * 140).tolist()))
    ok = probe.part_resample(str(saved), 20_000, 243, 3.5, (0.25, 0.5, 0.75, 1.0))
    assert ok["passes"] and ok["look_sizes"] == [61, 122, 183, 243]


# ------------------------------------------- paired noninferiority bands (2026-10-03, opt-in)

LEGACY_PLAN_KEYS = {
    "method", "mixes", "scripted_mix", "n_max", "look_sizes", "fractions", "family_alpha",
    "efficacy_alpha_per_mix", "efficacy_boundaries", "efficacy_nominal_p",
    "efficacy_alpha_spent", "ni_alpha", "ni_boundaries", "ni_nominal_p", "ni_alpha_spent",
    "required_successes", "mde", "futility_cp", "futility_policy", "band_policy",
    "band_margin_z", "spending", "multiplicity", "futility",
}  # fmt: skip


def _sha(plan_dict):
    text = json.dumps(plan_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _paired_plan(**overrides):
    kwargs = dict(
        band_policy="paired_ni_at_stop", band_ni_margin=0.05, band_alpha=0.05,
        band_bound="pointwise",
    )  # fmt: skip
    kwargs.update(overrides)
    return sequential_gate_plan(249, mde=30.0, **kwargs)


@pytest.mark.parametrize(
    "n_max, digest",
    [
        (243, "63f5e8eec780e20527481fa5321d877760773d1078fe619cf60bfbfc93b2c36c"),
        (249, "6eb9fb20cd6467c1e6309dd3a9868d34506302d023a0466aa96960e25b37a45b"),
    ],
)
def test_default_band_policy_plan_dict_and_hash_are_unchanged(n_max, digest):
    # Golden sha256 values were computed on main (5a82a6d) before paired bands existed.
    frozen = sequential_gate_plan(n_max, mde=30.0).as_dict()
    assert set(frozen) == LEGACY_PLAN_KEYS
    assert _sha(frozen) == digest


def test_paired_band_fields_are_part_of_the_hashed_plan():
    plan = _paired_plan()
    frozen = plan.as_dict()
    assert frozen["band_policy"] == "paired_ni_at_stop"
    assert frozen["band_ni_margin"] == 0.05 and frozen["band_alpha"] == 0.05
    assert frozen["band_bound"] == "pointwise" and frozen["band_floor"] is None
    assert frozen["band_nominal_p"] == [0.05] * 4
    assert set(frozen) == LEGACY_PLAN_KEYS | {
        "band_ni_margin", "band_alpha", "band_bound", "band_nominal_p", "band_floor",
    }  # fmt: skip
    digests = {
        _sha(p.as_dict())
        for p in (
            plan,
            _paired_plan(band_ni_margin=0.075),
            _paired_plan(band_alpha=0.10),
            _paired_plan(band_bound="rci_obf"),
            _paired_plan(band_floor=0.3),
            sequential_gate_plan(249, mde=30.0),
        )
    }
    assert len(digests) == 6
    assert json.loads(json.dumps(frozen)) == frozen
    # Efficacy, NI and futility parts are untouched by the band policy.
    legacy = sequential_gate_plan(249, mde=30.0).as_dict()
    for key in LEGACY_PLAN_KEYS - {"band_policy"}:
        assert frozen[key] == legacy[key]


def test_rci_band_levels_are_obf_spending_at_band_alpha():
    from src.evaluation.screen_stats import futility_plan

    plan = _paired_plan(band_bound="rci_obf", band_alpha=0.10)
    spent = futility_plan(plan.look_sizes, alpha=0.10, max_size=plan.n_max)
    assert plan.band_nominal_p == pytest.approx([N.cdf(-c) for c in spent.boundaries])
    assert all(a < b for a, b in zip(plan.band_nominal_p, plan.band_nominal_p[1:]))
    assert plan.band_nominal_p[-1] < 0.10
    assert sum(plan.band_nominal_p) >= 0.10  # any-look spending is at most the sum


def test_paired_band_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):  # paired fields under the default policy
        sequential_gate_plan(249, mde=30.0, band_ni_margin=0.05)
    with pytest.raises(ValueError):
        sequential_gate_plan(249, mde=30.0, band_floor=0.3)
    for bad in (
        dict(band_ni_margin=None),
        dict(band_ni_margin=0.0),
        dict(band_ni_margin=math.inf),
        dict(band_ni_margin=True),
        dict(band_alpha=None),
        dict(band_alpha=0.5),
        dict(band_bound=None),
        dict(band_bound="holm"),
        dict(band_floor=math.nan),
    ):
        with pytest.raises((TypeError, ValueError)):
            _paired_plan(**bad)
    with pytest.raises(ValueError):
        sequential_gate_plan(249, mde=30.0, band_policy="paired")


def test_paired_band_check_matches_the_scripted_ni_oracle():
    rng = np.random.default_rng(11)
    for trial in range(40):
        n = int(rng.integers(5, 250))
        inc = rng.uniform(0, 1, n)
        cand = np.clip(inc + rng.normal(-0.03 + 0.002 * trial, 0.23, n), 0, 1)
        margin, alpha = float(rng.choice([0.02, 0.05, 0.075])), float(rng.choice([0.05, 0.1]))
        mine = paired_band_check(cand.tolist(), inc.tolist(), margin=margin, nominal_p=alpha)
        oracle = scripted_noninferiority((cand - inc).tolist(), margin, alpha=alpha)
        assert mine["passes"] == oracle["passes"] == mine["passes_ni"]
        assert mine["lower_bound"] == pytest.approx(oracle["lower_bound"], abs=1e-12)
        assert mine["candidate_mean"] == pytest.approx(cand.mean())
        assert mine["incumbent_mean"] == pytest.approx(inc.mean())


def test_paired_band_check_is_strict_floor_and_zero_spread():
    inc = [0.5] * 10
    same = paired_band_check([0.45] * 10, inc, margin=0.05, nominal_p=0.05)
    assert same["sample_std"] == 0.0
    assert same["lower_bound"] == pytest.approx(-0.05)
    exact = paired_band_check([0.5 - 0.25] * 4, [0.5] * 4, margin=0.25, nominal_p=0.05)
    assert exact["lower_bound"] == -0.25 and not exact["passes"]  # strict >
    ok = paired_band_check([0.31, 0.29, 0.30], [0.30, 0.30, 0.30], margin=0.05, nominal_p=0.05)
    assert ok["passes"]
    floored = paired_band_check(
        [0.31, 0.29, 0.30], [0.30, 0.30, 0.30], margin=0.05, nominal_p=0.05, floor=0.35
    )
    assert floored["passes_ni"] and not floored["passes_floor"] and not floored["passes"]
    assert paired_band_check(
        [0.31, 0.29, 0.30], [0.30] * 3, margin=0.05, nominal_p=0.05, floor=0.30
    )["passes"]


def test_paired_band_check_validation():
    with pytest.raises(ValueError):
        paired_band_check([0.5, 0.6], [0.5], margin=0.05, nominal_p=0.05)
    with pytest.raises(ValueError):
        paired_band_check([0.5], [0.5], margin=0.05, nominal_p=0.05)
    with pytest.raises(ValueError):
        paired_band_check([0.5, math.nan], [0.5, 0.5], margin=0.05, nominal_p=0.05)
    with pytest.raises(ValueError):
        paired_band_check([0.5, 0.6], [0.5, 0.5], margin=0.0, nominal_p=0.05)
    with pytest.raises(TypeError):
        paired_band_check([0.5, 0.6], [0.5, 0.5], margin=True, nominal_p=0.05)
    with pytest.raises(ValueError):
        paired_band_check([0.5, 0.6], [0.5, 0.5], margin=0.05, nominal_p=0.5)
    with pytest.raises(ValueError):
        paired_band_check([0.5, 0.6], [0.5, 0.5], margin=0.05, nominal_p=0.05, floor=math.inf)
    for bad_floor in (True, "0.3"):
        with pytest.raises(ValueError):
            paired_band_check([0.5, 0.6], [0.5, 0.5], margin=0.05, nominal_p=0.05, floor=bad_floor)


def test_direct_plan_construction_rejects_inconsistent_band_fields():
    import dataclasses

    legacy = sequential_gate_plan(249, mde=30.0)
    with pytest.raises(ValueError):
        dataclasses.replace(legacy, band_ni_margin=0.05)
    paired = _paired_plan()
    with pytest.raises(ValueError):
        dataclasses.replace(paired, band_nominal_p=(0.05,))
    with pytest.raises(ValueError):
        dataclasses.replace(paired, band_alpha=None)
    assert dataclasses.replace(paired, band_floor=0.3).band_floor == 0.3


def test_plan_paired_band_check_uses_the_look_level_and_prefix():
    plan = _paired_plan(band_bound="rci_obf")
    rng = np.random.default_rng(5)
    inc = rng.uniform(0, 1, 249)
    cand = inc + rng.normal(0.05, 0.2, 249)
    for look, n in enumerate(plan.look_sizes):
        out = plan_paired_band_check(plan, look, cand[:n].tolist(), inc[:n].tolist())
        direct = paired_band_check(
            cand[:n].tolist(), inc[:n].tolist(), margin=0.05, nominal_p=plan.band_nominal_p[look]
        )
        assert out["look"] == look and out["band_bound"] == "rci_obf"
        assert out["lower_bound"] == direct["lower_bound"]
        assert out["passes"] == direct["passes"]
    with pytest.raises(ValueError):
        plan_paired_band_check(plan, 0, cand[:62].tolist(), inc[:62].tolist())
    with pytest.raises(ValueError):
        plan_paired_band_check(plan, 4, cand.tolist(), inc.tolist())
    with pytest.raises(ValueError):  # default plan has no paired band
        plan_paired_band_check(sequential_gate_plan(249, mde=30.0), 0, cand[:63], inc[:63])
    with pytest.raises(TypeError):
        plan_paired_band_check(plan.as_dict(), 0, cand[:63], inc[:63])


def _paired_band_verdicts(plan, cand, inc):
    return [
        all(plan_paired_band_check(plan, k, cand[m][:n], inc[m][:n])["passes"] for m in plan.mixes)
        for k, n in enumerate(plan.look_sizes)
    ]


@pytest.mark.parametrize("shift, decision", [(0.10, "STOP_PASS"), (-0.10, "STOP_FAIL_BANDS")])
def test_paired_bands_plug_into_the_sequential_decision(shift, decision):
    plan = _paired_plan()
    mass = _draw((300, 300, 300), 249, seed=7)
    rng = np.random.default_rng(8)
    inc = {m: rng.uniform(0.3, 0.9, 249).tolist() for m in MIXES}
    cand = {m: (np.asarray(inc[m]) + shift + rng.normal(0, 0.1, 249)).tolist() for m in MIXES}
    bands = _paired_band_verdicts(plan, cand, inc)
    out = run_sequential_gate(plan, mass, 3.5, bands)
    assert out["decision"] == decision and out["look"] == 0
    assert out["plan"]["band_policy"] == "paired_ni_at_stop"


def test_paired_band_pointwise_level_at_the_margin_by_simulation():
    # Vectorized replica of paired_band_check at a fixed look: P(pass | true delta = -M)
    # equals the nominal level (here 0.05) for normal per-world deltas.
    from src.scripts.eval_stats import student_t_isf

    rng = np.random.default_rng(9)
    n, margin, reps = 63, 0.05, 40000
    d = rng.normal(-margin, 0.23, size=(reps, n))
    lower = d.mean(axis=1) - student_t_isf(0.05, n - 1) * d.std(axis=1, ddof=1) / math.sqrt(n)
    rate = float((lower > -margin).mean())
    assert abs(rate - 0.05) < 0.006
    one = paired_band_check((d[0] + 0.5).tolist(), [0.5] * n, margin=margin, nominal_p=0.05)
    assert one["lower_bound"] == pytest.approx(lower[0], abs=1e-12)


# ------------------------------------- survival band v2: pooled_ni_continue (2026-10-06, opt-in)


@pytest.mark.parametrize(
    "kwargs, digest",
    [
        (
            dict(n_max=249, mde=30.0, band_bound="pointwise", band_floor=None),
            "9541c160417c93c9c922cc102ab07e1066d533b40c38de387384811c5de26a95",
        ),
        (
            dict(n_max=275, mde=65.0, band_bound="rci_obf", band_floor=0.3),
            "7a9505d144bc085e32a17e8c45b05c319f566a084d7e4212ad34aaf1637c8acd",
        ),
    ],
)
def test_paired_ni_at_stop_plan_hash_is_unchanged_by_band_v2(kwargs, digest):
    # Golden sha256 values computed on frp3-strict (4eb75af) before survival band v2; the
    # second is the FRP-v3 strict gate's frozen plan (research/frp3_strict_20261005).
    kw = dict(kwargs)
    n_max = kw.pop("n_max")
    frozen = sequential_gate_plan(
        n_max, band_policy="paired_ni_at_stop", band_ni_margin=0.05, band_alpha=0.05, **kw
    ).as_dict()
    assert "band_mix_margin" not in frozen
    assert _sha(frozen) == digest


def _pooled_plan(**overrides):
    kwargs = dict(
        band_policy="pooled_ni_continue", band_ni_margin=0.05, band_alpha=0.05,
        band_bound="rci_obf", band_floor=0.30, band_mix_margin=0.10,
    )  # fmt: skip
    kwargs.update(overrides)
    return sequential_gate_plan(275, mde=65.0, **kwargs)


def test_pooled_plan_fields_are_hashed_and_rci_levels():
    from src.evaluation.screen_stats import futility_plan

    plan = _pooled_plan()
    frozen = plan.as_dict()
    assert set(frozen) == LEGACY_PLAN_KEYS | {
        "band_ni_margin", "band_alpha", "band_bound", "band_nominal_p", "band_floor",
        "band_mix_margin",
    }  # fmt: skip
    assert frozen["band_policy"] == "pooled_ni_continue" and frozen["band_mix_margin"] == 0.10
    spent = futility_plan(plan.look_sizes, alpha=0.05, max_size=plan.n_max)
    assert plan.band_nominal_p == pytest.approx([N.cdf(-c) for c in spent.boundaries])
    assert plan.band_nominal_p == pytest.approx(
        (9.1224e-05, 5.6293e-3, 2.2124e-2, 4.2624e-2), rel=1e-3
    )
    digests = {
        _sha(p.as_dict())
        for p in (
            plan,
            _pooled_plan(band_mix_margin=0.075),
            _pooled_plan(band_ni_margin=0.04),
            _pooled_plan(band_floor=None),
            _pooled_plan(band_alpha=0.10),
            sequential_gate_plan(
                275, mde=65.0, band_policy="paired_ni_at_stop", band_ni_margin=0.05,
                band_alpha=0.05, band_bound="rci_obf", band_floor=0.3,
            ),
        )
    }  # fmt: skip
    assert len(digests) == 6
    legacy = sequential_gate_plan(275, mde=65.0).as_dict()
    for key in LEGACY_PLAN_KEYS - {"band_policy"}:
        assert frozen[key] == legacy[key]


def test_pooled_plan_rejects_bad_inputs():
    import dataclasses

    with pytest.raises(ValueError):  # pointwise is refused (bands judged at several looks)
        _pooled_plan(band_bound="pointwise")
    for bad in (dict(band_mix_margin=None), dict(band_mix_margin=0.0),
                dict(band_mix_margin=math.inf), dict(band_mix_margin=True),
                dict(band_ni_margin=None), dict(band_alpha=None)):  # fmt: skip
        with pytest.raises((TypeError, ValueError)):
            _pooled_plan(**bad)
    with pytest.raises(ValueError):  # mix margin under the other policies
        sequential_gate_plan(275, mde=65.0, band_mix_margin=0.1)
    with pytest.raises(ValueError):
        _paired_plan(band_mix_margin=0.1)
    with pytest.raises(ValueError):
        dataclasses.replace(_paired_plan(), band_mix_margin=0.1)
    with pytest.raises(ValueError):
        dataclasses.replace(_pooled_plan(), band_mix_margin=None)
    with pytest.raises(ValueError):
        dataclasses.replace(_pooled_plan(), band_bound="pointwise")


def _oracle_pooled(cand, inc, pooled_margin, mix_margin, p, floor):
    from src.scripts.eval_stats import student_t_isf

    d = np.asarray([np.asarray(cand[m]) - np.asarray(inc[m]) for m in MIXES])
    n = d.shape[1]
    t = student_t_isf(p, n - 1)
    pooled = d.mean(axis=0)
    plower = pooled.mean() - t * pooled.std(ddof=1) / math.sqrt(n)
    lowers = d.mean(axis=1) - t * d.std(axis=1, ddof=1) / math.sqrt(n)
    floor_ok = True if floor is None else all(np.mean(cand[m]) >= floor for m in MIXES)
    return (
        plower,
        lowers,
        bool(plower > -pooled_margin and np.all(lowers > -mix_margin) and floor_ok),
    )


def test_pooled_band_check_matches_a_numpy_oracle():
    rng = np.random.default_rng(21)
    for trial in range(40):
        n = int(rng.integers(5, 280))
        inc = {m: rng.uniform(0, 1, n).tolist() for m in MIXES}
        shift = rng.normal(-0.03, 0.04, 3)
        cand = {
            m: np.clip(np.asarray(inc[m]) + rng.normal(shift[j], 0.35, n), 0, 1).tolist()
            for j, m in enumerate(MIXES)
        }
        pm, mm = float(rng.choice([0.03, 0.05])), float(rng.choice([0.075, 0.10]))
        p = float(rng.choice([0.005, 0.0426, 0.1]))
        floor = [None, 0.3, 0.6][trial % 3]
        out = pooled_band_check(
            cand, inc, pooled_margin=pm, mix_margin=mm, nominal_p=p, floor=floor
        )
        plower, lowers, passes = _oracle_pooled(cand, inc, pm, mm, p, floor)
        assert out["pooled"]["lower_bound"] == pytest.approx(plower, abs=1e-12)
        for j, m in enumerate(MIXES):
            assert out["per_mix"][m]["lower_bound"] == pytest.approx(lowers[j], abs=1e-12)
        assert out["passes"] is passes
        assert out["n"] == n and out["df"] == n - 1 and out["mixes"] == list(MIXES)


def test_pooled_band_check_components_strict_and_floor():
    inc = {m: [0.5] * 6 for m in MIXES}
    # zero spread: bounds are the means; strict >
    at_margin = {m: [0.25] * 6 for m in MIXES}  # exactly representable: 0.25 - 0.5 = -0.25
    out = pooled_band_check(at_margin, inc, pooled_margin=0.25, mix_margin=0.5, nominal_p=0.05)
    assert out["pooled"]["lower_bound"] == -0.25 and not out["passes_pooled"]
    assert out["passes_mix"] and not out["passes"]
    # one mix down 0.15: pooled -0.05 fails, that mix's catastrophic floor fails too
    one = {"frozen": [0.5] * 6, "scripted": [0.35] * 6, "mixed": [0.5] * 6}
    out = pooled_band_check(one, inc, pooled_margin=0.06, mix_margin=0.10, nominal_p=0.05)
    assert out["passes_pooled"] and not out["per_mix"]["scripted"]["passes_ni"]
    assert not out["passes_mix"] and not out["passes"]
    # one mix down 0.08: tolerated (pooled -0.027, per mix above -0.10)
    ok = {"frozen": [0.5] * 6, "scripted": [0.42] * 6, "mixed": [0.5] * 6}
    assert pooled_band_check(ok, inc, pooled_margin=0.05, mix_margin=0.10, nominal_p=0.05)["passes"]
    floored = pooled_band_check(
        ok, inc, pooled_margin=0.05, mix_margin=0.10, nominal_p=0.05, floor=0.45
    )
    assert floored["passes_pooled"] and floored["passes_mix"] and not floored["passes_floor"]
    assert not floored["passes"] and not floored["per_mix"]["scripted"]["passes"]


def test_pooled_band_check_validation():
    inc = {m: [0.5, 0.5] for m in MIXES}
    cand = {m: [0.5, 0.6] for m in MIXES}
    kw = dict(pooled_margin=0.05, mix_margin=0.1, nominal_p=0.05)
    with pytest.raises(ValueError):
        pooled_band_check({**cand, "mixed": [0.5]}, inc, **kw)
    with pytest.raises(ValueError):
        pooled_band_check(cand, {"frozen": inc["frozen"]}, **kw)
    with pytest.raises(ValueError):
        pooled_band_check({m: [0.5] for m in MIXES}, {m: [0.5] for m in MIXES}, **kw)
    with pytest.raises(ValueError):
        pooled_band_check({**cand, "frozen": [0.5, math.nan]}, inc, **kw)
    with pytest.raises(TypeError):
        pooled_band_check(cand, inc, **{**kw, "mix_margin": True})
    with pytest.raises(ValueError):
        pooled_band_check(cand, inc, **{**kw, "pooled_margin": 0.0})
    with pytest.raises(ValueError):
        pooled_band_check(cand, inc, **{**kw, "nominal_p": 0.5})
    with pytest.raises(ValueError):
        pooled_band_check(cand, inc, **kw, floor=math.inf)


def test_plan_pooled_band_check_uses_plan_levels_and_prefix():
    plan = _pooled_plan()
    rng = np.random.default_rng(23)
    inc = {m: rng.uniform(0.3, 0.9, 275).tolist() for m in MIXES}
    cand = {m: (np.asarray(inc[m]) + rng.normal(0.0, 0.35, 275)).tolist() for m in MIXES}
    for look, n in enumerate(plan.look_sizes):
        out = plan_pooled_band_check(
            plan, look, {m: cand[m][:n] for m in MIXES}, {m: inc[m][:n] for m in MIXES}
        )
        direct = pooled_band_check(
            {m: cand[m][:n] for m in MIXES}, {m: inc[m][:n] for m in MIXES},
            pooled_margin=0.05, mix_margin=0.10, nominal_p=plan.band_nominal_p[look], floor=0.3,
        )  # fmt: skip
        assert out["look"] == look and out["policy"] == "pooled_ni_continue"
        assert out["passes"] == direct["passes"]
        assert out["pooled"] == direct["pooled"]
    with pytest.raises(ValueError):
        plan_pooled_band_check(plan, 0, {m: cand[m][:68] for m in MIXES},
                               {m: inc[m][:68] for m in MIXES})  # fmt: skip
    with pytest.raises(ValueError):
        plan_pooled_band_check(plan, 0, {"frozen": cand["frozen"][:69]}, inc)
    with pytest.raises(ValueError):
        plan_pooled_band_check(_paired_plan(), 0, cand, inc)
    with pytest.raises(TypeError):
        plan_pooled_band_check(plan.as_dict(), 0, cand, inc)


def _pooled_verdicts(plan, cand, inc):
    return [
        bool(
            plan_pooled_band_check(
                plan, k, {m: cand[m][:n] for m in MIXES}, {m: inc[m][:n] for m in MIXES}
            )["passes"]
        )
        for k, n in enumerate(plan.look_sizes)
    ]


def test_pooled_policy_continues_after_a_band_failure_and_passes_later():
    plan = _pooled_plan()
    mass = {m: [500.0 + (i % 7) for i in range(275)] for m in MIXES}  # qualifies at look 0
    bands = [False, False, True, True]
    out = run_sequential_gate(plan, mass, 9.9, bands)
    assert out["decision"] == "STOP_PASS" and out["look"] == 2 and out["passes"] and out["valid"]
    assert [h["decision"] for h in out["history"]] == [
        "CONTINUE_BANDS",
        "CONTINUE_BANDS",
        "STOP_PASS",
    ]
    assert all(h["bands_judged"] for h in out["history"])
    step = sequential_decision(plan, 1, {m: v[:138] for m, v in mass.items()}, 9.9, bands[:2])
    assert step["decision"] == "CONTINUE_BANDS" and not step["stopped"] and step["valid"]
    final = run_sequential_gate(plan, mass, 9.9, [False] * 4)
    assert final["decision"] == "FINAL_FAIL" and final["look"] == 3 and not final["passes"]
    last = run_sequential_gate(plan, mass, 9.9, [False, False, False, True])
    assert last["decision"] == "FINAL_PASS" and last["passes"]
    # a STOP_PASS earlier makes a later look invalid, as under every policy
    late = sequential_decision(plan, 3, mass, 9.9, [False, True, True, True])
    assert not late["valid"] and late["invalid_reason"].startswith("already_stopped_for_efficacy")
    # never STOP_FAIL_BANDS under this policy
    assert "STOP_FAIL_BANDS" not in {h["decision"] for h in final["history"]}


def test_pooled_policy_before_qualification_is_unchanged():
    plan = _pooled_plan()
    legacy = sequential_gate_plan(275, mde=65.0)
    for seed, means in ((31, (0.0, 0.0, 0.0)), (32, (40.0, 0.0, 0.0)), (33, (80.0, 80.0, 80.0))):
        mass = _draw(means, 275, sd=250.0, seed=seed)
        a = run_sequential_gate(plan, mass, 9.9, [True] * 4)
        b = run_sequential_gate(legacy, mass, 9.9, [True] * 4)
        assert (a["decision"], a["look"]) == (b["decision"], b["look"])


def test_pooled_policy_end_to_end_with_real_band_checks():
    plan = _pooled_plan()
    rng = np.random.default_rng(41)
    mass = _draw((400, 400, 400), 275, sd=150.0, seed=40)
    inc = {m: rng.uniform(0.3, 0.9, 275).tolist() for m in MIXES}
    good = {m: (np.asarray(inc[m]) + 0.06 + rng.normal(0, 0.05, 275)).tolist() for m in MIXES}
    out = run_sequential_gate(plan, mass, 9.9, _pooled_verdicts(plan, good, inc))
    assert out["decision"] == "STOP_PASS" and out["look"] == 0
    bad = {m: (np.asarray(inc[m]) - 0.2 + rng.normal(0, 0.05, 275)).tolist() for m in MIXES}
    out = run_sequential_gate(plan, mass, 9.9, _pooled_verdicts(plan, bad, inc))
    assert out["decision"] == "FINAL_FAIL" and out["look"] == 3
    assert out["history"][0]["decision"] == "CONTINUE_BANDS"


def test_pooled_rci_any_look_level_at_the_pooled_margin_by_simulation():
    # P(the pooled bound clears at ANY look | true pooled delta = -M) <= alpha (RCI).
    from src.scripts.eval_stats import student_t_isf

    plan = _pooled_plan()
    rng = np.random.default_rng(43)
    reps, margin = 20000, plan.band_ni_margin
    p = rng.normal(-margin, 0.2, size=(reps, plan.n_max))
    hit = np.zeros(reps, dtype=bool)
    for k, n in enumerate(plan.look_sizes):
        t = student_t_isf(plan.band_nominal_p[k], n - 1)
        seg = p[:, :n]
        hit |= seg.mean(axis=1) - t * seg.std(axis=1, ddof=1) / math.sqrt(n) > -margin
    rate = float(hit.mean())
    assert rate <= 0.05 + 0.006 and rate > 0.035
