"""M3 Phase R harness: pure parts (no game is played)."""

from __future__ import annotations

from research.redesign_m3_20261008 import phase_r, rule_spec
from research.sequential_phase_r import hooks


def test_pinned_plan_and_forwards():
    plan = phase_r.plan_for(smoke=False)
    assert plan.sha256() == phase_r.PLAN_SHA256 and plan.look_sizes == (11, 22, 32)
    assert phase_r.VECTOR61_FORWARD == "rowwise" and phase_r.EGO2S_FORWARD == "rowwise"
    assert phase_r.SIM_ENGINE == "grid" and phase_r.VETO == "v8" and phase_r.VETO_LAMBDA == 8.0


def test_heroes_cover_the_rule_cells():
    rule = rule_spec.m3_rule()
    labels = {h["hero"]: h["seeds"] for h in phase_r.heroes()}
    for cell in rule["cells"].values():
        for who in (cell["hero"], cell["baseline"]):
            assert set(cell["seeds"]) <= set(labels[who])
    assert labels[phase_r.CONTROL_ARM] == [0]  # reported only
    assert phase_r.CONTROL_ARM not in {c["hero"] for c in rule["cells"].values()}
    assert {c["hero"] for c in rule["cells"].values()} == {phase_r.CANDIDATE}


def test_episode_count_matches_the_preregistered_budget():
    plan = phase_r.plan_for(smoke=False)
    banks = {s: list(range(1000 * s, 1000 * s + 32)) for s in rule_spec.TRAINING_SEEDS}
    n = 0
    for look in range(3):
        for u in hooks.look_units(
            plan, look, banks, phase_r.heroes(), phase_r.controls(), phase_r.MAX_BATCH
        ):
            n += len(u["worlds"])
    # 12 hero-seed pairs x 3 mixes x 32 worlds + 11 control pairs x 3 mixes
    assert n == 12 * 3 * 32 + 11 * 3 == 1185


def test_prefix_identical_ignores_only_the_veto_probe():
    a = {"mass_integral": 1.0, "probes": {"safety_veto": {"x": 1}, "y": 2}}
    b = {"mass_integral": 1.0, "probes": {"safety_veto": {"x": 9}, "y": 2}}
    assert phase_r.prefix_identical(a, b)
    assert not phase_r.prefix_identical(a, dict(b, mass_integral=1.5))


def test_record_names_are_unique_per_key():
    keys = [
        (h, s, m, w)
        for h in (phase_r.CANDIDATE, phase_r.INCUMBENT, phase_r.CHAMPION, phase_r.CONTROL_ARM)
        for s in (0, 1)
        for m in ("frozen", "mixed")
        for w in (1, 2)
    ]
    names = {
        phase_r.record_name({"hero": h, "seed": s, "mix": m, "world_seed": w})
        for h, s, m, w in keys
    }
    assert len(names) == len(keys)
    assert phase_r.record_name(
        {"hero": "incumbent", "seed": 0, "mix": "frozen", "world_seed": 1, "control": True}
    ).startswith("control-")


def test_unit_rosters_come_from_the_full_bank():
    from research.apex_safety_20260926 import dev_screen

    plan = phase_r.plan_for(smoke=False)
    banks = phase_r.make_banks(plan.n_worlds, phase_r.PHASE_R_NS)
    full = {
        (s, r["mix"], int(r["world_seed"])): [x["member_sha256"] for x in r["slots"]]
        for s, bank in banks.items()
        for r in dev_screen._design_rows(bank)
    }
    for look in range(3):
        for u in hooks.look_units(
            plan, look, banks, phase_r.heroes(), phase_r.controls(), phase_r.MAX_BATCH
        ):
            rows = phase_r.bank_rows(banks[u["seed"]], u["mix"])
            for w in u["worlds"]:
                got = [x["member_sha256"] for x in rows[int(w)]["slots"]]
                assert got == full[(u["seed"], u["mix"], int(w))]
