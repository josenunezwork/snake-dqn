"""Frozen amendment-03 endpoint and decision tests."""

from research.bl_density.analysis import finalize_cpu, summarize


def _record(arm="A", mix="scripted", index=0, *, complete=True, avoidable=True, cpu=2.0):
    record = {
        "episode_id": f"{arm}/{mix}/{index}",
        "arm": arm,
        "mix": mix,
        "complete": complete,
        "decisions": 100,
        "cpu_seconds": cpu,
        "wall_seconds": 3.0,
        "h300_calls": 1,
        "h300_branch_decisions": 50,
        "h300_cpu_seconds": 0.5,
        "lengths": [100, 800],
        "sampled": [
            {
                "length": 800,
                "prefilter": True,
                "h300_doom": True,
                "natural_doom": True,
                "features": [float(index)],
                "witnessed_avoidable": avoidable,
                "death_cause": "DEATH_SELF",
                "all_snake_lengths": [800, 20, 20, 20, 20, 20],
                "total_occupied_cells": 900,
                "map_cells": 12035,
                "hero_occupancy": 800 / 12035,
                "nearest_opponent_head_manhattan": 5,
                "nearest_opponent_body_manhattan": 3,
                "wall_distance_ahead": 10,
            }
        ],
    }
    if arm == "C-P1":
        target = (900, 1200, 1500)[index % 3]
        record["construction"] = {"fixture": {"body_lengths": [target, 600, 400, 300, 150, 150]}}
    return record


def _full_records():
    rows = []
    for arm in ("A", "B70m", "B70p", "C-P1"):
        for mix in ("scripted", "frozen", "mixed"):
            for index in range(4):
                rows.append(_record(arm, mix, index, cpu=1.0 if arm == "A" else 0.2))
    return rows


def test_primary_rates_charge_overhead_and_preserve_moment_rate():
    value = summarize([_record()], ["q_gap"], overhead_cpu_seconds=2.0)
    cell = value["cells"]["A"]["scripted"]
    assert cell["doom_moments_per_cpu_second"] == 0.25
    assert cell["witnessed_avoidable_doom_roots_per_total_cpu_second"] == 0.25
    assert cell["distinct_doom_bearing_episodes_per_total_cpu_second"] == 0.25
    assert cell["doom_moments_per_hero_decision"] == 0.01
    assert value["total_cpu_seconds"] == 4.0


def test_unassessed_roots_do_not_become_non_avoidable():
    record = _record()
    record["sampled"].append({**record["sampled"][0], "witnessed_avoidable": None})
    cell = summarize([record], ["q_gap"])["cells"]["A"]["scripted"]
    assert cell["assessed_confirmed_doom_roots"] == 1
    assert cell["unassessed_confirmed_doom_roots"] == 1
    assert cell["witnessed_avoidable_fraction"] == 1.0


def test_incomplete_precedence_and_partial_cpu_are_visible_in_total_cost():
    value = summarize([_record(complete=False)], ["q_gap"])
    assert value["run_status"] == "INCOMPLETE"
    assert value["partial_episode_cpu_seconds"] == 2.0
    assert value["total_cpu_seconds"] == 2.0
    assert value["decisions"]["B70m"]["scripted"]["status"] == "INCOMPLETE"


def test_minimum_map_rule_rejects_before_small_sample_insufficiency():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "B70m":
            record["cpu_seconds"] = 1.0
            record["sampled"][0]["map_cells"] = 8349
            record["sampled"][0]["length"] = 1200
    value = summarize(rows, ["q_gap"])
    decision = value["decisions"]["B70m"]["scripted"]
    assert decision["status"] == "REJECTED"
    assert decision["reasons"] == ["minimum_map_rule_violation"]


def test_rejects_below_two_x_yield_after_valid_map_assessment():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "B70m":
            record["cpu_seconds"] = 1.0
    value = summarize(rows, ["q_gap"])
    decision = value["decisions"]["B70m"]["scripted"]
    assert decision["status"] == "REJECTED"
    assert "avoidable_doom_yield_below_2x_A" in decision["reasons"]


def test_pooled_zero_a_avoidable_is_insufficient_without_extension():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "A":
            record["sampled"][0]["witnessed_avoidable"] = False
    value = summarize(rows, ["q_gap"])
    assert value["decisions"]["B70p"]["mixed"] == {
        "status": "INSUFFICIENT",
        "reasons": ["pooled_A_zero_witnessed_avoidable_dooms"],
    }


def test_zero_a_mix_fraction_has_no_division_by_zero_or_ratio_claim():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "A" and record["mix"] == "scripted":
            record["sampled"][0]["witnessed_avoidable"] = False
    decision = summarize(rows, ["q_gap"])["decisions"]["B70p"]["scripted"]
    assert decision["avoidable_fraction_ratio_to_A"] is None


def test_yield_rejection_is_not_masked_by_insufficient_classifier():
    rows = _full_records()
    rows = [
        record
        for record in rows
        if not (
            record["arm"] == "B70m"
            and record["mix"] == "scripted"
            and "/0" not in record["episode_id"]
        )
    ]
    for record in rows:
        if record["arm"] == "B70m":
            record["cpu_seconds"] = 1.0
    decision = summarize(rows, ["q_gap"])["decisions"]["B70m"]["scripted"]
    assert decision["status"] == "REJECTED"
    assert "avoidable_doom_yield_below_2x_A" in decision["reasons"]


def test_overhead_is_equal_per_attempted_episode_and_partial_cpu_stays_in_its_cell():
    complete = _record(cpu=2.0)
    partial = _record("B70m", complete=False, cpu=2.0)
    value = summarize([complete, partial], ["q_gap"], overhead_cpu_seconds=2.0)
    assert value["cells"]["A"]["scripted"]["attributed_total_cpu_seconds"] == 3.0
    assert value["cells"]["B70m"]["scripted"]["attributed_total_cpu_seconds"] == 3.0


def test_c_p1_retains_realism_by_each_registered_start_length():
    value = summarize(_full_records(), ["q_gap"])
    by_length = value["realism"]["scripted"]["C-P1"]["by_injected_start_length"]
    assert set(by_length) == {"900", "1200", "1500"}


def test_b70p_geometry_uses_common_first_six_and_all_opponent_lengths():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "B70p":
            record["sampled"][0]["all_snake_lengths"].extend([20, 20])
    realism = summarize(rows, ["q_gap"])["realism"]["scripted"]["B70p"]
    assert realism["geometry_ks"]["status"] == "OK"
    assert realism["all_opponent_length_ks"]["fields"]["opponent_length"] == {
        "statistic": 0.0,
        "n_reference": 20,
        "n_candidate": 28,
    }


def test_missing_natural_rho95_is_insufficient_not_a_passing_map_assessment():
    rows = _full_records()
    for record in rows:
        if record["arm"] == "A":
            record["sampled"][0].pop("map_cells")
    decision = summarize(rows, ["q_gap"])["decisions"]["B70m"]["scripted"]
    assert decision == {"status": "INSUFFICIENT", "reasons": ["minimum_map_rho95_insufficient"]}


def test_finalize_cpu_reprices_rates_and_decisions_without_recomputing_realism():
    rows = _full_records()
    value = summarize(rows, ["q_gap"])
    fixed_realism = value["realism"]
    assert value["decisions"]["B70m"]["scripted"]["status"] == "ADMITTED_FOR_A_TRANSFER_TEST"
    finalized = finalize_cpu(value, rows, overhead_cpu_seconds=120.0)
    assert finalized["realism"] is fixed_realism
    assert finalized["decisions"]["B70m"]["scripted"]["status"] == "REJECTED"
    assert (
        "avoidable_doom_yield_below_2x_A" in finalized["decisions"]["B70m"]["scripted"]["reasons"]
    )
    denominators = sum(
        cell["attributed_total_cpu_seconds"]
        for arm in finalized["cells"].values()
        for cell in arm.values()
    )
    assert abs(denominators - finalized["total_cpu_seconds"]) < 1e-9
