"""T35 two-deviation rescue search: toy-world semantics plus one real-engine depth-1 check."""

from __future__ import annotations

import json

import pytest

from research.t35_two_deviation import events as ev
from research.t35_two_deviation import run as t35_run
from research.t35_two_deviation.search import (
    ControlMismatch,
    SearchParams,
    beam_key,
    select_beam,
    two_deviation_search,
)

Q = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]  # action 0 is always B8 greedy in the toy world


class ToyWorld:
    """Deterministic toy backend: a state is the tuple of deviations taken so far."""

    def __init__(self, fate, end: int = 5_000) -> None:
        self.fate = fate  # path -> death decision, or None for never
        self.end = end
        self.calls = {"expand": 0, "branch": 0, "confirm": 0}

    def _alive_through(self, path, decision: int, depth: int) -> tuple[bool, int | None, int]:
        death = self.fate(path)
        horizon = min(depth, self.end - decision)
        if death is None or death >= decision + horizon:
            return True, None, horizon
        return False, death, death - decision + 1

    def expand(self, state, decision, doom):
        self.calls["expand"] += 1
        outcomes, frames = {}, 0
        for action in range(1, 6):
            alive, death, executed = self._alive_through(
                state + ((decision, action),), decision, 300
            )
            outcomes[str(action)] = {
                "survives": alive,
                "death_decision": death,
                "frames_executed": executed,
                "horizon": min(300, self.end - decision),
            }
            frames += executed
        outcomes["0"] = {
            "survives": False,
            "death_decision": doom,
            "frames_executed": doom - decision + 1,
            "horizon": min(300, self.end - decision),
        }
        return {"greedy": 0, "q": Q, "outcomes": outcomes, "batch": {"actual_env_frames": frames}}

    def branch(self, state, decision, action, stop):
        self.calls["branch"] += 1
        path = state + ((decision, action),)
        death = self.fate(path)
        last = stop - 1 if death is None or death >= stop else death
        states = {t: path for t in range(decision + 1, last + 1)}
        return (None if last == stop - 1 and (death is None or death >= stop) else death,
                states, last - decision + 1)  # fmt: skip

    def confirm(self, state, decision, action, depth):
        self.calls["confirm"] += 1
        alive, death, executed = self._alive_through(state + ((decision, action),), decision, depth)
        return {"survives": alive, "death_decision": death, "env_frames": executed}


def _root(window):
    return {t: () for t in window}


def single_death(path) -> int:
    """Every single deviation (t, a) dies at 20 + a; the greedy doom is 20."""
    return 20 + path[0][1]


WINDOW = list(range(0, 21))  # 21 decisions -> 8 bins of 2-3 decisions
BEAM_T1 = [0, 3, 6, 8, 11, 14, 16, 19]  # earliest action-5 branch of each bin


def other_paths_die(fate_two):
    """Single deviations die at 20 + a; two-deviation paths die at 26 unless ``fate_two``."""

    def fate(path):
        if len(path) == 1:
            return single_death(path)
        special = fate_two(path)
        return 26 if special is False else special

    return fate


def test_known_single_deviation_rescue_is_found_at_depth_one():
    def fate(path):
        return None if path[0] == (5, 2) else single_death(path)

    world = ToyWorld(fate)
    result = two_deviation_search(world, _root(WINDOW), WINDOW, 20)
    assert result["status"] == "RESCUED_DEPTH1"
    assert result["depth"] == 1
    assert result["rescue"]["decision"] == 5 and result["rescue"]["action"] == 2
    assert result["layer1_survivors"] == [{"decision": 5, "action": 2}]
    assert result["beam"] == [] and world.calls["branch"] == 0


def test_two_deviation_rescue_inside_stratified_beam_reports_rank_and_second_deviation():
    fate = other_paths_die(lambda p: None if p == ((3, 5), (22, 1)) else False)
    result = two_deviation_search(ToyWorld(fate), _root(WINDOW), WINDOW, 20)
    assert result["status"] == "RESCUED_DEPTH2"
    assert result["rescue"] == {
        "rank": 2,
        "t1": 3,
        "a1": 5,
        "decision": 22,
        "action": 1,
        "confirm": {"survives": True, "death_decision": None, "env_frames": 600},
    }
    assert result["rescued_ranks"] == [2]
    assert [b["decision"] for b in result["beam"]] == BEAM_T1
    assert result["beam"][1]["window2"] == [4, 25, 22]
    assert len(result["layer2_candidates_300"]) == 1


def test_rescue_outside_the_beam_is_not_found_but_wider_beam_finds_it():
    fate = other_paths_die(lambda p: None if p == ((9, 5), (15, 3)) else False)
    narrow = two_deviation_search(ToyWorld(fate), _root(WINDOW), WINDOW, 20)
    assert narrow["status"] == "NOT_RESCUED" and narrow["rescue"] is None
    wide = two_deviation_search(ToyWorld(fate), _root(WINDOW), WINDOW, 20, SearchParams(beam=10))
    assert wide["status"] == "RESCUED_DEPTH2" and wide["rescue"]["rank"] == 5


def test_horizon_survivor_that_dies_before_confirmation_is_a_delay_not_a_rescue():
    fate = other_paths_die(lambda p: 24 + 450 if p == ((0, 5), (24, 1)) else False)
    result = two_deviation_search(ToyWorld(fate), _root(WINDOW), WINDOW, 20)
    assert result["status"] == "NOT_RESCUED"
    assert result["unconfirmed_survivors"][0]["decision"] == 24
    assert result["unconfirmed_survivors"][0]["confirm"]["death_decision"] == 474
    assert result["layer2_candidates_300"][0]["rank"] == 1  # secondary read-out keeps it


def test_episode_end_caps_both_horizon_and_confirmation():
    def fate(path):
        if path == ((4900, 5), (4925, 2)):
            return None  # alive at decision 5000 = episode survival
        return 4920 + path[0][1] if len(path) == 1 else 4930

    window = list(range(4900, 4921))
    result = two_deviation_search(ToyWorld(fate), _root(window), window, 4920)
    assert result["status"] == "RESCUED_DEPTH2"
    assert result["rescue"]["confirm"]["env_frames"] == 75


def test_branch_disagreeing_with_its_layer1_outcome_is_a_control_failure():
    class Drift(ToyWorld):
        def branch(self, state, decision, action, stop):
            death, states, steps = super().branch(state, decision, action, stop)
            return death + 1, states, steps

    with pytest.raises(ControlMismatch):
        two_deviation_search(Drift(single_death), _root(WINDOW), WINDOW, 20)


def test_confirmation_dying_before_the_expansion_horizon_is_a_control_failure():
    class Drift(ToyWorld):
        def confirm(self, state, decision, action, depth):
            return {"survives": False, "death_decision": decision + 10, "env_frames": 11}

    fate = other_paths_die(lambda p: None if p == ((0, 5), (22, 1)) else False)
    with pytest.raises(ControlMismatch):
        two_deviation_search(Drift(fate), _root(WINDOW), WINDOW, 20)


def test_cap_expiry_after_a_rescue_keeps_the_rescue():
    class Expires(ToyWorld):
        def branch(self, state, decision, action, stop):
            if self.calls["branch"] == 2:
                raise TimeoutError("cap")
            return super().branch(state, decision, action, stop)

    fate = other_paths_die(lambda p: None if p == ((0, 5), (22, 1)) else False)
    result = two_deviation_search(Expires(fate), _root(WINDOW), WINDOW, 20)
    assert result["status"] == "RESCUED_DEPTH2" and result["cap_expired"] is True
    assert result["completed_branches"] == 2 and result["rescue"]["rank"] == 1
    assert len(result["planned_beam"]) == 8


def test_cap_expiry_without_a_rescue_is_incomplete_not_negative():
    class Expires(ToyWorld):
        def expand(self, state, decision, doom):
            if self.calls["branch"] >= 3:
                raise TimeoutError("cap")
            return super().expand(state, decision, doom)

    result = two_deviation_search(
        Expires(other_paths_die(lambda p: False)), _root(WINDOW), WINDOW, 20
    )
    assert result["status"] == "INCOMPLETE_TIMEOUT" and result["cap_expired"] is True
    assert result["completed_branches"] == 2


def test_identity_failure_stops_before_layer_two():
    world = ToyWorld(other_paths_die(lambda p: False))
    result = two_deviation_search(
        world, _root(WINDOW), WINDOW, 20, after_layer1=lambda partial: [("7", "2")]
    )
    assert result["status"] == "INCOMPLETE_IDENTITY"
    assert result["identity_mismatches"] == [("7", "2")] and world.calls["branch"] == 0


def test_excluded_first_deviation_is_replaced_by_the_next_best_in_its_bin():
    world = ToyWorld(other_paths_die(lambda p: False))
    result = two_deviation_search(world, _root(WINDOW), WINDOW, 20, exclude=[(0, 5)])
    assert [b["decision"] for b in result["beam"]] == [1, *BEAM_T1[1:]]
    assert result["excluded_first_deviations"] == [[0, 5]]


def _candidate(t, a, death, q=0.0):
    return {"decision": t, "action": a, "death_decision": death, "q": q}


def test_beam_is_one_branch_per_bin_with_distinct_t1_not_adjacent_near_copies():
    window = list(range(100, 400))  # 300 decisions -> bins of 37-38
    # The 20 globally longest-surviving branches sit on 4 adjacent late decisions.
    candidates = [_candidate(t, a, 500 + a) for t in range(390, 394) for a in range(1, 6)]
    candidates += [_candidate(t, 1, 420 + t // 10) for t in range(100, 390, 7)]
    beam = select_beam(candidates, window, 8)
    t1 = [c["decision"] for c in beam]
    assert len(beam) == 8 and len(set(t1)) == 8
    assert sorted({(8 * (t - 100)) // 300 for t in t1}) == list(range(8))
    assert beam[0]["decision"] == 390 and beam[0]["action"] == 5  # best overall is rank 1


def test_beam_fills_empty_bins_with_distinct_t1_then_any_branch():
    candidates = [_candidate(t, a, 30 + a) for t in (0, 1, 2) for a in range(1, 6)]
    beam = select_beam(candidates, [0, 1, 2], 8)
    assert len(beam) == 8 and {c["decision"] for c in beam[:3]} == {0, 1, 2}
    assert beam[0]["action"] == 5 and len({(c["decision"], c["action"]) for c in beam}) == 8


def test_beam_key_orders_latest_death_then_earlier_decision_then_q_then_action():
    rows = [
        {"decision": 5, "action": 2, "q": 1.0, "death_decision": 30},
        {"decision": 4, "action": 3, "q": 0.0, "death_decision": 30},
        {"decision": 4, "action": 1, "q": 0.0, "death_decision": 30},
        {"decision": 4, "action": 4, "q": 2.0, "death_decision": 30},
        {"decision": 9, "action": 1, "q": 9.0, "death_decision": 31},
    ]
    ordered = sorted(rows, key=beam_key)
    assert [(r["decision"], r["action"]) for r in ordered] == [
        (9, 1),
        (4, 4),
        (4, 1),
        (4, 3),
        (5, 2),
    ]


def test_round_robin_selection_spreads_over_worlds_and_is_outcome_blind():
    events = [
        {"id": ev.event_id(w, s), "world": w, "start": s}
        for w, s in [("b", 10), ("a", 30), ("a", 5), ("b", 40), ("c", 7), ("a", 50)]
    ]
    assert ev.round_robin_order(events) == ["a@5", "b@10", "c@7", "a@30", "b@40", "a@50"]


@pytest.mark.skipif(not ev.M2_BANK.is_dir(), reason="T34 shadow M2 receipts not on this host")
def test_registered_primary_is_earliest_event_of_each_distinct_world():
    events = ev.all_die_events()
    order = ev.round_robin_order(events)
    assert tuple(order[:7]) == ev.PRIMARY_EVENTS
    worlds = {e["world"] for e in events}
    assert len(worlds) == 7 and {i.split("@")[0] for i in ev.PRIMARY_EVENTS} == worlds
    for event_id in ev.PRIMARY_EVENTS:
        world, start = ev.parse_event_id(event_id)
        assert start == min(e["start"] for e in events if e["world"] == world)
    assert t35_run.event_set("primary") == list(ev.PRIMARY_EVENTS)
    assert t35_run.event_set("extend") == order[7:] and len(order[7:]) == 26


@pytest.mark.skipif(not ev.M2_BANK.is_dir(), reason="T34 shadow M2 receipts not on this host")
def test_registered_exclusions_are_t3_changes_followed_by_an_all_die_case():
    by_id = {e["id"]: e for e in ev.all_die_events()}
    assert {i: ev.excluded_first_deviations(by_id[i]) for i in ev.PRIMARY_EVENTS} == {
        "b8_s0_21@3707": [(3736, 2)],
        "b8_s0_8@4318": [(4352, 2)],
        "b8_s3_19@4693": [],
        "b8_s3_26@3647": [(3674, 4)],
        "b8_s3_6@4562": [(4591, 0)],
        "b8_s3_8@4246": [(4255, 5)],
        "b8_s3_9@4188": [],
    }


def test_verdict_rule_on_seven_worlds_and_inconclusive_when_unknowns_can_flip_it():
    assert t35_run.verdict(3, 0, 7) == "REAL"
    assert t35_run.verdict(1, 0, 7) == "KILL"
    assert t35_run.verdict(2, 0, 7) == "EXTEND"
    assert t35_run.verdict(0, 1, 7) == "KILL"
    assert t35_run.verdict(1, 1, 7) == "INCONCLUSIVE"
    assert t35_run.verdict(3, 4, 7) == "REAL"
    with pytest.raises(ValueError):
        t35_run.verdict(3, 0, 10)


def test_summarize_refuses_receipts_from_more_than_one_commit(tmp_path):
    for event_id, commit in (("w_a@1", "a" * 40), ("w_b@2", "b" * 40)):
        path = tmp_path / "events" / f"{event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"status": "NOT_RESCUED", "provenance": {"commit": commit}, "search": {}})
        )
    args = t35_run.parse_args(["summarize", "--events", "w_a@1,w_b@2", "--out", str(tmp_path)])
    with pytest.raises(RuntimeError, match="more than one commit"):
        t35_run.summarize(args)


def test_identity_mismatch_detects_any_changed_layer1_field():
    fresh = {"layer1_outcomes": {"7": {"greedy": 1, "outcomes": {"2": {"survives": False}}}}}
    same = {"7": {"greedy": 1, "outcomes": {"2": {"survives": False}}}}
    assert t35_run.identity_mismatches(fresh, same) == []
    changed = {"7": {"greedy": 1, "outcomes": {"2": {"survives": True}}}}
    assert t35_run.identity_mismatches(fresh, changed) == [("7", "2")]
    assert t35_run.identity_mismatches(fresh, {}) == [("7", "greedy_or_missing")]


def test_trimmed_source_keeps_triggers_through_first_at_or_after_start():
    from research.t35_two_deviation.backend import trimmed_source

    row = {"x": 1, "trigger_rows": [{"decision": d} for d in (10, 15, 20, 25)]}
    assert [t["decision"] for t in trimmed_source(row, 15)["trigger_rows"]] == [10, 15]
    assert [t["decision"] for t in trimmed_source(row, 16)["trigger_rows"]] == [10, 15, 20]
    with pytest.raises(ValueError):
        trimmed_source(row, 26)


REAL_CASE = ("b8_s3_21", 1684, 1983, 1978, 0)  # M2 truth: LAD 1978, a* = 0


@pytest.mark.slow
@pytest.mark.skipif(not ev.M2_BANK.is_dir(), reason="T3 dense bank not on this host")
def test_real_engine_finds_m2_single_deviation_rescue_at_depth_one(monkeypatch):
    for name in t35_run.THREAD_VARS:
        monkeypatch.setenv(name, "1")
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t35_two_deviation.backend import BankBackend, Deadline, prepare_event

    configure_cpu()
    world, start, doom, t_star, a_star = REAL_CASE
    deadline = Deadline(900)
    states, prep = prepare_event(world, start, doom, deadline)
    assert prep["shadow_row_steps"] == doom - start + 1
    result = two_deviation_search(
        BankBackend(deadline), states, [t_star], doom, SearchParams(confirm_extra=0)
    )
    assert result["status"] == "RESCUED_DEPTH1"
    assert {"decision": t_star, "action": a_star} in result["layer1_survivors"]
    receipt = json.loads((ev.M2_BANK / f"{world}.json").read_text())
    case = next(c for c in receipt["cases"] if c["source_decision"] == start)
    probe = next(p for p in case["truth"]["probes"] if p["decision"] == t_star)
    for action, saved in probe["outcomes"].items():
        fresh = result["layer1_outcomes"][str(t_star)]["outcomes"][action]
        assert fresh["survives"] == saved["survives"]
        assert fresh["death_decision"] == saved["death_decision"]
    assert prep["saved_trigger_checks"] > 0
