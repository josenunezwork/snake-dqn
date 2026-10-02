"""research/trap_horizon_20261001/diagnose.py: own-move model, escape search, PNR walk.

No episode is ever played here: every episode runner (``run_simd_eval``, the live
``rollout``) is monkeypatched to raise for the whole module. ``BatchSim`` is stepped
directly in one test to check the hero model against the real dynamics.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from research.trap_horizon_20261001 import diagnose as d


@pytest.fixture(autouse=True)
def no_episodes(monkeypatch):
    """Make every episode runner fatal for the whole module."""
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    def boom(*_args, **_kwargs):
        raise AssertionError("an episode runner was called in a trap-horizon unit test")

    monkeypatch.setattr(ee, "run_simd_eval", boom)
    monkeypatch.setattr(te, "rollout", boom)


def test_runners_are_fatal():
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    with pytest.raises(AssertionError):
        ee.run_simd_eval()
    with pytest.raises(AssertionError):
        te.rollout()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def world(width, height, blocked=(), food=()):
    return d.StaticWorld(width, height, frozenset(blocked), frozenset(food))


def hero(cells, length=None, direction=1, boost_frames=0):
    cells = tuple(tuple(c) for c in cells)
    return d.HeroState(cells, len(cells) if length is None else length, direction, boost_frames)


def row_body(head_x, y, length):
    """Body along row ``y``, head at ``head_x`` heading right."""
    return [(head_x - i, y) for i in range(length)]


# Ring of 12 cells around a blocked 2x2 centre on a 4x4 board (clockwise order).
RING = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3)]
RING += [(0, 2), (0, 1)]
CENTRE = [(1, 1), (2, 1), (1, 2), (2, 2)]


def ring_hero(length):
    """Head at (0, 2) heading up, body counter-clockwise over 11 ring cells."""
    body = [RING[i] for i in range(10, -1, -1)]
    return hero(body, length=length, direction=0)


# ---------------------------------------------------------------------------
# Own-move model
# ---------------------------------------------------------------------------
def test_straight_move_pops_tail():
    w = world(20, 5)
    s = hero(row_body(5, 2, 4))
    nxt, cause = d.step_hero(w, s, 1)
    assert cause == "alive"
    assert nxt.body == ((6, 2), (5, 2), (4, 2), (3, 2))
    assert nxt.length == 4 and nxt.direction == 1


def test_entering_own_tail_cell_is_legal_without_growth_and_fatal_with_growth():
    w = world(4, 4, blocked=CENTRE)
    # 12 ring cells, length 11 (one free cell): straight twice, the second onto the tail
    # cell (0, 0), which the same move vacates.
    s = ring_hero(11)
    a, cause = d.step_hero(w, s, 1)
    assert cause == "alive" and a.body[0] == (0, 1)
    b, cause = d.step_hero(w, a, 1)
    assert cause == "alive" and b.body[0] == (0, 0) and b.body[-1] == (2, 0)
    # Pending growth of 2: the tail does not move, so entering it is a self collision.
    g = ring_hero(13)
    a, cause = d.step_hero(w, g, 1)
    assert cause == "alive" and len(a.body) == 12
    b, cause = d.step_hero(w, a, 1)
    assert b is None and cause == "self"


def test_wall_other_and_food():
    w = world(10, 3, blocked={(6, 0)}, food={(6, 1)})
    s = hero(row_body(5, 1, 3))
    nxt, cause = d.step_hero(w, s, 1)
    # Eaten after this move's pop (BatchSim order): the body grows on the next move.
    assert cause == "alive" and nxt.length == 4 and len(nxt.body) == 3
    nxt2, _ = d.step_hero(w, nxt, 1)
    assert nxt2.length == 4 and len(nxt2.body) == 4  # pellet consumed once
    _, cause = d.step_hero(w, s, 0)  # left of heading right is up: (5, 0)
    assert cause == "alive"
    edge = hero(row_body(9, 1, 3))
    assert d.step_hero(w, edge, 1) == (None, "wall")
    up = hero([(6, 1), (5, 1), (4, 1)])
    assert d.step_hero(w, up, 0) == (None, "other")


def test_boost_two_cells_burn_and_trail_pellet():
    w = world(30, 3)
    s = hero(row_body(10, 1, 6), boost_frames=2)
    nxt, cause = d.step_hero(w, s, 4)  # boost straight; third boost frame burns
    assert cause == "alive"
    assert nxt.body[0] == (12, 1) and nxt.body[1] == (11, 1)
    assert nxt.length == 5 and len(nxt.body) == 5 and nxt.boost_frames == 0
    assert nxt.body[-1] == (8, 1)
    assert nxt.food_added == frozenset({(7, 1)})  # the burned tail became a pellet
    short = hero(row_body(10, 1, 4))  # below min_boost_length: one cell, no counter
    nxt, _ = d.step_hero(w, short, 4)
    assert nxt.body[0] == (11, 1) and nxt.boost_frames == 0


def test_step_hero_matches_batchsim_exactly():
    """Random actions (boost included) on a food-rich board: every transition equal."""
    from src.simd_env.batch_sim import BatchSim, BatchSimConfig

    cfg = BatchSimConfig(
        num_envs=2, num_snakes=1, game_width=160, game_height=120, initial_food=80, max_food=80
    )
    sim = BatchSim(cfg, seeds=[3, 4], train_mode=False)
    meta = {
        "grid_width": sim.grid_w,
        "grid_height": sim.grid_h,
        "min_boost_length": cfg.min_boost_length,
        "boost_cost_frames": cfg.boost_length_cost_frames,
        "trail_food": sim.v2,
    }
    rng = random.Random(0)
    seen = {"alive": 0, "self": 0, "wall": 0, "boost": 0, "grew": 0}
    for _ in range(1200):
        pre = {}

        def select(s):
            actions = np.zeros((s.E, s.S), dtype=np.int64)
            for e in range(s.E):
                if s.alive[e, 0]:
                    actions[e, 0] = rng.randrange(6)
                    snap = {
                        "body": d._ordered_cells(s, e, 0),
                        "length": int(s.length[e, 0]),
                        "direction": int(s.direction[e, 0]),
                        "boost_frames": int(s.boost_frames[e, 0]),
                        "others": np.zeros((0, 2), np.int16),
                        "food": np.asarray(list(s.food_cells[e]), np.int16).reshape(-1, 2),
                    }
                    pre[e] = (snap, int(actions[e, 0]))
            return actions

        sim.step_with_policy(select)
        for e, (snap, action) in pre.items():
            w, s = d.world_and_state(snap, meta)
            nxt, cause = d.step_hero(w, s, action)
            if sim.get_done()[e, 0]:
                label = {1: "wall", 2: "self"}[int(sim.get_death_cause()[e, 0])]
                assert cause == label
                seen[label] += 1
                continue
            body = tuple(map(tuple, d._ordered_cells(sim, e, 0).tolist()))
            assert nxt is not None and nxt.body == body
            assert nxt.length == int(sim.length[e, 0])
            assert nxt.direction == int(sim.direction[e, 0])
            assert nxt.boost_frames == int(sim.boost_frames[e, 0])
            seen["alive"] += 1
            seen["boost"] += action >= 3 and s.length >= cfg.min_boost_length
            seen["grew"] += nxt.length > s.length
    assert seen["alive"] > 1000 and seen["self"] > 10 and seen["wall"] > 10
    assert seen["boost"] > 100 and seen["grew"] > 20


# ---------------------------------------------------------------------------
# Tail-aware count
# ---------------------------------------------------------------------------
def test_split_release_count_equals_reference():
    rng = random.Random(1)
    w = world(12, 9, blocked={(5, y) for y in range(2, 7)})
    checked = 0
    for trial in range(60):
        s = hero(row_body(4, 1 + trial % 6, 5), length=5 + trial % 30)
        for _ in range(60):
            for limit in (1, 4, 9, d.need_for(s.length), 40):
                assert d.tail_aware_count(w, s, limit) == d.tail_aware_count_reference(w, s, limit)
                checked += 1
            moves = [m for m in (d.step_hero(w, s, a)[0] for a in range(6)) if m is not None]
            if not moves:
                break
            s = rng.choice(moves)
    assert checked > 2000


# ---------------------------------------------------------------------------
# Escape search
# ---------------------------------------------------------------------------
def test_open_board_every_action_escapes_after_one_node():
    w = world(40, 40)
    s = hero(row_body(20, 20, 8))
    results = d.analyze_frame(w, s)
    for result in results.values():
        assert result.status == d.ESCAPE and result.kind == "count" and result.nodes == 1


def test_dead_end_strip_has_no_escape():
    w = world(10, 1)
    s = hero(row_body(5, 0, 6))
    for criterion in d.CRITERIA:
        results = d.analyze_frame(w, s, criterion=criterion)
        assert all(r.status == d.NO_ESCAPE for r in results.values()), criterion
    assert d.analyze_frame(w, s)[0].immediate == "wall"


def test_following_own_tail_is_an_escape_and_pending_growth_is_not():
    w = world(4, 4, blocked=CENTRE)
    free = d.analyze_frame(w, ring_hero(11))
    assert free[1].status == d.ESCAPE
    assert free[0].immediate == "wall" and free[2].immediate == "other"
    depth_only = d.analyze_frame(w, ring_hero(11), criterion=d.DEPTH_ONLY)
    assert depth_only[1].status == d.ESCAPE and depth_only[1].kind == "depth"
    grown = d.analyze_frame(w, ring_hero(13))
    assert all(r.status == d.NO_ESCAPE for r in grown.values())


def pocket_world(length, side=4):
    """``side``x``side`` pocket above a 1-wide corridor that holds the whole body.

    The head is at (0, side) heading up into the pocket; the corridor runs along row
    ``side`` and the rest of the rows above it are blocked beyond the pocket.
    """
    width = length + 10
    blocked = {(x, y) for x in range(side, width) for y in range(side)}
    w = world(width, side + 1, blocked=blocked)
    body = [(x, side) for x in range(length)]
    return w, hero(body, direction=0)


def test_long_hero_survives_pocket_but_cannot_reach_spacious_state():
    w, s = pocket_world(60)
    assert d.need_for(s.length) == 60
    kw = {"actions": (1,), "depth": 10, "budget": 200000}
    depth_only = d.analyze_frame(w, s, criterion=d.DEPTH_ONLY, **kw)[1]
    count_only = d.analyze_frame(w, s, criterion=d.COUNT_ONLY, **kw)[1]
    primary = d.analyze_frame(w, s, **kw)[1]
    assert depth_only.status == d.ESCAPE and depth_only.kind == "depth"
    assert count_only.status == d.NO_ESCAPE
    assert primary.status == d.ESCAPE and primary.kind == "depth"
    assert d.combine_status(depth_only.status, count_only.status) == d.ESCAPE
    # 16 cells cannot be survived for 40 frames: no escape at the default depth.
    assert d.analyze_frame(w, s, actions=(1,), criterion=d.DEPTH_ONLY)[1].status == d.NO_ESCAPE


def test_budget_exhaustion_is_unknown_not_guessed():
    w, s = pocket_world(60)
    kw = {"actions": (1,), "depth": 10, "budget": 5}
    result = d.analyze_frame(w, s, criterion=d.COUNT_ONLY, **kw)[1]
    assert result.status == d.UNKNOWN and result.nodes == 6


def test_combine_status_truth_table():
    E, N, U = d.ESCAPE, d.NO_ESCAPE, d.UNKNOWN
    assert d.combine_status(E, U) == E and d.combine_status(U, E) == E
    assert d.combine_status(N, N) == N
    assert d.combine_status(N, U) == U and d.combine_status(U, N) == U


def _random_states(seed, count):
    rng = random.Random(seed)
    w = world(9, 7, blocked={(4, 2), (4, 3), (4, 4), (1, 5), (7, 1)}, food={(2, 1), (6, 5)})
    out = []
    while len(out) < count:
        s = hero(row_body(3, rng.randrange(7), 4), length=4 + rng.randrange(25))
        for _ in range(rng.randrange(1, 40)):
            nxt, _ = d.step_hero(w, s, rng.randrange(6))
            if nxt is None:
                break
            s = nxt
        out.append((w, d.HeroState(s.body, s.length, s.direction, s.boost_frames)))
    return out


def test_combined_one_sided_searches_equal_direct_search_and_prune_is_sound():
    """combine(depth_only, count_only) == count_or_depth; prune never flips a verdict."""
    compared = 0
    for w, s in _random_states(2, 150):
        for depth in (6, 12):
            direct = d.analyze_frame(w, s, depth=depth, budget=200000)
            dep = d.analyze_frame(w, s, depth=depth, budget=200000, criterion=d.DEPTH_ONLY)
            cnt = d.analyze_frame(w, s, depth=depth, budget=200000, criterion=d.COUNT_ONLY)
            for crit, pruned in ((d.COUNT_OR_DEPTH, direct), (d.DEPTH_ONLY, dep)):
                plain = d.EscapeSearch(w, depth, 200000, criterion=crit, prune=False)
                for a in range(d.NUM_ACTIONS):
                    assert plain.run(s, a).status == pruned[a].status
            for a in range(d.NUM_ACTIONS):
                assert d.UNKNOWN not in (direct[a].status, dep[a].status, cnt[a].status)
                assert d.combine_status(dep[a].status, cnt[a].status) == direct[a].status
                compared += 1
    assert compared == 150 * 2 * d.NUM_ACTIONS


def brute_force(w, s, action, depth, criterion):
    """Reference: plain recursion over every action sequence, no memo, no prune."""

    def visit(state, step):
        spacious = d.tail_aware_count_reference(w, state, d.need_for(state.length))
        if criterion != d.DEPTH_ONLY and spacious >= d.need_for(state.length):
            return True
        if step >= depth:
            return criterion != d.COUNT_ONLY
        for a in range(d.NUM_ACTIONS):
            child, _ = d.step_hero(w, state, a)
            if child is not None and visit(child, step + 1):
                return True
        return False

    child, _ = d.step_hero(w, s, action)
    return d.ESCAPE if child is not None and visit(child, 1) else d.NO_ESCAPE


def test_memo_direction_per_criterion():
    assert d.memo_fails(d.COUNT_OR_DEPTH, 5, 7) and not d.memo_fails(d.COUNT_OR_DEPTH, 5, 3)
    assert d.memo_fails(d.DEPTH_ONLY, 5, 5) and not d.memo_fails(d.DEPTH_ONLY, 5, 4)
    assert d.memo_fails(d.COUNT_ONLY, 5, 3) and not d.memo_fails(d.COUNT_ONLY, 5, 7)
    # A count-only failure recorded with 3 frames left must not block a visit with 9
    # left, and an upward-monotone failure recorded with 9 left must not block 3.
    w = world(40, 40)
    s = hero(row_body(20, 20, 8))
    search = d.EscapeSearch(w, depth=12, criterion=d.COUNT_ONLY)
    search.failed[s.key] = 3
    assert search._visit(s, 12 - 9) is True
    search = d.EscapeSearch(w, depth=12, criterion=d.DEPTH_ONLY)
    search.failed[s.key] = 9
    assert search._visit(s, 12 - 3) is True


def test_search_matches_brute_force_on_small_worlds():
    compared = 0
    for w, s in _random_states(3, 40):
        for criterion in d.CRITERIA:
            got = d.analyze_frame(w, s, depth=4, budget=10**6, criterion=criterion)
            for a in range(d.NUM_ACTIONS):
                assert got[a].status == brute_force(w, s, a, 4, criterion), (criterion, a)
                compared += 1
    assert compared == 40 * 3 * d.NUM_ACTIONS


def test_doomed_prune_fires_and_matches_unpruned_on_long_hero():
    # Length 400 (need 160) in a 16-cell pocket: neither 40 frames of survival nor a
    # spacious state is possible, and the region bound proves it at the first node.
    w, s = pocket_world(400)
    for criterion in d.CRITERIA:
        pruned = d.EscapeSearch(w, 40, 200000, criterion=criterion)
        r1 = pruned.run(s, 1)
        assert r1.status == d.NO_ESCAPE and pruned.pruned >= 1 and r1.nodes == 1
        plain = d.EscapeSearch(w, 40, 20000, criterion=criterion, prune=False)
        r2 = plain.run(s, 1)
        assert r2.status in (d.NO_ESCAPE, d.UNKNOWN) and r2.nodes > 1000
    tiny = hero([(0, 0), (1, 0)], length=2)
    assert not d.doomed(world(5, 5), tiny, 10, d.COUNT_OR_DEPTH)  # short hero: never


# ---------------------------------------------------------------------------
# Backward PNR walk (synthetic statuses)
# ---------------------------------------------------------------------------
E, N, U = d.ESCAPE, d.NO_ESCAPE, d.UNKNOWN


def table(rows):
    """``rows[t]`` is a 6-char string over E/N/U, one per action."""
    code = {"E": E, "N": N, "U": U}
    calls = []

    def evaluate(t):
        calls.append(t)
        return {a: code[ch] for a, ch in enumerate(rows[t])}

    return evaluate, calls


def test_walk_exact_fatal_choice():
    rows = ["EEEEEE", "ENNNNN", "NNNNNN", "NNNNNN"]
    evaluate, calls = table(rows)
    out = d.walk_back(4, [1, 1, 1, 1], evaluate)
    assert out["status"] == "exact" and out["pnr_index"] == 1
    assert out["frames_before_death"] == 2 and out["frames_before_bounds"] == [2, 2]
    assert out["taken_status"] == N and out["escaping_actions"] == [0]
    assert out["fatal_choice"] is True
    assert calls == [3, 2, 1]  # stops at the PNR


def test_walk_taken_action_escaped_and_pnr_at_fatal_frame():
    evaluate, _ = table(["NNNNNN", "NENNNN"])
    out = d.walk_back(2, [0, 1], evaluate)
    assert out["pnr_index"] == 1 and out["frames_before_death"] == 0
    assert out["taken_status"] == E and out["fatal_choice"] is False


def test_walk_unknown_bounds_and_unknown_taken():
    evaluate, _ = table(["NNEUNN", "NNNNNN", "NNUNNN", "NNNNNN"])
    out = d.walk_back(4, [3, 0, 0, 0], evaluate)
    assert out["status"] == U and out["pnr_index"] == 0
    assert out["frames_before_bounds"] == [1, 3] and out["frames_before_lower"] == 1
    assert out["taken_status"] == U and out["fatal_choice"] is None
    assert d.horizon_bin(out) == "unknown_within_le_8"


def test_walk_beyond_window_and_unknown_without_escape():
    evaluate, _ = table(["NNNNNN"] * 50)
    out = d.walk_back(50, [1] * 50, evaluate)
    assert out["status"] == "beyond_window" and out["frames_before_lower"] == 50
    assert d.horizon_bin(out) == "gt_40"
    evaluate, _ = table(["NNNNNN"] * 10)
    short = d.walk_back(10, [1] * 10, evaluate)
    assert d.horizon_bin(short) == "unknown"  # >= 10 frames, cannot place in a bin
    evaluate, _ = table(["NNNNNN", "UNNNNN", "NNNNNN"])
    out = d.walk_back(3, [1] * 3, evaluate)
    assert out["status"] == U and out["pnr_index"] is None
    assert out["frames_before_lower"] == 1


def test_bins_and_summary():
    def walk(fb):
        rows = ["EEEEEE"] + ["NNNNNN"] * fb
        evaluate, _ = table(rows)
        return d.walk_back(fb + 1, [0] * (fb + 1), evaluate)

    deaths = [{"mix": "frozen", "walk": walk(fb)} for fb in (0, 8, 9, 20, 21, 40, 41)]
    assert [d.horizon_bin(x["walk"]) for x in deaths] == [
        "le_8",
        "le_8",
        "9_20",
        "9_20",
        "21_40",
        "21_40",
        "gt_40",
    ]
    evaluate, _ = table(["NNNNNN"] * 60)
    deaths.append({"mix": "mixed", "walk": d.walk_back(60, [0] * 60, evaluate)})
    summary = d.summarize_walks(deaths)
    assert summary["self_deaths"] == 8
    assert summary["bins"] == {"le_8": 2, "9_20": 2, "21_40": 2, "gt_40": 2, "unknown": 0}
    assert summary["quantiles_exact_only"]["median"] == 20.0
    assert summary["quantiles_exact_plus_beyond_window_as_inf"]["median"] == 20.0
    assert summary["fatal_choice"]["taken_action_escaped"] == 7
    assert summary["by_mix"]["mixed"]["gt_40"] == 1


# ---------------------------------------------------------------------------
# Whole-window analysis on a hand-made geometry
# ---------------------------------------------------------------------------
def corridor_window():
    """Length-6 hero runs straight from an open 8x3 area into a dead-end corridor.

    Corridor: row 1, x = 8..19, rows 0 and 2 blocked there; the hero dies on the wall
    at x = 20. Head positions 5..19 give 15 decisions (index 14 is the fatal one).
    """
    blocked = [(x, y) for x in range(8, 20) for y in (0, 2)]
    meta = {
        "grid_width": 20,
        "grid_height": 3,
        "min_boost_length": 5,
        "boost_cost_frames": 3,
        "trail_food": True,
    }
    w = world(20, 3, blocked=blocked)
    s = hero(row_body(5, 1, 6))
    snaps = []
    for t in range(15):
        snaps.append(
            {
                "frame": 100 + t,
                "body": np.asarray(s.body, dtype=np.int16),
                "length": s.length,
                "direction": s.direction,
                "boost_frames": s.boost_frames,
                "others": np.asarray(blocked, dtype=np.int16),
                "food": np.zeros((0, 2), np.int16),
                "v2_counts": np.asarray([0, 19 - s.body[0][0], 0]),
                "v2_need": 6,
                "base": 1,
                "final": 1,
                "q": np.zeros(6, np.float32),
                "mask": np.ones(6, bool),
            }
        )
        s, _ = d.step_hero(w, s, 1)
    return snaps, meta


def test_analyze_death_on_dead_end_corridor():
    snaps, meta = corridor_window()
    out = d.analyze_death(snaps, meta)
    assert out["model_check"]["mismatches"] == 0
    assert out["model_check"]["fatal_action_model_outcome"] == "wall"
    # Primary: straight keeps a spacious (area >= 6) next state while the head is at
    # x <= 13 (19 - x cells ahead), so the PNR is head x = 13, 6 frames before death,
    # and the taken action had that (area-only) escape.
    walk = out["walk"]
    assert walk["status"] == "exact" and walk["frames_before_death"] == 6
    assert walk["taken_status"] == d.ESCAPE and walk["fatal_choice"] is False
    row = out["pnr_row"]
    assert row[d.COUNT_ONLY]["1"]["status"] == d.ESCAPE
    assert row[d.COUNT_ONLY]["1"]["success_depth"] == 1
    # Survive-40 needs the open area: the last chance is turning at head x = 7.
    dep = out["sensitivity"][d.DEPTH_ONLY]["walk"]
    assert dep["status"] == "exact" and dep["frames_before_death"] == 12
    assert dep["taken_status"] == d.NO_ESCAPE and dep["fatal_choice"] is True
    assert out["sensitivity"][d.COUNT_ONLY]["walk"]["frames_before_death"] == 6
    assert len(out["taken_scan"]) == 15
    assert {s["status"] for s in out["taken_scan"][:9]} == {d.ESCAPE}
    assert out["pnr_v2"]["taken_direction_count"] == 6


def test_pack_unpack_roundtrip():
    snaps, _ = corridor_window()
    back = d.unpack_window(d.pack_window(snaps))
    assert len(back) == len(snaps)
    for a, b in zip(snaps, back):
        for key in ("body", "others", "food", "v2_counts", "q", "mask"):
            assert np.array_equal(np.asarray(a[key]), b[key]), key
        for key in ("frame", "length", "direction", "boost_frames", "v2_need", "base", "final"):
            assert int(a[key]) == b[key]


def test_recorder_captures_ordered_bodies_from_batchsim():
    from src.simd_env.batch_sim import BatchSim, BatchSimConfig

    sim = BatchSim(BatchSimConfig(num_envs=2, num_snakes=3), seeds=[5, 6], train_mode=False)
    for _ in range(30):
        sim.step(np.ones((2, 3), dtype=np.int64))
    rec = d.DecisionRecorder(window=3)
    veto = {
        "rows": np.array([[0, 0], [1, 0]]),
        "counts": np.array([[1, 2, 3], [4, 5, 6]]),
        "need": np.array([7, 8]),
        "base": np.array([1, 4]),
        "final": np.array([2, 4]),
    }
    for _ in range(4):
        rec.capture(sim, veto, np.zeros((2, 6), np.float32), np.ones((2, 6), bool))
    assert rec.decisions == {0: 4, 1: 4} and len(rec.buffers[0]) == 3
    snap = rec.buffers[1][-1]
    assert [tuple(c) for c in snap["body"].tolist()] == sim.get_bodies(1, 0)
    others = [c for s in (1, 2) if sim.alive[1, s] for c in sim.get_bodies(1, s)]
    assert [tuple(c) for c in snap["others"].tolist()] == others
    assert snap["final"] == 4 and snap["v2_need"] == 8
    assert rec.world_meta["grid_width"] == sim.grid_w


def test_world_seeds_are_fresh_and_disjoint():
    seeds = d.world_seeds(d.DOMAIN, d.WORLDS_PER_MIX)
    assert len(set(seeds)) == d.WORLDS_PER_MIX
    report = d.disjointness(seeds, d.DOMAIN)
    assert report["disjoint"] is True
    names = set(report["extra_namespaces"])
    assert any(n.startswith("apex-safety-screen-v1/") for n in names)
    assert any(n.startswith("apex-veto-v4-screen-v1/") for n in names)
    assert any(n.startswith("apex-veto-strict-final-v3/") for n in names)
    assert any(n.startswith(f"{d.SMOKE_DOMAIN}/") for n in names)
