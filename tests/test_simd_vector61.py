"""Parity tests: batched vector61 featurizer vs the live ``SnakeStateMixin.get_state``.

The reference is the live per-snake state builder run on real ``Snake`` objects
inside :class:`~src.simd_env.parity.PyRefGame`, which is held in bit-exact
dynamics lockstep with :class:`~src.simd_env.batch_sim.BatchSim` by the P2
parity harness. Every frame, every alive snake's live 61-D (or 58-D) float32
state is compared **bit for bit** with the batched featurizer's row, over a
recorded action log of several hundred frames per seed, on both respawn arms.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import (
    PyRefGame,
    SurvivorPolicy,
    _install_v2_config,
    build_parity_config,
    scripted_actions,
)
from src.simd_env.vector61_featurizer import (
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    Vector61Featurizer,
    Vector61Memory,
    Vector61Params,
    batched_capped_component_sizes,
    capped_component_sizes,
)

COLUMN_NAMES = (
    [f"dir{i}" for i in range(4)]
    + ["length"]
    + ["food_rel_x", "food_rel_y", "food_dist"]
    + [f"food_density{i}" for i in range(16)]
    + [f"danger{i}" for i in range(16)]
    + ["bound_left", "bound_right", "bound_top", "bound_bottom"]
    + [
        "enemy_x",
        "enemy_y",
        "enemy_size",
        "enemy_hdx",
        "enemy_hdy",
        "enemy_trend",
        "enemy2_x",
        "enemy2_y",
        "enemy2_size",
        "kill_opp",
    ]
    + ["pad_left", "pad_straight", "pad_right", "boost"]
    + ["free_left", "free_straight", "free_right"]
)


@pytest.fixture
def restore_config():
    """Restore the global config singleton and device after each test."""
    saved = get_config()
    DeviceManager.override_device(torch.device("cpu"))
    yield
    initialize_config(saved)
    DeviceManager.reset_for_testing()


def _install(cfg: BatchSimConfig, use_free_space: bool = True, max_length: int = 150) -> None:
    """Install the v2 parity world plus the vector61 network knobs."""
    _install_v2_config(cfg)
    base = get_config()
    network = replace(
        base.network,
        use_free_space=use_free_space,
        input_size=61 if use_free_space else 58,
    )
    game = replace(base.game, max_length=max_length)
    initialize_config(replace(base, network=network, game=game))


def _live_states(ref: PyRefGame, update: bool) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rows = [(0, s) for s, snake in enumerate(ref.snakes) if snake.is_alive]
    if not rows:
        return rows, np.zeros((0, 0), dtype=np.float32)
    states = [
        ref.snakes[s].get_state(ref.snakes, ref.food, update_enemy_memory=update).cpu().numpy()
        for _, s in rows
    ]
    return rows, np.stack(states).astype(np.float32)


def _assert_bit_equal(live: np.ndarray, batch: np.ndarray, rows, seed: int, frame: int) -> None:
    assert live.shape == batch.shape, (seed, frame, live.shape, batch.shape)
    if np.array_equal(live.view(np.uint32), batch.view(np.uint32)):
        return
    bad = np.argwhere(live.view(np.uint32) != batch.view(np.uint32))
    r, c = (int(x) for x in bad[0])
    name = COLUMN_NAMES[c] if c < len(COLUMN_NAMES) else str(c)
    raise AssertionError(
        f"seed={seed} frame={frame} row={rows[r]} col={c} ({name}): "
        f"live={live[r, c]!r} batch={batch[r, c]!r}; {len(bad)} mismatching cells"
    )


def _greedy_grow_actions(ref: PyRefGame, frame: int) -> np.ndarray:
    """Mask-following food chaser that never boosts, so snakes grow long.

    It has no flood-fill veto, so long snakes wander into pockets and self-trap:
    exactly the states that exercise fractional free-space features. Derived
    from the live reference only and applied identically to both sims.
    """
    from src.game.game_logic import GameLogic

    masks = ref.action_masks()
    out = np.ones(len(ref.snakes), dtype=np.int64)
    for s, snake in enumerate(ref.snakes):
        safe = [rel for rel in range(3) if masks[s][rel]]
        if not snake.is_alive or not safe:
            continue
        hx, hy = snake.head

        def score(rel: int) -> Tuple[float, int]:
            dx, dy = GameLogic.relative_to_absolute_direction(snake.direction, rel)
            nx, ny = hx + dx * snake.segment_size, hy + dy * snake.segment_size
            best = min(((fx - nx) ** 2 + (fy - ny) ** 2 for fx, fy in ref.food), default=0)
            # Deterministic, snake/frame-varying tie-break avoids lock-step loops.
            return (float(best), (rel + s + frame // 7) % 3)

        out[s] = min(safe, key=score)
    return out


def _run_lockstep(
    cfg: BatchSimConfig,
    seed: int,
    frames: int,
    policy: str,
    allow_respawn: bool = False,
    probe_no_update: bool = True,
) -> Dict[str, object]:
    """Step live reference + batch sim together and compare features every frame.

    Returns coverage counters plus the recorded action log so tests can assert
    that the comparison exercised the interesting feature branches.
    """
    params = Vector61Params.from_game_config(cfg)
    featurizer = Vector61Featurizer(params)
    S = cfg.num_snakes
    random.seed(seed)
    ref = PyRefGame(cfg, allow_respawn=allow_respawn)
    bat = BatchSim(
        replace(cfg, num_envs=1), seeds=[seed], train_mode=True, allow_respawn=allow_respawn
    )
    memory = Vector61Memory(1, S)
    survivor = SurvivorPolicy(seed, S) if policy == "survivor" else None
    greedy = policy == "greedy"
    scripted = scripted_actions(seed, frames, S)
    all_rows: List[np.ndarray] = []
    action_log: List[np.ndarray] = []
    compared_rows = 0
    respawns = 0

    def compare(frame: int) -> None:
        nonlocal compared_rows
        live_alive = [bool(sn.is_alive) for sn in ref.snakes]
        assert live_alive == bat.get_alive()[0].tolist(), (seed, frame, "alive desync")
        for s, sn in enumerate(ref.snakes):
            if sn.is_alive:
                head = (sn.head[0] // cfg.segment_size, sn.head[1] // cfg.segment_size)
                assert head == tuple(bat.get_heads()[0, s].tolist()), (seed, frame, s)
        passes = (False, True) if (probe_no_update and frame % 2) else (True,)
        for update in passes:
            rows, live = _live_states(ref, update)
            if not rows:
                return
            batch = featurizer.featurize(bat, np.array(rows), memory, update_memory=update)
            _assert_bit_equal(live, batch, rows, seed, frame)
            compared_rows += len(rows)
            if update:
                all_rows.append(batch)

    compare(0)
    for f in range(frames):
        if survivor is not None:
            a = survivor.actions(np.array(ref.action_masks(), dtype=bool))
        elif greedy:
            a = _greedy_grow_actions(ref, f)
        else:
            a = scripted[f]
        action_log.append(np.asarray(a, dtype=np.int64).copy())
        was_alive = bat.get_alive()[0].copy()
        ref.step(a)
        bat.step(np.asarray(a, dtype=np.int64).reshape(1, S))
        respawns += int(np.sum(~was_alive & bat.get_alive()[0]))
        memory.reset_where(bat.get_done())
        compare(f + 1)
    stacked = np.concatenate(all_rows, axis=0) if all_rows else np.zeros((0, params.input_size))
    return {
        "rows": stacked,
        "compared_rows": compared_rows,
        "actions": np.stack(action_log) if action_log else np.zeros((0, S)),
        "respawns": respawns,
    }


_CFG_SMALL = build_parity_config(
    num_snakes=4, game_width=300, game_height=200, initial_food=20, max_food=25
)
_CFG_MID = build_parity_config(
    num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
)
_CFG_TIGHT = build_parity_config(
    num_snakes=8, game_width=250, game_height=180, initial_food=15, max_food=20
)
# Wider than 2 * danger radius (30 cells) so the four wall-sampling branches are
# individually on/off rather than always on.
_CFG_WIDE = build_parity_config(
    num_snakes=4, game_width=900, game_height=700, initial_food=60, max_food=80
)


def test_constants_match_live_module():
    from src.game import snake_state

    assert FREE_SPACE_BFS_CAP == snake_state.FREE_SPACE_BFS_CAP
    assert FREE_SPACE_MIN_CAP == snake_state.FREE_SPACE_MIN_CAP


def _reference_capped_reach(free: np.ndarray, start: Tuple[int, int], cap: int) -> int:
    """Verbatim port of the live set-based capped flood fill."""
    gw, gh = free.shape

    def ok(c):
        return 0 <= c[0] < gw and 0 <= c[1] < gh and bool(free[c[0], c[1]])

    if not ok(start):
        return 0
    seen = {start}
    stack = [start]
    count = 0
    while stack and count < cap:
        gx, gy = stack.pop()
        count += 1
        for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
            if nb not in seen and ok(nb):
                seen.add(nb)
                stack.append(nb)
    return count


@pytest.mark.parametrize("density", [0.2, 0.45, 0.6])
def test_memoized_flood_fill_matches_live_algorithm(density):
    gen = np.random.default_rng(int(density * 100))
    for trial in range(20):
        gw, gh = int(gen.integers(5, 40)), int(gen.integers(5, 30))
        free = gen.random((gw, gh)) >= density
        starts = [(int(gen.integers(-1, gw + 1)), int(gen.integers(-1, gh + 1))) for _ in range(60)]
        got = capped_component_sizes(free, starts, FREE_SPACE_BFS_CAP)
        for start, value in zip(starts, got):
            for cap in (FREE_SPACE_MIN_CAP, 57, FREE_SPACE_BFS_CAP):
                assert min(value, cap) == _reference_capped_reach(free, start, cap), (
                    trial,
                    start,
                    cap,
                )


@pytest.mark.parametrize(
    "cfg,policy,allow_respawn,seeds,frames",
    [
        (_CFG_SMALL, "survivor", False, (0, 1, 2), 400),
        (_CFG_MID, "survivor", True, (3, 4), 400),
        (_CFG_TIGHT, "scripted", True, (5, 6), 300),
        (_CFG_WIDE, "survivor", False, (7,), 400),
    ],
)
def test_vector61_bit_exact_vs_live(restore_config, cfg, policy, allow_respawn, seeds, frames):
    """Every alive row, every frame, bit-identical to the live 61-D state."""
    _install(cfg)
    for seed in seeds:
        result = _run_lockstep(cfg, seed, frames, policy, allow_respawn=allow_respawn)
        assert result["compared_rows"] > 0
        assert result["actions"].shape == (frames, cfg.num_snakes)


def _serpentine(gw: int, gh: int) -> np.ndarray:
    """Maze of long 1-wide corridors (forces the windowed pre-pass to fall back)."""
    free = np.ones((gw, gh), dtype=bool)
    for x in range(1, gw, 2):
        free[x, :] = False
        free[x, 0 if (x // 2) % 2 else gh - 1] = True
    return free


@pytest.mark.parametrize("radius,max_layers", [(10, 40), (3, 5), (1, 1), (6, 200)])
def test_batched_flood_fill_matches_scalar(radius, max_layers):
    gen = np.random.default_rng(radius * 100 + max_layers)
    grids = [gen.random((37, 23)) >= dens for dens in (0.0, 0.25, 0.45, 0.6)]
    grids.append(_serpentine(37, 23))
    starts = np.array(
        [
            (int(gen.integers(0, len(grids))), int(gen.integers(-1, 38)), int(gen.integers(-1, 24)))
            for _ in range(400)
        ]
    )
    got = batched_capped_component_sizes(
        grids, starts, FREE_SPACE_BFS_CAP, radius=radius, max_layers=max_layers
    )
    for (g, x, y), value in zip(starts.tolist(), got.tolist()):
        assert value == capped_component_sizes(grids[g], [(x, y)], FREE_SPACE_BFS_CAP)[0]
        assert value == _reference_capped_reach(grids[g], (x, y), FREE_SPACE_BFS_CAP)


_CFG_GROW = build_parity_config(
    num_snakes=4, game_width=300, game_height=200, initial_food=60, max_food=80
)


def test_vector61_bit_exact_long_snakes_and_self_traps(restore_config):
    """Food-chasing snakes grow past 100 cells and self-trap; free space goes fractional."""
    _install(_CFG_GROW)
    result = _run_lockstep(_CFG_GROW, 12, 700, "greedy", allow_respawn=True)
    rows = result["rows"]
    free = rows[:, 58:61]
    # Coverage guards: the comparison really exercised the hard branches.
    assert rows[:, 4].max() * 150 >= 60  # long bodies (logical length >= 60)
    assert int(((free > 0) & (free < 1)).any(axis=1).sum()) >= 50  # fractional pockets
    assert int((free == 0).any(axis=1).sum()) >= 50  # blocked candidate cells
    assert result["respawns"] >= 10  # memory reset across deaths/respawns
    assert int((rows[:, 49] > 0).sum()) > 0 and int((rows[:, 49] < 0).sum()) > 0
    assert int(rows[:, 53].sum()) > 0  # kill-opportunity flag
    soft = (rows[:, 54:57] > 0) & (rows[:, 54:57] < 1)
    assert int(soft.sum()) > 0 and int((rows[:, 54:57] == 1).sum()) > 0


def test_vector58_mode_bit_exact(restore_config):
    """With use_free_space off the featurizer emits the 58-D contract."""
    _install(_CFG_SMALL, use_free_space=False, max_length=100)
    result = _run_lockstep(_CFG_SMALL, 21, 200, "survivor", allow_respawn=True)
    assert result["rows"].shape[1] == 58


def test_memory_none_is_stateless(restore_config):
    """memory=None acts like a fresh snake: trend 0 and nothing stored."""
    _install(_CFG_SMALL)
    params = Vector61Params.from_game_config(_CFG_SMALL)
    feat = Vector61Featurizer(params)
    sim = BatchSim(replace(_CFG_SMALL, num_envs=2), seeds=[1, 2], train_mode=True)
    for _ in range(5):
        sim.step(np.ones((2, _CFG_SMALL.num_snakes), dtype=np.int64))
    a = feat.featurize(sim)
    b = feat.featurize(sim)
    assert np.array_equal(a, b)
    assert np.all(a[:, 49] == 0.0)
    mem = Vector61Memory(2, _CFG_SMALL.num_snakes)
    feat.featurize(sim, memory=mem, update_memory=False)
    assert np.all(np.isinf(mem.prev_dist)) and np.all(mem.prev_id == -1)
    feat.featurize(sim, memory=mem, update_memory=True)
    alive = sim.get_alive()
    assert np.all(np.isfinite(mem.prev_dist[alive]))
    mem.reset_where(alive)
    assert np.all(np.isinf(mem.prev_dist)) and np.all(mem.prev_id == -1)


def test_dead_rows_are_zero_and_leave_memory(restore_config):
    _install(_CFG_SMALL)
    feat = Vector61Featurizer(Vector61Params.from_game_config(_CFG_SMALL))
    sim = BatchSim(replace(_CFG_SMALL, num_envs=1), seeds=[3], train_mode=True)
    sim.alive[0, 1] = False
    mem = Vector61Memory(1, _CFG_SMALL.num_snakes)
    out = feat.featurize(sim, np.array([[0, 1], [0, 0]]), mem)
    assert out.shape == (2, 61)
    assert np.all(out[0] == 0.0) and np.any(out[1] != 0.0)
    assert math.isinf(mem.prev_dist[0, 1])


def test_params_reject_unsupported_worlds():
    base = dict(
        game_width=300,
        game_height=200,
        segment_size=10,
        max_length=150,
        num_sectors=16,
        danger_max_distance=30,
        use_boundary_as_danger=True,
        min_boost_length=5,
        food_capacity=25,
    )
    Vector61Params(**base)
    with pytest.raises(NotImplementedError):
        Vector61Params(**{**base, "arena_type": "circular"})
    with pytest.raises(ValueError):
        Vector61Params(**{**base, "game_width": 305})
    with pytest.raises(ValueError):
        Vector61Params(**{**base, "actions": ((0, -1), (1, 0), (0, 1), (0, 1))})


# ---------------------------------------------------------------------------
# Synthetic-state parity on the deployment board (1450x830, 6 snakes).
# ---------------------------------------------------------------------------
_CFG_DEPLOY = build_parity_config(
    num_snakes=6, game_width=1450, game_height=830, initial_food=250, max_food=300
)


def _random_walk_body(gen, occupied: set, gw: int, gh: int, n: int) -> List[Tuple[int, int]]:
    """Self-avoiding random body (head first) of up to ``n`` cells."""
    for _ in range(50):
        head = (int(gen.integers(0, gw)), int(gen.integers(0, gh)))
        if head in occupied:
            continue
        body = [head]
        taken = {head}
        while len(body) < n:
            x, y = body[-1]
            options = [
                c
                for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                if 0 <= c[0] < gw and 0 <= c[1] < gh and c not in occupied and c not in taken
            ]
            if not options:
                break
            nxt = options[int(gen.integers(0, len(options)))]
            body.append(nxt)
            taken.add(nxt)
        return body
    return []


def _synthetic_world(gen, cfg: BatchSimConfig, num_envs: int, focus: Optional[Tuple] = None):
    """Build matching live Snake lists and a BatchSim with identical random states."""
    from src.game.snake import Snake

    ss = cfg.segment_size
    gw, gh = cfg.game_width // ss, cfg.game_height // ss
    if focus is not None:  # crowd everything into a small corner box
        gw, gh = focus
    sim = BatchSim(replace(cfg, num_envs=num_envs), seeds=list(range(num_envs)), train_mode=True)
    memory = Vector61Memory(num_envs, cfg.num_snakes)
    worlds = []
    for e in range(num_envs):
        occupied: set = set()
        snakes = []
        for s in range(cfg.num_snakes):
            n = int(gen.choice([1, 2, 3, 5, 12, 40, 90, 140]))
            body = _random_walk_body(gen, occupied, gw, gh, n) or [(s, 0)]
            occupied.update(body)
            snake = Snake(
                s,
                (255, 0, 0),
                (body[0][0] * ss, body[0][1] * ss),
                ss,
                cfg.game_width,
                cfg.game_height,
                food_capacity=cfg.max_food,
            )
            snake.segments = [(x * ss, y * ss) for x, y in body]
            snake.length = len(body) + int(gen.choice([0, 0, 0, 1, 2]))
            if len(body) > 1:
                d = (body[0][0] - body[1][0], body[0][1] - body[1][1])
            else:
                d = [(0, -1), (1, 0), (0, 1), (-1, 0)][int(gen.integers(0, 4))]
            snake.direction = d
            snake.is_alive = bool(gen.random() > 0.15)
            # Random trend baseline, sometimes exactly the current distance.
            if gen.random() < 0.5:
                pid = int(gen.integers(0, cfg.num_snakes))
                pdist = float(gen.choice([0.0, 10.0, 55.3, 300.0, 1e4]))
                snake._prev_nearest_enemy_dist = pdist
                snake._prev_nearest_enemy_id = pid
                memory.prev_dist[e, s] = pdist
                memory.prev_id[e, s] = pid
            snakes.append(snake)
            k = len(body)
            sim.bodies[e, s] = 0
            for i, cell in enumerate(body):
                sim.bodies[e, s, (-i) % sim.cap] = cell
            sim.head_ptr[e, s] = 0
            sim.seg_count[e, s] = k
            sim.length[e, s] = snake.length
            sim.alive[e, s] = snake.is_alive
            sim.direction[e, s] = [(0, -1), (1, 0), (0, 1), (-1, 0)].index(tuple(d))
        free = [(x, y) for x in range(gw) for y in range(gh) if (x, y) not in occupied]
        nfood = int(gen.choice([0, 1, 7, 60, 300]))
        picks = gen.permutation(len(free))[:nfood] if free else []
        food_cells = [free[int(i)] for i in picks]
        sim.food_cells[e] = list(food_cells)
        sim.food_set[e] = set(food_cells)
        worlds.append((snakes, [(x * ss, y * ss) for x, y in food_cells]))
    return sim, memory, worlds


@pytest.mark.parametrize("focus", [None, (40, 25)])
def test_vector61_bit_exact_synthetic_deployment_board(restore_config, focus):
    """Random long/coiled bodies, growth lag, dead bodies, 0..300 food, stale trends."""
    _install(_CFG_DEPLOY)
    feat = Vector61Featurizer(Vector61Params.from_game_config(_CFG_DEPLOY))
    gen = np.random.default_rng(2026 if focus is None else 926)
    fractional = 0
    for trial in range(6):
        sim, memory, worlds = _synthetic_world(gen, _CFG_DEPLOY, num_envs=3, focus=focus)
        rows = [(e, s) for e in range(3) for s in range(6) if worlds[e][0][s].is_alive]
        for update in (False, True):
            live = np.stack(
                [
                    worlds[e][0][s]
                    .get_state(worlds[e][0], worlds[e][1], update_enemy_memory=update)
                    .cpu()
                    .numpy()
                    for e, s in rows
                ]
            )
            batch = feat.featurize(sim, np.array(rows), memory, update_memory=update)
            _assert_bit_equal(live, batch, rows, trial, int(update))
            fractional += int(((batch[:, 58:61] > 0) & (batch[:, 58:61] < 1)).sum())
    if focus is not None:
        assert fractional > 0


def test_promotion_config_params_and_synthetic_parity(restore_config):
    """Params built from the real promotion YAML match the deployment world."""
    from src.core.config_loader import load_config
    from src.simd_env.eval_engine import _config_from_game_config

    initialize_config(load_config("configs/promotion_mechanics_v2.yaml"))
    sim_cfg = _config_from_game_config(6, 0.99)
    params = Vector61Params.from_game_config(sim_cfg)
    assert params.input_size == 61
    assert (params.game_width, params.game_height, params.segment_size) == (1450, 830, 10)
    assert params.max_length == 150 and params.food_capacity == 300
    feat = Vector61Featurizer(params)
    gen = np.random.default_rng(61)
    for trial in range(3):
        sim, memory, worlds = _synthetic_world(gen, sim_cfg, num_envs=2)
        rows = [(e, s) for e in range(2) for s in range(6) if worlds[e][0][s].is_alive]
        live = np.stack(
            [worlds[e][0][s].get_state(worlds[e][0], worlds[e][1]).cpu().numpy() for e, s in rows]
        )
        _assert_bit_equal(live, feat.featurize(sim, np.array(rows), memory), rows, trial, 0)
