"""Tests for the dual-scale ego-raster featurizer (``src/simd_env/featurizer``).

Covers shape/dtype contracts, heading-rotation invariance (food dead-ahead lands
on the same tactical cell for all 4 headings), channel semantics (own body vs
enemy vs food vs wall in the right channel), scalar values on constructed states
(hunger, boost_available, wall distances), strategic density sums, and a smoke
that ``build_observations`` runs on a real BatchSim batch and on a live
GameState via the adapter.
"""

import hashlib
from itertools import combinations

import numpy as np
import pytest

from src.model.obs_spec import RASTER31V3
from src.simd_env.featurizer import (
    SCALARS_DIM,
    STRATEGIC_CHANNELS,
    STRATEGIC_SIZE,
    TACTICAL_CHANNELS,
    TACTICAL_HEAD_COL,
    TACTICAL_HEAD_ROW,
    TACTICAL_SIZE,
    ObsInputs,
    build_observations,
    expand_tactical,
    obs_inputs_from_batch_sim,
    to_network_input,
)


# ---------------------------------------------------------------------------
# ObsInputs builders for hand-built single-agent scenarios
# ---------------------------------------------------------------------------
def _one_snake_inputs(
    head=(50, 50),
    heading=1,
    body=None,
    length=None,
    food=None,
    grid_w=145,
    grid_h=83,
):
    """Build a 1-env, 1-snake ObsInputs (plus a padded dummy snake slot).

    ``body`` is a list of (col, row) head->tail; defaults to just the head.
    ``food`` is a list of ((col,row), mass, is_corpse).
    """
    S = 1
    if body is None:
        body = [head]
    maxlen = max(len(body), 1)
    bodies = np.zeros((1, S, maxlen, 2), dtype=np.int64)
    for k, c in enumerate(body):
        bodies[0, 0, k] = c
    body_len = np.array([[len(body)]], dtype=np.int64)
    lengths = np.array([[length if length is not None else len(body)]], dtype=np.int64)

    food = food or []
    F = max(len(food), 1)
    food_cells = np.zeros((1, F, 2), dtype=np.int64)
    food_mass = np.zeros((1, F), dtype=np.float64)
    food_corpse = np.zeros((1, F), dtype=bool)
    for i, (cell, mass, corpse) in enumerate(food):
        food_cells[0, i] = cell
        food_mass[0, i] = mass
        food_corpse[0, i] = corpse

    return ObsInputs(
        heads=np.array([[head]], dtype=np.int64),
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=np.array([[True]], dtype=bool),
        heading=np.array([[heading]], dtype=np.int64),
        boost_frames=np.zeros((1, S), dtype=np.int64),
        frames_since_food=np.zeros((1, S), dtype=np.int64),
        boosting=np.zeros((1, S), dtype=bool),
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_corpse,
        grid_w=grid_w,
        grid_h=grid_h,
        max_snakes=S,
        starvation_max=500,
        max_length=100,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([0], dtype=np.int64),
        max_frames=5000,
    )


def _v2_golden_inputs() -> tuple[ObsInputs, np.ndarray]:
    """Fixed pre-v3 fixture world; keep its recipe separate from the renderer."""
    mask = np.array(
        [[[True, False, True, False, True, True], [False, True, True, True, False, True]]]
    )
    return (
        ObsInputs(
            heads=np.array([[[10, 10], [14, 8]]], dtype=np.int64),
            bodies=np.array(
                [[[[10, 10], [9, 10], [8, 10], [0, 0]], [[14, 8], [14, 9], [14, 10], [14, 11]]]]
            ),
            body_len=np.array([[3, 4]], dtype=np.int64),
            lengths=np.array([[3, 6]], dtype=np.int64),
            alive=np.array([[True, True]]),
            heading=np.array([[1, 2]], dtype=np.int64),
            boost_frames=np.array([[2, 1]], dtype=np.int64),
            frames_since_food=np.array([[17, 41]], dtype=np.int64),
            boosting=np.array([[False, True]]),
            food_cells=np.array([[[12, 10], [12, 10], [7, 12]]], dtype=np.int64),
            food_mass=np.array([[1.0, 3.0, 2.0]]),
            food_is_corpse=np.array([[False, True, False]]),
            grid_w=30,
            grid_h=24,
            max_snakes=2,
            starvation_max=80,
            max_length=40,
            min_boost_length=5,
            boost_cost_frames=3,
            frame=np.array([37], dtype=np.int64),
            max_frames=200,
        ),
        mask,
    )


# ---------------------------------------------------------------------------
# Shapes / dtypes
# ---------------------------------------------------------------------------
def test_output_shapes_and_dtypes():
    inp = _one_snake_inputs()
    obs = build_observations(inp)
    assert obs["tactical_uint8"].shape == (1, 1, 2, TACTICAL_SIZE, TACTICAL_SIZE)
    assert obs["tactical_uint8"].dtype == np.uint8
    assert obs["strategic_uint8"].shape == (
        1,
        1,
        STRATEGIC_CHANNELS,
        STRATEGIC_SIZE,
        STRATEGIC_SIZE,
    )
    assert obs["strategic_uint8"].dtype == np.uint8
    assert obs["scalars"].shape == (1, 1, SCALARS_DIM)
    assert obs["scalars"].dtype == np.float32
    assert obs["mask"].shape == (1, 1, 6)


def test_v2_golden_fixture_and_v3_shape_contract():
    golden = np.load("tests/fixtures/raster31v2_golden.npz")
    assert hashlib.sha256(
        open("tests/fixtures/raster31v2_golden.npz", "rb").read()
    ).hexdigest() == ("7698e2ce0cfd50eecb4f74137ce8efe31dbae7e439a8a6fe6fd7192cf79a36af")
    fixture_inp, fixture_mask = _v2_golden_inputs()
    rebuilt = build_observations(fixture_inp, mask=fixture_mask)
    for key in ("tactical_uint8", "strategic_uint8", "scalars", "mask"):
        assert np.array_equal(rebuilt[key], golden[key]), key
    inp = _one_snake_inputs(head=(0, 0), heading=1, grid_w=4, grid_h=4)
    v3 = build_observations(inp, obs_spec=RASTER31V3)
    assert v3["tactical_uint8"].shape == (1, 1, 2, TACTICAL_SIZE, TACTICAL_SIZE)
    assert v3["scalars"].shape[-1] == SCALARS_DIM
    assert v3["tactical_uint8"][0, 0, 0, TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL - 1] == 1


def test_batch_adapter_propagates_explicit_normalizations_not_ring_capacity():
    """Feature normalization values are caller data, not BatchSim storage capacity."""
    from src.simd_env.batch_sim import BatchSim, BatchSimConfig

    sim = BatchSim(BatchSimConfig(num_envs=1, num_snakes=2, max_capacity=17), seeds=[4])
    inp = obs_inputs_from_batch_sim(sim, max_frames=47, starvation_max=31, max_length=71)
    assert (inp.max_frames, inp.starvation_max, inp.max_length) == (47, 31, 71)


_ORACLE_CODE = {
    "ambient": 2,
    "corpse": 3,
    "prediction": 4,
    "own_body": 5,
    "enemy_body": 6,
    "enemy_head": 7,
    "own_head": 8,
    "wall": 1,
}
_ORACLE_RANK = {
    "ambient": 1,
    "corpse": 2,
    "prediction": 3,
    "own_body": 4,
    "enemy_body": 5,
    "enemy_head": 6,
    "own_head": 7,
    "wall": 8,
}


def _v3_contended_input(kinds, *, enemy_order=None):
    """Build an independently specified full-raster collision at one ego cell."""
    own_head = "own_head" in kinds
    target = np.array((20, 20), dtype=np.int64)
    observer_head = target if own_head else np.array((21, 20), dtype=np.int64)
    enemy_kinds = [kind for kind in kinds if kind.startswith("enemy") or kind == "prediction"]
    if enemy_order is None:
        enemy_order = tuple(range(len(enemy_kinds)))
    S, maxlen = 1 + len(enemy_kinds), 4
    heads = np.zeros((1, S, 2), dtype=np.int64)
    bodies = np.zeros((1, S, maxlen, 2), dtype=np.int64)
    body_len = np.ones((1, S), dtype=np.int64)
    lengths = np.ones((1, S), dtype=np.int64)
    alive = np.ones((1, S), dtype=bool)
    heading = np.ones((1, S), dtype=np.int64)
    heads[0, 0] = observer_head
    bodies[0, 0, 0] = observer_head
    if "own_body" in kinds:
        body_len[0, 0] = 2
        bodies[0, 0, 1] = target
    for source, kind in enumerate(np.asarray(enemy_kinds)[list(enemy_order)], start=1):
        if kind == "enemy_head":
            head = target
        elif kind == "prediction":
            head = target - np.array((1, 0))
        else:  # enemy body target is a trailing segment with a distinct head.
            head = target + np.array((0, 2))
            body_len[0, source] = 2 + source
            bodies[0, source, 1] = target
        heads[0, source] = head
        bodies[0, source, 0] = head
        lengths[0, source] = 2 + source
    foods = [kind for kind in kinds if kind in {"ambient", "corpse"}]
    food_cells = np.zeros((1, max(len(foods), 1), 2), dtype=np.int64)
    food_mass = np.zeros((1, max(len(foods), 1)), dtype=np.float64)
    food_is_corpse = np.zeros((1, max(len(foods), 1)), dtype=bool)
    for index, kind in enumerate(foods):
        food_cells[0, index] = target
        food_mass[0, index] = 1.0 + index
        food_is_corpse[0, index] = kind == "corpse"
    return (
        ObsInputs(
            heads=heads,
            bodies=bodies,
            body_len=body_len,
            lengths=lengths,
            alive=alive,
            heading=heading,
            boost_frames=np.zeros((1, S), dtype=np.int64),
            frames_since_food=np.zeros((1, S), dtype=np.int64),
            boosting=np.zeros((1, S), dtype=bool),
            food_cells=food_cells,
            food_mass=food_mass,
            food_is_corpse=food_is_corpse,
            grid_w=40,
            grid_h=40,
            max_snakes=S,
            starvation_max=100,
            max_length=100,
            min_boost_length=5,
            boost_cost_frames=3,
            frame=np.zeros(1, dtype=np.int64),
            max_frames=100,
        ),
        tuple(target),
        tuple(observer_head),
    )


@pytest.mark.parametrize("pair", list(combinations(tuple(_ORACLE_RANK)[:-1], 2)))
def test_v3_all_rendered_type_pairs_use_independent_oracle(pair):
    """Public v3 output follows the card's semantic ordering, not renderer ranks."""
    inp, target, observer = _v3_contended_input(pair)
    code = build_observations(inp, obs_spec=RASTER31V3)["tactical_uint8"]
    ahead, lateral = target[0] - observer[0], target[1] - observer[1]
    row, col = TACTICAL_HEAD_ROW - ahead, TACTICAL_HEAD_COL + lateral
    expected = max(pair, key=lambda kind: _ORACLE_RANK[kind])
    assert code[0, 0, 0, row, col] == _ORACLE_CODE[expected]


def test_v3_wall_beats_prediction_and_drops_out_of_world_prediction():
    """A prediction outside the world cannot overwrite the true wall cell."""
    inp = _one_snake_inputs(head=(0, 20), heading=1, grid_w=40, grid_h=40)
    # Add an enemy whose next cell is (-1, 20), outside the rectangular world.
    inp = ObsInputs(
        **{
            **inp.__dict__,
            "heads": np.array([[[0, 20], [-2, 20]]]),
            "bodies": np.array([[[[0, 20]], [[-2, 20]]]]),
            "body_len": np.ones((1, 2), dtype=np.int64),
            "lengths": np.ones((1, 2), dtype=np.int64),
            "alive": np.ones((1, 2), dtype=bool),
            "heading": np.array([[1, 1]], dtype=np.int64),
            "boost_frames": np.zeros((1, 2), dtype=np.int64),
            "frames_since_food": np.zeros((1, 2), dtype=np.int64),
            "boosting": np.zeros((1, 2), dtype=bool),
            "max_snakes": 2,
        }
    )
    out = build_observations(inp, obs_spec=RASTER31V3)["tactical_uint8"]
    assert out[0, 0, 0, TACTICAL_HEAD_ROW + 1, TACTICAL_HEAD_COL] == _ORACLE_CODE["wall"]


def test_v3_enemy_slot_permutation_and_same_type_max_byte_tie():
    """Two enemy body candidates have the same result after source-slot permutation."""
    kinds = ("enemy_body", "enemy_body")
    rendered = []
    for order in ((0, 1), (1, 0)):
        inp, target, observer = _v3_contended_input(kinds, enemy_order=order)
        out = build_observations(inp, obs_spec=RASTER31V3)["tactical_uint8"]
        row, col = (
            TACTICAL_HEAD_ROW - (target[0] - observer[0]),
            TACTICAL_HEAD_COL + target[1] - observer[1],
        )
        rendered.append(out[0, 0, :, row, col].copy())
    assert np.array_equal(rendered[0], rendered[1])
    assert rendered[0][0] == _ORACLE_CODE["enemy_body"]
    assert rendered[0][1] == 191


def test_expand_and_network_input_shapes():
    inp = _one_snake_inputs()
    obs = build_observations(inp)
    exp = expand_tactical(obs["tactical_uint8"])
    assert exp.shape == (1, 1, TACTICAL_CHANNELS, TACTICAL_SIZE, TACTICAL_SIZE)
    assert exp.dtype == np.float32
    net = to_network_input(obs)
    assert net["tactical"].shape == (1, 1, TACTICAL_CHANNELS, TACTICAL_SIZE, TACTICAL_SIZE)
    assert net["strategic"].shape == (1, 1, STRATEGIC_CHANNELS, STRATEGIC_SIZE, STRATEGIC_SIZE)
    assert net["scalars"].shape == (1, 1, SCALARS_DIM)
    assert np.all((exp >= 0) & (exp <= 1))


# ---------------------------------------------------------------------------
# Heading-rotation invariance: food dead-ahead lands on the same tactical cell
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("heading", [0, 1, 2, 3])
def test_food_dead_ahead_same_cell_all_headings(heading):
    """Food 3 cells directly ahead lands at (HEAD_ROW-3, HEAD_COL) for every heading."""
    head = (70, 40)  # cells (7,4) with segment 10 -> but we pass cells directly
    hcol, hrow = head[0] // 10, head[1] // 10
    # place food 3 cells ahead in WORLD frame per heading:
    #   up: row-3; right: col+3; down: row+3; left: col-3
    ahead_step = {0: (0, -3), 1: (3, 0), 2: (0, 3), 3: (-3, 0)}[heading]
    food_cell = (hcol + ahead_step[0], hrow + ahead_step[1])
    inp = _one_snake_inputs(head=(hcol, hrow), heading=heading, food=[(food_cell, 1.0, False)])
    obs = build_observations(inp)
    exp = expand_tactical(obs["tactical_uint8"])[0, 0]
    food_ch = exp[4]  # ambient food channel
    ys, xs = np.nonzero(food_ch > 0)
    assert len(ys) == 1
    assert (ys[0], xs[0]) == (TACTICAL_HEAD_ROW - 3, TACTICAL_HEAD_COL)


@pytest.mark.parametrize("heading", [0, 1, 2, 3])
def test_food_to_the_right_same_cell_all_headings(heading):
    """Food 2 cells to the snake's RIGHT lands at (HEAD_ROW, HEAD_COL+2) for all headings."""
    hcol, hrow = 7, 4
    # world offset that is "to the snake's right" per heading:
    #   up: col+2; right: row+2; down: col-2; left: row-2
    right_step = {0: (2, 0), 1: (0, 2), 2: (-2, 0), 3: (0, -2)}[heading]
    food_cell = (hcol + right_step[0], hrow + right_step[1])
    inp = _one_snake_inputs(head=(hcol, hrow), heading=heading, food=[(food_cell, 1.0, False)])
    exp = expand_tactical(build_observations(inp)["tactical_uint8"])[0, 0]
    ys, xs = np.nonzero(exp[4] > 0)
    assert (ys[0], xs[0]) == (TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL + 2)


# ---------------------------------------------------------------------------
# Channel semantics
# ---------------------------------------------------------------------------
def test_own_head_and_body_channels():
    # Snake heading right (1) with a 3-cell body: head at (7,4), tail west.
    body = [(7, 4), (6, 4), (5, 4)]
    inp = _one_snake_inputs(head=(7, 4), heading=1, body=body, length=3)
    exp = expand_tactical(build_observations(inp)["tactical_uint8"])[0, 0]
    # Own head at the fixed head cell.
    assert exp[7][TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL] > 0
    # Own body: the two trailing cells are BEHIND the head (row > HEAD_ROW).
    body_cells = np.argwhere(exp[0] > 0)
    assert len(body_cells) == 2
    assert all(r > TACTICAL_HEAD_ROW for r, _ in body_cells)
    # Nothing painted into the enemy channels.
    assert exp[1].sum() == 0 and exp[2].sum() == 0


def test_enemy_body_and_head_channels():
    # Two snakes in one env: observer (snake 0) heading right; enemy (snake 1).
    S = 2
    heads = np.array([[[7, 4], [10, 4]]], dtype=np.int64)  # enemy 3 cells ahead
    bodies = np.zeros((1, S, 2, 2), dtype=np.int64)
    bodies[0, 0, 0] = (7, 4)
    bodies[0, 1, 0] = (10, 4)
    bodies[0, 1, 1] = (11, 4)  # enemy body cell behind its head
    inp = ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=np.array([[1, 2]], dtype=np.int64),
        lengths=np.array([[4, 8]], dtype=np.int64),  # enemy bigger -> ratio ~2
        alive=np.array([[True, True]], dtype=bool),
        heading=np.array([[1, 3]], dtype=np.int64),
        boost_frames=np.zeros((1, S), dtype=np.int64),
        frames_since_food=np.zeros((1, S), dtype=np.int64),
        boosting=np.zeros((1, S), dtype=bool),
        food_cells=np.zeros((1, 1, 2), dtype=np.int64),
        food_mass=np.zeros((1, 1), dtype=np.float64),
        food_is_corpse=np.zeros((1, 1), dtype=bool),
        grid_w=145,
        grid_h=83,
        max_snakes=S,
        starvation_max=500,
        max_length=100,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([0], dtype=np.int64),
        max_frames=5000,
    )
    exp = expand_tactical(build_observations(inp)["tactical_uint8"])[0, 0]
    # Enemy head 3 cells ahead of observer -> (HEAD_ROW-3, HEAD_COL) in ch2.
    assert exp[2][TACTICAL_HEAD_ROW - 3, TACTICAL_HEAD_COL] > 0
    # Enemy body in ch1 (its trailing cell, 4 cells ahead).
    assert exp[1][TACTICAL_HEAD_ROW - 4, TACTICAL_HEAD_COL] > 0
    # Enemy larger than observer -> head value = clamp(8/4,0,2)/2 = 1.0.
    assert np.isclose(exp[2][TACTICAL_HEAD_ROW - 3, TACTICAL_HEAD_COL], 1.0, atol=0.01)


def test_food_ambient_vs_corpse_channels():
    inp = _one_snake_inputs(
        head=(7, 4),
        heading=1,
        food=[((10, 4), 1.0, False), ((10, 5), 1.0, True)],  # ambient + corpse
    )
    exp = expand_tactical(build_observations(inp)["tactical_uint8"])[0, 0]
    assert exp[4].sum() > 0  # ambient channel populated
    assert exp[5].sum() > 0  # corpse channel populated
    # The corpse pellet is not in the ambient channel and vice versa.
    assert exp[4][TACTICAL_HEAD_ROW - 3, TACTICAL_HEAD_COL] > 0  # ambient ahead
    assert exp[5].sum() == exp[5].max()  # exactly one corpse cell


def test_wall_channel_near_boundary():
    # Head near the west wall (col 0), heading up so west is to the LEFT.
    inp = _one_snake_inputs(head=(1, 40), heading=0, grid_w=145, grid_h=83)
    exp = expand_tactical(build_observations(inp)["tactical_uint8"])[0, 0]
    # Wall cells appear to the left of the head column (col < HEAD_COL region).
    assert exp[6].sum() > 0
    wall_cells = np.argwhere(exp[6] > 0)
    # With head at world col 1 heading up, only cols world -? out of bounds map
    # to the snake's LEFT (ego col < HEAD_COL).
    assert all(c < TACTICAL_HEAD_COL for _, c in wall_cells)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------
def test_scalar_boost_available_and_hunger():
    inp = _one_snake_inputs(head=(7, 4), heading=1, length=6)  # >= min_boost 5
    inp.frames_since_food[:] = 250  # half of starvation_max 500
    obs = build_observations(inp)
    sc = obs["scalars"][0, 0]
    # scalar 2 = boost_available, scalar 4 = hunger.
    assert sc[2] == 1.0
    assert np.isclose(sc[4], 0.5, atol=1e-6)

    inp2 = _one_snake_inputs(head=(7, 4), heading=1, length=3)  # < min_boost
    sc2 = build_observations(inp2)["scalars"][0, 0]
    assert sc2[2] == 0.0


def test_scalar_wall_distances_ego():
    # Head at world (5, 40) heading right (1). Grid 145x83.
    # world: left=5, right=139, up=40, down=42. Ego (right heading):
    #   ahead=right=139, right=down=42, behind=left=5, left=up=40.
    inp = _one_snake_inputs(head=(5, 40), heading=1, grid_w=145, grid_h=83)
    sc = build_observations(inp)["scalars"][0, 0]
    NORM = 64.0
    assert np.isclose(sc[8], min(1.0, 139 / NORM), atol=1e-5)  # ahead
    assert np.isclose(sc[9], 42 / NORM, atol=1e-5)  # right
    assert np.isclose(sc[10], 5 / NORM, atol=1e-5)  # behind
    assert np.isclose(sc[11], 40 / NORM, atol=1e-5)  # left


def test_scalar_length_and_world_position():
    inp = _one_snake_inputs(head=(72, 41), heading=1, length=50)
    sc = build_observations(inp)["scalars"][0, 0]
    assert np.isclose(sc[0], 50 / 100.0, atol=1e-6)  # length/max_length
    # world x / y normalized.
    assert np.isclose(sc[23], 72 / 144.0, atol=1e-5)
    assert np.isclose(sc[24], 41 / 82.0, atol=1e-5)


def test_scalar_nearest_food_excludes_in_raster():
    """Scalars 12-14 report the nearest food BEYOND the tactical raster only.

    A pellet inside the 31x31 tactical footprint must NOT be reported (it is
    already visible in the raster); a pellet beyond the footprint must be. When
    the only food is in-raster, the beyond-raster scalars are zero.
    """
    # Heading right (1): ego "ahead" is +col. Head far from walls on a big grid.
    grid_w, grid_h = 300, 83
    head = (40, 40)
    # In-raster pellet: 3 cells ahead (col 43) -> ego ahead 3 <= AHEAD_MAX 23.
    near = ((43, 40), 1.0, False)
    # Beyond-raster pellet: 60 cells ahead (col 100) -> ego ahead 60 > 23.
    far = ((100, 40), 1.0, False)

    # Only the in-raster pellet -> beyond-raster scalars are all zero.
    inp_near = _one_snake_inputs(head=head, heading=1, food=[near], grid_w=grid_w, grid_h=grid_h)
    sc_near = build_observations(inp_near)["scalars"][0, 0]
    assert sc_near[12] == 0.0 and sc_near[13] == 0.0 and sc_near[14] == 0.0

    # Both pellets present -> the FAR (beyond-raster) pellet is reported, not the
    # globally-nearest in-raster one. Heading right => far pellet is dead ahead:
    # ego lateral 0 (dx == 0), ego ahead > 0 (dy > 0), dist > 0.
    inp_both = _one_snake_inputs(
        head=head, heading=1, food=[near, far], grid_w=grid_w, grid_h=grid_h
    )
    sc_both = build_observations(inp_both)["scalars"][0, 0]
    diag = float(np.hypot(grid_w, grid_h))
    assert np.isclose(sc_both[12], 0.0, atol=1e-6)  # dx (lateral) == 0
    assert np.isclose(sc_both[13], 60.0 / diag, atol=1e-6)  # dy (ahead) == 60 cells
    assert np.isclose(sc_both[14], 60.0 / diag, atol=1e-6)  # dist == 60 cells


# ---------------------------------------------------------------------------
# Strategic density
# ---------------------------------------------------------------------------
def test_strategic_own_body_density_sum():
    # A long straight snake heading right. Own-body density (ch2) should hold
    # mass roughly equal to the number of body segments / normalization.
    n = 20
    body = [(30 - k, 40) for k in range(n)]
    inp = _one_snake_inputs(head=(30, 40), heading=1, body=body, length=n)
    strat = build_observations(inp)["strategic_uint8"][0, 0]
    # Own body channel = 2. Total painted mass (bytes) should be > 0.
    assert strat[2].sum() > 0
    # Enemy and food channels are empty in this single-snake scene.
    assert strat[0].sum() == 0
    # Head coarse cell (12,12) is populated.
    assert strat[2][12, 12] > 0


def test_strategic_food_density():
    food = [((30 + i, 40), 1.0, False) for i in range(10)]
    inp = _one_snake_inputs(head=(30, 40), heading=1, food=food)
    strat = build_observations(inp)["strategic_uint8"][0, 0]
    assert strat[1].sum() > 0  # food density channel


# ---------------------------------------------------------------------------
# Smoke: real BatchSim batch + live GameState via the adapter
# ---------------------------------------------------------------------------
def test_smoke_batch_sim():
    from src.simd_env import BatchSim, BatchSimConfig

    cfg = BatchSimConfig(num_envs=2, num_snakes=6)
    sim = BatchSim(cfg, seeds=[0, 1], train_mode=True)
    for _ in range(8):
        sim.step(np.ones((2, 6), dtype=np.int64))
    inp = obs_inputs_from_batch_sim(sim)
    obs = build_observations(inp, mask=sim.get_action_mask())
    assert obs["tactical_uint8"].shape == (2, 6, 2, 31, 31)
    assert obs["strategic_uint8"].shape == (2, 6, 3, 25, 25)
    assert obs["scalars"].shape == (2, 6, 26)
    assert obs["mask"].shape == (2, 6, 6)
    assert np.isfinite(obs["scalars"]).all()


def test_smoke_live_game_state(setup_config):
    from src.game.game_state import GameState
    from src.simd_env.live_adapter import game_state_to_obs_inputs

    gs = GameState(headless=True, num_snakes=4)
    for _ in range(5):
        gs.update(train_mode=False, learn=False)
    inp = game_state_to_obs_inputs(gs)
    obs = build_observations(inp)
    assert obs["tactical_uint8"].shape == (1, 4, 2, 31, 31)
    assert obs["scalars"].shape == (1, 4, 26)
    assert np.isfinite(obs["scalars"]).all()
    gs.full_cleanup()
