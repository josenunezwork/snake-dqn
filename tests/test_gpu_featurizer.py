"""Train/serve parity gate for the GPU featurizer (``src/simd_env/gpu_featurizer``).

The web app serves observations via the NumPy featurizer
(:func:`src.simd_env.featurizer.build_observations`); the PQN trainer will
featurize on the GPU via :func:`build_observations_gpu`. If the two diverge, a
GPU-trained policy sees different inputs at serve time. These tests are the gate:
over a stepped :class:`~src.simd_env.batch_sim.BatchSim` the GPU path must
produce, per the reference spec,

- ``expand_tactical(obs['tactical_uint8'])`` -> ``(E, S, 9, 31, 31)`` float
  planes that are **exactly equal** (byte-identical uint8, then the same ``/255``
  expansion),
- ``obs['strategic_uint8'] / 255`` -> ``(E, S, 3, 25, 25)`` planes exactly equal,
- ``obs['scalars']`` -> ``(E, S, 26)`` scalars within ``1e-4`` (float32 vs
  float64 accumulation only).

Coverage spans all four cardinal headings (bodies grown so enemies are in view),
boosting snakes (exercising the predicted-2-cell + own-head boost channels),
snake deaths, contested cells where two entities map to one ego cell (the overlap
priority winner must match), and the ``obs_inputs_to_torch`` bridge / shapes /
mask passthrough.

Runs on the CPU torch device so no GPU is required; ``build_observations_gpu`` is
device-agnostic, so CPU parity of the identical ops implies GPU parity.
"""

import time

import numpy as np
import torch

from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import (
    SCALARS_DIM,
    ObsInputs,
    build_observations,
    expand_tactical,
    obs_inputs_from_batch_sim,
)
from src.simd_env.gpu_featurizer import (
    GpuObsState,
    build_observations_gpu,
    obs_inputs_to_torch,
    raster_tensors_from_gpu_obs,
)

DEVICE = torch.device("cpu")
MAX_FRAMES = 5000

TACTICAL_SHAPE = (9, 31, 31)
STRATEGIC_SHAPE = (3, 25, 25)


# ---------------------------------------------------------------------------
# Reference builders (the exact spec the GPU path must match)
# ---------------------------------------------------------------------------
def _numpy_reference(sim):
    """CPU reference triple from the NumPy featurizer, in network-input form.

    Returns ``(ref_tac, ref_strat, ref_scal, mask)`` where ``ref_tac`` is the
    expanded ``(E, S, 9, 31, 31)`` float32 tactical raster, ``ref_strat`` the
    ``(E, S, 3, 25, 25)`` strategic raster ``/255``, and ``ref_scal`` the
    ``(E, S, 26)`` scalars — exactly as :func:`build_observations_gpu` emits them.
    """
    inp = obs_inputs_from_batch_sim(sim, max_frames=MAX_FRAMES)
    mask = sim.get_action_mask()
    obs = build_observations(inp, mask=mask)
    ref_tac = torch.as_tensor(expand_tactical(obs["tactical_uint8"]), dtype=torch.float32)
    ref_strat = torch.as_tensor(obs["strategic_uint8"].astype(np.float32) / 255.0)
    ref_scal = torch.as_tensor(np.asarray(obs["scalars"], dtype=np.float32))
    return ref_tac, ref_strat, ref_scal, mask


def _assert_parity(g, ref_tac, ref_strat, ref_scal, ctx=""):
    """Assert GPU planes are byte-identical and scalars within ``1e-4``.

    On divergence the message names the component and the offending
    ``(env, snake, ...)`` cell so a train/serve regression is immediately
    actionable.
    """
    if not torch.equal(g["tactical"], ref_tac):
        diff = (g["tactical"] - ref_tac).abs()
        loc = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            f"{ctx} tactical diverges at (env,snake,chan,row,col)={loc} "
            f"gpu={g['tactical'][loc].item()} numpy={ref_tac[loc].item()} "
            f"max_abs={diff.max().item():.3e}"
        )
    if not torch.equal(g["strategic"], ref_strat):
        diff = (g["strategic"] - ref_strat).abs()
        loc = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            f"{ctx} strategic diverges at (env,snake,chan,row,col)={loc} "
            f"gpu={g['strategic'][loc].item()} numpy={ref_strat[loc].item()} "
            f"max_abs={diff.max().item():.3e}"
        )
    scal_diff = torch.max(torch.abs(g["scalars"] - ref_scal)).item()
    if scal_diff >= 1e-4:
        diff = (g["scalars"] - ref_scal).abs()
        loc = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            f"{ctx} scalar diverges at (env,snake,scalar)={loc} "
            f"gpu={g['scalars'][loc].item()} numpy={ref_scal[loc].item()} "
            f"max_abs={scal_diff:.3e} (tol 1e-4)"
        )
    return scal_diff


# ---------------------------------------------------------------------------
# Action policies used to exercise the mechanics
# ---------------------------------------------------------------------------
def _growth_boost_actions(sim, rng):
    """Valid actions biased toward boost (grows bodies, drives all headings).

    Prefers a boost action (3-5) whenever the snake is long enough to boost and
    a boost action is safe, so the predicted-2-cell and own-head boost channels
    get exercised; otherwise a random safe action (falls back to 0 if trapped).
    """
    E, S = sim.E, sim.S
    mask = sim.get_action_mask()
    lengths = sim.get_lengths()
    actions = np.zeros((E, S), dtype=np.int64)
    for e in range(E):
        for s in range(S):
            valid = np.nonzero(mask[e, s])[0]
            if valid.size == 0:
                continue
            boost_valid = valid[valid >= 3]
            if lengths[e, s] >= 5 and boost_valid.size and rng.random() < 0.5:
                actions[e, s] = rng.choice(boost_valid)
            else:
                actions[e, s] = rng.choice(valid)
    return actions


# ---------------------------------------------------------------------------
# Core parity gate
# ---------------------------------------------------------------------------
def test_parity_over_stepped_batchsim_all_cases():
    """GPU planes byte-identical, scalars <=1e-4 over a rollout that exercises
    all four headings, boosting, and enemies-in-view (8 envs x 6 snakes)."""
    cfg = BatchSimConfig(num_envs=8, num_snakes=6, mechanics_version=2)
    sim = BatchSim(cfg, seeds=list(range(8)), train_mode=True)
    rng = np.random.default_rng(12345)

    boost_seen = False
    headings_seen = set()
    max_scal = 0.0

    for frame in range(80):
        ref_tac, ref_strat, ref_scal, mask = _numpy_reference(sim)
        state = obs_inputs_to_torch(sim, DEVICE, max_frames=MAX_FRAMES)
        g = build_observations_gpu(state, mask=torch.as_tensor(mask))
        max_scal = max(max_scal, _assert_parity(g, ref_tac, ref_strat, ref_scal, f"frame {frame}"))

        boost_seen = boost_seen or bool(sim.get_boosted_this_step().any())
        alive = sim.get_alive()
        headings_seen.update(int(h) for h in np.unique(sim.get_directions()[alive]))
        sim.step(_growth_boost_actions(sim, rng))

    # The rollout must actually have exercised the boost + all-heading paths,
    # otherwise the parity assertions above never touched those channels.
    assert boost_seen, "rollout never triggered a boost; boost channels untested"
    assert headings_seen == {0, 1, 2, 3}, f"headings not all exercised: {headings_seen}"
    assert max_scal < 1e-4


def test_parity_multiple_seeds_with_deaths():
    """Parity holds across seeds and after snakes die (crowded arena, unmasked
    random actions so snakes actually collide)."""
    death_seen = False
    for seed in (7, 23):
        cfg = BatchSimConfig(num_envs=6, num_snakes=10, mechanics_version=2)
        sim = BatchSim(cfg, seeds=[seed * 10 + i for i in range(6)], train_mode=True)
        rng = np.random.default_rng(seed)
        for frame in range(120):
            ref_tac, ref_strat, ref_scal, mask = _numpy_reference(sim)
            state = obs_inputs_to_torch(sim, DEVICE, max_frames=MAX_FRAMES)
            g = build_observations_gpu(state, mask=torch.as_tensor(mask))
            _assert_parity(g, ref_tac, ref_strat, ref_scal, f"seed {seed} frame {frame}")
            death_seen = death_seen or bool((~sim.get_alive()).any())
            # Fully-random (unmasked) actions provoke collisions -> dead snakes.
            sim.step(rng.integers(0, 6, size=(sim.E, sim.S)).astype(np.int64))
    assert death_seen, "no snake died; the dead-snake path went untested"


# ---------------------------------------------------------------------------
# Constructed overlap: two entities -> one ego cell; priority winner must match
# ---------------------------------------------------------------------------
def _obsinputs_to_gpustate(inp: ObsInputs) -> GpuObsState:
    """Mirror an :class:`ObsInputs` into a :class:`GpuObsState` on the CPU device.

    Lets a hand-built world snapshot feed both featurizers directly, bypassing
    the sim so a contested-cell scenario can be constructed deterministically.
    """

    def t(arr, dtype):
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=DEVICE)

    return GpuObsState(
        heads=t(inp.heads, torch.int64),
        bodies=t(inp.bodies, torch.int64),
        body_len=t(inp.body_len, torch.int64),
        lengths=t(inp.lengths, torch.int64),
        alive=t(inp.alive, torch.bool),
        heading=t(inp.heading, torch.int64),
        boost_frames=t(inp.boost_frames, torch.int64),
        frames_since_food=t(inp.frames_since_food, torch.int64),
        boosting=t(inp.boosting, torch.bool),
        food_cells=t(inp.food_cells, torch.int64),
        food_mass=t(inp.food_mass, torch.float64),
        food_is_corpse=t(inp.food_is_corpse, torch.bool),
        frame=t(inp.frame, torch.int64),
        grid_w=inp.grid_w,
        grid_h=inp.grid_h,
        max_snakes=inp.max_snakes,
        starvation_max=inp.starvation_max,
        max_length=inp.max_length,
        min_boost_length=inp.min_boost_length,
        boost_cost_frames=inp.boost_cost_frames,
        max_frames=inp.max_frames,
        arena_type_flag=inp.arena_type_flag,
        device=DEVICE,
    )


def test_constructed_overlap_priority_winner_matches():
    """Two entities mapping to one ego cell resolve to the same winner on both
    paths: enemy_body (code 6) beats enemy_pred (code 4); corpse (3) beats
    ambient (2)."""
    E, S, MAXLEN = 1, 3, 4
    heads = np.zeros((E, S, 2), np.int64)
    bodies = np.zeros((E, S, MAXLEN, 2), np.int64)
    body_len = np.ones((E, S), np.int64)
    lengths = np.ones((E, S), np.int64)
    alive = np.ones((E, S), bool)
    heading = np.zeros((E, S), np.int64)  # all facing up

    # Observer (snake 0) at (20, 20) heading up.
    heads[0, 0] = (20, 20)
    bodies[0, 0, 0] = (20, 20)

    # Snake 1 head at (25, 15) heading up -> predicted-next cell = (25, 14).
    heads[0, 1] = (25, 15)
    lengths[0, 1], body_len[0, 1] = 3, 3
    bodies[0, 1, 0] = (25, 15)
    bodies[0, 1, 1] = (25, 16)
    bodies[0, 1, 2] = (25, 17)

    # Snake 2 has a BODY segment exactly on (25, 14) == snake 1's predicted cell.
    heads[0, 2] = (10, 10)
    lengths[0, 2], body_len[0, 2] = 3, 3
    bodies[0, 2, 0] = (10, 10)
    bodies[0, 2, 1] = (25, 14)  # contested with snake 1's enemy_pred
    bodies[0, 2, 2] = (11, 10)

    # Two pellets stacked on (18, 18): one ambient, one corpse -> corpse wins.
    food_cells = np.array([[[18, 18], [18, 18], [30, 30]]], np.int64)
    food_mass = np.array([[1.0, 1.0, 1.0]], np.float64)
    food_is_corpse = np.array([[False, True, False]], bool)

    inp = ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=alive,
        heading=heading,
        boost_frames=np.zeros((E, S), np.int64),
        frames_since_food=np.zeros((E, S), np.int64),
        boosting=np.zeros((E, S), bool),
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=40,
        grid_h=40,
        max_snakes=S,
        starvation_max=500,
        max_length=400,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.zeros(E, np.int64),
        max_frames=MAX_FRAMES,
        arena_type_flag=0.0,
    )

    obs = build_observations(inp)
    ref_tac = torch.as_tensor(expand_tactical(obs["tactical_uint8"]), dtype=torch.float32)
    ref_strat = torch.as_tensor(obs["strategic_uint8"].astype(np.float32) / 255.0)
    ref_scal = torch.as_tensor(np.asarray(obs["scalars"], dtype=np.float32))
    g = build_observations_gpu(_obsinputs_to_gpustate(inp))
    _assert_parity(g, ref_tac, ref_strat, ref_scal, "constructed-overlap")

    # Contested cell (25,14) in observer 0's ego frame: ahead=20-14=6,
    # lateral=25-20=5 -> row=23-6=17, col=15+5=20. enemy_body must win, pred off.
    assert ref_tac[0, 0, 1, 17, 20].item() > 0, "enemy_body absent at contested cell"
    assert ref_tac[0, 0, 3, 17, 20].item() == 0, "enemy_pred should be overwritten by body"
    assert torch.equal(g["tactical"][0, 0, 1, 17, 20], ref_tac[0, 0, 1, 17, 20])
    assert torch.equal(g["tactical"][0, 0, 3, 17, 20], ref_tac[0, 0, 3, 17, 20])

    # Stacked food at (18,18): ahead=2, lateral=-2 -> row=21, col=13. Corpse wins.
    assert ref_tac[0, 0, 5, 21, 13].item() > 0, "corpse_food absent at stacked cell"
    assert ref_tac[0, 0, 4, 21, 13].item() == 0, "ambient_food should be overwritten by corpse"
    assert torch.equal(g["tactical"][0, 0, 5, 21, 13], ref_tac[0, 0, 5, 21, 13])
    assert torch.equal(g["tactical"][0, 0, 4, 21, 13], ref_tac[0, 0, 4, 21, 13])


def test_food_overlap_priority_corpse_at_lower_index():
    """Corpse wins a contested ego cell regardless of pellet ARRAY order.

    Regression for the food-paint ``krank`` bug: the GPU ``_paint_food_tactical``
    must rank colliding pellets by type code (corpse code 3 > ambient code 2), not
    pass a constant rank. With a constant rank every colliding pellet is a "winner"
    and the survivor falls to torch's duplicate-index scatter order (pellet array
    order), so a corpse at a LOWER array index than an ambient pellet on the same
    ego cell lost to the ambient one — diverging from NumPy's code-priority. This
    is the mirror of :func:`test_constructed_overlap_priority_winner_matches`
    (which stacks the corpse at the HIGHER index and therefore passed even with the
    bug). Both orderings must resolve to corpse on both paths.
    """
    E, S, MAXLEN = 1, 3, 4
    heads = np.zeros((E, S, 2), np.int64)
    bodies = np.zeros((E, S, MAXLEN, 2), np.int64)
    body_len = np.ones((E, S), np.int64)
    lengths = np.ones((E, S), np.int64)
    alive = np.ones((E, S), bool)
    heading = np.zeros((E, S), np.int64)

    # Observer (snake 0) at (20, 20) heading up; the other snakes are elsewhere.
    heads[0, 0] = (20, 20)
    bodies[0, 0, 0] = (20, 20)
    heads[0, 1] = (10, 10)
    bodies[0, 1, 0] = (10, 10)
    heads[0, 2] = (11, 11)
    bodies[0, 2, 0] = (11, 11)

    # Two pellets stacked on (18, 18) with the CORPSE at the LOWER array index.
    food_cells = np.array([[[18, 18], [18, 18], [30, 30]]], np.int64)
    food_mass = np.array([[1.0, 1.0, 1.0]], np.float64)
    food_is_corpse = np.array([[True, False, False]], bool)  # corpse FIRST

    inp = ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=alive,
        heading=heading,
        boost_frames=np.zeros((E, S), np.int64),
        frames_since_food=np.zeros((E, S), np.int64),
        boosting=np.zeros((E, S), bool),
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=40,
        grid_h=40,
        max_snakes=S,
        starvation_max=500,
        max_length=400,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.zeros(E, np.int64),
        max_frames=MAX_FRAMES,
        arena_type_flag=0.0,
    )

    obs = build_observations(inp)
    ref_tac = torch.as_tensor(expand_tactical(obs["tactical_uint8"]), dtype=torch.float32)
    ref_strat = torch.as_tensor(obs["strategic_uint8"].astype(np.float32) / 255.0)
    ref_scal = torch.as_tensor(np.asarray(obs["scalars"], dtype=np.float32))
    g = build_observations_gpu(_obsinputs_to_gpustate(inp))
    _assert_parity(g, ref_tac, ref_strat, ref_scal, "corpse-lower-index")

    # Stacked food at (18,18): ahead=2, lateral=-2 -> row=21, col=13. Corpse (code
    # 3) must win over ambient (code 2) even though it is the lower array index.
    assert ref_tac[0, 0, 5, 21, 13].item() > 0, "NumPy: corpse must win at stacked cell"
    assert ref_tac[0, 0, 4, 21, 13].item() == 0, "NumPy: ambient must be overwritten"
    assert torch.equal(g["tactical"][0, 0, 5, 21, 13], ref_tac[0, 0, 5, 21, 13])
    assert torch.equal(g["tactical"][0, 0, 4, 21, 13], ref_tac[0, 0, 4, 21, 13])


# ---------------------------------------------------------------------------
# Bridge / shapes / dtypes / mask
# ---------------------------------------------------------------------------
def test_shapes_dtypes_and_mask_passthrough():
    """Output dict has the right shapes/dtypes and passes the mask through."""
    cfg = BatchSimConfig(num_envs=3, num_snakes=4, mechanics_version=2)
    sim = BatchSim(cfg, seeds=[0, 1, 2], train_mode=True)
    E, S = sim.E, sim.S
    state = obs_inputs_to_torch(sim, DEVICE)
    assert isinstance(state, GpuObsState)
    assert state.E == E and state.S == S

    mask = sim.get_action_mask()
    g = build_observations_gpu(state, mask=torch.as_tensor(mask))
    assert g["tactical"].shape == (E, S, *TACTICAL_SHAPE)
    assert g["strategic"].shape == (E, S, *STRATEGIC_SHAPE)
    assert g["scalars"].shape == (E, S, SCALARS_DIM)
    assert g["mask"].shape == (E, S, 6)
    assert g["tactical"].dtype == torch.float32
    assert g["strategic"].dtype == torch.float32
    assert g["scalars"].dtype == torch.float32
    assert g["mask"].dtype == torch.bool
    assert torch.equal(g["mask"], torch.as_tensor(mask, dtype=torch.bool))

    r = raster_tensors_from_gpu_obs(g)
    assert set(r.keys()) == {"tactical", "strategic", "scalars"}
    assert r["tactical"] is g["tactical"]


def test_default_mask_all_true():
    """Omitting the mask yields an all-True (E, S, 6) mask on device."""
    cfg = BatchSimConfig(num_envs=2, num_snakes=3, mechanics_version=2)
    sim = BatchSim(cfg, seeds=[0, 1], train_mode=True)
    state = obs_inputs_to_torch(sim, DEVICE)
    g = build_observations_gpu(state)
    assert g["mask"].dtype == torch.bool
    assert bool(g["mask"].all())


def test_micro_timing_smoke():
    """Rough CPU micro-timing (informational, not a hard assert).

    On CPU the two paths are comparable; the real win is that the torch path runs
    on the otherwise-idle GPU at train time. This only guards that the GPU path
    runs end-to-end at a realistic size (128 envs x 6 snakes) and prints timings.
    """
    cfg = BatchSimConfig(num_envs=128, num_snakes=6, mechanics_version=2)
    sim = BatchSim(cfg, seeds=list(range(128)), train_mode=True)
    rng = np.random.default_rng(0)
    for _ in range(8):
        sim.step(_growth_boost_actions(sim, rng))

    # Warm up both paths.
    build_observations(obs_inputs_from_batch_sim(sim, max_frames=MAX_FRAMES))
    build_observations_gpu(obs_inputs_to_torch(sim, DEVICE, max_frames=MAX_FRAMES))

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        build_observations(obs_inputs_from_batch_sim(sim, max_frames=MAX_FRAMES))
    t_np = (time.perf_counter() - t0) / reps * 1000.0

    t0 = time.perf_counter()
    for _ in range(reps):
        build_observations_gpu(obs_inputs_to_torch(sim, DEVICE, max_frames=MAX_FRAMES))
    t_gpu = (time.perf_counter() - t0) / reps * 1000.0

    print(f"\n[micro-timing 128x6 CPU] numpy={t_np:.1f}ms torch-cpu={t_gpu:.1f}ms")
    assert t_np > 0 and t_gpu > 0
