"""T4d telescoping target checked against actual simulator/trainer rewards."""

from __future__ import annotations

import numpy as np
import pytest

from research.redesign_m3c_20261008.train_m3c import mass_term
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def short_branch(kind: str, gamma: float, kappa: float) -> dict:
    """Roll a controlled short branch, using production reward assembly each step."""
    sim = GridBatchSim(
        BatchSimConfig(
            num_envs=1,
            num_snakes=1,
            initial_food=0,
            max_food=0,
            gamma=gamma,
            kill_scale=0.0,
            death_value=-3.0,
            max_capacity=32,
        ),
        seeds=[42819],
        train_mode=True,
        allow_respawn=False,
        jit=False,
    )
    head = 142 if kind == "death" else 12
    cells = [(head - offset, 10) for offset in range(6)]
    sim.bodies[0, 0] = 0
    sim.bodies[0, 0, :6] = np.asarray(cells[::-1])
    sim.head_ptr[0, 0] = 5
    sim.seg_count[0, 0] = sim.length[0, 0] = 6
    sim.alive[0, 0] = True
    sim.direction[0, 0] = 1
    sim.boost_frames[0, 0] = sim.frames_since_food[0, 0] = 0
    sim._reward_prev_length[0, 0] = 6
    food = [(head + 1, 10)] if kind in {"growth", "death"} else []
    sim.food_cells[0] = food.copy()
    sim.food_set[0] = set(food)
    sim.corpse_cells[0] = set()
    sim._rebuild_env_grids(0)
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()
    actions = [4] * 3 if kind == "boost" else [1] * 4
    lengths, rewards = [], []
    for action in actions:
        sim.step(np.asarray([[action]], dtype=np.int64))
        alive = sim.get_alive()[:, 0] & ~sim.get_done()[:, 0]
        reward = sim.get_reward()[:, 0] + mass_term(
            sim.get_lengths()[:, 0], alive, gamma, kappa
        )
        rewards.append(float(reward[0]))
        lengths.append(int(sim.length[0, 0]) if alive[0] else 0)
        if not alive[0]:
            break
    died = not bool(sim.alive[0, 0])
    assert died == (kind == "death")
    if kind == "growth":
        assert lengths[0] > 6
    if kind == "boost":
        assert lengths[-1] < 6
    return {
        "T": len(rewards),
        "died": died,
        "S": sum(gamma**t * length / 10.0 for t, length in enumerate(lengths)),
        "L0": 6,
        "L_T": lengths[-1],
        "discounted_rewards": sum(gamma**t * r for t, r in enumerate(rewards)),
    }


@pytest.mark.parametrize("kind", ["survivor", "growth", "boost", "death"])
@pytest.mark.parametrize("gamma,kappa", [(0.998, 1.0), (0.9, 0.4), (0.998, 0.0)])
def test_return_formula_equals_actual_trainer_rewards(kind: str, gamma: float, kappa: float) -> None:
    """Pin Phi(dead)=0, death exponent T-1, mass timing, and PBRS telescoping."""
    branch = short_branch(kind, gamma, kappa)
    t, died = branch["T"], branch["died"]
    target = (
        kappa * (1 - gamma) * branch["S"]
        - branch["L0"] / 10
        + (0 if died else gamma**t * branch["L_T"] / 10)
        - (3 * gamma ** (t - 1) if died else 0)
    )
    assert target == pytest.approx(branch["discounted_rewards"], abs=2e-14)
