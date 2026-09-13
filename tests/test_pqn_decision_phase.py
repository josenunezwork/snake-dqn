"""Decision-phase regression tests for the opt-in Watch-pre-move rollout path."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.simd_env.batch_sim import BatchSim
from src.training.pqn_trainer import (
    DECISION_PHASE_PRE_TRANSITION_V1,
    DECISION_PHASE_WATCH_PRE_MOVE_V1,
    PQNConfig,
    PQNTrainer,
)
from src.training.rollout_policies import FixedPolicySource


def _config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 1,
        "num_snakes": 1,
        "rollout_len": 2,
        "max_frames": 8,
        "initial_food": 0,
        "max_food": 0,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
    }
    values.update(overrides)
    return PQNConfig(**values)


def _obs_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(left[key], right[key]) for key in ("tactical", "strategic", "scalars"))


def _obs(trainer: PQNTrainer, sim=None) -> dict[str, torch.Tensor]:
    return trainer._current_obs(sim)[0]


def test_legacy_default_keeps_frame_zero_pre_transition_first_observation() -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_PRE_TRANSITION_V1))
    before = _obs(trainer)
    assert trainer.sim.frame.tolist() == [0]

    roll = trainer._rollout()

    assert trainer.cfg.decision_phase_mode == DECISION_PHASE_PRE_TRANSITION_V1
    assert torch.equal(roll["tactical"][0], before["tactical"])
    assert torch.equal(roll["strategic"][0], before["strategic"])
    assert torch.equal(roll["scalars"][0], before["scalars"])


def test_watch_first_stored_observation_is_callback_prepared_frame_one() -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1))
    prepared = copy.deepcopy(trainer.sim)

    class CapturePrepared(Exception):
        pass

    with pytest.raises(CapturePrepared):
        prepared.step_with_policy(lambda _: (_ for _ in ()).throw(CapturePrepared()))
    expected = _obs(trainer, prepared)

    roll = trainer._rollout()
    assert trainer.sim.frame.tolist() == [2]
    assert torch.equal(roll["tactical"][0], expected["tactical"])
    assert torch.equal(roll["strategic"][0], expected["strategic"])
    assert torch.equal(roll["scalars"][0], expected["scalars"])


def test_watch_final_successor_preview_does_not_mutate_actual_simulator_or_rng() -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1))
    roll = trainer._rollout()
    snapshot = copy.deepcopy(trainer.sim)
    rng_state = copy.deepcopy([rng._rng.getstate() for rng in trainer.sim._rngs])

    class CapturePrepared(Exception):
        pass

    prepared = copy.deepcopy(trainer.sim)
    with pytest.raises(CapturePrepared):
        prepared.step_with_policy(lambda _: (_ for _ in ()).throw(CapturePrepared()))

    # ``final_obs`` requires a preview, but obtaining it must leave the real
    # successor and RNG precisely where the actual rollout left them.
    assert _obs_equal(roll["final_obs"], _obs(trainer, prepared))
    assert np.array_equal(snapshot.frame, trainer.sim.frame)
    assert np.array_equal(snapshot.bodies, trainer.sim.bodies)
    assert rng_state == [rng._rng.getstate() for rng in trainer.sim._rngs]


def test_watch_shifts_prepared_successor_obs_and_mask_into_next_rollout_step() -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1))
    first = trainer._rollout()
    second = trainer._rollout()

    assert torch.equal(first["final_obs"]["tactical"], second["tactical"][0])
    assert torch.equal(first["final_obs"]["strategic"], second["strategic"][0])
    assert torch.equal(first["final_obs"]["scalars"], second["scalars"][0])
    assert torch.equal(first["final_mask"], second["decision_mask"][0])


def test_watch_max_frame_one_never_prepares_frame_two() -> None:
    trainer = PQNTrainer(
        _config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1, max_frames=1, rollout_len=2)
    )
    roll = trainer._rollout()
    assert roll["tactical"].shape[0] == 1
    assert trainer.sim.frame.tolist() == [1]
    assert roll["final_obs"]["scalars"].shape[0] == 1


def test_watch_target_uses_shifted_prepared_successor_mask_and_q() -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1, lambda_=0.0))
    roll = trainer._rollout()
    assert torch.equal(roll["next_mask"][0], roll["decision_mask"][1])
    # Q(lambda) retains its established no-extra-forward path: for every
    # non-final transition it takes the successor Q from hero_q[t + 1].
    targets = trainer._compute_targets(roll)
    mask = roll["decision_mask"][1, 0, 0]
    expected = roll["rewards"][0, 0, 0] + trainer.cfg.gamma * roll["hero_q"][1, 0, 0][mask].max()
    assert targets[0, 0, 0].item() == pytest.approx(float(expected))


def test_watch_preview_propagates_non_sentinel_failure(monkeypatch) -> None:
    trainer = PQNTrainer(_config(decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1))

    def broken_preview(*args, **kwargs):
        raise RuntimeError("preview boom")

    monkeypatch.setattr(trainer, "_capture_watch_preview", broken_preview)
    with pytest.raises(RuntimeError, match="preview boom"):
        trainer._rollout()


def test_watch_fixed_source_callback_sees_the_prepared_simulator_frame() -> None:
    class Fixed:
        identity = "fixed:prepared-frame-test:v1"

        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def actions(self, masks, sim, slots):
            self.frames.append(sim.frame.copy())
            return np.ones(len(slots), dtype=np.int64)

    fixed = Fixed()
    config = _config(
        num_snakes=2,
        hero_frac=0.0,
        rollout_policy_mode="fixed",
        fixed_policy_identity=fixed.identity,
        decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1,
    )
    PQNTrainer(config, fixed_policy=FixedPolicySource(fixed, fixed.identity))._rollout()
    assert [frames.tolist() for frames in fixed.frames] == [[1], [2]]


def test_watch_per_env_completion_freezes_completed_lane_and_prepares_continuing_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = dict(
        num_envs=2,
        max_frames=3,
        decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1,
        episode_reset_mode="per_env_autoreset_v1",
        episode_seed_mode="derived_env_episode_v1",
    )
    trainer = PQNTrainer(_config(**options, rollout_len=2))
    trainer.sim.frame[0] = 2
    terminal: dict[str, object] = {}
    real_step_with_policy = BatchSim.step_with_policy

    def capture_after_terminal(self, selector, active_env_mask=None):
        real_step_with_policy(self, selector, active_env_mask)
        # The final-observation preview operates on a deep copy.  Capture only
        # the real simulator's terminal post-transition state; otherwise an
        # instance-bound wrapper would accidentally invoke the original sim
        # from the clone's copied closure.
        if self is trainer.sim and self.frame[0] == trainer.cfg.max_frames and not terminal:
            terminal["obs"] = _obs(trainer)
            terminal["frame"] = self.frame.copy()
            terminal["body"] = self.bodies[0].copy()
            terminal["alive"] = self.alive[0].copy()
            terminal["direction"] = self.direction[0].copy()
            terminal["length"] = self.length[0].copy()
            terminal["food"] = list(self.food_cells[0])
            terminal["rng"] = copy.deepcopy(self._rngs[0]._rng.getstate())

    monkeypatch.setattr(BatchSim, "step_with_policy", capture_after_terminal)
    roll = trainer._rollout()

    assert trainer.sim.frame.tolist() == [3, 2]
    # Env 0 completes after t=0 and is frozen at terminal post-transition state.
    # Env 1 continues to an independent prepared decision at t=1.
    assert torch.equal(roll["scalars"][1, 0], terminal["obs"]["scalars"][0])
    assert trainer.sim.frame[0] == terminal["frame"][0]
    assert np.array_equal(trainer.sim.bodies[0], terminal["body"])
    assert np.array_equal(trainer.sim.alive[0], terminal["alive"])
    assert np.array_equal(trainer.sim.direction[0], terminal["direction"])
    assert np.array_equal(trainer.sim.length[0], terminal["length"])
    assert list(trainer.sim.food_cells[0]) == terminal["food"]
    assert trainer.sim._rngs[0]._rng.getstate() == terminal["rng"]
    assert not torch.equal(roll["scalars"][0, 1], roll["scalars"][1, 1])
