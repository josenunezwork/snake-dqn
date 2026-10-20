"""Native chronological windows and publication under a moving live clock."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_pathb.batch_v2 import evaluate_decisions
from research.t34_pathb.engine_v2 import (
    PathBSearchEngine,
    PreparedDecision,
    SearchOutcome,
    SearchRequest,
    _capture_and_step,
)


@pytest.mark.parametrize("mix", ["scripted", "frozen", "mixed"])
@pytest.mark.skipif(
    not Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
    ).is_file(),
    reason="released B8 gameplay artifacts are not installed",
)
def test_branch_point_batch_matches_independent_reference(mix):
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared
    from research.t34_shadow.identity import _new_pair
    from research.t34_shadow.shadow import clone_world_exact

    _live, world, _specs, _identity = _new_pair(991827, mix)
    decisions = [_capture_and_step(world) for _ in range(3)]
    horizons = [20, 10, 3]
    actual = evaluate_decisions(decisions, horizons)
    completed = []

    def stop_at_short_source(index, outcomes):
        completed.append(index)
        return index == 2

    partial = evaluate_decisions(decisions, horizons, on_complete=stop_at_short_source)
    assert completed[-1] == 2
    assert partial[2] == actual[2]
    assert partial[0] == {} and partial[1] == {}
    for decision, horizon, outcomes in zip(decisions, horizons, actual):
        assert set(outcomes) == {i for i, legal in enumerate(decision.legal) if legal}
        for action, outcome in outcomes.items():
            branch = clone_world_exact(decision.world)
            step_prepared(branch, action)
            alive = int(branch.sim.alive[0, 0])
            mass = float(branch.sim.length[0, 0]) if alive else 0.0
            for _ in range(1, horizon):
                if not branch.sim.alive[0, 0]:
                    break
                choose_and_step(branch, capture_obs=False)
                alive += int(branch.sim.alive[0, 0])
                mass += float(branch.sim.length[0, 0]) if branch.sim.alive[0, 0] else 0.0
            assert (outcome.survives, outcome.alive_steps, outcome.mass) == (
                bool(branch.sim.alive[0, 0]) and alive == horizon,
                alive,
                mass,
            )


@pytest.mark.parametrize("invalidate", [False, True])
def test_window_publishes_first_still_future_result(monkeypatch, invalidate):
    import research.t34_pathb.engine_v2 as engine

    root = SimpleNamespace(frame=0, sim=SimpleNamespace(alive=np.array([[True]])))
    progress = [0, 0]

    def capture(current):
        current.frame += 1
        return PreparedDecision(
            current.frame, object(), 0, (9, 8, 0, 0, 0, 0), (True, True, False, False, False, False)
        )

    monkeypatch.setattr(engine, "clone_world_exact", lambda world: world)
    monkeypatch.setattr(engine, "_capture_and_step", capture)

    def evaluate(window, horizons, **kwargs):
        assert [item.frame for item in window] == list(range(1, 9))
        progress[:] = [4, int(invalidate)]  # Four real ticks passed during the native window.
        return [
            {
                0: SearchOutcome(0, False, 1, 1, horizon),
                1: SearchOutcome(1, True, horizon, 100, horizon),
            }
            for horizon in horizons
        ]

    monkeypatch.setattr("research.t34_pathb.batch_v2.evaluate_decisions", evaluate)
    request = SearchRequest(
        7,
        "source",
        1,
        capture(SimpleNamespace(frame=0)),
        20,
        replay_root=root,
        cutoff_frame=0,
        doom_frame=9,
        latest_progress=lambda: tuple(progress),
        source_epoch=0,
    )
    result = PathBSearchEngine.run_search(request)
    if invalidate:
        assert result.status == "STALE"
    else:
        assert result.status == "IMPROVING"
        assert result.target_frame == 5
        assert result.action == 1
