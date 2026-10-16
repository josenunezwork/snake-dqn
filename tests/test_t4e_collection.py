"""T4e collection selection and split contracts."""

import numpy as np
import pytest

from research.t4e_return_20261016 import collection


def row(index, *, length=800, q=None, holdout=False):
    return {
        "plan_index": index,
        "source_decision": index * 5,
        "length": length,
        "q": np.asarray(q if q is not None else [1, 2, 3, 4, 5, 6], np.float32),
        "state_packed": np.asarray([index], np.uint8),
        "state_scalars": np.asarray([index], np.float32),
        "fingerprint": f"fingerprint-{index}",
        "holdout": holdout,
    }


def test_genuine_decisiveness_requires_b8_greedy_death_and_surviving_alternative():
    q, legal = np.array([1, 9, 3, 0, 0, 0]), np.array([0, 1, 2])
    assert collection.masked_greedy(q, legal) == 1
    assert collection.genuinely_decisive(q, legal, {0: False, 1: True, 2: True})
    assert not collection.genuinely_decisive(q, legal, {0: True, 1: False, 2: False})
    assert not collection.genuinely_decisive(q, legal, {0: True, 1: True, 2: True})


def test_outside_selection_uses_new_decisive_quota_and_never_cross_substitutes(monkeypatch):
    monkeypatch.setattr(
        collection, "fixed_stratum", lambda length, q, cuts: "A" if length < 900 else "B"
    )
    decisive = [row(1, length=800), row(2, length=1000)]
    candidates = [row(8, length=800), row(9, length=1000), row(10, length=800)]
    selected, receipt = collection.select_outside(decisive, candidates, np.zeros(9), set())
    assert [value["plan_index"] for value in selected] == [8, 9]
    assert receipt["quota_by_stratum"] == {"A": 1, "B": 1}
    with pytest.raises(RuntimeError, match="INCOMPLETE_MATCHING"):
        collection.select_outside(decisive, [row(8, length=800)], np.zeros(9), set())


def test_overlap_and_holdout_candidates_cannot_enter_training_selection(monkeypatch):
    monkeypatch.setattr(collection, "fixed_stratum", lambda *_: "A")
    decisive = [row(1)]
    held = row(2, holdout=True)
    overlapping = row(3)
    with pytest.raises(RuntimeError, match="INCOMPLETE_MATCHING"):
        collection.select_outside(
            decisive, [held, overlapping], np.zeros(9), {collection._fingerprint(overlapping)}
        )


def test_holdout_is_world_disjoint_and_hash_deterministic():
    plans = [
        {"arm": "B8_s0", "mix": "scripted", "seed": index, "world_index": index}
        for index in range(200)
    ]
    flags = collection.assign_world_holdout(plans)
    assert sum(flags) == 20
    assert flags == collection.assign_world_holdout(plans)


def test_dedupe_reserves_a_holdout_identity_over_matching_training_state():
    train = row(1)
    held = row(2, holdout=True)
    held["fingerprint"] = train["fingerprint"]
    assert collection.dedupe_decisive([train, held]) == [held]


def test_phase_b_binding_is_plan_identity_ordered_and_rejects_holdout_sources():
    jobs = [
        {"plan_index": 2, "holdout": False, "seed": 12},
        {"plan_index": 1, "holdout": False, "seed": 11},
    ]
    selected = [row(2), row(1)]
    selected[0]["plan_index"] = 2
    selected[1]["plan_index"] = 1
    assert [job["plan_index"] for job, _ in collection.phase_b_jobs(jobs, selected)] == [1, 2]
    jobs[0]["holdout"] = True
    with pytest.raises(RuntimeError, match="training source"):
        collection.phase_b_jobs(jobs, selected)


def test_phase_b_replays_one_world_prefix_for_two_selected_states(monkeypatch):
    from types import SimpleNamespace

    from research.t3_rollout_labels_20261015 import bank_engine
    from research.t4d_return_regression_20261016 import collector as t4d_collector

    world = SimpleNamespace(
        sim=SimpleNamespace(alive=np.array([[True]])),
        runtime=SimpleNamespace(prepare=lambda _: None),
    )
    built, stepped = [], []
    monkeypatch.setattr(
        bank_engine, "build_world", lambda *args, **kwargs: built.append(1) or world
    )
    monkeypatch.setattr(
        bank_engine,
        "choose_and_step",
        lambda current, **_kwargs: (
            current.runtime.prepare(current.sim),
            stepped.append(1),
            np.array([[1]]),
            None,
            None,
        )[2:],
    )

    def snapshot(_world, decision):
        return {"decision": decision, "q": np.array([0, 1, 0, 0, 0, 0]), "legal": np.array([0, 1])}

    monkeypatch.setattr(t4d_collector, "_snapshot", snapshot)
    monkeypatch.setattr(t4d_collector, "rollout_branch", lambda *_args: {"T": 300, "died": False})
    monkeypatch.setattr(
        collection,
        "_candidate",
        lambda state, _job, _cuts: {
            "fingerprint": f"f-{state['decision']}",
            "state_packed": np.array([state["decision"]]),
            "state_scalars": np.array([state["decision"]], np.float32),
            "legal": state["legal"],
            "q": state["q"],
        },
    )
    job = {"arm": "B8_s0", "student": "/tmp/student", "plan_index": 0}
    candidates = [
        {
            "source_decision": 1,
            "fingerprint": "f-1",
            "state_packed": [1],
            "state_scalars": [1],
            "legal": [0, 1],
            "q": [0, 1, 0, 0, 0, 0],
        },
        {
            "source_decision": 3,
            "fingerprint": "f-3",
            "state_packed": [3],
            "state_scalars": [3],
            "legal": [0, 1],
            "q": [0, 1, 0, 0, 0, 0],
        },
    ]
    rows = collection.replay_selected_outside_world(job, candidates, np.zeros(9))
    assert len(built) == 1
    assert len(stepped) == 4
    assert [row["fingerprint"] for row in rows] == ["f-1", "f-3"]


def test_completion_requires_8000_equal_training_states():
    decisive = [row(i) for i in range(8000)]
    outside = [row(i + 9000) for i in range(8000)]
    result = collection.completion_receipt(decisive, outside, 12.5)
    assert result["training_decisive_states"] == 8000
    with pytest.raises(RuntimeError, match="equal"):
        collection.completion_receipt(decisive, outside[:-1], 1)
