"""Budget ablation must not consume the Phase 0 selection bank."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.phase0_search_20261019 import teacher
from research.t33_batched_teacher import ablation, reference


def test_calibration_allocation_is_audited_and_disjoint() -> None:
    rows = ablation.calibration_rows()
    bank = json.loads(ablation.DEFAULT_BANK.read_text())
    assert {r["mix"] for r in rows} == {"scripted", "frozen", "mixed"}
    assert not {r["world_seed"] for r in rows} & {r["world_seed"] for r in bank["selection"]}


@pytest.mark.parametrize("mutation", ["overlap", "audit", "digest"])
def test_rejects_bank_tampering(tmp_path: Path, mutation: str) -> None:
    value = json.loads(ablation.DEFAULT_BANK.read_text())
    audit = json.loads(ablation.DEFAULT_BANK.with_name("SEED_AUDIT.json").read_text())
    if mutation == "overlap":
        value["fixtures"]["calibration"][0]["world_seed"] = value["selection"][0]["world_seed"]
    elif mutation == "audit":
        audit["pass"] = False
    else:
        value["allocation_semantic_sha256"] = "0" * 64
    path = tmp_path / "WORLD_BANK.json"
    path.write_text(json.dumps(value))
    (tmp_path / "SEED_AUDIT.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        ablation.calibration_rows(path)


def test_reference_independent_rng_matches_phase0_and_crn_repeats() -> None:
    if not ablation.B8.exists():
        pytest.skip("local B8 inventory absent")
    teacher.configure_cpu()
    row = ablation.calibration_rows(smoke=True)[1]
    world = ablation.fixture(row, ablation.B8, 800)
    captured = {}

    class Read(Exception):
        pass

    def select(prepared):
        world.runtime.prepare(prepared)
        config = teacher.SearchConfig(rollouts=2, horizon=4, epsilon=0.5, length_gate=0)
        actions = list(map(int, np.flatnonzero(prepared.get_legal_action_mask()[0, 0])))
        decision = int(prepared.frame[0]) - 1
        old = teacher._reference_scores(world, actions, config, decision)
        new = reference.scores(world, actions, config, decision, common_random=False)
        assert old == new
        first = reference.scores(world, actions, config, decision)
        assert first == reference.scores(world, actions, config, decision)
        captured["done"] = True
        raise Read

    with pytest.raises(Read):
        world.sim.step_with_policy(select)
    assert captured["done"]


def test_summary_regret_and_agreement_use_reference_scores() -> None:
    rows = [
        {
            "variants": [
                {
                    "id": "one",
                    "searched": True,
                    "agreement": False,
                    "regret": 3.0,
                    "legacy_agreement": True,
                    "legacy_regret": 0.0,
                    "seconds": 1.0,
                }
            ]
        }
    ]
    rows.append(copy.deepcopy(rows[0]))
    rows[1]["variants"][0].update(agreement=True, regret=0.0)
    summary = ablation.summarize(rows)["one"]
    assert summary["agreement"] == 0.5
    assert summary["regret"] == 1.5
    assert summary["max_regret"] == 3.0
    assert summary["legacy_agreement"] == 1.0


def test_value_leaf_uses_dueling_mean_instead_of_masked_max(monkeypatch) -> None:
    from types import SimpleNamespace

    sim = SimpleNamespace(
        E=1,
        S=1,
        alive=np.array([[True]]),
        length=np.array([[10]]),
        get_resolved_action_mask=lambda: np.ones((1, 1, 6), dtype=bool),
        step=lambda grid: None,
    )
    sim.step_with_policy = lambda selector, **kwargs: selector(sim)
    runtime = SimpleNamespace(prepare=lambda s: None, observe_step=lambda s: None)
    replica = SimpleNamespace(sim=sim, runtime=runtime, heroes=[None], opponents=[[]])
    monkeypatch.setattr(reference, "_replicate", lambda *args: replica)
    monkeypatch.setattr(
        reference,
        "_q_and_legal",
        lambda *args: (
            np.arange(6, dtype=np.float32),
            np.ones(6, dtype=bool),
            np.ones(6, dtype=bool),
        ),
    )
    from src.simd_env import eval_engine

    monkeypatch.setattr(eval_engine, "_dispatch_actions", lambda *args: None)
    world = SimpleNamespace(seed=1, mix="scripted", kinds=[])
    config = teacher.SearchConfig(horizon=1, rollouts=1)
    assert reference.scores(world, [0], config, 0, leaf="max") == {0: 15.0}
    assert reference.scores(world, [0], config, 0, leaf="value") == {0: 12.5}
