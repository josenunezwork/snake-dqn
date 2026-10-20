"""V2 metadata survives the original immutable world/replay envelope."""

from pathlib import Path

import pytest

from research.t34_exit_b8.exports import load_world, write_world
from tests.test_t34_exit_replay import _checkpoint, _result


def test_v2_label_roles_and_lad_provenance_roundtrip(tmp_path: Path) -> None:
    result = _result(_checkpoint(tmp_path))
    result["teacher_v2"] = {"rule": "teacher-bisection-refine3", "shadow_depth": 300}
    result["labels"][0].update(
        label_role="decisive",
        shadow_id="seed17/fork0",
        shadow_confirmed=True,
        t_star=0,
        a_star=2,
        F=[0, 3],
        q=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        lad={"doom": 20, "global_last_avoidable_proven": False},
        searched=True,
        decisive=True,
        kl_anchor_only=False,
        ranked_actions=[2, 0, 3],
        outcome_kind="horizon_rescue",
        delay_only=False,
        label_weight=1.0,
        horizon=300,
    )
    path = tmp_path / "world.json"
    write_world(path, result)
    loaded = load_world(path)
    assert loaded["teacher_v2"] == result["teacher_v2"]
    for key in (
        "label_role",
        "shadow_id",
        "shadow_confirmed",
        "t_star",
        "a_star",
        "F",
        "q",
        "lad",
        "searched",
        "decisive",
        "kl_anchor_only",
        "ranked_actions",
        "outcome_kind",
        "delay_only",
        "label_weight",
        "horizon",
    ):
        assert loaded["labels"][0][key] == result["labels"][0][key]


def test_v2_metadata_cannot_leak_host_paths(tmp_path: Path) -> None:
    result = _result(_checkpoint(tmp_path))
    result["teacher_v2"] = {"source": "/Users/private/checkpoint"}
    with pytest.raises(ValueError, match="filesystem paths"):
        write_world(tmp_path / "world.json", result)


def test_teacher_roles_adapt_to_integrated_fitcache_boundary(tmp_path: Path) -> None:
    """The fit adapter sees each transition at its own decisive decision."""
    from types import SimpleNamespace

    import numpy as np

    from research.t34_exit_b8.teacher_v2 import _label
    from research.t34_exit_b8_fit.interface import normalize_row

    result = _result(_checkpoint(tmp_path))
    evaluation = SimpleNamespace(q=np.arange(6, dtype=np.float32), legal=np.ones(6, dtype=bool))
    rows = [
        _label(
            decision=0,
            evaluation=evaluation,
            student_action=1,
            role="shadow_confirm",
            shadow_id=0,
            shadow_frame=0,
            shadow_confirmed=True,
            kl_anchor_only=True,
        ),
        _label(
            decision=1,
            evaluation=evaluation,
            student_action=1,
            role="lad_transition",
            shadow_id=0,
            shadow_frame=1,
            shadow_confirmed=True,
            searched=True,
            decisive=True,
            t_star=1,
            a_star=2,
            forbidden=[1],
            per_action={"1": {"survives": False}, "2": {"survives": True}},
        ),
    ]
    result["labels"] = rows
    path = tmp_path / "world.json"
    write_world(path, result)
    labels = load_world(path)["labels"]
    adapted = [normalize_row(dict(label, q_prev=np.asarray(label["q"]))) for label in labels]
    assert adapted[0]["kl_anchor_only"] and not adapted[0]["searched"]
    assert not adapted[0]["forbidden_mask"].any()
    assert adapted[1]["decisive"] and adapted[1]["searched"]
    assert adapted[1]["forbidden_mask"].tolist() == [False, True, False, False, False, False]


def test_ranked_delay_roundtrips_with_zero_fit_loss(tmp_path: Path) -> None:
    """All-die diagnostics survive export/cache admission but teach no action."""
    import numpy as np
    import torch

    from research.t34_exit_b8_fit.data import _stack, sample_weights, to_device
    from research.t34_exit_b8_fit.interface import normalize_row
    from research.t34_exit_b8_fit.loss import exit_fit_loss
    from tests.test_t34_fitcache_recipe import _v2

    raw = _v2("transition")
    raw.update(
        student_action=1,
        outcome_kind="all_die_delay",
        delay_only=True,
        label_weight=0.0,
        decisive=False,
        per_action={
            str(a): {
                "survives": False,
                "alive_steps": 2 + a,
                "mass": 10.0,
                "horizon": 300,
                "censored": False,
                "tested": True,
            }
            for a in range(6)
        },
        ranked_actions=[5, 4, 3, 2, 1, 0],
        a_star=5,
        action=5,
        executed_action=1,
        outcomes={},
    )
    raw["outcomes"] = raw["per_action"]
    result = _result(_checkpoint(tmp_path))
    result["labels"] = [raw]
    path = tmp_path / "world.json"
    write_world(path, result)
    exported = load_world(path)["labels"][0]
    assert exported["ranked_actions"] == [5, 4, 3, 2, 1, 0]
    assert exported["outcomes"][5]["horizon"] == 300
    assert exported["outcomes"][5]["censored"] is False
    row = normalize_row(dict(raw, **exported))
    assert not row["decisive"] and row["delay_only"]
    assert not row["forbidden_mask"].any()
    assert sample_weights([row]).tolist() == [0.0]
    batch = to_device(_stack([dict(row, _sample_weight=0.0)]), torch.device("cpu"))
    q = torch.zeros((1, 6), requires_grad=True)
    loss, _ = exit_fit_loss(q, batch)
    loss.backward()
    assert loss.item() == 0.0
    assert np.count_nonzero(q.grad.numpy()) == 0

    rescue = dict(
        row,
        outcome_kind="horizon_rescue",
        decisive=True,
        delay_only=False,
        label_weight=1.0,
    )
    assert sample_weights([row, rescue]).tolist() == [0.0, 10.0]


def test_ranked_outcome_adapter_rejects_false_rescue() -> None:
    from research.t34_exit_b8_fit.interface import normalize_row
    from tests.test_t34_fitcache_recipe import _v2

    raw = dict(
        _v2("lad"),
        student_action=0,
        outcome_kind="horizon_rescue",
        per_action={
            "0": {"survives": True, "tested": True},
            "2": {"survives": True, "tested": True},
        },
    )
    with pytest.raises(ValueError, match="dying bare"):
        normalize_row(raw)


def test_ranked_outcome_adapter_ignores_untested_placeholders() -> None:
    from research.t34_exit_b8_fit.interface import normalize_row
    from tests.test_t34_fitcache_recipe import _v2

    raw = dict(
        _v2("lad"),
        student_action=0,
        outcome_kind="horizon_rescue",
        per_action={
            "0": {"survives": False, "tested": True},
            "1": {"survives": False, "tested": False},
            "2": {"survives": True, "tested": True},
        },
    )
    assert normalize_row(raw)["forbidden_mask"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    raw["per_action"]["2"]["tested"] = False
    with pytest.raises(ValueError, match="must be tested"):
        normalize_row(raw)
    del raw["label_kind"]
    with pytest.raises(ValueError, match="require label_kind"):
        normalize_row(raw)
