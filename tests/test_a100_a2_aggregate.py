"""A2 decision math and no-look guard, using tiny planted predictions only."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from research.a100_a2.aggregate import _bootstrap, analyze, kill_decisions
from research.a100_a2.protocol import NAMESPACE, empty_ledger


def test_world_bootstrap_and_paired_metric():
    world = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([[0, 1, 0, 1, 1, 0]] * 6)
    p = y * 0.7 + 0.15
    row = dict(
        world=world,
        died=y,
        legal=np.ones_like(y, bool),
        observed=np.ones_like(y, bool),
        p_d300=p,
        decisive=np.zeros(6, bool),
    )
    summaries, draws = _bootstrap({"a": row, "b": row}, 12)
    assert summaries["a"]["auc"] == roc_auc_score(y.flatten(), p.flatten()) == 1
    assert summaries["a"]["parent_worlds"] == 3
    np.testing.assert_array_equal(draws["a"]["auc"], draws["b"]["auc"])
    np.testing.assert_array_equal(draws["a"]["brier"], draws["b"]["brier"])


def test_incomplete_union_refuses_before_outcome_read(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(empty_ledger(NAMESPACE, "x")))
    with pytest.raises(ValueError, match="480"):
        analyze(ledger, "x", tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_frozen_kill_boundaries():
    assert kill_decisions(0.019, 0, 0.009, 0.099)["scale_closed"]
    assert not kill_decisions(0.02, 0, 0.009, 0.1)["scale_closed"]
    assert not kill_decisions(0.019, 0.001, 0.009, 0.1)["scale_closed"]
    assert not kill_decisions(0.019, 0, 0.01, 0.1)["scale_closed"]
    assert kill_decisions(0.019, 0, 0.009, 0.1)["target"] == "KEEP_HAZARDS"
    assert kill_decisions(0.019, 0, 0.009, 0.099)["target"] == "KEEP_SINGLE_HORIZON"


def test_complete_factorial_one_look_with_synthetic_units(tmp_path, monkeypatch):
    """480 unit schemas traverse the production reducer; no fitting or history."""
    from research.a100_a2 import aggregate as agg
    from research.a100_a2.job import synthetic_table
    from research.a100_a2.protocol import allocation
    from research.bl_ppo.exports import export_json

    assert agg.TABLE_COUNTS == {"t5": (19976, 342), "a5": (296, None)}
    monkeypatch.setattr(agg, "TABLE_COUNTS", {"t5": (10, 5), "a5": (10, 5)})
    monkeypatch.setattr(agg, "A5_DECISIVE_COUNTS", (10, 5))
    table = synthetic_table()
    # Every held-out world contains positives and negatives in this planted fixture.
    table["died"][:] = [False, True, False, True, False, True]
    table["alive_steps"][:] = [300, 29, 300, 29, 300, 29]
    p = np.where(table["died"], 0.8, 0.2).astype(np.float32)
    specs = allocation()
    units = {}
    rows = {}
    for u in specs:
        i = u["index"]
        path = tmp_path / f"fit-{i:04d}.npz"
        path.write_bytes(b"x")
        units[i] = dict(path=str(path), size=1, sha256="a" * 64)
        idx = np.flatnonzero(table["fold"] == u["fold"])
        row = {k: v[idx] for k, v in table.items() if k not in ("local", "global", "scalars")}
        row.update({k: np.asarray(u[k]) for k in ("table", "encoder", "target", "dose", "seed")})
        row.update(
            namespace=np.asarray(NAMESPACE),
            allocation=np.asarray("allocation"),
            index=np.asarray(i),
            row_index=idx,
            p_d300=p[idx],
        )
        if u["target"] == "hazard":
            row["p_hazards"] = p[idx, :, None] * np.array([0.1, 0.2, 1 / 3, 2 / 3, 1])
        rows[str(path)] = row
    export_json(tmp_path / "materialization.json", {"synthetic_fixture": True})
    monkeypatch.setattr(agg, "read_ledger", lambda *a: units)
    monkeypatch.setattr(agg, "verify_export", lambda *a: "a" * 64)
    monkeypatch.setattr(agg, "digest", lambda *a: "a" * 64)

    class Archive:
        def __init__(self, values):
            self.values = values
            self.files = list(values)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def __getitem__(self, k):
            return self.values[k]

    monkeypatch.setattr(agg.np, "load", lambda path, **kw: Archive(rows[str(path)]))
    result = agg.analyze(tmp_path / "ledger", "allocation", tmp_path / "result.json", n_boot=8)
    assert result["scale_decision"] == "CLOSED_WP4_ONLY_WP2_COUNTERFACTUAL_LABELS"
    assert result["target_decision"] == "KEEP_SINGLE_HORIZON"
    assert result["tables"]["t5"]["arms"]["s_random/d300/100"]["auc"] == 1
    # Uniformly shrunken units cannot turn into a scientific result.
    monkeypatch.setattr(agg, "TABLE_COUNTS", {"t5": (11, 5), "a5": (10, 5)})
    with pytest.raises(ValueError, match="coverage"):
        agg.analyze(tmp_path / "ledger", "allocation", tmp_path / "must-not-exist.json", n_boot=8)


def test_missing_teacher_action_never_becomes_zero_agreement():
    y = np.array([[0, 1, 0, 1, 0, 1]] * 3)
    row = dict(
        world=np.array(["a", "b", "c"]),
        died=y,
        legal=np.ones_like(y, bool),
        observed=np.ones_like(y, bool),
        p_d300=y * 0.6 + 0.2,
        decisive=np.ones(3, bool),
        a_star=np.full(3, -1),
        q=np.full((3, 6), np.nan),
    )
    summary, _ = _bootstrap({"t5": row}, 4)
    assert summary["t5"]["decisive_agreement"] is None
