"""Focused collection-format and return-statistic tests for T4d."""

import numpy as np

from research.t4d_return_regression_20261016 import collector


def test_hand_computed_dead_branch_statistics_and_target():
    # L0=100; transition lengths 110, 120, then death.  S excludes the terminal zero.
    got = collector.branch_summary(100, [(110, True), (120, True), (0, False)])
    expected_s = 11.0 + collector.GAMMA * 12.0
    assert got == {"T": 3, "died": True, "S": expected_s, "L0": 100, "L_T": 0}
    expected_y = (1.0 - collector.GAMMA) * expected_s - 10.0 - 3.0 * collector.GAMMA**2
    assert collector.realised_target(got) == expected_y


def test_hand_computed_survivor_branch_statistics_and_target():
    got = collector.branch_summary(100, [(110, True), (120, True)])
    expected_s = 11.0 + collector.GAMMA * 12.0
    assert got == {"T": 2, "died": False, "S": expected_s, "L0": 100, "L_T": 120}
    expected_y = (1.0 - collector.GAMMA) * expected_s - 10.0 + collector.GAMMA**2 * 12.0
    assert collector.realised_target(got) == expected_y


def test_write_bank_refuses_empty_rows(tmp_path):
    try:
        collector.write_bank(tmp_path / "return_bank.npz", [], {"test": True})
    except ValueError as error:
        assert "real branches" in str(error)
    else:  # pragma: no cover
        raise AssertionError("empty banks would be synthetic evidence")


def test_cli_accepts_future_mps_launch_flag_without_starting_collection(tmp_path, monkeypatch):
    called = {}

    def fake_collect(*args, **kwargs):
        called["args"] = args
        return {"status": "PLANNED"}

    monkeypatch.setattr(collector, "collect", fake_collect)
    monkeypatch.setattr(collector, "_configure_device", lambda _: None)
    collector.main(
        [
            "--student",
            str(tmp_path / "student.pth"),
            "--out",
            str(tmp_path / "out"),
            "--round",
            "future",
            "--device",
            "mps",
            "--record-returns",
        ]
    )
    assert called["args"][5] == collector.DEPTH


def test_outside_distance_is_fixed_300_even_with_short_rollout():
    assert not collector.outside_eligible(10, 309)
    assert collector.outside_eligible(10, 310)


def test_global_matcher_deduplicates_and_caps_across_worlds(monkeypatch, tmp_path):
    from research.t4a_posthoc_20261016 import spillover
    from research.t4c_20261016 import data

    monkeypatch.setattr(data, "bound_files", lambda _: {"source": "hash"})
    monkeypatch.setattr(data, "target_definition", lambda _: (np.zeros(9), {"A": 2}, {"bank"}))
    monkeypatch.setattr(data, "strata", lambda *args: np.array(["A"]))
    monkeypatch.setattr(
        spillover,
        "observation_fingerprints",
        lambda local, glob, scalars: np.array([str(scalars[0, 0])]),
    )
    matcher = collector.OutsideMatcher(tmp_path)

    def row(value):
        return {
            "length": 700,
            "q": np.zeros(6),
            "obs": {"local": None, "global": None, "scalars": np.array([[value]])},
        }

    assert matcher.admit(row(1))
    assert not matcher.admit(row(1))
    assert matcher.admit(row(2))
    assert not matcher.admit(row(3))
    assert matcher.receipt()["deficits"] == {}
    assert matcher.receipt()["selected_counts"] == {"A": 2}


def test_streaming_world_bounds_clones_and_requires_outside_opt_in(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from research.t3_rollout_labels_20261015 import bank_engine

    monkeypatch.setattr(bank_engine, "HORIZON", 340)

    def build(*args, **kwargs):
        return SimpleNamespace(
            sim=SimpleNamespace(alive=np.array([[True]])),
            hero=SimpleNamespace(net=SimpleNamespace(to=lambda _: None)),
            runtime=SimpleNamespace(prepare=lambda _: None),
        )

    def step(world, **kwargs):
        world.runtime.prepare(world.sim)

    monkeypatch.setattr(bank_engine, "build_world", build)
    monkeypatch.setattr(bank_engine, "choose_and_step", step)
    monkeypatch.setattr(
        collector,
        "_snapshot",
        lambda world, d: {"decision": d, "legal": [0, 1, 2], "obs": {}, "length": 700},
    )
    rolled = []

    def branch(snapshot, action, depth):
        rolled.append((snapshot["decision"], action, depth))
        return {"T": depth, "died": False}

    monkeypatch.setattr(collector, "rollout_branch", branch)
    plan = {"seed": 1, "mix": "scripted"}
    rows, receipt = collector.collect_world(plan, tmp_path / "student")
    assert rows == [] and rolled == []
    assert receipt["peak_pending_clones"] <= 61
    matcher = SimpleNamespace(admit=lambda _: True)
    rows, receipt = collector.collect_world(
        plan, tmp_path / "student", 2, matcher, max_states=2, state_budget=[0]
    )
    assert len(rows) == 6
    assert {r[0] for r in rolled} == {0, 5}
    assert {r[1] for r in rolled} == {0, 1, 2}
    assert all(r["row_weight"] == 0.5 for r in rows)
    assert receipt["peak_pending_clones"] <= 61
    # A shared global budget prevents a second world from restarting the cap.
    rows, _ = collector.collect_world(
        plan, tmp_path / "student", 2, matcher, max_states=2, state_budget=[2]
    )
    assert rows == []


def test_freshness_scan_is_literal_whole_word_and_ignores_binary(tmp_path, monkeypatch):
    from research.t4d_return_regression_20261016.planning import concrete_hits

    (tmp_path / "values.txt").write_text("123456 1234567 a+b\n")
    (tmp_path / "binary.bin").write_bytes(b"\0 555555\0")
    config = tmp_path / "rg-config"
    config.write_text("--glob=!values.txt\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    assert concrete_hits(tmp_path, ["123456", "23456", "555555"], words=True) == {"123456"}
    assert concrete_hits(tmp_path, ["a+b"], words=False) == {"a+b"}


def test_smoke_fixture_escape_hatch_requires_bounded_partial_mode(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="partial smoke"):
        collector.collect(
            tmp_path / "unused",
            tmp_path / "out",
            "fixture",
            2,
            ["scripted"],
            300,
            None,
            smoke_seeds=[1, 2],
        )
