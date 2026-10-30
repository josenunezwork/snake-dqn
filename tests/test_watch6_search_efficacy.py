"""Efficacy decision boundaries, product config, and outcome-blind resume fences."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.watch6_search_efficacy import analyze
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy import resume, run
from tools.ops.package_spec import empty_ledger, publish_json


def rows(death_delta: int = -2, relative: float = 0) -> list[dict]:
    result = []
    for index, allocation in enumerate(p.allocation()):
        bare = {
            "frames": p.HORIZON,
            "mirror_identity_frames": p.HORIZON,
            "mass_sum": (500 + index * 20) * p.HORIZON,
            "deaths": 4 + index % 2,
        }
        dense = {
            **bare,
            "mass_sum": bare["mass_sum"] * (1 + relative),
            "deaths": bare["deaths"] + death_delta + (index % 2 if death_delta else 0),
        }
        result.append(
            {
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "allocation": allocation,
                "fixture": False,
                "unit_id": run.unit_id(p.NAMESPACE, "watch6", index),
                "arms": {"bare": bare, "exact_stop": dense},
            }
        )
    return result


def test_co_primary_success_and_mass_failure() -> None:
    assert analyze.decide(rows())["verdict"] == "GO_EFFICACY_ONLY"
    assert analyze.decide(rows(relative=-0.1))["verdict"] == "FAIL_EFFICACY"
    assert analyze.decide(rows(death_delta=0))["verdict"] == "INCONCLUSIVE"
    assert analyze.decide(rows(death_delta=1))["verdict"] == "FAIL_EFFICACY"


def test_mass_boundary_and_ratio_of_means() -> None:
    decision = analyze.decide(rows(relative=-0.04))
    assert decision["mass_pass"]
    assert decision["mass_relative_difference"] == pytest.approx(-0.04)
    assert decision["mass_relative_ci95_fieller"][0] == pytest.approx(-0.04)


def test_unbounded_fieller_cannot_pass() -> None:
    values = rows()
    for row in values:
        row["arms"]["bare"]["mass_sum"] = 0
    assert not analyze.decide(values)["mass_pass"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "fixture", "foreign", "short"])
def test_incomplete_or_foreign_never_gets_a_look(mutation: str) -> None:
    values = rows()
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = copy.deepcopy(values[0])
    elif mutation == "fixture":
        values[0]["fixture"] = True
    elif mutation == "foreign":
        values[0]["allocation"]["seed"] += 1
    else:
        values[0]["arms"]["exact_stop"]["frames"] -= 1
    with pytest.raises(ValueError):
        analyze.decide(values)


def test_ledger_exact_skip_and_foreign_refusal(tmp_path: Path) -> None:
    ledger = empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity())
    publish_json(tmp_path / "empty.json", ledger)
    assert run.completed(tmp_path / "empty.json") == []
    ledger["units"] = [
        {
            "index": 3,
            "unit_id": run.unit_id(p.NAMESPACE, "watch6", 3),
            "sha256": "a" * 64,
            "size": 12,
        }
    ]
    publish_json(tmp_path / "one.json", ledger)
    assert [r["index"] for r in run.completed(tmp_path / "one.json")] == [3]
    ledger["units"] *= 2
    publish_json(tmp_path / "duplicate.json", ledger)
    with pytest.raises(ValueError):
        run.completed(tmp_path / "duplicate.json")
    ledger["namespace"] = "foreign"
    publish_json(tmp_path / "foreign.json", ledger)
    with pytest.raises(ValueError):
        run.completed(tmp_path / "foreign.json")


def test_outcome_blind_metadata_builder_refuses_foreign(tmp_path: Path) -> None:
    payload = tmp_path / "exports" / "efficacy" / "pair-00.json"
    descriptor = publish_json(payload, {"sealed_science": "must not be parsed"})
    publish_json(
        payload.with_suffix(".meta.json"),
        {
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "allocation": p.allocation()[0],
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
        },
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"stop_verified": True, "cleanup_error": None, "output_dir": str(tmp_path)})
    )
    ledger = resume.build([payload], [receipt])
    assert ledger["units"][0]["index"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        resume.build([payload, payload], [receipt])
    receipt.write_text(json.dumps({"stop_verified": False}))
    with pytest.raises(ValueError, match="stop"):
        resume.build([payload], [receipt])


def test_allocation_and_fixture_isolation() -> None:
    assert len({r["seed"] for r in p.allocation()}) == p.COUNT
    assert not {r["seed"] for r in p.allocation()} & set(p.FIXTURE_SEEDS)
    assert not {r["seed"] for r in p.allocation()} & {
        r["seed"]
        for r in __import__(
            "research.bl_served_search.c1_bank", fromlist=["allocation"]
        ).allocation()
    }


def runtime_inputs(tmp_path: Path) -> dict:
    from research.watch6_search_efficacy.mirror import execution_identity
    from tools.ops.package_spec import digest

    publish_json(
        tmp_path / "verified-ledger.json",
        empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()),
    )
    publish_json(
        tmp_path / "runtime-contract.json", {"science": p.identity(), "allocation": p.allocation()}
    )
    # Synthetic integrity fixtures only. These never leave pytest's tmp_path.
    publish_json(
        tmp_path / "mac-mirror.json",
        {
            "status": "PASS",
            "platform": "Darwin",
            "execution_identity": execution_identity(),
            "seed": p.FIXTURE_SEEDS[0],
            "horizon": p.HORIZON,
            "mirror_identity_frames": p.HORIZON,
            "mirror_mismatches": 0,
            "respawns_per_slot": [2, 1, 0, 0, 0, 0],
        },
    )
    publish_json(
        tmp_path / "judge-approval.json",
        {
            "status": "RATIFIED",
            "judge": "Astra",
            "science": p.identity(),
            "registration_sha256": digest(p.registration()),
            "ruling_sha256": digest(p.registration().with_name("RULING.md")),
        },
    )
    return {
        "WATCH6_SCIENCE": p.identity(),
        "WATCH6_SHARD": "0",
        "WATCH6_FIXTURE": "1",
        "WATCH6_WORKERS": "4",
        "WATCH6_SHARDS": "6",
        "WATCH6_BASE_MINUTES": "15",
    }


def test_execute_time_admission_pending_and_foreign(tmp_path: Path) -> None:
    from research.watch6_search_efficacy.admission import verify_runtime

    env = runtime_inputs(tmp_path)
    verify_runtime(tmp_path, env)  # recorded Astra ruling accepted
    with pytest.raises(ValueError, match="measured"):
        verify_runtime(tmp_path, {**env, "WATCH6_FIXTURE": "0"})
    with pytest.raises(ValueError, match="shard"):
        verify_runtime(tmp_path, {**env, "WATCH6_SHARD": "6"})
    approval_path = tmp_path / "judge-approval.json"
    approval = json.loads(approval_path.read_text())
    approval_path.write_text(json.dumps({**approval, "status": "PENDING"}))
    with pytest.raises(ValueError, match="ratification"):
        verify_runtime(tmp_path, env)
    approval_path.write_text(json.dumps(approval))
    contract = tmp_path / "runtime-contract.json"
    contract.write_text(json.dumps({"science": "foreign"}))
    with pytest.raises(ValueError, match="source"):
        verify_runtime(tmp_path, env)


def test_partial_payload_is_never_a_verified_unit(tmp_path: Path) -> None:
    from tools.ops.package_spec import verify_immutable

    payload = tmp_path / "pair-00.json"
    payload.write_text('{"partial":true}')
    with pytest.raises(ValueError):
        verify_immutable(payload)


def test_real_fixture_serial_equals_parallel(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    saved = p.ROOT.parent / "snake-dqn/saved_snakes"
    if not (saved / "b8_s3_t7_20261016.pth").is_file():
        pytest.skip("local pinned checkpoints required for real integration fixture")
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", run.ledger_identity()))
    env = {
        **os.environ,
        "SNAKE_CKPT_DIR": str(saved),
        "SNAKE_DQN_DEVICE": "cpu",
        **{
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    for workers in (1, 2):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.watch6_search_efficacy.run",
                "--ledger",
                str(ledger),
                "--out",
                str(tmp_path / f"w{workers}"),
                "--fixture",
                "--fixture-horizon",
                "3",
                "--workers",
                str(workers),
            ],
            env=env,
            check=True,
            capture_output=True,
            timeout=120,
        )
    for workers in (2,):
        for index in range(3):
            assert (tmp_path / f"w1/pair-{index:02d}.json").read_bytes() == (
                tmp_path / f"w{workers}/pair-{index:02d}.json"
            ).read_bytes()


def performance_report(seconds: float = 0.6) -> dict:
    """Explicitly synthetic cost-only unit fixture, never dispatch evidence."""
    profiles = []
    for workers in p.WORKERS:
        rows_ = [
            {
                "index": i,
                "arms": {
                    arm: {"main_seconds_per_frame": 0.02, "mirror_identity_frames": 300}
                    for arm in ("bare", "exact_stop")
                },
                "branches": [
                    {
                        "seed": seed,
                        "seconds": seconds,
                        "branch_seconds_upper": seconds,
                        "frames_executed": [300],
                        "native_v8": {
                            "kernel_rows": 1,
                            "stock_rows": 0,
                            "fallback_rows": 0,
                            "noncandidate_rows": 0,
                        },
                    }
                    for seed in p.FIXTURE_SEEDS
                ],
            }
            for i in range(workers)
        ]
        profiles.append(
            {
                "workers": workers,
                "branch_seconds_upper": seconds,
                "main_seconds_per_frame": {"bare": 0.02, "exact_stop": 0.02},
                "rows": rows_,
            }
        )
    return {
        "schema": "watch6-target-performance/v1",
        "science": p.identity(),
        "fixture": True,
        "horizon": 300,
        "cpu_count": 12,
        "profiles": profiles,
    }


def test_measured_forecast_uses_all_costs_margin_and_w8() -> None:
    import math

    from research.watch6_search_efficacy.performance import forecast

    selected = forecast(performance_report())
    assert selected["workers"] == 8
    assert selected["shards"] == 6
    assert selected["pairs_per_shard"] == 16
    pair_seconds = 14400 * 0.04 + 2880 * 6 * 0.6
    assert selected["pair_seconds_upper_forecast"] == pytest.approx(pair_seconds)
    assert selected["base_minutes_per_shard"] == math.ceil(20 + 1.25 * 2 * pair_seconds / 60)
    assert selected["effective_minutes"] == 2 * selected["base_minutes_per_shard"]
    assert selected["effective_minutes"] <= 1320


def test_w6_can_win_on_measured_contention() -> None:
    from research.watch6_search_efficacy.performance import forecast

    report = performance_report(0.5)
    for profile in report["profiles"]:
        if profile["workers"] == 8:
            profile["branch_seconds_upper"] *= 2
            for row in profile["rows"]:
                for batch in row["branches"]:
                    batch["seconds"] *= 2
                    batch["branch_seconds_upper"] *= 2
    assert forecast(report)["workers"] == 6


@pytest.mark.parametrize("mutation", ["slow", "stale", "cores", "missing", "nan", "loop", "short"])
def test_forecast_fails_closed(mutation: str) -> None:
    from research.watch6_search_efficacy.performance import forecast

    report = performance_report(100 if mutation == "slow" else 0.6)
    if mutation == "stale":
        report["science"] = "foreign"
    elif mutation == "cores":
        report["cpu_count"] = 8
    elif mutation == "missing":
        report["profiles"].pop()
    elif mutation == "nan":
        report["profiles"][0]["branch_seconds_upper"] = float("nan")
    elif mutation == "loop":
        report["profiles"][0]["main_seconds_per_frame"]["bare"] = 0
    elif mutation == "short":
        report["profiles"][0]["rows"][0]["branches"][0]["frames_executed"] = [12]
    with pytest.raises(ValueError):
        forecast(report)


def test_main_preparation_refuses_without_measured_canary(tmp_path: Path) -> None:
    from research.watch6_search_efficacy.prepare import prepare

    with pytest.raises(ValueError, match="measured.*canary"):
        prepare(tmp_path / "never-created", tmp_path / "missing.json", "a" * 40, main_shards=True)
    assert not (tmp_path / "never-created").exists()


def test_execute_time_rederives_forecast_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.watch6_search_efficacy.admission import runtime_admit, verify_runtime
    from research.watch6_search_efficacy.performance import forecast

    env = runtime_inputs(tmp_path)
    report = performance_report()
    selected = forecast(report)
    env.update(
        WATCH6_FIXTURE="0",
        WATCH6_MEASUREMENTS=json.dumps(report),
        WATCH6_WORKERS=str(selected["workers"]),
        WATCH6_SHARDS=str(selected["shards"]),
        WATCH6_BASE_MINUTES=str(selected["base_minutes_per_shard"]),
    )
    verify_runtime(tmp_path, env)
    with pytest.raises(ValueError, match="sizing"):
        verify_runtime(tmp_path, {**env, "WATCH6_WORKERS": "4"})
    monkeypatch.setenv("COLAB_INPUT_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    runtime_admit(workers=8, shards=6, shard=0)
    with pytest.raises(ValueError, match="CLI"):
        runtime_admit(workers=4, shards=6, shard=0)


def test_inconclusive_is_replay_no_go_and_no_degenerate_superiority() -> None:
    decision = analyze.decide(rows(death_delta=0))
    assert decision["verdict"] == "INCONCLUSIVE"
    assert decision["replay_serving_decision"] == "NO_GO_FOR_REPLAY_SERVING_UNDER_THIS_REGISTRATION"
    values = rows()
    for row in values:
        row["arms"]["exact_stop"]["deaths"] = row["arms"]["bare"]["deaths"] - 1
    assert analyze.decide(values)["verdict"] == "INCONCLUSIVE"


def test_completion_preserves_frozen_layout_and_refuses_larger_wall() -> None:
    from research.watch6_search_efficacy.performance import forecast

    original = forecast(performance_report())
    frozen = {key: original[key] for key in ("workers", "shards", "base_minutes_per_shard")}
    faster = forecast(performance_report(0.5), frozen)
    assert faster["base_minutes_per_shard"] == original["base_minutes_per_shard"]
    assert faster["workers"] == original["workers"]
    assert faster["shards"] == original["shards"]
    with pytest.raises(ValueError, match="frozen shard wall"):
        forecast(performance_report(0.7), frozen)


@pytest.mark.parametrize("fault", [False, True])
def test_pair_cancels_idle_worker_watchdog(monkeypatch: pytest.MonkeyPatch, fault: bool) -> None:
    import faulthandler

    from research.t1_commitment_20261015 import replay
    from research.watch6_search_efficacy import engine

    calls = []
    monkeypatch.setattr(replay, "configure_cpu", lambda: None)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **kw: calls.append("arm"))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    monkeypatch.setattr(run, "metrics", lambda raw: {})

    def arm(*args):
        if fault:
            raise ValueError("mechanical fixture error")
        return {}

    monkeypatch.setattr(engine, "run_arm", arm)
    if fault:
        with pytest.raises(ValueError, match="mechanical"):
            run.pair(p.allocation()[0], fixture=True)
    else:
        run.pair(p.allocation()[0], fixture=True)
    assert calls == ["arm", "cancel"]
