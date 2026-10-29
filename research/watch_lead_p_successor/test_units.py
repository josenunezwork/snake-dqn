"""Hermetic custody/control tests; no closed data or scientific worlds."""

import multiprocessing as mp
import signal
import time
from pathlib import Path

import pytest

from research.watch_lead_p_successor import units


def bound(engineering: bool = True, count: int = 2) -> dict:
    return units.binding(
        units.allocation("p-watch-lead-successor-fixture-v1", count, engineering=engineering),
        {"fixture": "a" * 64},
    )


def row(b: dict, index: int) -> dict:
    host = {"cpu": "test-host", "software": "frozen-test-software"}
    return {
        "binding": b,
        "index": index,
        "unit_id": units.unit_id(b, index),
        "host": host,
        "arms": {
            arm: {
                "arm": arm,
                "seed": b["bank"]["seeds"][index],
                "frames": 6000,
                "host": host,
                "initial_state": "a" * 64,
                "final_state": "b" * 64,
                "trace_sha256": "c" * 64,
                "deaths": 10 if arm == "bare" else 8,
                "mass": 100,
            }
            for arm in units.ARMS
        },
    }


def unit(tmp: Path, b: dict, i: int) -> Path:
    path = tmp / "units" / f"session-{i:02d}.json"
    units.publish(path, row(b, i), b, i)
    return path


def test_empty_skip_and_foreign_binding(tmp_path) -> None:
    b = bound()
    assert units.read_ledger(units.empty_ledger(b), b) == {}
    p = unit(tmp_path, b, 0)
    ledger = units.make_ledger([p], b)
    assert list(units.read_ledger(ledger, b)) == [0]
    wrong = bound()
    wrong["bank"]["study"] += "foreign"
    with pytest.raises(ValueError):
        units.read_ledger(ledger, wrong)
    ledger["units"].append(ledger["units"][0])
    with pytest.raises(ValueError, match="duplicate"):
        units.read_ledger(ledger, b)


def test_make_ledger_never_opens_outcome(tmp_path, monkeypatch) -> None:
    b = bound()
    p = unit(tmp_path, b, 0)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path == p:
            raise AssertionError("OUTCOME OPENED")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    assert units.make_ledger([p], b)["units"][0]["index"] == 0


@pytest.mark.parametrize("change", ["payload", "marker", "sidecar", "metadata", "symlink"])
def test_corrupt_or_partial_units_refused(tmp_path, change) -> None:
    b = bound()
    p = unit(tmp_path, b, 0)
    if change == "payload":
        p.write_text("{}")
    elif change == "marker":
        Path(str(p) + ".complete").unlink()
    elif change == "sidecar":
        Path(str(p) + ".sha256").write_text("0" * 64 + " foreign.json\n")
    elif change == "metadata":
        Path(str(p) + ".metadata.json").write_text("{}")
    else:
        moved = p.with_name("other")
        p.rename(moved)
        p.symlink_to(moved)
    with pytest.raises(ValueError):
        units.make_ledger([p], b)


def test_marker_last_and_immutable(tmp_path, monkeypatch) -> None:
    b = bound()
    p = tmp_path / "session-00.json"
    original = units.export_bytes

    def checked(path, payload):
        assert Path(str(path) + ".metadata.json.complete").is_file()
        assert not Path(str(path) + ".complete").exists()
        return original(path, payload)

    monkeypatch.setattr(units, "export_bytes", checked)
    units.publish(p, row(b, 0), b, 0)
    with pytest.raises(FileExistsError):
        units.publish(p, row(b, 0), b, 0)


def test_missing_N_gate_before_payload_parse(tmp_path, monkeypatch) -> None:
    b = bound(False, 32)
    p = unit(tmp_path, b, 0)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path == p:
            raise AssertionError("PARTIAL UNION OUTCOME OPENED")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    monkeypatch.setattr(units.platform, "system", lambda: "Darwin")
    result = units.aggregate([p], b, tmp_path / "decision.json")
    assert result["decision"] == "INVALID/INCOMPLETE" and not result["P48_primary"]


def test_duplicate_union_refused(tmp_path) -> None:
    b = bound()
    p = unit(tmp_path, b, 0)
    with pytest.raises(ValueError, match="duplicate"):
        units.make_ledger([p, p], b)


@pytest.mark.parametrize("change", ["host", "initial", "seed", "frames", "nan", "arms"])
def test_publisher_rejects_invalid_pair(tmp_path, change) -> None:
    b = bound()
    r = row(b, 0)
    arm = r["arms"]["P48"]
    if change == "host":
        arm["host"] = {"cpu": "foreign"}
    elif change == "initial":
        arm["initial_state"] = "d" * 64
    elif change == "seed":
        arm["seed"] += 1
    elif change == "frames":
        arm["frames"] = 5999
    elif change == "nan":
        arm["mass"] = float("nan")
    else:
        r["arms"].pop("P0")
    with pytest.raises(ValueError):
        units.publish(tmp_path / "session-00.json", r, b, 0)
    assert not list(tmp_path.iterdir())


def test_complete_N_synthetic_reducer_matches_independent_constant(tmp_path, monkeypatch) -> None:
    b = bound(False, 32)
    paths = [unit(tmp_path, b, i) for i in range(32)]
    monkeypatch.setattr(units.platform, "system", lambda: "Darwin")
    report = units.aggregate(paths, b, tmp_path / "decision.json")
    assert report["decision"] == "GO to real paced development"
    assert report["P48_primary"]["benefit"] == pytest.approx(0.2)
    assert report["P48_primary"]["mass_relative"] == 0
    with pytest.raises(FileExistsError):
        units.aggregate(paths, b, tmp_path / "decision.json")


def test_zero_death_bootstraps_conservative() -> None:
    from research.watch_lead_p.protocol import decision, statistics

    b = bound(False, 32)
    rows = [row(b, i) for i in range(32)]
    for r in rows:
        for a in r["arms"].values():
            a["deaths"] = 0
    metrics = statistics(rows)
    assert metrics["zero_baseline_bootstraps"] == 10000
    assert metrics["benefit_lb90"] is None and metrics["benefit_ub90"] is None
    assert decision(metrics, valid=True, complete=True) == "INCONCLUSIVE"


def test_completion_limit_and_mechanical_checks() -> None:
    b = bound()
    prior = {
        "attempt": 0,
        "binding": b,
        "indices": [0, 1],
        "failure_class": "timeout",
        "stop_verified": True,
        "disappearance_verified": True,
    }
    assert units.completion_attempt(1, prior, units.empty_ledger(b), b, [0, 1]) == [0, 1]
    for attempt in (0, 3, True):
        with pytest.raises(ValueError):
            units.completion_attempt(attempt, prior, units.empty_ledger(b), b, [0, 1])
    prior["stop_verified"] = False
    with pytest.raises(ValueError):
        units.completion_attempt(1, prior, units.empty_ledger(b), b, [0, 1])


def fake_worker(tasks, b, checkpoints, output):
    while True:
        i = tasks.get()
        if checkpoints.name == "stall" and i == 1:
            time.sleep(60)
        units.publish(output / "units" / f"session-{i:02d}.json", row(b, i), b, i)
        units.export_json(
            output / "timing" / f"session-{i:02d}.json", {"index": i, "engineering_only": True}
        )


def test_deadline_retains_finished_session_and_exact_skip(tmp_path, monkeypatch) -> None:
    from research.watch_lead_p_successor import runtime

    monkeypatch.setattr(runtime, "worker", fake_worker)
    b = bound()
    output = tmp_path / "first"
    before = {p.pid for p in mp.active_children()}
    result = runtime.execute(
        b, units.empty_ledger(b), [0, 1], Path("stall"), output, workers=1, work_seconds=1.5
    )
    assert result["status"] == "INCOMPLETE" and result["completed"] == [0]
    assert not ({p.pid for p in mp.active_children()} - before)
    p = output / "units/session-00.json"
    ledger = units.make_ledger([p], b)
    resumed = runtime.execute(
        b, ledger, [0, 1], Path("fast"), tmp_path / "second", workers=1, work_seconds=10
    )
    assert resumed["completed"] == [1] and resumed["skipped"] == [0]
    assert not (tmp_path / "second/units/session-00.json").exists()


def test_signal_during_publication_finishes_unit_and_reaps(tmp_path, monkeypatch) -> None:
    from research.watch_lead_p_successor import runtime

    monkeypatch.setattr(runtime, "worker", fake_worker)
    real_verify = runtime.verify_metadata

    def stopped(*args, **kwargs):
        signal.raise_signal(signal.SIGTERM)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(runtime, "verify_metadata", stopped)
    b = bound()
    result = runtime.execute(
        b,
        units.empty_ledger(b),
        [0, 1],
        Path("stall"),
        tmp_path / "out",
        workers=1,
        work_seconds=10,
    )
    assert result["completed"] == [0]
    units.verify_metadata(tmp_path / "out/units/session-00.json", b)


def test_serial_parallel_unit_bytes_equal(tmp_path, monkeypatch) -> None:
    from research.watch_lead_p_successor import runtime

    monkeypatch.setattr(runtime, "worker", fake_worker)
    b = bound()
    for w in (1, 2):
        runtime.execute(
            b,
            units.empty_ledger(b),
            [0, 1],
            Path("fast"),
            tmp_path / f"w{w}",
            workers=w,
            work_seconds=10,
        )
    for i in (0, 1):
        assert (tmp_path / f"w1/units/session-{i:02d}.json").read_bytes() == (
            tmp_path / f"w2/units/session-{i:02d}.json"
        ).read_bytes()


def test_transfer_margin_and_fresh_bank() -> None:
    from research.watch_lead_p_successor.delivery import require_transfer_margin
    from research.watch_lead_p_successor.prepare_session import collision_audit, fresh_bank

    assert collision_audit(fresh_bank())["collisions"] == []
    assert (
        require_transfer_margin({"transfer_seconds": 105, "elapsed_seconds": 200})["margin_seconds"]
        == 45
    )
    for seconds in (141.505044, float("nan"), True, 0, 105.01):
        with pytest.raises(ValueError):
            require_transfer_margin({"transfer_seconds": seconds, "elapsed_seconds": 200})


def test_scientific_runtime_stays_closed(tmp_path) -> None:
    from research.watch_lead_p_successor import runtime

    b = bound(False, 32)
    with pytest.raises(ValueError, match="ratification"):
        runtime.run_session(b, 0, tmp_path)
    with pytest.raises(ValueError, match="ratification"):
        runtime.execute(
            b, units.empty_ledger(b), [0], tmp_path, tmp_path / "out", workers=1, work_seconds=1
        )
    assert not (tmp_path / "out").exists()


def test_instrumented_recipe_reports_real_search_support(monkeypatch, tmp_path) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from research.watch_lead_p import engine, live, uploaded
    from research.watch_lead_p_successor import runtime
    from research.watch_lead_p_successor.test_profile import TinyWorld

    monkeypatch.setattr(
        TinyWorld, "hero", property(lambda _self: SimpleNamespace(is_alive=True, length=1000))
    )
    monkeypatch.setattr(live, "LiveWorld", TinyWorld)
    monkeypatch.setattr(engine, "LiveWorld", TinyWorld)
    monkeypatch.setattr(engine, "search", lambda *_: {"chosen": 1})
    monkeypatch.setattr(uploaded, "uploaded_checkpoints", lambda _: nullcontext())
    monkeypatch.setattr(runtime, "host_identity", lambda: {"torch_threads": 1, "torch_interop": 1})
    _, timing = runtime.run_session(bound(), 0, tmp_path, smoke_frames=3)
    assert timing["smoke"] is True
    assert timing["arms"]["P0"]["root_lengths"][0]["search_calls"] == 1
    assert timing["arms"]["P0"]["root_lengths"][0]["length"] == 1000
    assert timing["arms"]["bare"]["search_calls"] == 0
    assert timing["session_inclusive_seconds"] > 0


def test_long_packet_binder_refuses_missing_transport_pin(tmp_path) -> None:
    import json

    from research.watch_lead_p_successor.bind_timing import bind

    (tmp_path / "transport-target.json").write_text(json.dumps({"target_transport_tree": None}))
    with pytest.raises(ValueError, match="re-pin"):
        bind(tmp_path, tmp_path / "receipt", tmp_path / "qualified", tmp_path / "new-packet")
    assert not (tmp_path / "new-packet").exists()
