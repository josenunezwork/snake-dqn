"""Bank generation integrity, product compatibility, and within-session resume."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.watch_replay import engine_l1_os
from research.watch_replay import run_l1_os as bank
from research.watch_replay.install import extract_controller_bank, install
from research.watch_replay.qualify import SUPPORTED_CONTROLLERS, scan_bank_headers
from research.watch_replay.resume import build
from tools.ops.package_spec import PackageRefused, empty_ledger, publish_json
from web.backend.watch_replay import Recording, resolve_replay_controller, validate_frame


@pytest.fixture
def real_env(monkeypatch):
    from research.t1_commitment_20261015.replay import configure_cpu

    ckpts = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    if not (ckpts / "b8_s3_t7_20261016.pth").exists():
        pytest.skip("released checkpoint fixture unavailable")
    monkeypatch.setenv("SNAKE_CKPT_DIR", str(ckpts))
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    configure_cpu()


def contract(count=1, horizon=24):
    return {
        **bank.allocation("t34-l1os-bank-test-fixture", count, horizon=horizon),
        "commit": "a" * 40,
    }


def ledger(tmp_path, value):
    path = tmp_path / "ledger.json"
    publish_json(path, empty_ledger(value["namespace"], "sessions", bank.identity(value)))
    return path


def test_resumed_gzip_equals_uninterrupted_and_raw_controller(tmp_path, monkeypatch, real_env):
    value = contract()
    row = value["rows"][0]
    whole = bank.run_unit(value, row, tmp_path / "whole", lambda: None)
    raw = engine_l1_os.run_arm(row["seed"], lambda: None, horizon=value["horizon"])
    rec = Recording(tmp_path / "whole/000000.watch.gz", product=False)
    assert rec.header["controller"] == bank.CONTROLLER
    assert rec.info()["label"] == "Recorded assisted hero"
    stats = rec.header["stats"]
    all_batches = raw["batches"] + raw["open_stop_batches"]
    assert stats["hero_deaths"] == sum(f["death"] for f in raw["frames"])
    assert stats["hero_mass_integral"] == sum(f["mass"] for f in raw["frames"]) / 24
    assert stats["interventions"] == sum(b["chosen"] != b["bare"] for b in all_batches)
    assert stats["search_roots"] == len(all_batches)
    assert stats["mirror_identity_frames"] == 24
    for index in range(1, 25):
        validate_frame(rec.frame, index)
        if index < 24:
            rec.step()
    rec.close()
    seen = {"n": 0}
    original = bank._normalize

    def counting(snapshot):
        seen["n"] += 1
        return original(snapshot)

    def stop():
        if seen["n"] >= 15:
            raise InterruptedError("preemption fixture")

    monkeypatch.setattr(bank, "_normalize", counting)
    out = tmp_path / "partial"
    with pytest.raises(InterruptedError):
        bank.run_unit(value, row, out, stop, checkpoint_frames=frozenset({8}))
    assert not (out / "000000.watch.gz.complete").exists()
    resumed = bank.run_unit(value, row, out, lambda: None)
    assert resumed["resumed_from_frame"] == 15
    assert resumed["checkpoints_rejected"] == 0
    assert resumed["sha256"] == whole["sha256"]
    assert (out / "000000.watch.gz").read_bytes() == (
        tmp_path / "whole/000000.watch.gz"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        bank.run_unit(value, row, out, lambda: None)


def test_foreign_checkpoint_refused_but_canary_skips(tmp_path):
    value = contract()
    cp = tmp_path / "000000/frame-000008.json"
    publish_json(
        cp, {"schema": bank.CHECKPOINT_SCHEMA, "allocation": {"contract_sha256": "f" * 64}}
    )
    with pytest.raises(PackageRefused, match="foreign"):
        bank.checkpoint_candidates([tmp_path], 0, value)
    assert bank.checkpoint_candidates([tmp_path], 0, value, foreign="skip") == []


def test_hash_mismatch_restarts_without_publishing_bad_prefix(tmp_path, real_env):
    value = contract(horizon=16)
    row = value["rows"][0]
    whole = bank.run_unit(value, row, tmp_path / "whole", lambda: None)
    cp = {
        "schema": bank.CHECKPOINT_SCHEMA,
        "allocation": bank.identity(value),
        "index": 0,
        "arm": bank.CONTROLLER,
        "seed": row["seed"],
        "frame": 10,
        "frames_sha256": "f" * 64,
        "counters": dict.fromkeys(bank.COUNTERS, 0),
        "overrides": {},
        "hardware": bank.hardware_fingerprint(),
        "compute_seconds": 0,
    }
    publish_json(tmp_path / "partial/checkpoints/000000/frame-000010.json", cp)
    resumed = bank.run_unit(value, row, tmp_path / "partial", lambda: None)
    assert resumed["checkpoints_rejected"] == 1
    assert resumed["resumed_from_frame"] is None
    assert whole["sha256"] == resumed["sha256"]


def test_parallel_bytes_match_serial_and_verified_ledger_skips(tmp_path, real_env):
    value = contract(count=2, horizon=12)
    lp = ledger(tmp_path, value)
    serial = bank.run(value, lp, tmp_path / "serial", 1, isolate=False)
    parallel = bank.run(value, lp, tmp_path / "parallel", 2)
    assert serial["completed_now"] == parallel["completed_now"] == [0, 1]
    for i in range(2):
        assert (tmp_path / "serial" / f"{i:06d}.watch.gz").read_bytes() == (
            tmp_path / "parallel" / f"{i:06d}.watch.gz"
        ).read_bytes()
    verified = tmp_path / "verified.json"
    publish_json(verified, build(value, [tmp_path / "serial"]))
    skipped = bank.run(value, verified, tmp_path / "retry", 1)
    assert skipped["completed"] == 0 and skipped["skipped"] == [0, 1]
    assert not list((tmp_path / "retry").glob("*.watch.gz"))
    foreign = contract(count=1)
    with pytest.raises(PackageRefused):
        bank.run(foreign, verified, tmp_path / "foreign", 1)


def test_product_install_qualify_and_filter_l1_os(tmp_path, monkeypatch):
    # Synthetic full-length product fixture: exercise install and qualification readers,
    # without 6,000 frames of search or reading any existing study output.
    from tests.test_watch_replay import recording

    value = contract(horizon=6000)
    root = tmp_path / "sessions"
    root.mkdir()
    publish_json(root / "contract.json", value)
    recording(
        root,
        "000000",
        controller=bank.CONTROLLER,
        seed=value["rows"][0]["seed"],
        unit_id=f"{value['namespace']}/sessions/000000",
        index=0,
        allocation=bank.identity(value),
        source_commit=value["commit"],
    )
    destination = tmp_path / "installed"
    assert install(value, [root], destination) == 1
    extracted = extract_controller_bank(destination, bank.CONTROLLER, tmp_path / "pure")
    assert extracted["copied_count"] == 1
    assert bank.CONTROLLER in SUPPORTED_CONTROLLERS
    scan = scan_bank_headers(destination)
    assert scan["by_controller"] == {bank.CONTROLLER: 1}
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l1")
    assert resolve_replay_controller() == ("exact-stop-l1", None)
    assert resolve_replay_controller(bank.CONTROLLER) == (bank.CONTROLLER, None)
    assert bank.CONTROLLER in resolve_replay_controller("invalid")[1]


@pytest.mark.parametrize("interval", [0, -1, 1801, float("inf"), float("nan")])
def test_checkpoint_interval_bound(tmp_path, interval):
    value = contract()
    with pytest.raises(ValueError, match="checkpoint interval"):
        bank.run_unit(value, value["rows"][0], tmp_path, lambda: None, checkpoint_seconds=interval)


def test_frozen_engine_files_unchanged():
    import subprocess

    for name in ("engine.py", "run.py"):
        path = f"research/watch_replay/{name}"
        before = subprocess.check_output(["git", "show", f"e649385a:{path}"])
        assert hashlib.sha256(Path(path).read_bytes()).digest() == hashlib.sha256(before).digest()


def waiting_unit(contract, unit, out, check, **kwargs):
    """Spawn-safe interruption fixture; never completes a replay."""
    import time

    while True:
        check()
        time.sleep(0.02)


def test_session_deadline_reaps_owned_workers(tmp_path):
    import multiprocessing

    value = contract(count=2)
    lp = ledger(tmp_path, value)
    before = {child.pid for child in multiprocessing.active_children()}
    result = bank.run(
        value,
        lp,
        tmp_path / "stopped",
        2,
        session_seconds=0.5,
        runner="tests.test_run_l1_os_bank:waiting_unit",
    )
    assert result["stopped"] == "session budget"
    assert result["completed"] == 0
    assert not list((tmp_path / "stopped").glob("*.watch.gz.complete"))
    assert {child.pid for child in multiprocessing.active_children()} <= before
