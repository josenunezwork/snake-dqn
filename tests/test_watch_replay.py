"""Replay integrity, protocol parity, source isolation and completed-unit resume."""

from __future__ import annotations

import copy
import gzip
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research.watch_replay.install import install
from research.watch_replay.prepare import release_bundle
from research.watch_replay.resume import build
from research.watch_replay.run import allocation, completed, identity, run
from tools.ops.package_spec import empty_ledger, publish_immutable, publish_json
from web.backend.session import GameSession
from web.backend.watch_replay import B8_SHA, FRP3_SHA, SCHEMA, Recording, ReplayBank, canonical


@pytest.fixture(autouse=True)
def close_replay_workers(monkeypatch):
    """Every test owns and reaps its bank workers, including assertion failures."""
    banks = []
    original = ReplayBank.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        banks.append(self)

    monkeypatch.setattr(ReplayBank, "__init__", initialize)
    yield
    for bank in banks:
        bank.close()


def frame(index: int) -> dict:
    return {
        "type": "frame",
        "protocol_version": 2,
        "frame": index,
        "mechanics_version": 2,
        "paused": False,
        "viewer_count": 0,
        "obs_spec": "ego2s-bv8",
        "architecture": "CNN",
        "serving_contract": None,
        "checkpoint_name": "b8.pth",
        "hero_raster": None,
        "arena": {
            "width": 800,
            "height": 600,
            "segment": 10,
            "wall": 10,
            "arena_type": "rectangular",
        },
        "snakes": [
            {
                "id": i,
                "segments": [[i * 10, index % 50]],
                "head": [i * 10, index % 50],
                "length": 1,
                "alive": True,
                "boosting": False,
                "color": [1, 2, 3],
                "is_hero": i == 0,
                "name": str(i),
            }
            for i in range(6)
        ],
        "food": [[30, 30]],
        "stats": {"frame": index},
        "session": {"mode": "watch", "watch_roster": "mixed", "hero_id": 0, "playing": True},
        "play": None,
        "inspector": {"action": 1},
        "netviz": None,
    }


def recording(root: Path, name: str = "a", horizon: int = 6000, **overrides: object) -> Path:
    header = {
        "schema": SCHEMA,
        "controller": "exact-stop-l2-prime",
        "horizon": horizon,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "protocol_version": 2,
        "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
        "seed": 42,
        "unit_id": name,
        "fixture": horizon != 6000,
        "stats": {},
        **overrides,
    }
    payload = canonical(header) + b"".join(canonical(frame(i)) for i in range(1, horizon + 1))
    path = root / f"{name}.watch.gz"
    publish_immutable(path, gzip.compress(payload, mtime=0))
    return path


def pick_ready(bank: ReplayBank) -> Recording | None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = bank.pick()
        if record is not None or not bank.pending:
            return record
        time.sleep(0.005)
    raise AssertionError("replay prefetch timed out")


def shell_session() -> GameSession:
    """Isolate replay controls without constructing a network or a global world."""
    session = GameSession.__new__(GameSession)
    session._lock = threading.RLock()
    session.mode = "watch"
    session.playing = True
    session.speed = 24
    session.speed_ceiling = 60
    session.viewer_count = 2
    session.last_error = None
    session.pathb = None
    session.hero_id = 0
    return session


def test_protocol_frames_survive_recording_exactly_and_controls_overlay(tmp_path):
    record = Recording(recording(tmp_path, horizon=3), product=False)
    session = shell_session()
    original = copy.deepcopy(record.frame)
    got = record.snapshot(session)
    for key in original.keys() - {"paused", "viewer_count", "session"}:
        assert got[key] == original[key]
    assert got["session"]["watch_source"] == "replay"
    assert got["session"]["replay"]["controller"] == "exact-stop-l2-prime"
    assert got["viewer_count"] == 2
    assert record.frame == original
    session.playing = False
    assert record.snapshot(session)["paused"] is True
    assert record.step() and record.frame["frame"] == 2
    assert record.step() and not record.step()


@pytest.mark.parametrize(
    "damage", ["hash", "marker", "truncated", "foreign", "fixture", "protocol"]
)
def test_bank_refuses_corrupt_incomplete_or_foreign_recording(tmp_path, damage):
    if damage == "fixture":
        path = recording(tmp_path, horizon=2)
    elif damage == "foreign":
        path = recording(tmp_path, controller="unknown")
    else:
        path = recording(tmp_path)
    if damage == "hash":
        path.write_bytes(path.read_bytes() + b"garbage")
    elif damage == "marker":
        Path(str(path) + ".complete").unlink()
    elif damage in {"truncated", "protocol"}:
        lines = gzip.decompress(path.read_bytes()).splitlines(keepends=True)
        if damage == "truncated":
            lines.pop()
        else:
            value = json.loads(lines[50])
            value["protocol_version"] = 999
            lines[50] = canonical(value)
        path.unlink()
        Path(str(path) + ".sha256").unlink()
        Path(str(path) + ".complete").unlink()
        publish_immutable(path, gzip.compress(b"".join(lines), mtime=0))
    bank = ReplayBank(tmp_path)
    assert pick_ready(bank) is None
    assert bank.error


def test_bank_has_no_immediate_repeats_even_for_duplicate_files(tmp_path):
    recording(tmp_path, "a")
    recording(tmp_path, "b")
    bank = ReplayBank(tmp_path)
    ids = [pick_ready(bank).header["unit_id"] for _ in range(5)]
    assert all(a != b for a, b in zip(ids, ids[1:]))
    assert len(set(ids[:2])) == 2


def test_single_session_returns_to_live_after_end_instead_of_repeating(tmp_path):
    recording(tmp_path)
    bank = ReplayBank(tmp_path)
    assert pick_ready(bank) is not None
    assert pick_ready(bank) is None


def test_session_steps_replay_without_live_and_pause_does_not_advance(tmp_path):
    session = shell_session()
    session._watch_replay = Recording(recording(tmp_path, horizon=3), product=False)
    session.game = SimpleNamespace(update=Mock(side_effect=AssertionError("live tick in replay")))
    session._replay_bank = None
    session.step()
    assert session.snapshot()["frame"] == 2
    session.set_playing(False)
    session.step()
    assert session.snapshot()["frame"] == 2
    session.set_speed(30)
    assert session.control_state()["speed"] == 30
    session.set_hero(2)
    assert session.hero_id == 0
    assert "hero #0" in session.last_error
    session.game.update.assert_not_called()


def test_empty_bank_falls_back_and_explicit_model_controls_leave_replay(tmp_path):
    session = shell_session()
    session._replay_bank = ReplayBank(tmp_path)
    pick_ready(session._replay_bank)
    session._select_watch_replay()
    assert session._watch_replay is None
    assert "live B8" in session.last_error
    session._watch_replay = Recording(recording(tmp_path, horizon=3), product=False)
    session.policy = SimpleNamespace(epsilon=0)
    session.set_epsilon(0.2)
    assert session._watch_replay is None
    assert session.policy.epsilon == 0.2


def test_resume_skips_exact_verified_ids_and_refuses_foreign_allocation(tmp_path, monkeypatch):
    contract = {**allocation("test-replay", 3, "exact-stop-l2-prime"), "commit": "1" * 40}
    ledger = tmp_path / "ledger.json"
    value = empty_ledger(contract["namespace"], "sessions", identity(contract))
    prior = tmp_path / "prior"
    prior.mkdir()
    publish_json(prior / "contract.json", contract)
    publish_immutable(prior / "000001.watch.gz", b"payload-not-interpreted-by-ledger")
    value = build(contract, [prior])
    publish_json(ledger, value)
    calls = []

    def generate(task):
        calls.append(task[1]["index"])
        return {"index": task[1]["index"], "sha256": "0" * 64, "size": 1}

    monkeypatch.setattr("research.watch_replay.run.generate", generate)
    run(contract, ledger, tmp_path / "next", 1)
    assert calls == [0, 2]
    with pytest.raises(ValueError, match="foreign"):
        completed(ledger, {**contract, "namespace": "foreign"})
    with pytest.raises(ValueError, match="duplicate"):
        build(contract, [prior, prior])
    with pytest.raises(ValueError, match="foreign"):
        build({**contract, "controller": "bare-b8"}, [prior])


def test_partial_export_never_counts_as_verified_unit(tmp_path):
    contract = {**allocation("partial", 1, "bare-b8"), "commit": "1" * 40}
    publish_json(tmp_path / "contract.json", contract)
    (tmp_path / "000000.watch.gz").write_bytes(b"partial")
    with pytest.raises(ValueError, match="missing"):
        build(contract, [tmp_path])


@pytest.mark.parametrize("failure", [False, True])
def test_end_or_playback_failure_ticks_live_fallback_in_same_step(tmp_path, failure):
    session = shell_session()
    session._replay_bank = None
    session._watch_replay = Recording(recording(tmp_path, horizon=1), product=False)
    if failure:
        session._watch_replay.step = Mock(side_effect=ValueError("corrupt cursor"))
    session.game = SimpleNamespace(update=Mock(), snakes=[])
    session._is_b8_release_monitoring = Mock(return_value=False)
    session._enforce_food_target = Mock()
    session.step()
    session.game.update.assert_called_once_with(train_mode=False, learn=False, allow_respawn=True)
    assert session._watch_replay is None


def test_play_mode_always_clears_recorded_source(tmp_path):
    session = shell_session()
    session.obs_spec = "ego2s-bv8"
    session.checkpoint_path = "b8.pth"
    session._watch_replay = Recording(recording(tmp_path, horizon=1), product=False)
    session._replay_bank = None

    def build(checkpoint, mode, **kwargs):
        assert session._watch_replay is None
        session.mode = mode

    session._build = build
    session.set_mode("play")
    assert session.mode == "play" and session._watch_replay is None


def test_release_bundle_recreated_for_main_keeps_canary_input_bytes(tmp_path):
    from tools.ops.package_spec import digest

    source = tmp_path / "receipt.json"
    source.write_text('{"verdict":"PASS"}\n')
    pins = {
        "checkpoints": {
            "b8-s3": {
                "strict_receipt": {
                    "root": str(tmp_path),
                    "decision": {"path": source.name, "sha256": digest(source)},
                }
            }
        }
    }
    first = tmp_path / "canary-evidence.tar.gz"
    second = tmp_path / "main-evidence.tar.gz"
    release_bundle(first, pins)
    release_bundle(second, pins)
    assert first.read_bytes() == second.read_bytes()


def test_install_requires_complete_frozen_union_then_publishes_bank(tmp_path):
    contract = {**allocation("bank-fixture", 1, "exact-stop-l2-prime"), "commit": "1" * 40}
    exports = tmp_path / "exports"
    exports.mkdir()
    publish_json(exports / "contract.json", contract)
    recording(
        exports,
        "000000",
        unit_id="bank-fixture/sessions/000000",
        allocation=identity(contract),
        source_commit=contract["commit"],
        seed=contract["rows"][0]["seed"],
    )
    bank = tmp_path / "bank"
    assert install(contract, [exports], bank) == 1
    assert pick_ready(ReplayBank(bank)).header["unit_id"] == "bank-fixture/sessions/000000"
    with pytest.raises(ValueError, match="foreign"):
        install({**contract, "controller": "bare-b8"}, [exports], tmp_path / "foreign")
    refused = json.loads((tmp_path / "foreign/operational.json").read_text())
    assert refused["completed"] == 0 and refused["exclusions"]


def test_install_records_missing_units_and_always_writes_operational(tmp_path):
    contract = {**allocation("bank-missing", 1, "bare-b8"), "commit": "1" * 40}
    publish_json(tmp_path / "contract.json", contract)
    bank = tmp_path / "bank"
    assert install(contract, [tmp_path], bank) == 0
    report = json.loads((bank / "operational.json").read_text())
    assert report["missing_indices"] == [0] and report["completed"] == 0


def test_corrupt_file_is_skipped_and_verification_is_cached(tmp_path, monkeypatch, caplog):
    import web.backend.watch_replay as replay

    recording(tmp_path, "bad", horizon=2)
    recording(tmp_path, "good")
    verify = Mock(wraps=replay.verify_payload)
    monkeypatch.setattr(replay, "verify_payload", verify)
    bank = ReplayBank(tmp_path)
    assert pick_ready(bank).header["unit_id"] == "good"
    assert pick_ready(bank) is None
    assert verify.call_count == 2  # bad fixture and good; good rotation uses SHA cache
    assert "Skipping replay" in caplog.text
    bank.close()


def test_rotation_tick_never_waits_for_bank_io(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    first = Recording(recording(tmp_path, horizon=1), product=False)
    entered, release = threading.Event(), threading.Event()
    original = replay.Recording

    def slow_recording(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(replay, "Recording", slow_recording)
    bank = ReplayBank(tmp_path)
    assert entered.wait(5)
    session = shell_session()
    session._replay_bank, session._watch_replay = bank, first
    session.game = SimpleNamespace(update=Mock(side_effect=AssertionError("live during prefetch")))
    start = time.monotonic()
    session.step()  # end of recording: worker is still blocked in bank I/O
    session.step()  # waiting at the last recorded frame also stays nonblocking
    assert time.monotonic() - start < 0.1
    assert session._replay_pending and session._watch_replay is first
    release.set()
    bank.close()


def test_cold_start_keeps_live_ticking_until_first_record_is_ready():
    session = shell_session()
    session._watch_replay = None
    session._replay_pending = True
    session._replay_bank = SimpleNamespace(pick=lambda: None, pending=True, error=None)
    session.game = SimpleNamespace(update=Mock(), snakes=[])
    session._is_b8_release_monitoring = Mock(return_value=False)
    session._enforce_food_target = Mock()
    session.step()
    session.step()
    assert session._replay_pending and session._watch_replay is None
    assert session.game.update.call_count == 2


def test_discovery_queues_first_record_before_verifying_second(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    for name in ("a", "b"):
        (tmp_path / f"{name}.watch.gz").touch()
    entered, release = threading.Event(), threading.Event()

    def load(path, **kwargs):
        if path.stem == "b.watch":
            entered.set()
            assert release.wait(5)
        return SimpleNamespace(header={"unit_id": path.name})

    monkeypatch.setattr(replay, "Recording", load)
    bank = ReplayBank(tmp_path)
    try:
        assert entered.wait(2)
        assert bank.pick().header["unit_id"] == "a.watch.gz"
        assert bank.pending
    finally:
        release.set()
        bank.close()
    assert not bank._worker.is_alive()


def test_close_interrupts_verification_between_frames(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    recording(tmp_path)
    entered = threading.Event()
    original = replay.validate_frame

    def validate(*args):
        original(*args)
        entered.set()

    monkeypatch.setattr(replay, "validate_frame", validate)
    bank = ReplayBank(tmp_path)
    assert entered.wait(2)
    bank.close()
    assert not bank._worker.is_alive() and bank._done.is_set()


def test_full_queue_wakes_on_pick_and_close_without_lost_signal(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"{name}.watch.gz").touch()
    monkeypatch.setattr(
        replay, "Recording", lambda path, **kwargs: SimpleNamespace(header={"unit_id": path.name})
    )
    bank = ReplayBank(tmp_path)
    ids = [pick_ready(bank).header["unit_id"] for _ in range(30)]
    assert all(a != b for a, b in zip(ids, ids[1:]))
    deadline = time.monotonic() + 2
    while not bank._ready.full() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert bank._ready.full()
    bank.close()
    assert not bank._worker.is_alive()


def test_rotation_does_not_revalidate_known_sha_and_rejects_replacement(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    path = recording(tmp_path, horizon=2)
    cache = {}
    Recording(path, product=False, cache=cache)
    monkeypatch.setattr(replay, "verify_payload", Mock(side_effect=AssertionError("revalidated")))
    assert Recording(path, product=False, cache=cache).frame["frame"] == 1
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA mismatch"):
        Recording(path, product=False, cache=cache)


@pytest.mark.parametrize(
    "key,value",
    [("length", "long"), ("alive", 1), ("segments", [[True, 0]]), ("color", [300, 0, 0])],
)
def test_frame_field_types_are_checked(key, value):
    from web.backend.watch_replay import validate_frame

    value_frame = frame(1)
    value_frame["snakes"][0][key] = value
    with pytest.raises(ValueError, match="field types"):
        validate_frame(value_frame, 1)


def test_install_keeps_verified_subset_and_records_bad_names(tmp_path):
    contract = {**allocation("subset", 2, "exact-stop-l2-prime"), "commit": "1" * 40}
    exports = tmp_path / "exports"
    exports.mkdir()
    publish_json(exports / "contract.json", contract)
    recording(
        exports,
        "000000",
        unit_id="subset/sessions/000000",
        allocation=identity(contract),
        source_commit=contract["commit"],
        seed=contract["rows"][0]["seed"],
    )
    (exports / "000001.watch.gz").write_bytes(b"bad")
    (exports / "999999.watch.gz").write_bytes(b"out of range")
    (exports / "invalid.watch.gz").write_bytes(b"bad name")
    bank = tmp_path / "bank"
    assert install(contract, [exports], bank) == 1
    report = json.loads((bank / "operational.json").read_text())
    assert report["missing_indices"] == [1]
    assert len(report["exclusions"]) == 3


def test_generator_excludes_failed_seed_but_writes_operational(tmp_path, monkeypatch):
    contract = {**allocation("failed-seed", 2, "bare-b8"), "commit": "1" * 40}
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(contract["namespace"], "sessions", identity(contract)))

    def generate(task):
        if task[1]["index"] == 0:
            raise ValueError("mirror mismatch")
        return {"index": 1, "sha256": "a" * 64, "size": 1}

    monkeypatch.setattr("research.watch_replay.run.generate", generate)
    rows = run(contract, ledger, tmp_path / "out", 1)
    assert [r["index"] for r in rows] == [1]
    report = json.loads((tmp_path / "out/operational.json").read_text())
    assert report["completed"] == 1 and report["exclusions"][0]["index"] == 0


def test_generator_fails_size_limit_before_computing_next_frame(tmp_path, monkeypatch):
    import research.watch_replay.engine as engine
    import research.watch_replay.run as generator

    contract = {**allocation("early-size", 1, "bare-b8", 2), "commit": "1" * 40}
    calls = []

    def run_arm(seed, dense, check, horizon, record):
        calls.append(1)
        record(frame(1))
        calls.append(2)
        raise AssertionError("computed another frame")

    monkeypatch.setattr(engine, "run_arm", run_arm)
    monkeypatch.setattr(generator, "MAX_DECODED", generator.MAX_LINE + 1)
    with pytest.raises(ValueError, match="incremental"):
        generator.generate((contract, contract["rows"][0], str(tmp_path)))
    assert calls == [1]
    assert not list(tmp_path.glob("*.complete"))


def test_ready_rotation_is_nonblocking_and_all_bank_reads_are_background(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    recording(tmp_path, "first")
    recording(tmp_path, "second")
    original = replay.Recording
    threads = []

    def track(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(replay, "Recording", track)
    bank = ReplayBank(tmp_path)
    first = pick_ready(bank)
    deadline = time.monotonic() + 15
    while bank._ready.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not bank._ready.empty()
    session = shell_session()
    session._watch_replay, session._replay_bank = first, bank
    first.step = Mock(return_value=False)
    start = time.monotonic()
    session.step()
    assert time.monotonic() - start < 0.1
    assert session._watch_replay.header["unit_id"] != first.header["unit_id"]
    assert threads and set(threads) == {"watch-replay-prefetch"}
    bank.close()


@pytest.mark.parametrize("damage", ["length", "roots", "frames", "failed"])
def test_canary_long_body_segment_coverage_guard(tmp_path, damage):
    from research.watch_replay.preflight import segment_proof

    stats = {"max_hero_length": 900, "search_roots": 4, "mirror_identity_frames": 3000}
    report = {"completed": 1, "exclusions": [], "units": [{"stats": stats}]}
    contract = {"horizon": 3000, "controller": "exact-stop-l2-prime"}
    if damage == "length":
        stats["max_hero_length"] = 599
    elif damage == "roots":
        stats["search_roots"] = 0
    elif damage == "frames":
        stats["mirror_identity_frames"] = 2999
    else:
        report["completed"] = 0
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    path = tmp_path / "operational.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="canary"):
        segment_proof(path)


def test_uploads_pack_and_stage_metadata_deterministically(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    from research.t34_exit_b8 import public_inputs
    from research.watch_replay.uploads import (
        CHECKPOINTS,
        MAX_ESTIMATED_TRANSFER_SECONDS,
        MAX_TRANSFERS,
        METADATA,
        SMALL_PAYLOADS,
        pack_inputs,
        stage_metadata,
    )
    from tools.ops.package_spec import Input, stage_inputs, verify_immutable

    assert MAX_TRANSFERS == 8
    assert MAX_ESTIMATED_TRANSFER_SECONDS == 60.0

    b8_bytes = b"ckpt:b8_s3_t7_20261016.pth\n"
    frp3_bytes = b"ckpt:frp3_m3_s12_u60000_20261005.pth\n"
    unused_bytes = b"ckpt:unused_apex.pth\n"
    league_buf = io.BytesIO()
    with tarfile.open(fileobj=league_buf, mode="w:gz") as tar:
        for member_name, payload in (
            ("checkpoints/frp3_m3_s12_u60000_20261005.pth", frp3_bytes),
            ("checkpoints/best_apex_fs.pth", unused_bytes),
        ):
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    league_bytes = league_buf.getvalue()

    catalog_payload = {
        "inputs": [
            {
                "name": "B8_s3_final.pth",
                "dest": "B8_s3_final.pth",
                "size_bytes": len(b8_bytes),
                "sha256": hashlib.sha256(b8_bytes).hexdigest(),
            },
            {
                "name": "frp3_m3_s12_u60000_20261005.pth",
                "dest": "checkpoints/frp3_m3_s12_u60000_20261005.pth",
                "size_bytes": len(frp3_bytes),
                "sha256": hashlib.sha256(frp3_bytes).hexdigest(),
            },
        ],
        "assets": {
            "B8_s3_final.pth": {
                "name": "B8_s3_final.pth",
                "url": "https://example.invalid/B8_s3_final.pth",
                "size_bytes": len(b8_bytes),
                "sha256": hashlib.sha256(b8_bytes).hexdigest(),
            },
            "league.tar.gz": {
                "name": "league.tar.gz",
                "url": "https://example.invalid/league.tar.gz",
                "size_bytes": len(league_bytes),
                "sha256": hashlib.sha256(league_bytes).hexdigest(),
            },
        },
    }
    monkeypatch.setattr(
        public_inputs,
        "_load",
        lambda _path: (catalog_payload["inputs"], catalog_payload["assets"]),
    )

    src = tmp_path / "src"
    src.mkdir()
    inputs = []
    for name in SMALL_PAYLOADS:
        path = src / name
        path.write_bytes(f"payload:{name}\n".encode())
        inputs.append(Input(path, name))

    raw = stage_inputs(tmp_path / "staged", inputs)
    assert len(raw) == 18
    archive_a = tmp_path / "a.tar.gz"
    archive_b = tmp_path / "b.tar.gz"
    packed_a = pack_inputs(raw, archive_a, verify_pins=False)
    packed_b = pack_inputs(raw, archive_b, verify_pins=False)
    assert packed_a[0]["sha256"] == packed_b[0]["sha256"]
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert [row["dest"] for row in packed_a] == [METADATA]

    remote = tmp_path / "remote_inputs"
    remote.mkdir(parents=True)
    for row in packed_a:
        target = remote / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row["path"]).read_bytes())

    by_url = {
        "https://example.invalid/B8_s3_final.pth": b8_bytes,
        "https://example.invalid/league.tar.gz": league_bytes,
    }

    def fake_opener(url: str, timeout: int = 60):
        assert timeout == 60
        return io.BytesIO(by_url[url])

    stage_metadata(remote, packed_a[0]["sha256"], verify_pins=False, opener=fake_opener)
    for name in SMALL_PAYLOADS:
        assert verify_immutable(remote / name)["size"] == len(f"payload:{name}\n")
    assert verify_immutable(remote / "checkpoints/b8_s3_t7_20261016.pth")["size"] == len(b8_bytes)
    assert verify_immutable(remote / "checkpoints/frp3_m3_s12_u60000_20261005.pth")["size"] == len(
        frp3_bytes
    )
    assert not (remote / "checkpoints/best_apex_fs.pth").exists()
    assert set(CHECKPOINTS) == {"b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth"}
    with pytest.raises(ValueError, match="already exists"):
        stage_metadata(remote, packed_a[0]["sha256"], verify_pins=False, opener=fake_opener)


def test_seal_main_follows_ready_queue_retry_chain_and_writes_once(tmp_path, monkeypatch):
    from research.colab_cli import canary
    from research.watch_replay.seal_main import resolve_canary_receipt, seal
    from tools.ops.package_spec import digest

    ops = tmp_path / "ops"
    (ops / "manifests").mkdir(parents=True)
    retry_out = tmp_path / "canary-out-retry-1"
    retry_out.mkdir()
    receipt = retry_out / "receipt.json"
    receipt.write_text('{"status": "PASS"}\n')
    (ops / "manifests/watch-replay-bank05-canary-01-r1-retry-1.json").write_text(
        json.dumps({"output_dir": str(retry_out)}) + "\n"
    )
    (ops / "ready-queue.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "watch-replay-bank05-canary-01",
                        "status": "FAILED",
                        "retry_id": "watch-replay-bank05-canary-01-r1",
                        "manifest": "/unused.json",
                        "argv": ["dispatch.sh", "watch-replay-bank05-canary-01"],
                    },
                    {
                        "id": "watch-replay-bank05-canary-01-r1",
                        "retry_of": "watch-replay-bank05-canary-01",
                        "status": "PASS",
                        "manifest": "/orig.json",
                        "argv": ["dispatch.sh", "watch-replay-bank05-canary-01-r1"],
                    },
                ]
            }
        )
        + "\n"
    )
    assert resolve_canary_receipt(ops, "watch-replay-bank05-canary-01") == receipt

    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "main-unsealed.json").write_text(json.dumps({"job_id": "main-test"}) + "\n")
    checked = []
    monkeypatch.setattr(canary, "check", lambda manifest, broker: checked.append(manifest))
    out = seal(packet, tmp_path, ops=ops, canary_id="watch-replay-bank05-canary-01")
    assert out == packet / "main.json"
    sealed = json.loads(out.read_text())
    assert sealed["canary"] == {
        "schema": canary.SCHEMA,
        "receipt": {"path": str(receipt.resolve()), "sha256": digest(receipt)},
    }
    assert len(checked) == 1
    with pytest.raises(FileExistsError):
        seal(packet, tmp_path, canary_receipt=receipt)


def test_seed_audit_detects_collisions_without_opening_exports(tmp_path):
    from research.watch_replay.seed_audit import DEFAULT_NAMESPACES, audit

    fake_artifacts = tmp_path / "artifacts"
    # Poison an exports/ and pair-*.json file with a candidate seed; audit must NEVER open them.
    report_probe_seed = audit(
        namespaces=DEFAULT_NAMESPACES,
        artifacts=tmp_path / "empty-artifacts",
        extra_worktrees=[],
    )
    assert report_probe_seed["status"] == "CLEAR"
    assert report_probe_seed["total_seeds"] == 36

    first_seed = report_probe_seed["banks"][DEFAULT_NAMESPACES[0]][0]["seed"]
    exports_dir = fake_artifacts / "t34-sol/watch6-v3/output/exports"
    exports_dir.mkdir(parents=True)
    (exports_dir / "pair-000000.json").write_text(json.dumps({"seed": first_seed}) + "\n")
    for idx in range(7):
        inv = fake_artifacts / f"frp-{idx}/namespaces/training_exclusions.json"
        inv.parent.mkdir(parents=True)
        inv.write_text(json.dumps([idx + 100]) + "\n")

    clear = audit(namespaces=DEFAULT_NAMESPACES, artifacts=fake_artifacts, extra_worktrees=[])
    assert clear["status"] == "CLEAR"

    with pytest.raises(ValueError, match="disjointness"):
        audit(
            namespaces=["t34-watch-replay-fix-20261030-bank01"],
            artifacts=fake_artifacts,
            extra_worktrees=[],
        )


def test_prepare_a100_defaults_to_w6_and_single_metadata_input(tmp_path, monkeypatch):
    from research.watch_replay import prepare as prep

    commit = "1" * 40
    monkeypatch.setattr(
        prep.subprocess,
        "check_output",
        lambda argv, **kw: ("  origin/main\n" if "branch" in argv else b""),
    )
    monkeypatch.setattr(
        prep,
        "check_uploads",
        lambda manifest: {
            "status": "PASS",
            "normal_transfer_count": len(manifest["inputs"]) + 2,
            "transfer_count": len(manifest["inputs"]) + 3,
            "estimated_transfer_seconds": 21.6,
        },
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps({"commit": commit, "branch_timings": [{"seconds": 9.9, "branches": 6}]}) + "\n"
    )
    out = tmp_path / "packet-bank05-a100"
    forecast = prep.prepare(
        out,
        commit,
        benchmark,
        "t34-watch-replay-20261031-bank05",
        12,
        "exact-stop-l2-prime",
        gpu="A100",
    )
    assert forecast["gpu"] == "A100"
    assert forecast["workers"] == 6
    canary_manifest = json.loads((out / "canary.json").read_text())
    main_manifest = json.loads((out / "main-unsealed.json").read_text())
    assert canary_manifest["requirements"]["gpu"] == "A100"
    assert canary_manifest["requirements"]["allowed_gpus"] == ["A100"]
    assert canary_manifest["requirements"]["min_vcpu"] == 12
    assert canary_manifest["budget"]["rate_units_per_hour"] == 8.0
    assert canary_manifest["env"]["WATCH_REPLAY_WORKERS"] == "1"
    assert main_manifest["requirements"]["gpu"] == "A100"
    assert main_manifest["requirements"]["allowed_gpus"] == ["A100"]
    assert main_manifest["budget"]["rate_units_per_hour"] == 8.0
    assert main_manifest["env"]["WATCH_REPLAY_WORKERS"] == "6"
    assert len(canary_manifest["inputs"]) == 1
    assert canary_manifest["inputs"][0]["dest"] == "input-metadata.tar.gz"
    assert main_manifest["inputs"] == canary_manifest["inputs"]
