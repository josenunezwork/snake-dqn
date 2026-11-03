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
    bank = ReplayBank(tmp_path, shuffle_discovery=False)
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


def test_validate_header_and_recording_info_accept_exact_stop_l1_assisted_label(tmp_path):
    from web.backend.watch_replay import validate_header

    base_header = {
        "schema": SCHEMA,
        "controller": "exact-stop-l1",
        "horizon": 6000,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "protocol_version": 2,
        "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
        "seed": 123,
        "unit_id": "bank14l1/sessions/000000",
        "fixture": False,
        "stats": {},
    }
    validate_header(base_header, product=True)
    with pytest.raises(ValueError, match="foreign replay metadata"):
        validate_header({**base_header, "controller": "exact-stop-alllen"}, product=True)

    l1_path = recording(tmp_path, "l1_unit", controller="exact-stop-l1")
    bare_path = recording(tmp_path, "bare_unit", controller="bare-b8")
    l1_rec = Recording(l1_path)
    bare_rec = Recording(bare_path)
    assert l1_rec.info()["controller"] == "exact-stop-l1"
    assert l1_rec.info()["label"] == "Recorded assisted hero"
    assert bare_rec.info()["label"] == "Recorded bare B8"

    session = shell_session()
    snap = l1_rec.snapshot(session)
    assert snap["session"]["replay"]["controller"] == "exact-stop-l1"
    assert snap["session"]["replay"]["label"] == "Recorded assisted hero"

    bank = ReplayBank(tmp_path)
    picked = {pick_ready(bank).header["controller"], pick_ready(bank).header["controller"]}
    assert picked == {"exact-stop-l1", "bare-b8"}


def test_canary_exact_stop_l1_segment_coverage_guard(tmp_path):
    from research.watch_replay.preflight import segment_proof

    stats = {"max_hero_length": 22, "search_roots": 60, "mirror_identity_frames": 300}
    report = {"completed": 1, "exclusions": [], "units": [{"stats": stats}]}
    contract = {"horizon": 300, "controller": "exact-stop-l1"}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    path = tmp_path / "operational.json"
    path.write_text(json.dumps(report))
    assert segment_proof(path)["stats"]["search_roots"] == 60

    for bad_contract, bad_stats in (
        ({"horizon": 3000, "controller": "exact-stop-l1"}, stats),
        (contract, {**stats, "mirror_identity_frames": 299}),
        (contract, {**stats, "search_roots": 29}),
    ):
        (tmp_path / "contract.json").write_text(json.dumps(bad_contract))
        path.write_text(json.dumps({**report, "units": [{"stats": bad_stats}]}))
        with pytest.raises(ValueError, match="canary"):
            segment_proof(path)


def test_prepare_exact_stop_l1_a100_budgets_2_8x_search_cost_and_2x_timeouts(tmp_path, monkeypatch):
    from research.watch_replay import prepare as prep
    from research.watch_replay.source import source_paths

    paths = source_paths()
    assert "research/watch6_alllen_screen/__init__.py" in paths
    assert "research/watch6_alllen_screen/protocol.py" in paths

    commit = "2" * 40
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
        json.dumps({"commit": commit, "branch_timings": [{"seconds": 6.6, "branches": 6}]}) + "\n"
    )
    l2_out = tmp_path / "packet-l2-a100"
    l1_out = tmp_path / "packet-bank14l1-a100"
    l2_forecast = prep.prepare(
        l2_out,
        commit,
        benchmark,
        "t34-watch-replay-20261031-bank08",
        12,
        "exact-stop-l2-prime",
        gpu="A100",
    )
    l1_forecast = prep.prepare(
        l1_out,
        commit,
        benchmark,
        "t34-watch-replay-20261101-bank14l1",
        12,
        "exact-stop-l1",
        gpu="A100",
    )
    expected_ratio = 1268.5208333333333 / 449.6041666666667
    assert l1_forecast["search_cost_multiplier_vs_l2_prime"] == pytest.approx(
        expected_ratio, rel=1e-5
    )
    assert l1_forecast["search_colab_seconds_per_session"] == pytest.approx(
        l2_forecast["search_colab_seconds_per_session"] * expected_ratio
    )
    assert l1_forecast["canary_horizon"] == 300
    canary_contract = json.loads((l1_out / "canary-contract.json").read_text())
    assert canary_contract["controller"] == "exact-stop-l1"
    assert canary_contract["horizon"] == 300
    main_manifest = json.loads((l1_out / "main-unsealed.json").read_text())
    assert main_manifest["timeout_minutes"] == 2 * main_manifest["requirements"]["est_minutes"]
    assert main_manifest["timeout_minutes"] == l1_forecast["effective_minutes"]
    assert main_manifest["budget"]["unit_cap"] == pytest.approx(
        2 * main_manifest["watch_replay"]["base_unit_cap"]
    )


def test_exact_stop_l1_reuses_alllen_arm_spec_and_is_byte_identical_to_screen_arm(
    tmp_path, monkeypatch
):
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_alllen_screen.engine import run_arm as screen_run_arm
    from research.watch6_alllen_screen.protocol import ARM_SPECS, ROOT
    from research.watch_replay.engine import controller_arm_spec
    from research.watch_replay.engine import run_arm as replay_run_arm
    from web.backend.watch_replay import validate_frame

    controller, spec = controller_arm_spec("exact-stop-l1")
    assert controller == "exact-stop-l1"
    assert spec is ARM_SPECS["exact_stop_alllen"]

    saved = None
    for candidate in (ROOT.parent / "snake-dqn/saved_snakes", ROOT / "saved_snakes"):
        if (candidate / "b8_s3_t7_20261016.pth").is_file() and (
            candidate / "frp3_m3_s12_u60000_20261005.pth"
        ).is_file():
            saved = candidate
            break
    if saved is None:
        pytest.skip("product checkpoints unavailable in test environment")
    monkeypatch.setenv("SNAKE_CKPT_DIR", str(saved))
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")

    configure_cpu()
    seed = 42424242
    horizon = 12
    recorded_frames: list[dict] = []
    replay_out = replay_run_arm(
        seed,
        "exact-stop-l1",
        lambda: None,
        horizon=horizon,
        record=recorded_frames.append,
    )
    screen_out = screen_run_arm(
        seed,
        "exact_stop_alllen",
        lambda: None,
        horizon=horizon,
        audit_live_mirror=True,
        include_raw=True,
    )
    assert len(replay_out["batches"]) == 3
    assert screen_out["search_roots"] == 3
    assert canonical(replay_out["frames"]) == canonical(screen_out["raw"]["frames"])
    assert canonical(replay_out["decisions"]) == canonical(screen_out["raw"]["decisions"])
    assert replay_out["batches"][0]["root_sha256"] == screen_out["raw"]["batches"][0]["root_sha256"]

    def strip_rng(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "root_sha256"} for row in rows]

    assert canonical(strip_rng(replay_out["batches"])) == canonical(
        strip_rng(screen_out["raw"]["batches"])
    )
    assert canonical(replay_out["roster"]) == canonical(screen_out["roster"])
    assert replay_out["mirror_identity_frames"] == screen_out["mirror_identity_frames"] == horizon
    assert len(recorded_frames) == horizon
    for idx, dto in enumerate(recorded_frames, 1):
        validate_frame(dto, idx)

    contract = {
        **allocation("t34-watch-replay-l1-unit-test", 1, "exact-stop-l1", horizon),
        "commit": "3" * 40,
    }
    contract["rows"][0]["seed"] = seed
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(contract["namespace"], "sessions", identity(contract)))
    rows = run(contract, ledger, tmp_path / "sessions", 1)
    assert len(rows) == 1
    rec = Recording(tmp_path / "sessions/000000.watch.gz", product=False)
    assert rec.header["controller"] == "exact-stop-l1"
    assert rec.info()["label"] == "Recorded assisted hero"
    assert rec.header["stats"]["search_roots"] == screen_out["search_roots"] == 3
    assert rec.header["stats"]["hero_mass_integral"] == pytest.approx(screen_out["mass_integral"])
    assert rec.header["stats"]["hero_deaths"] == screen_out["deaths"]


def test_shuffle_discovery_default_randomizes_cycle_one_and_false_is_lexicographic(
    tmp_path, monkeypatch
):
    import web.backend.watch_replay as replay

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"{name}.watch.gz").touch()
    monkeypatch.setattr(
        replay, "Recording", lambda path, **kwargs: SimpleNamespace(header={"unit_id": path.stem})
    )

    ordered = ReplayBank(tmp_path, shuffle_discovery=False)
    got_ordered = [pick_ready(ordered).header["unit_id"] for _ in range(4)]
    assert got_ordered == ["a.watch", "b.watch", "c.watch", "d.watch"]
    ordered.close()

    shuffle_calls = []

    class ReversingRng:
        def shuffle(self, items):
            shuffle_calls.append(len(items))
            items.reverse()

    monkeypatch.setattr(replay.random, "SystemRandom", ReversingRng)
    shuffled = ReplayBank(tmp_path)
    got_shuffled = [pick_ready(shuffled).header["unit_id"] for _ in range(4)]
    assert shuffle_calls and shuffle_calls[0] == 4
    assert got_shuffled == ["d.watch", "c.watch", "b.watch", "a.watch"]
    shuffled.close()


def test_controller_filter_line_one_skips_unmatched_without_verifying_payload(
    tmp_path, monkeypatch
):
    import web.backend.watch_replay as replay

    recording(tmp_path, "00_l2a", controller="exact-stop-l2-prime")
    recording(tmp_path, "01_l1a", controller="exact-stop-l1")
    recording(tmp_path, "02_l2b", controller="exact-stop-l2-prime")
    recording(tmp_path, "03_bare", controller="bare-b8")

    verified_controllers = []
    original_verify = replay.verify_payload

    def spy_verify(payload, **kwargs):
        header = original_verify(payload, **kwargs)
        verified_controllers.append(header["controller"])
        return header

    monkeypatch.setattr(replay, "verify_payload", spy_verify)

    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l2-prime")
    l2_bank = ReplayBank(tmp_path, shuffle_discovery=False)
    l2_ids = [pick_ready(l2_bank).header["unit_id"] for _ in range(3)]
    l2_bank.close()
    assert set(l2_ids) == {"00_l2a", "02_l2b"}
    assert all(a != b for a, b in zip(l2_ids, l2_ids[1:]))
    assert verified_controllers == ["exact-stop-l2-prime", "exact-stop-l2-prime"]

    verified_controllers.clear()
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l1")
    l1_bank = ReplayBank(tmp_path, shuffle_discovery=False)
    first_l1 = pick_ready(l1_bank)
    assert first_l1 is not None and first_l1.header["unit_id"] == "01_l1a"
    assert pick_ready(l1_bank) is None
    l1_bank.close()
    assert verified_controllers == ["exact-stop-l1"]

    verified_controllers.clear()
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "all")
    all_bank = ReplayBank(tmp_path, shuffle_discovery=False)
    all_controllers = {pick_ready(all_bank).header["controller"] for _ in range(4)}
    all_bank.close()
    assert all_controllers == {"exact-stop-l2-prime", "exact-stop-l1", "bare-b8"}
    assert len(verified_controllers) == 4

    verified_controllers.clear()
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "bogus-controller")
    bad_bank = ReplayBank(tmp_path, shuffle_discovery=False)
    assert pick_ready(bad_bank) is None
    assert bad_bank.error is not None and "Invalid SNAKE_WATCH_REPLAY_CONTROLLER" in bad_bank.error
    assert verified_controllers == []
    bad_bank.close()


def test_controller_filter_rejects_pinned_payload_controller_swap(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay

    recording(tmp_path, "swap", controller="exact-stop-l1")
    monkeypatch.setattr(
        replay,
        "peek_header",
        lambda path, product=True: {"controller": "exact-stop-l2-prime"},
    )
    bank = ReplayBank(tmp_path, controller="exact-stop-l2-prime", shuffle_discovery=False)
    assert pick_ready(bank) is None
    assert bank.error is not None
    bank.close()


def test_shallow_recording_snapshot_does_not_mutate_pinned_frame(tmp_path):
    record = Recording(recording(tmp_path, horizon=2), product=False)
    pinned_before = copy.deepcopy(record.frame)
    session = shell_session()
    session.playing = False
    session.speed = 12
    session.viewer_count = 7
    session.last_error = "transient notice"

    snap_a = record.snapshot(session)
    snap_a["paused"] = False
    snap_a["session"]["watch_source"] = "mutated"
    snap_a["session"]["speed"] = 999

    assert record.frame == pinned_before
    session.playing = True
    session.speed = 24
    session.last_error = None
    snap_b = record.snapshot(session)
    assert snap_b["paused"] is False
    assert snap_b["session"]["watch_source"] == "replay"
    assert snap_b["session"]["speed"] == 24
    assert snap_b["session"]["error"] is None
    assert record.frame == pinned_before


def test_rejected_train_mode_preserves_active_watch_replay(tmp_path, monkeypatch):
    import web.backend.session as session_module

    active = Recording(recording(tmp_path, horizon=3), product=False)

    for obs_spec, expected_msg in (
        ("ego2s-bv8", "ego2s students are served forward-only"),
        ("raster31v2", "Raster checkpoints are served forward-only"),
    ):
        session = shell_session()
        session.obs_spec = obs_spec
        session.checkpoint_path = "forward_only.pth"
        session.last_error_overridable = False
        session._watch_replay = active
        session._replay_pending = False
        session._build = Mock(side_effect=AssertionError("unexpected rebuild"))
        session.set_mode("train")
        assert session.mode == "watch"
        assert session._watch_replay is active
        assert expected_msg in (session.last_error or "")
        assert session.last_error_overridable is False

    monkeypatch.setattr(session_module, "_reward_contract_mismatch", lambda _path: True)
    mismatch_session = shell_session()
    mismatch_session.obs_spec = "vector61"
    mismatch_session.checkpoint_path = "v1_ckpt.pth"
    mismatch_session.last_error_overridable = False
    mismatch_session._watch_replay = active
    mismatch_session._replay_pending = False
    mismatch_session._build = Mock(side_effect=AssertionError("unexpected rebuild"))
    mismatch_session.set_mode("train")
    assert mismatch_session.mode == "watch"
    assert mismatch_session._watch_replay is active
    assert mismatch_session.last_error_overridable is True


def _install_local_release_fixtures(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """Wire synthetic release-shaped B8 + FRP3 checkpoints without workstation I/O."""
    from dataclasses import replace

    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
    from src.training.apex_policy import ApexPolicy
    from web.backend import served_checkpoint as registry
    from web.backend import session as session_module

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    b8 = ckpt_dir / registry.B8_S3_FILENAME
    frp3 = ckpt_dir / registry.FRP3_S12_FILENAME
    champion = ckpt_dir / registry.CHAMPION_FILENAME
    save_ego2s_checkpoint(b8, Ego2sNet(local_channels=7, n_scalars=60), {"fixture": "b8"})
    opponent = ApexPolicy(input_size=61, hidden_size=512, output_size=6, training=False)
    torch.save(opponent.get_state_dict(), frp3)
    torch.save(opponent.get_state_dict(), champion)

    local_registry = {
        name: replace(entry, artifact_path=None) for name, entry in registry.REGISTRY.items()
    }
    monkeypatch.setattr(registry, "strict_evidence_problems", lambda *args, **kwargs: [])

    def receipt(files):
        return {
            "verdict": "STRICT_PASS",
            "root": str(ckpt_dir),
            **{key: {"path": f"{key}.json", "sha256": "0" * 64} for key in files},
        }

    pins = {
        "schema": registry.PINS_SCHEMA,
        "checkpoints": {
            registry.NAME_FRP3_S12: {
                "checkpoint_sha256": registry.FRP3_S12_SHA256,
                "veto_variant": registry.V8_VARIANT,
                "veto_method": registry.V8_METHOD,
                "strict_receipt": receipt(registry.STRICT_FILES),
            },
            registry.NAME_B8_S3: {
                "checkpoint_sha256": registry.B8_S3_SHA256,
                "veto_variant": "none",
                "veto_method": "none",
                "strict_receipt": receipt(registry.T7_PIN_FILES),
            },
        },
    }
    pins_path = ckpt_dir / "pins.json"
    pins_path.write_text(json.dumps(pins))
    expected_hashes = {
        b8: registry.B8_S3_SHA256,
        frp3: registry.FRP3_S12_SHA256,
        champion: registry.CHAMPION_SHA256,
    }

    def fixture_digest(path):
        resolved = Path(path)
        assert resolved in expected_hashes, f"unexpected checkpoint access: {path}"
        return expected_hashes[resolved]

    original_load = session_module._load_checkpoint_once

    def load_fixture_checkpoint(path):
        resolved = Path(path)
        assert resolved in expected_hashes, f"unexpected checkpoint access: {path}"
        payload, blob, _ = original_load(path)
        return payload, blob, expected_hashes[resolved]

    monkeypatch.setattr(registry, "REGISTRY", local_registry)
    monkeypatch.setattr(registry, "SAVED_DIR", str(ckpt_dir))
    monkeypatch.setattr(registry, "PINS_PATH", str(pins_path))
    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(champion))
    monkeypatch.setattr(registry, "_sha256_file", fixture_digest)
    monkeypatch.setattr(session_module, "_load_checkpoint_once", load_fixture_checkpoint)
    monkeypatch.setattr(session_module, "served_resolution", lambda *a, **kw: None)
    monkeypatch.setattr(session_module, "watch_tick", lambda *a, **kw: None)
    return {"b8": b8, "frp3": frp3, "champion": champion}


def test_replay_opt_out_scoped_to_epsilon_and_food_while_play_and_new_game_resume(
    tmp_path, monkeypatch
):
    from web.backend import app as app_mod

    for key in (
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_WATCH_REPLAY_CONTROLLER",
    ):
        monkeypatch.delenv(key, raising=False)
    _install_local_release_fixtures(monkeypatch, tmp_path)

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    recording(bank_dir, "00_l2a", controller="exact-stop-l2-prime")
    recording(bank_dir, "01_l2b", controller="exact-stop-l2-prime")

    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "replay")
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_DIR", str(bank_dir))
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l2-prime")

    session = GameSession()
    try:
        assert session._replay_bank is not None
        assert session._is_clean_mixed_watch6_b8()
        deadline = time.monotonic() + 10
        while session._watch_replay is None and time.monotonic() < deadline:
            session.step()
            time.sleep(0.01)
        assert session._watch_replay is not None
        assert session.snapshot()["session"]["watch_source"] == "replay"

        # Rejected Train click in Watch mode must keep active Watch replay intact.
        active_before = session._watch_replay
        session.set_mode("train")
        assert session.mode == "watch"
        assert session._watch_replay is active_before

        # Malformed set_epsilon / set_food must not spuriously opt out of Watch replay.
        app_mod._apply_control(session, {"type": "control", "action": "set_epsilon", "value": "x"})
        assert session._watch_replay is active_before
        assert session._replay_opted_out is False
        app_mod._apply_control(session, {"type": "control", "action": "set_food", "value": "bad"})
        assert session._watch_replay is active_before
        assert session._replay_opted_out is False

        # Play -> Watch must NOT opt out of Watch replay.
        session.set_mode("play")
        assert session.mode == "play" and session._watch_replay is None
        assert session._replay_opted_out is False
        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        session.set_mode("watch")
        assert session.mode == "watch"
        assert session._watch_replay is not None

        # set_epsilon opts out and stays live across reset_game; new_game re-enables replay.
        session.set_epsilon(0.25)
        assert session._replay_opted_out is True
        assert session._watch_replay is None
        assert session.snapshot()["session"]["watch_source"] == "live"
        session.reset_game()
        assert session._watch_replay is None
        assert session._replay_opted_out is True
        assert session.snapshot()["session"]["watch_source"] == "live"

        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        app_mod._apply_control(session, {"type": "control", "action": "new_game"})
        assert session._replay_opted_out is False
        assert session._watch_replay is not None

        # set_food also opts out until _build or new_game clears _replay_opted_out.
        session.set_food(42)
        assert session._replay_opted_out is True
        assert session._watch_replay is None
        session.reset_game()
        assert session._watch_replay is None
        assert session.game.food_manager.max_food == 42

        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        session.set_mode("play")
        session.set_mode("watch")
        assert session._replay_opted_out is False
        assert session._watch_replay is not None
    finally:
        session.close()


def test_watch_replay_gates_on_clean_mixed_watch6_b8(tmp_path, monkeypatch):
    from dataclasses import replace

    import web.backend.session as session_module

    for key in (
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_WATCH_REPLAY_CONTROLLER",
    ):
        monkeypatch.delenv(key, raising=False)
    _install_local_release_fixtures(monkeypatch, tmp_path)

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    recording(bank_dir, "00_l2a", controller="exact-stop-l2-prime")
    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "replay")
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_DIR", str(bank_dir))

    # 1. Rollback checkpoint (frp3-s12) must bypass ReplayBank.
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "frp3-s12")
    frp_session = GameSession()
    try:
        assert not frp_session._is_clean_mixed_watch6_b8()
        assert frp_session._replay_bank is None
        assert frp_session._watch_replay is None
    finally:
        frp_session.close()
    monkeypatch.delenv("SNAKE_SERVE_CHECKPOINT")

    # 2. All-B8 roster rollback must bypass ReplayBank.
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "all_b8")
    all_b8_session = GameSession()
    try:
        assert not all_b8_session._is_clean_mixed_watch6_b8()
        assert all_b8_session._replay_bank is None
        assert all_b8_session._watch_replay is None
    finally:
        all_b8_session.close()
    monkeypatch.delenv("SNAKE_SERVE_WATCH_ROSTER")

    # 3. Non-Watch6 roster count (Watch8) must bypass ReplayBank.
    original_loader = session_module.load_config
    monkeypatch.setattr(
        session_module,
        "load_config",
        lambda path: replace(
            original_loader(path),
            game=replace(original_loader(path).game, num_snakes=8),
        ),
    )
    watch8_session = GameSession()
    try:
        assert len(watch8_session.game.snakes) == 8
        assert not watch8_session._is_clean_mixed_watch6_b8()
        assert watch8_session._replay_bank is None
        assert watch8_session._watch_replay is None
    finally:
        watch8_session.close()


def test_qualify_bank_header_scan_and_reverse_filter_proof(tmp_path):
    from research.watch_replay.qualify import (
        prove_line1_filter_skips_l1_payloads,
        scan_bank_headers,
    )

    recording(tmp_path, "000000", unit_id="u0", controller="exact-stop-l2-prime", seed=101)
    recording(tmp_path, "000001", unit_id="u1", controller="exact-stop-l2-prime", seed=102)
    recording(tmp_path, "000002", unit_id="u2", controller="exact-stop-l1", seed=201)

    scan = scan_bank_headers(tmp_path)
    assert scan["total_files"] == 3
    assert scan["by_controller"] == {"exact-stop-l1": 1, "exact-stop-l2-prime": 2}
    proof = prove_line1_filter_skips_l1_payloads(tmp_path)
    assert proof["l1_headers_peeked_and_skipped_before_first_pick"] == 1
    assert proof["non_target_verify_payload_calls"] == 0
    assert proof["first_picked_controller"] == "exact-stop-l2-prime"


def test_set_hero_during_live_fallback_opts_out_and_cancels_pending_replay(tmp_path, monkeypatch):
    import web.backend.watch_replay as replay
    from web.backend import app as app_mod

    for key in (
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_WATCH_REPLAY_CONTROLLER",
    ):
        monkeypatch.delenv(key, raising=False)
    _install_local_release_fixtures(monkeypatch, tmp_path)

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    recording(bank_dir, "00_l2a", controller="exact-stop-l2-prime")
    recording(bank_dir, "01_l2b", controller="exact-stop-l2-prime")

    entered, release = threading.Event(), threading.Event()
    original_recording = replay.Recording

    def gated_recording(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_recording(*args, **kwargs)

    monkeypatch.setattr(replay, "Recording", gated_recording)
    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "replay")
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_DIR", str(bank_dir))
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l2-prime")

    session = GameSession()
    try:
        assert entered.wait(5)
        assert session._watch_replay is None
        assert session._replay_pending is True
        assert session._replay_opted_out is False

        # Malformed, same-hero, and unknown-id requests must not opt out or drop pending replay.
        app_mod._apply_control(session, {"type": "control", "action": "set_hero", "value": None})
        assert session.last_error is not None
        session.last_error = None
        session.set_hero(0)
        session.set_hero(99)
        assert session.hero_id == 0
        assert session._replay_opted_out is False
        assert session._replay_pending is True

        # Re-targeting the hero during live fallback opts out and cancels pending replay.
        session.set_hero(1)
        assert session.hero_id == 1
        assert session._replay_opted_out is True
        assert session._replay_pending is False
        assert session._watch_replay is None

        release.set()
        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not session._replay_bank._ready.empty()

        # Subsequent ticks must keep live Watch serving hero #1 instead of cutting in a recording.
        session.step()
        session.step()
        assert session._watch_replay is None
        assert session._replay_pending is False
        assert session.hero_id == 1
        assert session.snapshot()["session"]["watch_source"] == "live"

        # Returning to hero #0 stays opted out across reset_game until explicit new_game.
        session.set_hero(0)
        assert session.hero_id == 0
        assert session._replay_opted_out is True
        session.reset_game()
        assert session._watch_replay is None
        assert session._replay_opted_out is True

        session.new_game()
        assert session._replay_opted_out is False
        assert session._watch_replay is not None
        assert session.snapshot()["session"]["watch_source"] == "replay"
    finally:
        release.set()
        session.close()


def test_set_mode_exception_in_watch_reselects_replay_under_clean_gate(tmp_path, monkeypatch):
    for key in (
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_WATCH_REPLAY_CONTROLLER",
    ):
        monkeypatch.delenv(key, raising=False)
    _install_local_release_fixtures(monkeypatch, tmp_path)

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    for name in ("00_l2a", "01_l2b", "02_l2c"):
        recording(bank_dir, name, controller="exact-stop-l2-prime")

    monkeypatch.setenv("SNAKE_WATCH_SOURCE", "replay")
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_DIR", str(bank_dir))
    monkeypatch.setenv("SNAKE_WATCH_REPLAY_CONTROLLER", "exact-stop-l2-prime")

    session = GameSession()
    try:
        deadline = time.monotonic() + 10
        while (
            session._watch_replay is None or session._replay_bank._ready.empty()
        ) and time.monotonic() < deadline:
            session.step()
            time.sleep(0.01)
        assert session._watch_replay is not None
        assert not session._replay_bank._ready.empty()

        original_build_impl = session._build_impl
        mutate_mode_before_fail = False

        def failing_play_build(checkpoint, mode, override_reward_contract=False):
            if mode == "play":
                if mutate_mode_before_fail:
                    session.mode = "play"
                    raise RuntimeError("late play build boom")
                raise RuntimeError("play build boom")
            return original_build_impl(checkpoint, mode, override_reward_contract)

        monkeypatch.setattr(session, "_build_impl", failing_play_build)

        # 1. Clean Watch6 b8-s3: failed set_mode("play") re-selects replay and keeps error.
        first_replay = session._watch_replay
        session.set_mode("play")
        assert session.mode == "watch"
        assert session._watch_replay is not None
        assert session._watch_replay is not first_replay
        assert session.last_error == "Cannot enter play mode: play build boom"
        assert session.last_error_overridable is False
        snap = session.snapshot()
        assert snap["session"]["watch_source"] == "replay"
        assert snap["session"]["error"] == "Cannot enter play mode: play build boom"

        # 2. Opted-out Watch: failed set_mode("play") preserves _replay_opted_out (stays live).
        session.set_epsilon(0.2)
        assert session._replay_opted_out is True
        assert session._watch_replay is None
        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        session.set_mode("play")
        assert session.mode == "watch"
        assert session._replay_opted_out is True
        assert session._watch_replay is None
        assert session._replay_pending is False
        assert session.last_error == "Cannot enter play mode: play build boom"

        # 3. Unclean Watch gate (watch_roster != mixed): does not re-select replay.
        session.new_game()
        assert session._watch_replay is not None
        deadline = time.monotonic() + 10
        while session._replay_bank._ready.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        session.watch_roster = "all_b8"
        assert not session._is_clean_mixed_watch6_b8()
        session.set_mode("play")
        assert session.mode == "watch"
        assert session._watch_replay is None
        assert session._replay_pending is False
        assert session.last_error == "Cannot enter play mode: play build boom"

        # 4. Post-mutation failure falling back via _build(..., mode=MODE_WATCH) also re-selects.
        mutate_mode_before_fail = True
        session.set_mode("play")
        assert session.mode == "watch"
        assert session._is_clean_mixed_watch6_b8()
        assert session._watch_replay is not None
        assert session.last_error == "Cannot enter play mode: late play build boom"
    finally:
        session.close()


def test_verify_payload_and_recording_step_do_not_promote_frame_containers_to_older_gc_generations(
    tmp_path,
):
    import gc

    import web.backend.watch_replay as replay

    saved_thresholds = gc.get_threshold()
    saved_enabled = gc.isenabled()
    gc_events: list[int] = []

    def on_gc(phase: str, info: dict) -> None:
        if phase == "start":
            gc_events.append(int(info["generation"]))

    gc.callbacks.append(on_gc)
    try:
        gc.enable()
        gc.set_threshold(700, 10, 10)
        replay._ensure_frame_gc_thresholds()
        t0, t1, t2 = gc.get_threshold()
        assert t0 >= replay.GC_GEN0_THRESHOLD
        assert t1 >= replay.GC_GEN1_THRESHOLD
        assert t2 >= replay.GC_GEN2_THRESHOLD

        # Nested/re-entrant suppression must restore prior enabled/disabled state accurately.
        replay._enter_gc_suppress()
        replay._enter_gc_suppress()
        assert not gc.isenabled()
        replay._exit_gc_suppress()
        assert not gc.isenabled()
        replay._exit_gc_suppress()
        assert gc.isenabled()

        gc.disable()
        replay._enter_gc_suppress()
        replay._exit_gc_suppress()
        assert not gc.isenabled()
        gc.enable()

        # Build a 64-frame heavy payload where each frame allocates >1,800 [x, y] lists
        # (>2x CPython's default 700 Gen-0 threshold per frame; >115,000 lists total).
        horizon = 64
        header = {
            "schema": SCHEMA,
            "controller": "exact-stop-l1",
            "horizon": horizon,
            "world": "watch6-mechanics-v2-continuous-respawn",
            "protocol_version": 2,
            "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
            "seed": 77,
            "unit_id": "heavy_gc_unit",
            "fixture": True,
            "stats": {},
        }
        heavy_frames = []
        for idx in range(1, horizon + 1):
            dto = frame(idx)
            for s_idx, snake in enumerate(dto["snakes"]):
                snake["segments"] = [[s_idx * 10 + k, (idx + k) % 500] for k in range(250)]
                snake["length"] = 250
            dto["food"] = [[k, (idx + k) % 500] for k in range(350)]
            heavy_frames.append(canonical(dto))
        payload = gzip.compress(canonical(header) + b"".join(heavy_frames), mtime=0)
        path = tmp_path / "heavy_gc_unit.watch.gz"
        publish_immutable(path, payload)

        gc.collect()
        gc_events.clear()
        g0_before_verify, g1_before_verify, g2_before_verify = gc.get_count()

        stop = threading.Event()
        verified_header = replay.verify_payload(payload, product=False, stop=stop)
        assert verified_header["unit_id"] == "heavy_gc_unit"
        assert gc.isenabled()
        g0_after_verify, g1_after_verify, g2_after_verify = gc.get_count()
        assert [g for g in gc_events if g >= 1] == []
        assert g1_after_verify == g1_before_verify
        assert g2_after_verify == g2_before_verify
        assert abs(g0_after_verify - g0_before_verify) < 500

        # Stepping through the entire Recording must also release each prior frame in Gen 0
        # without promoting transient frame containers into Gen 1 / Gen 2.
        rec = Recording(path, product=False)
        g0_frame1 = gc.get_count()[0]
        steps = 0
        while rec.step():
            steps += 1
        assert steps == horizon - 1
        assert rec.frame["frame"] == horizon
        assert gc.isenabled()
        g0_end, g1_end, g2_end = gc.get_count()
        assert [g for g in gc_events if g >= 1] == []
        assert g1_end == g1_before_verify
        assert g2_end == g2_before_verify
        assert abs(g0_end - g0_frame1) < 500

        # Error paths inside verify_payload must always restore GC state.
        bad_dto = frame(1)
        bad_dto["snakes"][0]["length"] = -1
        bad_payload = gzip.compress(canonical(header) + canonical(bad_dto), mtime=0)
        with pytest.raises(ValueError, match="invalid recorded snake"):
            replay.verify_payload(bad_payload, product=False)
        assert gc.isenabled()
    finally:
        if on_gc in gc.callbacks:
            gc.callbacks.remove(on_gc)
        gc.set_threshold(*saved_thresholds)
        if saved_enabled:
            gc.enable()
        else:
            gc.disable()
