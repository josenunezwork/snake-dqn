"""Focused evidence-contract tests for the paced identity journal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_paced_identity.events import (
    CandidateIdentity,
    Event,
    EventValidationError,
    Request,
)
from research.bl_paced_identity.record import (
    Journal,
    JournalError,
    JournalOverflow,
    export_json,
    read_export,
    read_journal,
)

SHA = "a" * 64


def test_request_and_identity_bind_all_delivery_fields() -> None:
    request = Request("session-1", 2, 12, 16, SHA, 3, SHA, "request-1", SHA, 999, 1000)
    assert request.as_dict()["target_frame"] == 16
    warmup = {"frames": [32]}
    identity = CandidateIdentity(
        SHA,
        "edf",
        [32, 96, 192, 300],
        "b8-q-index",
        1,
        1,
        "default",
        warmup,
        "bare",
        "v1",
        {"profile": "fixture"},
        SHA,
    )
    warmup["frames"].append(96)
    assert identity.warmup == {"frames": [32]}
    assert identity.authorized()
    multi = CandidateIdentity(
        SHA,
        "edf",
        300,
        "b8-q-index",
        4,
        8,
        "default",
        {"frames": 1},
        "bare",
        "v1",
        {"profile": "fixture"},
        SHA,
    )
    assert not multi.authorized()
    assert multi.authorized(owner_multi_worker=True)
    with pytest.raises(EventValidationError):
        multi.require_authorized()
    with pytest.raises(EventValidationError):
        CandidateIdentity(SHA, "edf", 300, "tie", 4, 3, "q", {"frames": 1}, "bare", "v1", {}, SHA)
    with pytest.raises(EventValidationError):
        Request("session", 0, 3, 2, SHA, 0, SHA, "id", SHA, 1, 2)


def test_event_from_dict_is_strict_and_finite() -> None:
    assert Event.from_dict({"seq": 0, "time_ns": 4, "kind": "clock_read", "data": {}}).seq == 0
    with pytest.raises(EventValidationError):
        Event.from_dict({"seq": 0, "time_ns": 4, "kind": "clock_read", "data": {}, "extra": 1})
    with pytest.raises(EventValidationError):
        Event(0, 0, "bad", {"x": float("nan")})


def test_journal_round_trip_hash_chain_and_sidecars(tmp_path: Path) -> None:
    identity = {"candidate": {"sha": SHA}}
    state = {"frame": [11]}
    journal = Journal(tmp_path / "journal", identity, state, capacity=3, chunk_size=2)
    emitted = journal.emit(10, "submit", request_id="a", nested={"steps": [1]})
    identity["candidate"]["sha"] = "b" * 64
    state["frame"].append(12)
    emitted.data["nested"]["steps"].append(2)
    journal.emit(10, "visible", request_id="a")
    assert len(journal.drain()) == 1
    journal.emit(12, "consumed", request_id="a", action=2)
    seal = journal.seal()
    header, events, read_seal = read_journal(tmp_path / "journal")
    assert header["identity"] == {"candidate": {"sha": SHA}}
    assert header["initial_state"] == {"frame": [11]}
    assert [event.kind for event in events] == ["submit", "visible", "consumed"]
    assert events[0].data["nested"] == {"steps": [1]}
    assert read_seal == seal
    for path in (tmp_path / "journal").glob("*.json"):
        assert Path(str(path) + ".complete").read_text() == "complete\n"


def test_journal_binds_frozen_manifest_digest_in_header_chain(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "bound", {}, {}, binding_sha256=SHA)
    journal.emit(1, "submit")
    journal.seal()
    header, _, _ = read_journal(tmp_path / "bound")
    assert header["binding_sha256"] == SHA
    with pytest.raises(JournalError, match="binding_sha256"):
        Journal(tmp_path / "invalid-binding", {}, {}, binding_sha256="not-a-sha")


def test_journal_fails_closed_on_overflow_and_backwards_time(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "overflow", {}, {}, capacity=1, chunk_size=1)
    journal.emit(9, "submit")
    with pytest.raises(JournalOverflow):
        journal.emit(10, "result_visible")
    with pytest.raises(JournalError, match="invalid"):
        journal.drain()
    timed = Journal(tmp_path / "time", {}, {}, capacity=2, chunk_size=1)
    timed.emit(8, "submit")
    with pytest.raises(JournalError, match="backwards"):
        timed.emit(7, "result_visible")
    with pytest.raises(JournalError, match="invalid"):
        timed.seal()
    malformed = Journal(tmp_path / "malformed", {}, {}, capacity=2, chunk_size=1)
    with pytest.raises(JournalError, match="finite canonical JSON"):
        malformed.emit(1, "submit", impossible={"nan": float("nan")})
    with pytest.raises(JournalError, match="invalid"):
        malformed.seal()


def test_reader_rejects_corruption_missing_or_out_of_order_chunks(tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    journal = Journal(root, {}, {}, capacity=3, chunk_size=1)
    journal.emit(1, "submit")
    journal.drain()
    journal.emit(2, "consumed")
    journal.seal()
    chunk = root / "chunk-00000000.json"
    value = json.loads(chunk.read_text())
    value["events"][0]["seq"] = 9
    chunk.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(JournalError):
        read_journal(root)

    missing = tmp_path / "missing"
    journal = Journal(missing, {}, {}, capacity=2, chunk_size=1)
    journal.emit(1, "submit")
    journal.seal()
    (missing / "chunk-00000000.json").unlink()
    with pytest.raises(JournalError, match="missing"):
        read_journal(missing)

    reordered = tmp_path / "reordered"
    journal = Journal(reordered, {}, {}, capacity=2, chunk_size=1)
    journal.emit(1, "submit")
    journal.emit(2, "consumed")
    journal.seal()
    first = reordered / "chunk-00000000.json"
    second = reordered / "chunk-00000001.json"
    first.rename(reordered / "chunk-00000002.json")
    second.rename(first)
    with pytest.raises(JournalError):
        read_journal(reordered)


def test_exports_are_create_only_and_verified(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    digest = export_json(path, {"z": 1, "a": 2})
    assert len(digest) == 64
    assert read_export(path) == {"a": 2, "z": 1}
    with pytest.raises(FileExistsError):
        export_json(path, {})
    Path(str(path) + ".complete").write_text("partial\n")
    with pytest.raises(JournalError):
        read_export(path)
