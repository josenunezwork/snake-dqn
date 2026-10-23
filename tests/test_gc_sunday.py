"""Untimed Sunday protocol/reducer checks; never start a real server."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from research.bl_gc import sunday_regression as sunday


def _state(policy: str, count: int, checkpoint: str | None = None) -> dict:
    ceiling = 120.0 if checkpoint else sunday.CEILINGS[count]
    return {
        "error": None,
        "mode": "watch",
        "gc_policy": policy,
        "num_snakes": count,
        "checkpoint": (
            "champion_a5_freespace_20260621.pth" if checkpoint else "b8_s3_t7_20261016.pth"
        ),
        "speed": ceiling,
        "speed_ceiling": ceiling,
        "watch_roster": "mixed",
        "playing": True,
    }


def test_matched_gc_uses_previous_drain_window() -> None:
    row = {
        "client": {"gaps": [{"start": 10.04, "end": 10.10, "gap_ms": 60.0}]},
        "watch_ticks": [
            {"timestamp": 10.0, "gc_gen2_pause_ms": 0.0},
            {"timestamp": 10.15, "gc_gen2_pause_ms": 5.0},
        ],
    }
    assert sunday._recompute_matches(row)
    row["watch_ticks"][1]["gc_gen2_pause_ms"] = 0.0
    assert not sunday._recompute_matches(row)


def test_exact_watch_ceilings_and_champion_rollback() -> None:
    for count in (6, 8, 12):
        sunday._validate_state(_state("freeze_startup", count), "freeze_startup", count, None)
    sunday._validate_state(_state("default", 6, "champion"), "default", 6, "champion")
    state = _state("freeze_startup", 6)
    state["speed"] = 12
    with pytest.raises(RuntimeError):
        sunday._validate_state(state, "freeze_startup", 6, None)


def test_raw_run_cli_does_not_look_or_require_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from research import mac_lease

    @contextmanager
    def lease(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test"})

    published = []
    monkeypatch.setattr(mac_lease, "held", lease)
    monkeypatch.setattr(sunday, "run", lambda *args: {"status": "COMPLETE_ONE_LOOK_PENDING"})
    monkeypatch.setattr(
        sunday.common, "write_result", lambda path, result: published.append(result)
    )
    assert sunday.main(["--run", "--out", str(tmp_path)]) == 0
    assert "verdict" not in published[0]


def test_incomplete_sunday_receipt_cannot_pass() -> None:
    with pytest.raises(ValueError):
        sunday.check({"schema": sunday.SCHEMA + "/raw"})


def test_websocket_control_waits_for_ack_amid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    import websockets

    # No host clock or server is used: provide the protocol frames directly.
    class FakeSocket:
        def __init__(self):
            self.messages = [{"type": "frame", "frame": 5}]
            self.closed = False
            self.action = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def recv(self):
            return json.dumps(self.messages.pop(0))

        async def send(self, raw):
            action = json.loads(raw)["action"]
            state = _state("default", 6)
            state["playing"] = action != "pause"
            self.messages.extend([{"type": "frame", "frame": 7}, {"type": "ack", "state": state}])
            if action == "pause":
                self.messages.extend([{"type": "frame", "frame": 7, "paused": True}] * 3)
            elif action == "play":
                self.messages.extend([{"type": "frame", "frame": n} for n in (8, 9, 10)])
            elif action == "reset":
                self.messages.append({"type": "frame", "frame": 0, "checkpoint_name": "b8"})

    sockets = [FakeSocket(), FakeSocket()]
    sockets[1].messages = [{"type": "frame", "frame": 0, "checkpoint_name": "b8"}]
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: sockets.pop(0))
    result = asyncio.run(sunday._client(12345, 0, "default", 6, None))
    assert [c["action"] for c in result["control_states"]] == [
        "set_speed",
        "pause",
        "play",
        "reset",
    ]
    assert result["reconnected"] and result["reset_observed"]


def test_sunday_incomplete_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from research import mac_lease

    @contextmanager
    def lease(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test"})

    monkeypatch.setattr(mac_lease, "held", lease)
    monkeypatch.setattr(sunday, "run", lambda *args: {"status": "INCOMPLETE"})
    monkeypatch.setattr(sunday.common, "write_result", lambda *args: None)
    assert sunday.main(["--run", "--out", str(tmp_path)]) == 2
