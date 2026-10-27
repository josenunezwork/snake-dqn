"""Focused checks for the isolated real-Hub production trace arm."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_served_search import production_probe as probe


def test_settings_only_allows_frozen_or_smaller_smoke() -> None:
    assert probe._settings({"seed": 2, "warmup_ticks": 0, "measured_ticks": 2}) == (2, 0, 2, 24)
    with pytest.raises(ValueError, match="frozen"):
        probe._settings({"seed": 2, "fps": 12})
    with pytest.raises(ValueError, match="frozen"):
        probe._settings({"seed": 2, "measured_ticks": probe.MEASURED_TICKS + 1})


def test_frame_id_preserves_product_payload_shape() -> None:
    assert probe._frame_id({"frame": 7}) == 7
    assert probe._frame_id({"game": {"frame": 8}}) == 8
    assert probe._frame_id({"frame": "8"}) is None


def test_merge_span_preserves_snapshot_lock_breakdown() -> None:
    left = {
        "calls": 1,
        "wall_ns": 3,
        "wall_exclusive_ns": 2,
        "cpu_ns": 4,
        "cpu_exclusive_ns": 3,
        "first_wall_start_ns": 20,
        "last_wall_end_ns": 30,
        "first_cpu_start_ns": 100,
        "last_cpu_end_ns": 104,
    }
    right = {**left, "calls": 2, "wall_ns": 5, "first_wall_start_ns": 10, "last_wall_end_ns": 40}
    merged = probe._merge_span(left, right)
    assert merged["calls"] == 3
    assert merged["wall_ns"] == 8
    assert merged["wall_exclusive_ns"] == 4
    assert merged["cpu_ns"] == 8
    assert merged["cpu_exclusive_ns"] == 6
    assert "thread_cpu_ns" not in merged
    assert "first_cpu_start_ns" not in merged
    assert "last_cpu_end_ns" not in merged
    assert merged["cpu_clock_kind"] == "aggregate_independent_thread_durations"
    assert merged["components"] == {"step": left, "snapshot": right}
    assert merged["first_wall_start_ns"] == 10
    assert merged["last_wall_end_ns"] == 40


def test_client_retries_connect_only_and_refuses_midstream_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that closes after admission is evidence, never a new session."""

    class EarlyClose(Exception):
        pass

    class Socket:
        def __init__(self) -> None:
            self.received = False

        async def recv(self) -> str:
            if not self.received:
                self.received = True
                return '{"frame": 0}'
            raise EarlyClose()

        async def send(self, _: str) -> None:
            return None

        async def close(self) -> None:
            return None

    attempts = 0

    async def connect(*_: object, **__: object) -> Socket:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("server still starting")
        return Socket()

    monkeypatch.setitem(
        sys.modules, "websockets", SimpleNamespace(connect=connect, ConnectionClosed=EarlyClose)
    )
    with pytest.raises(RuntimeError, match="closed before all engine frames"):
        asyncio.run(probe._collect_client(12345, wanted_frames=2, timeout_seconds=0.2))
    assert attempts == 2


def test_client_receive_timeout_fails_without_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connected socket may not be replaced after its receive deadline expires."""

    class EarlyClose(Exception):
        pass

    class Socket:
        async def recv(self) -> str:
            await asyncio.Event().wait()
            return "unreachable"

        async def send(self, _: str) -> None:
            return None

        async def close(self) -> None:
            return None

    attempts = 0

    async def connect(*_: object, **__: object) -> Socket:
        nonlocal attempts
        attempts += 1
        return Socket()

    monkeypatch.setitem(
        sys.modules, "websockets", SimpleNamespace(connect=connect, ConnectionClosed=EarlyClose)
    )
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(probe._collect_client(12345, wanted_frames=1, timeout_seconds=0.02))
    assert attempts == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not all(
        Path("saved_snakes", name).is_file()
        for name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth")
    ),
    reason="actual Hub smoke requires the local released checkpoint artifacts",
)
def test_smoke_runs_actual_hub_and_loopback_websocket() -> None:
    result = probe.run_production_arm(
        {"arm": "production", "seed": 2026102601, "warmup_ticks": 0, "measured_ticks": 2},
        timeout_seconds=120,
    )
    assert result["clock_kind"] == "hub_relative_sleep_not_absolute"
    assert result["absolute_overrun_applicable"] is False
    assert len(result["trace"]) == 2
    assert all(row["spans"]["game_update"]["calls"] == 1 for row in result["trace"])
    assert all("ipc_lock" in row["spans"] for row in result["trace"])
    assert all(len(row["gc_overlap_ns"]) == 3 for row in result["trace"])
    assert len(result["frame_send"]) == 2
    assert result["engine_error"] is None
    assert sum(row["source"] == "connect_snapshot" for row in result["client_receive"]) == 1
    assert sum(row["source"] == "hub_broadcast" for row in result["client_receive"]) == 2


def test_client_startup_timeout_is_independent_of_whole_panel_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    attempts = 0

    async def connect(*_: object, **__: object) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("startup refused")

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    monkeypatch.setattr(probe, "STARTUP_TIMEOUT_SECONDS", 0.02)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="could not connect"):
        asyncio.run(probe._collect_client(12345, wanted_frames=1, timeout_seconds=7200))
    assert attempts >= 1
    assert time.monotonic() - started < 1
