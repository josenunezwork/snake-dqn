"""Hermetic public-release input admission checks."""

from __future__ import annotations

import hashlib
import signal
import time

import pytest

from research.colab_cli import inputs, supervisor


class _Reply:
    def __init__(self, payload: bytes) -> None:
        self.payload, self.offset = payload, 0

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        value = self.payload[self.offset :]
        self.offset = len(self.payload)
        return value


def test_public_preflight_returns_verified_measurement() -> None:
    payload = b"release-input"
    row = {
        "url": "https://example.test/release/input.bin",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "dest": "input.bin",
    }
    result = inputs.preflight([row], opener=lambda *_args, **_kwargs: _Reply(payload))
    assert result[0]["sha256"] == row["sha256"]
    assert result[0]["bytes_per_second"] > 0


def test_public_preflight_refuses_hash_mismatch_before_allocation() -> None:
    with pytest.raises(inputs.InputError, match="mismatch"):
        inputs.preflight(
            [
                {
                    "url": "https://example.test/release/input.bin",
                    "sha256": "0" * 64,
                    "size_bytes": 3,
                }
            ],
            opener=lambda *_args, **_kwargs: _Reply(b"bad"),
        )


def test_contents_only_input_preflight_does_not_require_http_timer() -> None:
    assert supervisor.input_preflight({"inputs": [{"path": "/unused", "dest": "input"}]}) == []


def test_mac_total_deadline_interrupts_blocked_read_and_restores_alarm() -> None:
    class BlockedReply(_Reply):
        def read(self, _size: int) -> bytes:
            time.sleep(1)
            return b""

    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(inputs.InputError, match="total timeout"):
        inputs.preflight(
            [
                {
                    "url": "https://example.test/release/input.bin",
                    "sha256": "0" * 64,
                    "size_bytes": 3,
                }
            ],
            opener=lambda *_args, **_kwargs: BlockedReply(b""),
            total_timeout_seconds=0.02,
        )
    assert time.monotonic() - started < 0.5
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
