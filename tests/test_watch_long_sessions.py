"""Tests for opt-in H=14,400 long Watch sessions and the Mac engineering gate tooling."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools.watch_long_sessions import (
    PILOT_CHECKPOINT_FRAMES,
    PILOT_COUNT,
    PILOT_HORIZON,
    PILOT_MAX_SESSION_BYTES,
    PILOT_NAMESPACE,
    audit_pilot_seeds,
    build_pilot_package_spec,
    install_long_bank,
    verify_bank_directory,
    write_synthetic_session,
)
from web.backend import watch_replay
from web.backend.watch_replay import (
    DEFAULT_HORIZON,
    DEFAULT_HORIZONS,
    ENV_REPLAY_HORIZON,
    ENV_REPLAY_LONG_DIR,
    ENV_REPLAY_LONG_SESSIONS,
    LONG_HORIZON,
    PRODUCT_HORIZONS,
    Recording,
    ReplayBank,
    peek_header,
    resolve_replay_horizons,
    validate_header,
    verify_payload,
)


def _pick_ready(bank: ReplayBank, timeout_s: float = 15.0) -> Recording | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rec = bank.pick()
        if rec is not None or not bank.pending:
            return rec
        time.sleep(0.005)
    raise AssertionError("replay prefetch timed out")


@pytest.fixture(autouse=True)
def _fast_cooperative_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cooperative yield active while avoiding multi-second sleeps in unit tests."""
    monkeypatch.setattr(watch_replay, "COOPERATIVE_YIELD_MIN_WAIT_SECONDS", 0.00001)
    monkeypatch.setattr(watch_replay, "COOPERATIVE_YIELD_DUTY_FACTOR", 0.01)


def test_resolve_replay_horizons_defaults_and_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_REPLAY_LONG_SESSIONS, raising=False)
    monkeypatch.delenv(ENV_REPLAY_HORIZON, raising=False)

    assert resolve_replay_horizons() == (DEFAULT_HORIZONS, False, None)
    assert resolve_replay_horizons(allow_long=False) == (DEFAULT_HORIZONS, False, None)
    assert resolve_replay_horizons(allow_long=True) == (PRODUCT_HORIZONS, True, None)
    assert resolve_replay_horizons(horizons=14400) == (frozenset({LONG_HORIZON}), True, None)
    assert resolve_replay_horizons(horizons="6000,14400") == (PRODUCT_HORIZONS, True, None)

    monkeypatch.setenv(ENV_REPLAY_LONG_SESSIONS, "1")
    assert resolve_replay_horizons() == (PRODUCT_HORIZONS, True, None)

    monkeypatch.setenv(ENV_REPLAY_LONG_SESSIONS, "0")
    assert resolve_replay_horizons() == (DEFAULT_HORIZONS, False, None)

    # Conflicting SNAKE_WATCH_LONG_SESSIONS=0 + SNAKE_WATCH_REPLAY_HORIZON=14400 fails closed.
    monkeypatch.setenv(ENV_REPLAY_HORIZON, "14400")
    h_conflict, long_conflict, err_conflict = resolve_replay_horizons()
    assert h_conflict is None and long_conflict is False and err_conflict is not None

    monkeypatch.delenv(ENV_REPLAY_LONG_SESSIONS, raising=False)
    monkeypatch.setenv(ENV_REPLAY_HORIZON, "14400")
    assert resolve_replay_horizons() == (frozenset({LONG_HORIZON}), True, None)

    monkeypatch.setenv(ENV_REPLAY_HORIZON, "long")
    assert resolve_replay_horizons() == (frozenset({LONG_HORIZON}), True, None)

    monkeypatch.setenv(ENV_REPLAY_HORIZON, "all")
    assert resolve_replay_horizons() == (PRODUCT_HORIZONS, True, None)


def test_resolve_replay_horizons_fails_closed_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_REPLAY_HORIZON, raising=False)
    monkeypatch.setenv(ENV_REPLAY_LONG_SESSIONS, "sometimes")
    horizons, allow_long, reason = resolve_replay_horizons()
    assert horizons is None
    assert allow_long is False
    assert reason is not None and ENV_REPLAY_LONG_SESSIONS in reason

    monkeypatch.delenv(ENV_REPLAY_LONG_SESSIONS, raising=False)
    monkeypatch.setenv(ENV_REPLAY_HORIZON, "7200")
    horizons2, allow_long2, reason2 = resolve_replay_horizons()
    assert horizons2 is None
    assert allow_long2 is False
    assert reason2 is not None and ENV_REPLAY_HORIZON in reason2


def test_header_and_payload_validation_default_vs_long(tmp_path: Path) -> None:
    write_synthetic_session(
        tmp_path / "long",
        namespace=PILOT_NAMESPACE,
        count=1,
        index=0,
        horizon=LONG_HORIZON,
        snake_lengths=(20, 15, 12, 10, 10, 10),
        unique_frames=4,
    )
    long_path = tmp_path / "long" / "000000.watch.gz"
    raw = long_path.read_bytes()

    # Default (allow_long=False) must reject H=14,400 product recordings.
    with pytest.raises(ValueError, match="foreign replay metadata"):
        peek_header(long_path, product=True)
    with pytest.raises(ValueError, match="foreign replay metadata"):
        verify_payload(raw, product=True)
    with pytest.raises(ValueError, match="foreign replay metadata"):
        Recording(long_path, product=True)

    # Opt-in allow_long=True accepts H=14,400 product recordings.
    header = peek_header(long_path, product=True, allow_long=True)
    assert header["horizon"] == LONG_HORIZON
    assert header["fixture"] is False
    validate_header(header, product=True, allow_long=True)

    verified = verify_payload(raw, product=True, allow_long=True)
    assert verified["horizon"] == LONG_HORIZON

    rec = Recording(long_path, product=True, allow_long=True)
    try:
        assert rec.header["horizon"] == LONG_HORIZON
        assert rec.frame["frame"] == 1
    finally:
        rec.close()


def test_replay_bank_skips_unselected_long_session_on_line_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mixed_dir = tmp_path / "mixed"
    write_synthetic_session(
        mixed_dir,
        namespace="test-mixed-short-v1",
        count=2,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(12, 10, 8, 8, 8, 8),
        unique_frames=2,
    )
    write_synthetic_session(
        mixed_dir,
        namespace="test-mixed-long-v1",
        count=2,
        index=1,
        horizon=LONG_HORIZON,
        snake_lengths=(12, 10, 8, 8, 8, 8),
        unique_frames=2,
    )
    short_file = mixed_dir / "000000.watch.gz"
    long_file = mixed_dir / "000001.watch.gz"

    assert watch_replay._is_unselected_long_header(long_file) is True
    assert watch_replay._is_unselected_long_header(short_file) is False

    # Default ReplayBank serves only H=6,000, skips the H=14,400 file on line 1
    # without calling verify_payload on it, and logs no warning.
    verify_spy = Mock(wraps=watch_replay.verify_payload)
    monkeypatch.setattr(watch_replay, "verify_payload", verify_spy)
    bank_default = ReplayBank(mixed_dir, shuffle_discovery=False)
    try:
        rec_default = _pick_ready(bank_default)
        assert rec_default is not None
        try:
            assert rec_default.header["horizon"] == DEFAULT_HORIZON
            assert _pick_ready(bank_default) is None
        finally:
            rec_default.close()
        assert verify_spy.call_count == 1
        assert "Skipping replay" not in caplog.text
    finally:
        bank_default.close()

    # Opt-in long-only ReplayBank serves only the H=14,400 recording.
    bank_long = ReplayBank(
        mixed_dir,
        shuffle_discovery=False,
        horizons=frozenset({LONG_HORIZON}),
    )
    try:
        rec_long = _pick_ready(bank_long)
        assert rec_long is not None
        try:
            assert rec_long.header["horizon"] == LONG_HORIZON
        finally:
            rec_long.close()
    finally:
        bank_long.close()


def test_replay_bank_long_root_and_env_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_dir = tmp_path / "default_bank"
    long_dir = tmp_path / "long_bank"
    write_synthetic_session(
        default_dir,
        namespace="test-default-bank-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=2,
    )
    write_synthetic_session(
        long_dir,
        namespace="test-long-bank-v1",
        count=1,
        index=0,
        horizon=LONG_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=2,
    )

    monkeypatch.setenv(ENV_REPLAY_LONG_DIR, str(long_dir))
    monkeypatch.setenv(ENV_REPLAY_HORIZON, "14400")

    bank = ReplayBank(default_dir, shuffle_discovery=False)
    try:
        rec = _pick_ready(bank)
        assert rec is not None
        try:
            assert rec.header["horizon"] == LONG_HORIZON
            assert rec.path.resolve().is_relative_to(long_dir.resolve())
        finally:
            rec.close()
    finally:
        bank.close()


def test_verify_and_install_long_bank_tooling(tmp_path: Path) -> None:
    src_dir = tmp_path / "src_long"
    for idx in range(2):
        write_synthetic_session(
            src_dir,
            namespace="test-install-long-v1",
            count=2,
            index=idx,
            horizon=LONG_HORIZON,
            snake_lengths=(12, 10, 8, 8, 8, 8),
            unique_frames=2,
        )

    summary = verify_bank_directory(
        src_dir,
        expected_horizon=LONG_HORIZON,
        allow_long=True,
        full_payload=True,
    )
    assert summary["count"] == 2
    assert summary["horizons"] == [LONG_HORIZON]

    dest_dir = tmp_path / "installed_long"
    installed = install_long_bank(
        src_dir,
        dest_dir,
        expected_count=2,
        expected_horizon=LONG_HORIZON,
        full_payload=False,
    )
    assert Path(installed["installed_dir"]).is_dir()
    assert len(list(dest_dir.glob("*.watch.gz"))) == 2

    with pytest.raises(FileExistsError):
        install_long_bank(src_dir, dest_dir, expected_count=2, expected_horizon=LONG_HORIZON)


def test_pilot_seed_disjointness_and_package_spec() -> None:
    audit = audit_pilot_seeds()
    assert audit["pilot_namespace"] == PILOT_NAMESPACE
    assert audit["pilot_count"] == PILOT_COUNT
    assert audit["pilot_horizon"] == PILOT_HORIZON
    assert audit["seed_overlap_count"] == 0
    assert audit["disjoint_from_installed_132"] is True
    assert len(audit["pilot_rows"]) == 12

    spec = build_pilot_package_spec()
    assert spec["namespace"] == PILOT_NAMESPACE
    assert spec["count"] == 12
    assert spec["horizon"] == 14400
    assert spec["checkpoints"]["cadence_frames"] == PILOT_CHECKPOINT_FRAMES
    assert spec["checkpoints"]["expected_checkpoints_per_session"] == [
        1800,
        3600,
        5400,
        7200,
        9000,
        10800,
        12600,
    ]
    assert spec["size_caps"]["max_session_compressed_bytes"] == PILOT_MAX_SESSION_BYTES
    resume = spec["manifest_resume_declaration"]["resume"]
    assert resume == {
        "unit_kind": "sessions",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    g4 = spec["compute_and_expected_units"]["colab_g4_48vcpu_pilot"]
    assert g4["rate_units_per_hour_g4"] == pytest.approx(10.05)
    assert g4["expected_units_forecast_range"] == [60.3, 80.4]
