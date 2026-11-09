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
    extrapolate_from_installed_bank,
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
    MAX_DECODED,
    MAX_LONG_DECODED,
    PRODUCT_HORIZONS,
    Recording,
    ReplayBank,
    max_decoded_bytes,
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

    # Conflicting SNAKE_WATCH_LONG_SESSIONS=1/long/14400 + SNAKE_WATCH_REPLAY_HORIZON=6000
    # must fail closed.
    for long_val in ("1", "long", "14400", "true", "yes", "on"):
        for short_h in ("6000", "default", "short"):
            monkeypatch.setenv(ENV_REPLAY_LONG_SESSIONS, long_val)
            monkeypatch.setenv(ENV_REPLAY_HORIZON, short_h)
            h_rev, long_rev, err_rev = resolve_replay_horizons()
            assert h_rev is None and long_rev is False and err_rev is not None

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
        # When horizons={14400} and long_root is set, _discover_paths must include only long_root.
        discovered = [p.resolve() for p in bank._discover_paths(opt_in_horizon=True)]
        assert discovered == [(long_dir / "000000.watch.gz").resolve()]
        rec = _pick_ready(bank)
        assert rec is not None
        try:
            assert rec.header["horizon"] == LONG_HORIZON
            assert rec.path.resolve().is_relative_to(long_dir.resolve())
        finally:
            rec.close()
    finally:
        bank.close()


def test_discover_paths_root_selection_and_resolved_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_dir = tmp_path / "default_root"
    long_dir = tmp_path / "long_root"
    default_dir.mkdir(parents=True)
    long_dir.mkdir(parents=True)

    # Populate default_dir with 1,030 dummy .watch.gz files to ensure long-only mode
    # never scans default_dir or truncates at 1,024 before reaching long_dir.
    for idx in range(1030):
        (default_dir / f"{idx:06d}.watch.gz").write_bytes(b"")
    long_file = long_dir / "000000.watch.gz"
    long_file.write_bytes(b"")

    monkeypatch.setattr(ReplayBank, "_prefetch", lambda self: self._done.set())

    # 1. Long-only (horizons={14400}) with long_root: scans ONLY long_root, never default_dir.
    bank_long_only = ReplayBank(
        default_dir,
        long_root=long_dir,
        horizons=frozenset({LONG_HORIZON}),
        shuffle_discovery=False,
    )
    try:
        paths_long = bank_long_only._discover_paths(opt_in_horizon=True)
        assert [p.resolve() for p in paths_long] == [long_file.resolve()]
    finally:
        bank_long_only.close()

    # 2. Short-only (horizons={6000}) with allow_long=True and long_root: scans ONLY default_dir.
    bank_short_only = ReplayBank(
        default_dir,
        long_root=long_dir,
        allow_long=True,
        horizons=frozenset({DEFAULT_HORIZON}),
        shuffle_discovery=False,
    )
    try:
        paths_short = bank_short_only._discover_paths(opt_in_horizon=True)
        assert len(paths_short) == 1024
        assert all(p.resolve().is_relative_to(default_dir.resolve()) for p in paths_short)
    finally:
        bank_short_only.close()

    # 3. Long-only (horizons={14400}) WITHOUT long_root: falls back to scanning self.root.
    bank_no_long_root = ReplayBank(
        long_dir,
        long_root=None,
        horizons=frozenset({LONG_HORIZON}),
        shuffle_discovery=False,
    )
    try:
        paths_fallback = bank_no_long_root._discover_paths(opt_in_horizon=True)
        assert [p.resolve() for p in paths_fallback] == [long_file.resolve()]
    finally:
        bank_no_long_root.close()

    # 4. Deduplication on resolved paths when root and long_root point to the same directory
    # via symlink or non-normalized relative path.
    symlink_long_dir = tmp_path / "symlink_to_long"
    symlink_long_dir.symlink_to(long_dir)
    bank_dedup = ReplayBank(
        long_dir,
        long_root=symlink_long_dir,
        horizons=PRODUCT_HORIZONS,
        shuffle_discovery=False,
    )
    try:
        paths_dedup = bank_dedup._discover_paths(opt_in_horizon=True)
        assert len(paths_dedup) == 1
        assert paths_dedup[0].resolve() == long_file.resolve()
    finally:
        bank_dedup.close()


def test_scaled_decoded_bound_applies_only_to_long_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MAX_DECODED == 512 * 1024**2
    assert MAX_LONG_DECODED == (MAX_DECODED * LONG_HORIZON) // DEFAULT_HORIZON
    assert max_decoded_bytes() == MAX_DECODED
    assert max_decoded_bytes(allow_long=False, horizon=LONG_HORIZON) == MAX_DECODED
    assert max_decoded_bytes(allow_long=True, horizon=DEFAULT_HORIZON) == MAX_DECODED
    assert max_decoded_bytes(allow_long=True, horizon=LONG_HORIZON) == MAX_LONG_DECODED

    s_short = write_synthetic_session(
        tmp_path / "short",
        namespace="test-cap-short-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    s_long = write_synthetic_session(
        tmp_path / "long",
        namespace="test-cap-long-v1",
        count=1,
        index=0,
        horizon=LONG_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    short_path = tmp_path / "short" / "000000.watch.gz"
    long_path = tmp_path / "long" / "000000.watch.gz"

    # Set MAX_DECODED below s_long["decoded_bytes"] (and below s_short["decoded_bytes"] + 100)
    # so that MAX_DECODED < s_long["decoded_bytes"] < (MAX_DECODED * 14400 // 6000).
    tight_cap = int(s_long["decoded_bytes"] * 0.60)
    assert tight_cap < s_long["decoded_bytes"] < (tight_cap * LONG_HORIZON) // DEFAULT_HORIZON
    monkeypatch.setattr(watch_replay, "MAX_DECODED", tight_cap)

    # H=14,400 with allow_long=True succeeds because its decoded bound is scaled 2.4x.
    hdr = verify_payload(long_path.read_bytes(), product=True, allow_long=True)
    assert hdr["horizon"] == LONG_HORIZON
    rec = Recording(long_path, product=True, allow_long=True)
    try:
        assert rec.step() is True
    finally:
        rec.close()

    # H=6,000 keeps the unscaled MAX_DECODED even when allow_long=True.
    monkeypatch.setattr(watch_replay, "MAX_DECODED", s_short["decoded_bytes"] - 10)
    with pytest.raises(ValueError, match="replay decompression limit"):
        verify_payload(short_path.read_bytes(), product=True, allow_long=True)

    # H=14,400 also fails when its decoded size exceeds the 2.4x scaled bound.
    monkeypatch.setattr(
        watch_replay,
        "MAX_DECODED",
        int(s_long["decoded_bytes"] * DEFAULT_HORIZON // LONG_HORIZON) - 100,
    )
    with pytest.raises(ValueError, match="replay decompression limit"):
        verify_payload(long_path.read_bytes(), product=True, allow_long=True)


def test_verify_and_install_long_bank_tooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    # Add a stray unverified file matching *.watch.gz* that install_long_bank must ignore.
    stray = src_dir / "000000.watch.gz.tmp"
    stray.write_bytes(b"stray-temp")

    summary = verify_bank_directory(
        src_dir,
        expected_horizon=LONG_HORIZON,
        allow_long=True,
        full_payload=True,
    )
    assert summary["count"] == 2
    assert summary["horizons"] == [LONG_HORIZON]
    assert summary["verified_files"] == ["000000.watch.gz", "000001.watch.gz"]

    # Track link order to verify (.watch.gz, .sha256, .complete) order per session.
    linked_targets: list[str] = []
    orig_link = watch_replay.os.link

    def spy_link(src: Path, dst: Path) -> None:
        linked_targets.append(Path(dst).name)
        orig_link(src, dst)

    monkeypatch.setattr("tools.watch_long_sessions.os.link", spy_link)

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
    assert not (dest_dir / "000000.watch.gz.tmp").exists()
    assert linked_targets == [
        "000000.watch.gz",
        "000000.watch.gz.sha256",
        "000000.watch.gz.complete",
        "000001.watch.gz",
        "000001.watch.gz.sha256",
        "000001.watch.gz.complete",
    ]

    with pytest.raises(FileExistsError):
        install_long_bank(src_dir, dest_dir, expected_count=2, expected_horizon=LONG_HORIZON)

    # Corrupting .sha256 sidecar format must fail verify_bank_directory even with
    # full_payload=False.
    (src_dir / "000000.watch.gz.sha256").write_text("not-a-valid-sha256  wrong.watch.gz\n")
    with pytest.raises(ValueError, match="invalid .sha256 sidecar format"):
        verify_bank_directory(
            src_dir,
            expected_horizon=LONG_HORIZON,
            allow_long=True,
            full_payload=False,
        )


def test_extrapolate_from_installed_bank_quantile_sampling_includes_largest(
    tmp_path: Path,
) -> None:
    bank_dir = tmp_path / "installed_sample"
    lengths_by_idx = [
        (10, 8, 8, 8, 8, 8),
        (20, 12, 10, 8, 8, 8),
        (30, 16, 12, 10, 8, 8),
        (40, 20, 14, 12, 10, 8),
        (55, 25, 18, 14, 12, 10),
        (75, 35, 24, 18, 14, 12),
        (110, 50, 32, 24, 18, 14),
    ]
    for idx, lengths in enumerate(lengths_by_idx):
        write_synthetic_session(
            bank_dir,
            namespace="test-extrapolate-bank-v1",
            count=len(lengths_by_idx),
            index=idx,
            horizon=DEFAULT_HORIZON,
            snake_lengths=lengths,
            unique_frames=1,
        )
    res = extrapolate_from_installed_bank(bank_dir, sample_sessions_for_frame_curve=4)
    assert res["installed_bank_available"] is True
    sampled = res["frame_size_growth_profile"]["sampled_sessions"]
    assert sampled[0] == "000000.watch.gz"
    assert sampled[-1] == "000006.watch.gz"
    ext = res["extrapolated_h14400"]
    assert ext["max_reader_long_decoded_cap_mib"] == pytest.approx(1228.8)
    assert ext["fits_reader_max_long_decoded_1228_8_mib"] is True
    assert ext["decoded_headroom_below_long_cap_mib"]["superlinear_max"] > 0.0


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
    assert spec["size_caps"]["max_reader_default_decoded_bytes"] == MAX_DECODED
    assert spec["size_caps"]["max_reader_decoded_bytes"] == MAX_LONG_DECODED
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
    assert g4["base_unit_cap"] == pytest.approx(80.4)
    assert g4["effective_unit_cap_2x_rule"] == pytest.approx(160.8)
    assert g4["base_timeout_minutes"] == 480
    assert g4["effective_timeout_minutes_2x_rule"] == 960
    assert "three_hour_session_guard_option" not in g4
