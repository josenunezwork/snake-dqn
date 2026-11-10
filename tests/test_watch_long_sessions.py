"""Tests for opt-in H=14,400 long Watch sessions and the Mac engineering gate tooling."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools.watch_long_sessions import (
    ENV_INSTALLED_H6000_BANK,
    INSTALLED_H6000_BANK_DEFAULT,
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
    (src_dir / "contract.json").write_text(
        json.dumps(
            {
                **watch_replay_long_allocation_for_test("test-install-long-v1", 2),
                "commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    installed = install_long_bank(
        src_dir,
        dest_dir,
        expected_count=2,
        expected_horizon=LONG_HORIZON,
        full_payload=False,
    )
    assert installed["contract_verified"] is True
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
    bad_fmt_dir = tmp_path / "bad_fmt"
    bad_fmt_dir.mkdir()
    shutil.copy2(src_dir / "000000.watch.gz", bad_fmt_dir / "000000.watch.gz")
    (bad_fmt_dir / "000000.watch.gz.sha256").write_text("not-a-valid-sha256  wrong.watch.gz\n")
    (bad_fmt_dir / "000000.watch.gz.complete").write_text("complete\n")
    with pytest.raises(ValueError, match="invalid .sha256 sidecar format"):
        verify_bank_directory(
            bad_fmt_dir,
            expected_horizon=LONG_HORIZON,
            allow_long=True,
            full_payload=False,
        )


def watch_replay_long_allocation_for_test(namespace: str, count: int) -> dict:
    from tools.watch_long_sessions import PRODUCT_CONTROLLER, long_allocation

    return long_allocation(namespace, count, PRODUCT_CONTROLLER, LONG_HORIZON)


def test_install_long_bank_enforces_sha256_controller_allocation_and_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tools.watch_long_sessions import PRODUCT_CONTROLLER, long_allocation, main

    # 1. Header-only verify_bank_directory(full_payload=False) stays lightweight and
    # does NOT read/hash full payloads, while install_long_bank always enforces SHA-256
    # even when full_payload=False, and CLI install-long-bank rejects --header-only.
    valid_dir = tmp_path / "valid_src"
    for idx in range(2):
        write_synthetic_session(
            valid_dir,
            namespace="test-strict-install-v1",
            count=2,
            index=idx,
            horizon=LONG_HORIZON,
            snake_lengths=(12, 10, 8, 8, 8, 8),
            unique_frames=2,
        )
    corrupt_dir = tmp_path / "corrupt_payload"
    corrupt_dir.mkdir()
    raw_bytes = (valid_dir / "000000.watch.gz").read_bytes()
    (corrupt_dir / "000000.watch.gz").write_bytes(raw_bytes[:-8] + b"\x00" * 8)
    shutil.copy2(
        valid_dir / "000000.watch.gz.sha256",
        corrupt_dir / "000000.watch.gz.sha256",
    )
    (corrupt_dir / "000000.watch.gz.complete").write_text("complete\n")
    header_only_summary = verify_bank_directory(
        corrupt_dir,
        expected_horizon=LONG_HORIZON,
        allow_long=True,
        full_payload=False,
    )
    assert header_only_summary["count"] == 1
    assert header_only_summary["full_payload_verified"] is False
    assert "verified_rows" not in header_only_summary
    with pytest.raises(ValueError, match="SHA mismatch"):
        install_long_bank(
            corrupt_dir,
            tmp_path / "dest_corrupt",
            expected_count=1,
            expected_horizon=LONG_HORIZON,
            full_payload=False,
            require_contract=False,
        )

    with pytest.raises(SystemExit):
        main(["install-long-bank", "--header-only", str(valid_dir), str(tmp_path / "cli_dest")])

    # 2. Wrong controller (exact-stop-l1 instead of exact-stop-l1-os) is rejected.
    wrong_ctrl_dir = tmp_path / "wrong_ctrl"
    for idx in range(2):
        write_synthetic_session(
            wrong_ctrl_dir,
            namespace="test-wrong-ctrl-v1",
            count=2,
            index=idx,
            horizon=LONG_HORIZON,
            controller="exact-stop-l1",
            snake_lengths=(12, 10, 8, 8, 8, 8),
            unique_frames=1,
        )
    with pytest.raises(ValueError, match="controller"):
        install_long_bank(
            wrong_ctrl_dir,
            tmp_path / "dest_wrong_ctrl",
            expected_count=2,
            expected_horizon=LONG_HORIZON,
        )

    # 3. Heterogeneous allocations across sessions are rejected.
    mixed_alloc_dir = tmp_path / "mixed_alloc"
    write_synthetic_session(
        mixed_alloc_dir,
        namespace="test-alloc-a-v1",
        count=2,
        index=0,
        horizon=LONG_HORIZON,
        snake_lengths=(12, 10, 8, 8, 8, 8),
        unique_frames=1,
    )
    write_synthetic_session(
        mixed_alloc_dir,
        namespace="test-alloc-b-v1",
        count=2,
        index=1,
        horizon=LONG_HORIZON,
        snake_lengths=(12, 10, 8, 8, 8, 8),
        unique_frames=1,
    )
    with pytest.raises(ValueError, match="allocation"):
        install_long_bank(
            mixed_alloc_dir,
            tmp_path / "dest_mixed_alloc",
            expected_count=2,
            expected_horizon=LONG_HORIZON,
        )

    # 4. Missing contract.json fails closed by default; --no-contract explicitly overrides
    # and records contract_verified=False; mismatched contract.json fails; matching passes.
    with pytest.raises(ValueError, match="missing contract.json"):
        install_long_bank(
            valid_dir,
            tmp_path / "dest_missing_contract",
            expected_count=2,
            expected_horizon=LONG_HORIZON,
        )

    capsys.readouterr()
    rc = main(
        [
            "install-long-bank",
            "--no-contract",
            "--count",
            "2",
            str(valid_dir),
            str(tmp_path / "dest_no_contract_cli"),
        ]
    )
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["contract_verified"] is False

    bad_contract_dir = tmp_path / "bad_contract"
    shutil.copytree(valid_dir, bad_contract_dir)
    wrong_contract = {
        **long_allocation("test-other-ns-v1", 2, PRODUCT_CONTROLLER, LONG_HORIZON),
        "commit": "a" * 40,
    }
    (bad_contract_dir / "contract.json").write_text(json.dumps(wrong_contract), encoding="utf-8")
    with pytest.raises(ValueError, match="contract.json"):
        install_long_bank(
            bad_contract_dir,
            tmp_path / "dest_bad_contract",
            expected_count=2,
            expected_horizon=LONG_HORIZON,
        )

    good_contract = {
        **long_allocation("test-strict-install-v1", 2, PRODUCT_CONTROLLER, LONG_HORIZON),
        "commit": "a" * 40,
    }
    (valid_dir / "contract.json").write_text(json.dumps(good_contract), encoding="utf-8")
    installed = install_long_bank(
        valid_dir,
        tmp_path / "dest_good_contract",
        expected_count=2,
        expected_horizon=LONG_HORIZON,
    )
    assert installed["contract_verified"] is True
    assert "verified_rows" not in installed
    assert (Path(installed["installed_dir"]) / "contract.json").is_file()


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


def test_pilot_seed_disjointness_and_package_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.watch_long_sessions as wls

    assert wls.INSTALLED_H6000_BANK_DEFAULT == Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch-replay-bank-live-132-l1os-installed"
    )
    synth_bank = tmp_path / "synthetic_default_bank"
    write_synthetic_session(
        synth_bank,
        namespace="test-pilot-default-bank-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    monkeypatch.setenv(ENV_INSTALLED_H6000_BANK, str(synth_bank))
    expected_rel = wls._repo_relpath(synth_bank)

    audit = audit_pilot_seeds()
    assert audit["pilot_namespace"] == PILOT_NAMESPACE
    assert audit["pilot_count"] == PILOT_COUNT
    assert audit["pilot_horizon"] == PILOT_HORIZON
    assert audit["installed_bank_dir"] == expected_rel
    assert audit["default_bank_dir"] == expected_rel
    assert audit["default_bank_present"] is True
    assert audit["default_bank_seed_count"] == 1
    assert audit["seed_overlap_count"] == 0
    assert audit["disjoint_from_installed_132"] is True
    assert len(audit["pilot_rows"]) == 12

    spec = build_pilot_package_spec()
    assert spec["namespace"] == PILOT_NAMESPACE
    assert spec["seed_disjointness"]["installed_bank_dir"] == expected_rel
    assert spec["seed_disjointness"]["default_bank_dir"] == expected_rel
    assert spec["seed_disjointness"]["default_bank_present"] is True
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


@pytest.mark.skipif(
    not INSTALLED_H6000_BANK_DEFAULT.is_dir(),
    reason="real 132-session H=6,000 bank not present on disk",
)
def test_real_installed_h6000_bank_disjointness(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.watch_long_sessions as wls

    monkeypatch.delenv(ENV_INSTALLED_H6000_BANK, raising=False)
    expected_rel = wls._repo_relpath(INSTALLED_H6000_BANK_DEFAULT)
    audit = audit_pilot_seeds()
    assert audit["installed_bank_dir"] == expected_rel
    assert audit["default_bank_dir"] == expected_rel
    assert audit["default_bank_present"] is True
    assert audit["default_bank_seed_count"] == 132
    assert audit["seed_overlap_count"] == 0
    assert audit["disjoint_from_installed_132"] is True


def test_audit_pilot_seeds_and_gate_thread_installed_bank_and_sessions_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.watch_long_sessions as wls

    clean_default_bank = tmp_path / "clean_default_bank"
    write_synthetic_session(
        clean_default_bank,
        namespace="test-clean-default-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    monkeypatch.setenv(ENV_INSTALLED_H6000_BANK, str(clean_default_bank))

    # 1. sessions/ subdirectory layout is scanned by audit_pilot_seeds and detects collisions.
    nested_bank = tmp_path / "nested_bank"
    write_synthetic_session(
        nested_bank / "sessions",
        namespace="test-disjoint-nested-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    audit = audit_pilot_seeds(installed_bank_dir=nested_bank)
    expected_rel = wls._repo_relpath(nested_bank)
    assert not expected_rel.startswith("/")
    assert audit["installed_bank_dir"] == expected_rel
    assert audit["installed_disk_seed_count"] == 1
    spec_nested = build_pilot_package_spec(installed_bank_dir=nested_bank)
    assert spec_nested["seed_disjointness"]["installed_bank_dir"] == expected_rel
    spec_in_repo = build_pilot_package_spec(
        installed_bank_dir=wls.REPO_ROOT / "replays" / "nonexistent_test_bank"
    )
    assert spec_in_repo["seed_disjointness"]["installed_bank_dir"] == (
        "replays/nonexistent_test_bank"
    )

    # Mixed layout (both top-level *.watch.gz and sessions/*.watch.gz) scans the union,
    # catching collisions inside sessions/ even when top-level files exist (F11).
    mixed_layout_bank = tmp_path / "mixed_layout_bank"
    write_synthetic_session(
        mixed_layout_bank,
        namespace="test-disjoint-top-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    write_synthetic_session(
        mixed_layout_bank / "sessions",
        namespace=PILOT_NAMESPACE,
        count=PILOT_COUNT,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    with pytest.raises(ValueError, match="seed collision"):
        audit_pilot_seeds(installed_bank_dir=mixed_layout_bank)

    # Auditing a custom --installed-bank still checks INSTALLED_H6000_BANK_DEFAULT
    # when present on disk (F12).
    fake_default_bank = tmp_path / "fake_default_bank"
    write_synthetic_session(
        fake_default_bank,
        namespace=PILOT_NAMESPACE,
        count=PILOT_COUNT,
        index=1,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    with monkeypatch.context() as mp:
        mp.delenv(ENV_INSTALLED_H6000_BANK, raising=False)
        mp.setattr(wls, "INSTALLED_H6000_BANK_DEFAULT", fake_default_bank)
        with pytest.raises(ValueError, match="seed collision"):
            audit_pilot_seeds(installed_bank_dir=nested_bank)

    colliding_bank = tmp_path / "colliding_bank"
    write_synthetic_session(
        colliding_bank / "sessions",
        namespace=PILOT_NAMESPACE,
        count=PILOT_COUNT,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    with pytest.raises(ValueError, match="seed collision"):
        build_pilot_package_spec(installed_bank_dir=colliding_bank)

    # 2. run_mac_engineering_gate forwards installed_bank_dir to build_pilot_package_spec.
    monkeypatch.setattr(
        wls,
        "run_short_local_generator_check",
        lambda **_kw: {
            "sessions_generated": 2,
            "frames_replayed": [6, 6],
            "resumed_bit_identical": True,
        },
    )
    monkeypatch.setattr(
        wls,
        "run_h14400_scale_synthetic_gate",
        lambda: {
            "synthetic_h14400_session0": {
                "within_max_compressed_128_mib": True,
                "within_max_long_decoded_1228_8_mib": True,
                "within_max_line_2048_kib": True,
                "peak_line_kib": 35.0,
            },
            "verification_and_prefetch_timing": {
                "cold_first_pick_production_seconds": 1.0,
                "playback_duration_24fps_seconds": 600.0,
                "prefetch_headroom_ratio": 10.0,
                "full_verify_payload_slots_verified": 3,
            },
            "memory": {
                "peak_python_heap_delta_mib": 4.0,
                "pinned_mmap_3_slots_mib": 12.0,
            },
            "default_h6000_byte_identity_and_isolation": {
                "default_rejects_h14400_header": True,
                "default_rejects_h14400_peek": True,
                "unselected_long_header_detected": True,
                "unselected_long_header_skip_ms": 1.0,
                "mixed_dir_default_served_horizon": DEFAULT_HORIZON,
                "mixed_dir_skipped_long_without_verify": True,
                "opt_in_first_horizon": LONG_HORIZON,
                "opt_in_rotated_horizon": LONG_HORIZON,
                "long_only_dual_bank_horizon": LONG_HORIZON,
            },
        },
    )
    monkeypatch.setattr(
        wls,
        "extrapolate_from_installed_bank",
        lambda _dir=None: {"installed_bank_available": False},
    )
    with pytest.raises(ValueError, match="seed collision"):
        wls.run_mac_engineering_gate(installed_bank_dir=colliding_bank)

    # R1: Importing from a nested checkout not sibling to snake-dqn-artifacts still
    # points INSTALLED_H6000_BANK_DEFAULT to the original absolute default path from main.
    import importlib.util

    nested_copy = tmp_path / "deep" / "nested_checkout" / "tools" / "watch_long_sessions.py"
    nested_copy.parent.mkdir(parents=True)
    shutil.copy2(Path(wls.__file__), nested_copy)
    spec_mod = importlib.util.spec_from_file_location("_wls_nested_test", nested_copy)
    assert spec_mod is not None and spec_mod.loader is not None
    nested_mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(nested_mod)
    assert nested_mod.INSTALLED_H6000_BANK_DEFAULT == wls.INSTALLED_H6000_BANK_DEFAULT

    # R1: _repo_relpath uses os.path.abspath (not resolve()) so symlinked artifact
    # directories keep their portable lexical path instead of following symlinks.
    real_target = tmp_path / "elsewhere" / "real_bank"
    real_target.mkdir(parents=True)
    sym_parent = tmp_path / "sym_artifacts"
    sym_parent.mkdir(parents=True)
    sym_bank = sym_parent / "watch-replay-bank-live-132-l1os-installed"
    sym_bank.symlink_to(real_target)
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    with monkeypatch.context() as mp:
        mp.setattr(wls, "REPO_ROOT", checkout_dir)
        assert (
            wls._repo_relpath(sym_bank)
            == "../sym_artifacts/watch-replay-bank-live-132-l1os-installed"
        )

    # R1: Missing default H=6,000 bank sets default_bank_present=False and fails the gate;
    # pilot_seeds_disjoint_from_installed_132 stays True (default_h6000_bank_present is separate);
    # SNAKE_WATCH_INSTALLED_H6000_BANK overrides the default lookup while output stays relative.
    missing_default = tmp_path / "missing_default_bank"
    with monkeypatch.context() as mp:
        mp.setenv(wls.ENV_INSTALLED_H6000_BANK, str(missing_default))
        audit_missing = audit_pilot_seeds(installed_bank_dir=nested_bank)
        assert audit_missing["default_bank_present"] is False
        assert audit_missing["default_bank_seed_count"] == 0
        assert audit_missing["default_bank_dir"] == wls._repo_relpath(missing_default)
        gate_missing = wls.run_mac_engineering_gate(installed_bank_dir=nested_bank)
        assert gate_missing["verdict"] == "FAIL"
        assert gate_missing["checks"]["default_h6000_bank_present"] is False
        assert gate_missing["checks"]["pilot_seeds_disjoint_from_installed_132"] is True
        assert [k for k, v in gate_missing["checks"].items() if not v] == [
            "default_h6000_bank_present"
        ]
        assert wls.main(["gate", "--installed-bank", str(nested_bank)]) == 1

    present_default = tmp_path / "present_default_bank"
    write_synthetic_session(
        present_default,
        namespace="test-present-default-v1",
        count=1,
        index=0,
        horizon=DEFAULT_HORIZON,
        snake_lengths=(10, 8, 8, 8, 8, 8),
        unique_frames=1,
    )
    with monkeypatch.context() as mp:
        mp.setenv(wls.ENV_INSTALLED_H6000_BANK, str(present_default))
        audit_present = audit_pilot_seeds(installed_bank_dir=nested_bank)
        assert audit_present["default_bank_present"] is True
        assert audit_present["default_bank_seed_count"] == 1
        gate_present = wls.run_mac_engineering_gate(installed_bank_dir=nested_bank)
        assert gate_present["verdict"] == "PASS"
        assert gate_present["checks"]["default_h6000_bank_present"] is True
        assert wls.main(["gate", "--installed-bank", str(nested_bank)]) == 0
