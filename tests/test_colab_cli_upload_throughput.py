"""Receipt-latency replay through real upload code; synthetic bytes, no provider."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, transfers
from tests.test_colab_cli_transfers import FakeCLI

PROFILE = json.loads(
    (Path(__file__).parent / "fixtures/f4_upload_latency_0bd6f058.json").read_text()
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.Lock()

    def monotonic(self) -> float:
        with self.lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.now += seconds


class _ReceiptCLI(FakeCLI):
    """Charge saved CLI times, overlapping exactly two immutable PUTs per window."""

    def __init__(self, remote: Path, clock: _Clock) -> None:
        super().__init__(remote)
        self.clock = clock
        self.row: dict[str, Any] = {}
        self.latencies: list[float] = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2, action=self._finish_window)
        self.active = 0
        self.max_active = 0
        self.overlap_windows = 0

    def _finish_window(self) -> None:
        self.clock.advance(max(self.latencies))
        self.latencies.clear()
        self.overlap_windows += 1

    def call(self, *args: str, timeout: float = 45) -> str:
        assert args[0] == "upload"
        index = int(Path(args[-1]).stem) if args[-1].endswith(".part") else 0
        latency = self.row["put_seconds"][index]
        assert latency < timeout
        result = super().call(*args, timeout=timeout)
        if args[-1].endswith(".part"):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.latencies.append(latency)
            self.barrier.wait(timeout=5)
            with self.lock:
                self.active -= 1
        else:
            self.clock.advance(latency)
        return result

    def rpc(self, code: str, timeout: float = 20) -> Any:
        result = super().rpc(code, timeout=timeout)
        name = next(
            name
            for name in ("CREATE_DIR", "UPLOAD_ASSEMBLE", "VERIFY_FILE")
            if "SNAKE_TRANSFER_" + name in code
        )
        tag = {"CREATE_DIR": "mkdir", "UPLOAD_ASSEMBLE": "assemble", "VERIFY_FILE": "verify"}[name]
        latency = self.row["control_seconds"][tag]
        if name != "CREATE_DIR":
            latency += self.row["wrapper_overhead_seconds"]
        assert latency < timeout
        self.clock.advance(latency)
        return result


def test_receipt_profile_replays_real_concurrent_verified_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the 39.505s outlier, real chunk/full hashing and unchanged 150/300 bars."""
    clock = _Clock()
    cli = _ReceiptCLI(tmp_path / "remote", clock)
    monkeypatch.setattr(transfers.time, "monotonic", clock.monotonic)
    measurement = PROFILE["measurement"]
    wrapper_total = sum(row["elapsed_seconds"] for row in PROFILE["files"])
    wrapper_gap = measurement["transfer_seconds"] - wrapper_total
    old = (
        sum(
            sum(row["put_seconds"])
            + sum(row["control_seconds"].values())
            + row["wrapper_overhead_seconds"]
            for row in PROFILE["files"]
        )
        + wrapper_gap
    )
    assert old == pytest.approx(207.348055665032)
    # Execute the exact reviewed serial implementation against the same fake
    # filesystem and saved call costs. This is a local git-object read only.
    legacy_source = subprocess.check_output(
        ["git", "show", "0bd6f058a34afb5817ff30b61c2a333ad9758fd4:research/colab_cli/transfers.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )
    legacy = types.ModuleType("f4_reviewed_serial_transport")
    assert hashlib.sha256(legacy_source.encode()).hexdigest() == (
        "eeb490abf9442b9b519f4e8daeaf6126022586256c51e22b2ef3749d26d14d90"
    )
    exec(compile(legacy_source, "<reviewed-serial-transport>", "exec"), legacy.__dict__)

    class SerialCLI(_ReceiptCLI):
        def call(self, *args: str, timeout: float = 45) -> str:
            result = FakeCLI.call(self, *args, timeout=timeout)
            index = int(Path(args[-1]).stem) if args[-1].endswith(".part") else 0
            self.clock.advance(self.row["put_seconds"][index])
            return result

        def rpc(self, code: str, timeout: float = 20) -> Any:
            if "REMOVE_TREE" in code:
                result = FakeCLI.rpc(self, code, timeout)
                self.clock.advance(self.row["control_seconds"]["cleanup"] / 2)
                return result
            return super().rpc(code, timeout)

    serial = SerialCLI(tmp_path / "serial-remote", clock)
    for row in PROFILE["files"]:
        serial.row = row
        source = tmp_path / row["name"]
        with source.open("wb") as handle:
            handle.truncate(row["size_bytes"])
        remaining_renewal = [row["control_seconds"]["renewal"]]

        def legacy_renew() -> None:
            clock.advance(remaining_renewal[0])
            remaining_renewal[0] = 0

        legacy.upload_file(serial, source, "/content/" + source.name, legacy_renew, deadline=300)
    clock.advance(wrapper_gap)
    assert clock.monotonic() == pytest.approx(old)
    clock.now = 0
    old_path = tmp_path / "old-measurement.json"
    old_path.write_text(json.dumps(measurement))
    with pytest.raises(acceptance.supervisor.Refused, match="150-second bar"):
        acceptance._d1024_staging_measurement(old_path)

    # Model prospective coordinator renewal at the production 30s throttle.
    # Last old foreground renewal completed 14.733484s before staging began.
    last_renewal = -14.733484333963133
    renewals = []

    def renew() -> None:
        nonlocal last_renewal
        assert threading.current_thread() is threading.main_thread()
        if clock.monotonic() - last_renewal < 30:
            return
        elapsed = PROFILE["heartbeat_seconds"][len(renewals)]
        clock.advance(elapsed)
        renewals.append(elapsed)
        last_renewal = clock.monotonic()

    for row in PROFILE["files"]:
        cli.row = row
        source = tmp_path / row["name"]
        # Full-size deterministic synthetic bytes; never a scientific payload.
        with source.open("wb") as handle:
            handle.truncate(row["size_bytes"])
        digest = transfers.upload_file(cli, source, "/content/" + source.name, renew, deadline=300)
        assert digest == transfers._digest(source)
        assert transfers._digest(cli._path("/content/" + source.name)) == digest
    clock.advance(wrapper_gap)
    actual = clock.monotonic()
    projected = (
        sum(
            sum(max(row["put_seconds"][i : i + 2]) for i in range(0, len(row["put_seconds"]), 2))
            + sum(row["control_seconds"][tag] for tag in ("mkdir", "assemble", "verify"))
            + row["wrapper_overhead_seconds"]
            for row in PROFILE["files"]
        )
        + wrapper_gap
    )
    assert projected == pytest.approx(89.16094970621634)
    assert actual == pytest.approx(projected + sum(renewals))
    assert actual < 110  # receipt-based prospective renewal model, with margin
    # Separate stress allowance: charge the entire worst observed heartbeat,
    # rather than subtracting its historically overlapping interval twice.
    conservative = actual + max(PROFILE["heartbeat_seconds"])
    assert conservative == pytest.approx(128.22566428827122)
    assert conservative < 130 < acceptance.D1024_TRANSFER_BAR_SECONDS
    new_path = tmp_path / "new-measurement.json"
    new_path.write_text(
        json.dumps(
            {
                **measurement,
                "transfer_seconds": actual,
                "elapsed_seconds": actual
                + measurement["elapsed_seconds"]
                - measurement["transfer_seconds"],
            }
        )
    )
    assert acceptance._d1024_staging_measurement(new_path)["transfer_seconds"] == actual
    assert cli.max_active == transfers.UPLOAD_WORKERS == 2
    assert cli.overlap_windows == 13
    assert all("REMOVE_TREE" not in code for code in cli.rpc_codes)
    assert not list((tmp_path / "remote").rglob("*.part"))
    print(f"receipt replay old={old:.6f}s new={actual:.6f}s conservative={conservative:.6f}s")


def test_failed_worker_is_joined_before_remote_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 32)
    entered = threading.Barrier(2)
    finished = threading.Event()
    source = tmp_path / "input"
    source.write_bytes(b"x" * 64)

    class FailedCLI(FakeCLI):
        def call(self, *args: str, timeout: float = 45) -> str:
            entered.wait(timeout=5)
            if args[-1].endswith("00000000.part"):
                raise RuntimeError("authentication denied")
            try:
                return super().call(*args, timeout=timeout)
            finally:
                finished.set()

        def rpc(self, code: str, timeout: float = 20) -> Any:
            if "REMOVE_TREE" in code:
                assert finished.is_set()
            assert "UPLOAD_ASSEMBLE" not in code
            return super().rpc(code, timeout)

    cli = FailedCLI(tmp_path / "remote")
    with pytest.raises(transfers.TransferError, match="authentication denied"):
        transfers.upload_file(cli, source, "/content/result", lambda: None)
    assert finished.is_set()
    assert not cli._path("/content/result").exists()
    assert not list((tmp_path / "remote").rglob("*.part"))


def test_chunk_digest_checked_independently_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 32)
    source = tmp_path / "input"
    source.write_bytes(b"x" * 64)

    class WrongChunkDigest(FakeCLI):
        def rpc(self, code: str, timeout: float = 20) -> Any:
            if "UPLOAD_ASSEMBLE" in code:
                # Full expected SHA is still correct, but the per-chunk contract
                # is deliberately inconsistent: assembly must reject it too.
                digest = hashlib.sha256(b"x" * 32).hexdigest()
                code = code.replace(digest, "0" * 64)
            return super().rpc(code, timeout)

    cli = WrongChunkDigest(tmp_path / "remote")
    with pytest.raises(transfers.TransferError, match="chunk digest mismatch"):
        transfers.upload_file(cli, source, "/content/result", lambda: None)
    assert not cli._path("/content/result").exists()
    assert not list((tmp_path / "remote").rglob("*.part"))
    assert not list((tmp_path / "remote").rglob("*.assembling"))


@pytest.mark.parametrize("expire", [False, True])
def test_parallel_workers_inherit_real_broker_staging_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire: bool
) -> None:
    from research.colab_cli import supervisor

    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 32)
    clock = _Clock()
    monkeypatch.setattr(transfers.time, "monotonic", clock.monotonic)
    source = tmp_path / "input"
    source.write_bytes(b"x" * 64)
    seen = []

    class DeadlineCLI(FakeCLI, supervisor.CLI):
        poll_rpc = FakeCLI.rpc
        recoverable_rpc = FakeCLI.rpc

        def call(self, *args: str, timeout: float = 45) -> str:
            seen.append((threading.current_thread().name, self._control_context.deadline, timeout))
            result = FakeCLI.call(self, *args, timeout=timeout)
            if expire:
                clock.advance(8)
            return result

    cli = DeadlineCLI(tmp_path / "remote")
    supervisor.CLI.__init__(cli, "fake", tmp_path / "config", "fake-session")
    if expire:
        with pytest.raises(transfers.TransferError, match="total upload time cap"):
            transfers.upload_file(cli, source, "/content/result", lambda: None, deadline=7)
        assert not cli._path("/content/result").exists()
    else:
        transfers.upload_file(cli, source, "/content/result", lambda: None, deadline=7)
        assert cli._path("/content/result").read_bytes() == source.read_bytes()
    assert seen and all(name.startswith("colab-put") for name, _, _ in seen)
    assert all(deadline == 7 and 0 < timeout <= 7 for _, deadline, timeout in seen)
    if expire:
        assert not any("UPLOAD_ASSEMBLE" in code for code in cli.rpc_codes)
    assert getattr(cli._control_context, "deadline", float("inf")) == float("inf")


@pytest.mark.parametrize("staging,force", [(True, False), (True, True), (False, False)])
def test_production_renewal_selects_fenced_contents_only_during_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging: bool, force: bool
) -> None:
    import contextlib

    from research.colab_cli import supervisor

    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    puts, rpcs = [], []
    monkeypatch.setattr(cli, "_put_lease", lambda *args: puts.append(args))
    monkeypatch.setattr(cli, "poll_rpc", lambda code: rpcs.append(code) or {"state": "ready"})
    with cli.staging_controls(107) if staging else contextlib.nullcontext():
        cli.renew_runtime_lease("/content/job", "generation-42", force=force)
    if staging and not force:
        assert puts == [("/content/job", "generation-42", 7)]
        assert not rpcs
    else:
        assert not puts
        assert len(rpcs) == 1
        assert "read_state(" in rpcs[0] if force else "renew_lease(" in rpcs[0]
    assert cli.operation_records[-1]["operation"] == "renewal"
