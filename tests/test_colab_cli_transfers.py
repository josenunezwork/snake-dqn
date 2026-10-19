"""Hermetic transfer tests: no real Colab allocation, exec, or stop."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from research.colab_cli import transfers


class FakeCLI:
    session = "fake-session"

    def __init__(
        self, remote: Path, *, corrupt_download: bool = False, transient_failures: int = 0
    ) -> None:
        self.remote = remote
        self.remote.mkdir()
        (self.remote / "content").mkdir()
        self.calls: list[tuple[str, ...]] = []
        self.rpc_codes: list[str] = []
        self.transfer_records: list[dict[str, object]] = []
        self.corrupt_download = corrupt_download
        self.transient_failures = transient_failures

    def _path(self, value: str) -> Path:
        return self.remote / value.lstrip("/")

    def call(self, *args: str, timeout: float = 45) -> str:
        self.calls.append(args)
        if self.transient_failures:
            self.transient_failures -= 1
            raise RuntimeError("transient ContentsClient connection reset")
        if args[0] == "upload":
            source, remote = Path(args[-2]), self._path(args[-1])
            if not remote.parent.is_dir():
                raise RuntimeError("ContentsClient cannot create upload parents")
            shutil.copyfile(source, remote)
        elif args[0] == "download":
            remote, local = self._path(args[-2]), Path(args[-1])
            data = remote.read_bytes()
            if self.corrupt_download and ".part" in remote.name:
                data += b"x"
            local.write_bytes(data)
        else:
            raise AssertionError(args)
        return ""

    def rpc(self, code: str, timeout: float = 20):
        self.rpc_codes.append(code)
        spec = json.loads(re.search(r"spec = (\{.*\})", code).group(1))
        if "UPLOAD_ASSEMBLE" in code:
            data = b"".join(
                self._path(f"{spec['parts']}/{index:08d}.part").read_bytes()
                for index in range(spec["chunks"])
            )
            target = self._path(spec["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if "DOWNLOAD_SPLIT" in code:
            data = self._path(spec["source"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == spec["sha256"]
            directory = self._path(spec["parts"])
            directory.mkdir(parents=True)
            chunks = [
                data[i : i + spec["chunk_bytes"]] for i in range(0, len(data), spec["chunk_bytes"])
            ]
            for index, piece in enumerate(chunks):
                (directory / f"{index:08d}.part").write_bytes(piece)
            return {"sha256": spec["sha256"], "size": len(data), "chunks": len(chunks)}
        if "REMOVE_TREE" in code:
            shutil.rmtree(self._path(spec["path"]), ignore_errors=True)
            return True
        if "CREATE_DIR" in code:
            self._path(spec["path"]).mkdir(parents=True)
            return True
        if "VERIFY_FILE" in code:
            path = self._path(spec["path"])
            data = path.read_bytes()
            return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        raise AssertionError(code)


def test_large_round_trip_is_chunked_and_digest_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 1024)
    monkeypatch.setattr(transfers, "DOWNLOAD_CHUNK_BYTES", 1024)
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 13)
    cli = FakeCLI(tmp_path / "remote")
    renewals: list[None] = []

    digest = transfers.upload_file(
        cli, source, "/content/job/input.bin", lambda: renewals.append(None)
    )
    target = tmp_path / "out.bin"
    assert (
        transfers.download_file(
            cli,
            "/content/job/input.bin",
            target,
            source.stat().st_size,
            lambda: renewals.append(None),
            expected_sha256=digest,
        )
        == digest
    )
    assert target.read_bytes() == source.read_bytes()
    uploads = [call for call in cli.calls if call[0] == "upload"]
    downloads = [call for call in cli.calls if call[0] == "download"]
    assert len(uploads) == len(downloads) == 4
    assert all(".snake-transfer-" in call[-1] for call in uploads)
    assert all(".snake-transfer-" in call[-2] for call in downloads)
    assert renewals
    assert not list((tmp_path / "remote").rglob("*.part"))


def test_small_files_use_direct_cli_contents_call(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote")
    digest = transfers.upload_file(cli, source, "/content/small", lambda: None)
    target = tmp_path / "out"
    transfers.download_file(cli, "/content/small", target, 5, expected_sha256=digest)
    assert [call[0] for call in cli.calls] == ["upload", "download"]
    assert len(cli.rpc_codes) == 1
    assert "VERIFY_FILE" in cli.rpc_codes[0]


def test_contents_transfer_retries_one_transient_failure(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote", transient_failures=1)
    renewals: list[None] = []
    digest = transfers.upload_file(cli, source, "/content/small", lambda: renewals.append(None))
    assert digest == hashlib.sha256(b"small").hexdigest()
    assert [call[0] for call in cli.calls] == ["upload", "upload"]
    assert len(renewals) == 3  # before failed call, before + after successful call


def test_contents_transfer_does_not_retry_non_transient_error(tmp_path: Path) -> None:
    class RejectingCLI(FakeCLI):
        def call(self, *args: str, timeout: float = 45) -> str:
            self.calls.append(args)
            raise RuntimeError("authentication denied")

    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = RejectingCLI(tmp_path / "remote")
    with pytest.raises(RuntimeError, match="authentication denied"):
        transfers.upload_file(cli, source, "/content/small", lambda: None)
    assert len(cli.calls) == 1


def test_renew_failure_after_success_does_not_repeat_contents_write(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote")

    renewals = 0

    def failed_renew() -> None:
        nonlocal renewals
        renewals += 1
        if renewals == 2:
            raise RuntimeError("lease renewal refused")

    with pytest.raises(RuntimeError, match="lease renewal refused"):
        transfers.upload_file(cli, source, "/content/small", failed_renew)
    assert len(cli.calls) == 1


def test_upload_refuses_expired_shared_deadline(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote")
    with pytest.raises(transfers.TransferError, match="total upload time cap"):
        transfers.upload_file(
            cli,
            source,
            "/content/small",
            lambda: None,
            deadline=transfers.time.monotonic() - 0.01,
        )
    assert not cli.calls


def test_later_upload_gets_its_own_deadline_and_records_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-staging transfer must not inherit the already-expired staging cap."""
    source = tmp_path / "resume.bin"
    source.write_bytes(b"resume-input" * 100)
    cli = FakeCLI(tmp_path / "remote")
    # Simulate a renewal/resume upload more than five minutes after initial
    # input staging.  A default transfer deadline is created at this instant.
    now = 301.0
    monkeypatch.setattr(transfers.time, "monotonic", lambda: now)

    digest = transfers.upload_file(cli, source, "/content/resume.bin", lambda: None)

    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert cli._path("/content/resume.bin").read_bytes() == source.read_bytes()
    assert cli.transfer_records[-1]["direction"] == "upload"
    assert cli.transfer_records[-1]["remote"] == "/content/resume.bin"
    assert cli.transfer_records[-1]["size_bytes"] == source.stat().st_size
    assert cli.transfer_records[-1]["elapsed_seconds"] >= 0
    assert cli.transfer_records[-1]["bytes_per_second"] > 0
    assert cli.transfer_records[-1]["status"] == "verified"
    assert cli.transfer_records[-1]["call_site"]


def test_expired_staging_deadline_reports_transfer_call_site(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote")

    with pytest.raises(
        transfers.TransferError,
        match=(
            r"upload /content/small called at .*test_colab_cli_transfers.py.*"
            r"total upload time cap reached at upload /content/small"
        ),
    ):
        transfers.upload_file(
            cli,
            source,
            "/content/small",
            lambda: None,
            deadline=transfers.time.monotonic() - 0.01,
        )


def test_control_upload_retry_budget_obeys_renewal_lease_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renewal credentials get retries, but cannot consume the whole lease."""
    source = tmp_path / "credential.json"
    source.write_text('{"access_token":"short-lived"}')
    now = [301.0]
    seen_timeouts: list[float] = []

    class SlowTransientCLI(FakeCLI):
        def call(self, *args: str, timeout: float = 45) -> str:
            seen_timeouts.append(timeout)
            now[0] += min(60.0, timeout)
            raise RuntimeError("ContentsClient connection reset")

    monkeypatch.setattr(transfers.time, "monotonic", lambda: now[0])
    cli = SlowTransientCLI(tmp_path / "remote")

    with pytest.raises(
        transfers.TransferError,
        match=(
            r"control upload /content/release.json called at .*test_colab_cli_transfers.py.*"
            r"total upload time cap reached"
        ),
    ):
        transfers.upload_contents(cli, source, "/content/release.json", deadline=391.0)

    # First attempt has its normal chunk allowance, the retry is clipped to
    # the 30 seconds left in the 90-second renewal window, and a third PUT is
    # refused before it can begin.
    assert seen_timeouts == [60.0, 30.0]
    assert now[0] == pytest.approx(391.0)


def test_upload_and_download_chunk_sizes_are_separate() -> None:
    assert transfers.UPLOAD_CHUNK_BYTES == 2 * 1024 * 1024
    assert transfers.DOWNLOAD_CHUNK_BYTES == 8 * 1024 * 1024


def test_upload_rejects_remote_digest_mismatch(tmp_path: Path) -> None:
    class CorruptVerifierCLI(FakeCLI):
        def rpc(self, code: str, timeout: float = 20):
            if "VERIFY_FILE" in code:
                return {"sha256": "0" * 64, "size": 5}
            return super().rpc(code, timeout)

    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = CorruptVerifierCLI(tmp_path / "remote")
    with pytest.raises(transfers.TransferError, match="remote upload digest"):
        transfers.upload_file(cli, source, "/content/small", lambda: None)


def test_download_rejects_corrupt_chunk_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "DOWNLOAD_CHUNK_BYTES", 32)
    cli = FakeCLI(tmp_path / "remote", corrupt_download=True)
    data = b"safe-export" * 12
    remote = cli._path("/content/export.bin")
    remote.parent.mkdir(parents=True, exist_ok=True)
    remote.write_bytes(data)
    target = tmp_path / "published.bin"
    with pytest.raises(transfers.TransferError, match="unexpected size"):
        transfers.download_file(
            cli,
            "/content/export.bin",
            target,
            len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
        )
    assert not target.exists()
    assert len([call for call in cli.calls if call[0] == "download"]) == 1


def test_download_requires_inventory_digest(tmp_path: Path) -> None:
    cli = FakeCLI(tmp_path / "remote")
    with pytest.raises(transfers.TransferError, match="requires immutable"):
        transfers.download_file(cli, "/content/nope", tmp_path / "out", 0)


def test_chunk_timeout_scales_and_is_bounded() -> None:
    assert transfers._timeout(1) == 60
    assert transfers._timeout(80 * 1024 * 1024) == 120


@pytest.mark.parametrize(
    "message",
    [
        "colab upload failed: requests.exceptions.ConnectionError: connection dropped",
        "colab upload failed: NameResolutionError: Failed to resolve 'runtime.test'",
        "colab upload failed: RemoteDisconnected('Remote end closed connection')",
    ],
)
def test_cli_wrapped_connection_errors_are_retryable(message: str) -> None:
    assert transfers._is_retryable_contents_error(RuntimeError(message))
