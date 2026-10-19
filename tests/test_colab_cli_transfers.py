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
        raise AssertionError(code)


def test_large_round_trip_is_chunked_and_digest_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "CHUNK_BYTES", 1024)
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
    assert not cli.rpc_codes


def test_contents_transfer_retries_one_transient_failure(tmp_path: Path) -> None:
    source = tmp_path / "small"
    source.write_bytes(b"small")
    cli = FakeCLI(tmp_path / "remote", transient_failures=1)
    renewals: list[None] = []
    digest = transfers.upload_file(cli, source, "/content/small", lambda: renewals.append(None))
    assert digest == hashlib.sha256(b"small").hexdigest()
    assert [call[0] for call in cli.calls] == ["upload", "upload"]
    assert len(renewals) == 3  # before failed call, before + after successful call


def test_download_rejects_corrupt_chunk_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfers, "CHUNK_BYTES", 32)
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
