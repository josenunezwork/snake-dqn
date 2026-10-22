"""Unit coverage for bounded batched immutable Colab export recovery."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from research.colab_cli import export_batches


class Kernel:
    """A fake kernel that executes the generated batch operation locally."""

    session = "batch-test"

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.remote.mkdir()
        self.calls: list[tuple[str, ...]] = []
        self.renewals = 0
        self.malicious_archive: bytes | None = None
        self.bad_metadata = False

    def renew(self) -> None:
        self.renewals += 1

    def _path(self, value: str) -> Path:
        return self.remote / value.lstrip("/")

    def recoverable_rpc(self, code: str, timeout: float = 0):
        assert "# SNAKE_EXPORT_BATCH" in code
        translated = code.replace("'/content", repr(str(self.remote / "content"))[:-1])
        scope: dict[str, object] = {"__name__": "__main__"}
        exec(compile(translated, "<batch>", "exec"), scope)
        result = scope["result"]
        assert isinstance(result, dict)
        remote_archive = Path(str(result["path"]))
        if self.malicious_archive is not None:
            remote_archive.write_bytes(self.malicious_archive)
            result["size"] = len(self.malicious_archive)
            result["sha256"] = hashlib.sha256(self.malicious_archive).hexdigest()
        result["path"] = "/content" + str(remote_archive).split("/content", 1)[1]
        if self.bad_metadata:
            result["members"] = []
        return result

    def call(self, *args: str, timeout: float = 0) -> str:
        self.calls.append(args)
        assert args[0] == "download"
        source, local = self._path(args[-2]), Path(args[-1])
        shutil.copyfile(source, local)
        return ""


def _entry(path: str, payload: bytes) -> dict[str, object]:
    return {"path": path, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _remote_export(kernel: Kernel, root: str, entry: dict[str, object], payload: bytes) -> None:
    target = kernel._path(root + "/output/" + str(entry["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    Path(str(target) + ".sha256").write_text(str(entry["sha256"]) + "  payload\n")
    Path(str(target) + ".complete").write_bytes(b"complete\n")


def test_small_batches_preserve_order_and_leave_oversize_singleton() -> None:
    entries = [
        _entry("a", b"a"),
        _entry("big", b"x" * (export_batches.MAX_BATCH_PAYLOAD_BYTES + 1)),
        _entry("b", b"b"),
    ]
    groups = export_batches.small_batches(entries)
    assert [[row["path"] for row in group] for group in groups] == [["a"], ["big"], ["b"]]


def test_export_batch_executes_remote_pack_and_publishes_original_bytes(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "remote")
    root = "/content/run"
    entries = [_entry("nested/a.json", b'{"a":1}'), _entry("b.bin", b"binary\x00payload")]
    for entry, payload in zip(entries, (b'{"a":1}', b"binary\x00payload")):
        _remote_export(kernel, root, entry, payload)
    published: list[str] = []

    export_batches.export_batch(
        kernel, root, entries, tmp_path / "out", 1024, lambda entry: published.append(entry["path"])
    )

    assert (tmp_path / "out/exports/nested/a.json").read_bytes() == b'{"a":1}'
    assert (tmp_path / "out/exports/b.bin.sha256").read_bytes().endswith(b"payload\n")
    assert (tmp_path / "out/exports/b.bin.complete").read_bytes() == b"complete\n"
    assert published == ["nested/a.json", "b.bin"]
    assert len([call for call in kernel.calls if call[0] == "download"]) == 1


def test_malicious_archive_is_rejected_before_publication(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "remote")
    root = "/content/run"
    entry = _entry("safe.bin", b"safe")
    _remote_export(kernel, root, entry, b"safe")
    archive = tmp_path / "crafted.zip"
    with __import__("zipfile").ZipFile(archive, "w") as crafted:
        crafted.writestr("safe.bin", b"safe")
        crafted.writestr("safe.bin.sha256", str(entry["sha256"]))
        crafted.writestr("safe.bin.complete", b"complete")
        crafted.writestr("extra", b"unexpected")
    kernel.malicious_archive = archive.read_bytes()

    with pytest.raises(export_batches.ExportBatchError, match="member set"):
        export_batches.export_batch(kernel, root, [entry], tmp_path / "out", 1024)
    assert not (tmp_path / "out/exports/safe.bin").exists()


def test_symlink_member_archive_is_rejected_before_publication(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "remote")
    root = "/content/run"
    entry = _entry("safe.bin", b"safe")
    _remote_export(kernel, root, entry, b"safe")
    archive = tmp_path / "symlink.zip"
    with __import__("zipfile").ZipFile(archive, "w") as crafted:
        crafted.writestr("safe.bin", b"safe")
        crafted.writestr("safe.bin.sha256", str(entry["sha256"]))
        link = __import__("zipfile").ZipInfo("safe.bin.complete")
        link.external_attr = 0o120777 << 16
        crafted.writestr(link, b"target")
    kernel.malicious_archive = archive.read_bytes()

    with pytest.raises(export_batches.ExportBatchError, match="unsupported member type"):
        export_batches.export_batch(kernel, root, [entry], tmp_path / "out", 1024)


def test_invalid_remote_archive_metadata_is_rejected_before_download(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "remote")
    kernel.bad_metadata = True
    root = "/content/run"
    entry = _entry("safe.bin", b"safe")
    _remote_export(kernel, root, entry, b"safe")

    with pytest.raises(export_batches.ExportBatchError, match="archive inventory"):
        export_batches.export_batch(kernel, root, [entry], tmp_path / "out", 1024)
    assert not kernel.calls


def test_local_symlink_destination_is_rejected_without_following_it(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "remote")
    root = "/content/run"
    entry = _entry("nested/safe.bin", b"safe")
    _remote_export(kernel, root, entry, b"safe")
    output, outside = tmp_path / "out", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "exports").symlink_to(outside, target_is_directory=True)

    with pytest.raises(export_batches.ExportBatchError, match="symlink ancestor"):
        export_batches.export_batch(kernel, root, [entry], output, 1024)
    assert not list(outside.iterdir())


def test_existing_destination_and_partial_publication_do_not_overwrite_evidence(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "remote")
    root = "/content/run"
    first, second = _entry("first", b"first"), _entry("second", b"second")
    _remote_export(kernel, root, first, b"first")
    _remote_export(kernel, root, second, b"second")
    output = tmp_path / "out"
    (output / "exports").mkdir(parents=True)
    (output / "exports/second").write_bytes(b"old")
    verified: list[str] = []

    with pytest.raises(export_batches.ExportBatchError, match="already exists"):
        export_batches.export_batch(
            kernel, root, [first, second], output, 1024, lambda row: verified.append(row["path"])
        )
    assert (output / "exports/first.complete").read_bytes() == b"complete\n"
    assert (output / "exports/second").read_bytes() == b"old"
    assert verified == ["first"]
