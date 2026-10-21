"""Runtime-export compatibility tests for O1's final evidence bundle."""

from __future__ import annotations

import hashlib
import tarfile

import pytest

from research.colab_cli.runtime import _exports
from research.t34_o1_harvest.export import ExportRefused, bundle


def test_bundle_is_reproducible_and_has_runtime_sidecars(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    (run_dir / "world-000.json").write_text('{"complete": true}\n')
    (run_dir / "decision.json").write_text('{"verdict": "KILL"}\n')
    first = (tmp_path / "first.tar.gz").resolve()
    second = (tmp_path / "second.tar.gz").resolve()
    first_digest, second_digest = bundle(run_dir, first), bundle(run_dir, second)
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name("first.tar.gz.complete").read_text() == "complete\n"
    assert (
        first.with_name("first.tar.gz.sha256").read_text().split()[0]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["decision.json", "world-000.json"]


def test_bundle_refuses_output_inside_run_directory(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}\n")
    with pytest.raises(ExportRefused, match="outside"):
        bundle(run_dir, run_dir / "o1.tar.gz")


def test_failed_fixture_export_is_discovered_only_with_verified_sidecars(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    (run_dir / "failure.json").write_text('{"status":"INCOMPLETE"}\n')
    output = (tmp_path / "output" / "o1.tar.gz").resolve()
    digest = bundle(run_dir, output)
    assert _exports(tmp_path, 64 * 1024**2) == [
        {"path": "o1.tar.gz", "sha256": digest, "size": output.stat().st_size}
    ]
    marker = output.with_name(output.name + ".complete")
    marker.unlink()
    assert _exports(tmp_path, 64 * 1024**2) == []
    marker.write_text("complete\n")
    checksum = output.with_name(output.name + ".sha256")
    checksum.unlink()
    with pytest.raises(RuntimeError, match="invalid completed export marker"):
        _exports(tmp_path, 64 * 1024**2)
    checksum.write_text("0" * 64)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _exports(tmp_path, 64 * 1024**2)
