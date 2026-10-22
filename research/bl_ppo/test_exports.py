"""Exercise the actual supervisor export discovery, including mandatory sidecars."""

from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, verify_export
from research.colab_cli.runtime import _exports


def test_supervisor_discovers_only_complete_immutable_exports(tmp_path: Path) -> None:
    payload = tmp_path / "output" / "checkpoint.json"
    receipt = export_json(payload, {"counter": 123})
    assert Path(str(payload) + ".sha256").is_file()
    assert Path(str(payload) + ".complete").read_text() == "complete\n"
    assert verify_export(payload) == receipt["sha256"]
    cache = {}
    assert _exports(tmp_path, 1024, cache) == [
        {"path": "checkpoint.json", "sha256": receipt["sha256"], "size": receipt["size"]}
    ]
    with pytest.raises(FileExistsError):
        export_json(payload, {"counter": 124})
    payload.write_text("altered")
    with pytest.raises(RuntimeError, match="changed"):
        _exports(tmp_path, 1024, cache)


def test_missing_sidecar_rejected(tmp_path: Path) -> None:
    payload = tmp_path / "output" / "receipt.json"
    export_json(payload, {"complete": True})
    Path(str(payload) + ".sha256").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_export(payload)
    with pytest.raises(RuntimeError, match="marker"):
        _exports(tmp_path, 1024)


def test_completion_marker_is_required_and_checked_last(tmp_path: Path) -> None:
    payload = tmp_path / "output" / "diagnostics.json"
    export_json(payload, {"scientific": False, "late_body_decisions": 7})
    marker = Path(str(payload) + ".complete")
    marker.unlink()
    assert _exports(tmp_path, 1024) == []
    with pytest.raises(ValueError, match="missing"):
        verify_export(payload)
    marker.write_text("pending\n")
    with pytest.raises(ValueError, match="completion"):
        verify_export(payload)


def test_immutable_export_rejects_renamed_sidecar_and_symlinks(tmp_path: Path) -> None:
    payload = tmp_path / "ledger.json"
    export_json(payload, {"both_alive": 12, "candidate_only": 3, "b8_only": 1})
    sidecar = Path(str(payload) + ".sha256")
    checksum = verify_export(payload)
    sidecar.write_text(f"{checksum}  another-name.json\n")
    with pytest.raises(ValueError, match="name mismatch"):
        verify_export(payload)
    sidecar.unlink()
    external = tmp_path / "other.sha256"
    external.write_text(f"{checksum}  {payload.name}\n")
    sidecar.symlink_to(external)
    with pytest.raises(ValueError, match="unsafe"):
        verify_export(payload)


def test_oversize_snapshot_refused_before_publication(tmp_path: Path) -> None:
    from research.bl_ppo.exports import export_torch

    target = tmp_path / "too-large.pt"
    with pytest.raises(ValueError, match="byte bound"):
        export_torch(target, {"state": "bounded"}, max_payload_bytes=1)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_job_export_budget_counts_payloads_and_sidecars_before_publication(tmp_path: Path) -> None:
    from research.bl_ppo.exports import export_bytes, limit_exports

    root = tmp_path / "output"
    # a.pt payload=10, SHA sidecar=71, completion marker=9 =>90 bytes.
    with limit_exports(root, total_bytes=90, file_bytes=10):
        export_bytes(root / "a.pt", b"0123456789")
        with pytest.raises(ValueError, match="max_total_export_bytes"):
            export_bytes(root / "b.pt", b"a")
        with pytest.raises(ValueError, match="max_export_bytes"):
            export_bytes(root / "c.pt", b"01234567890")
        with pytest.raises(ValueError, match="escaped"):
            export_bytes(tmp_path / "outside.pt", b"a")
    assert sum(p.stat().st_size for p in root.iterdir()) == 90
    assert not (root / "b.pt").exists()
    # Context is scoped to the invocation, never process-global afterwards.
    export_bytes(tmp_path / "outside.pt", b"a")
