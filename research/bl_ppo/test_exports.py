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
