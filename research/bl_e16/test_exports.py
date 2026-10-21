import pytest
import torch

from research.bl_ppo.exports import export_torch, verify_export
from research.colab_cli.runtime import _exports


def test_full_state_has_create_only_sidecars(tmp_path):
    p = tmp_path / "output" / "state.pth"
    r = export_torch(
        p, {"optimizer": {"step": 2}, "actor": {"weight": torch.ones(1)}, "rng": {"seed": 7}}
    )
    assert verify_export(p) == r["sha256"]
    assert _exports(tmp_path, 1024 * 1024) == [
        {"path": "state.pth", "sha256": r["sha256"], "size": r["size"]}
    ]
    with pytest.raises(FileExistsError):
        export_torch(p, {})


def test_truncated_state_is_rejected(tmp_path):
    p = tmp_path / "output" / "state.pth"
    export_torch(p, {"full_state": torch.zeros(2)})
    p.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="digest"):
        verify_export(p)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _exports(tmp_path, 1024 * 1024, {})
