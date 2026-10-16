"""Focused contracts for the T5B fuse diagnostic."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from research.t5b_risk_and_fuse_20261016.fuse import (  # noqa: E402
    design,
    forward_final,
    saved_folds,
)


def _data() -> dict[str, np.ndarray]:
    return {
        "scalars": np.zeros((3, 60), np.float32),
        "features": np.zeros((3, 24), np.float32),
        "legal": np.array([[1, 1, 0, 1, 0, 1], [1] * 6, [1] * 6], bool),
        "died": np.array([[0, 1, 0, 1, 0, 0], [1] * 6, [0] * 6], bool),
    }


def test_design_preserves_legal_branch_labels_and_action_context() -> None:
    data = _data()
    fuse = np.zeros((3, 256), np.float32)
    q = np.zeros((3, 6), np.float32)
    for full, width in ((False, 268), (True, 352)):
        x, y, state, action = design(data, fuse, q, full)
        assert x.shape == (16, width)
        np.testing.assert_array_equal(y, data["died"][state, action])
        np.testing.assert_array_equal(x[:, -6:], np.eye(6)[action])


def test_forward_hook_reads_post_relu_fuse_output(tmp_path) -> None:
    import torch

    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    net = Ego2sNet(local_channels=7, n_scalars=60)
    with torch.no_grad():
        net.fuse[0].weight.zero_()
        net.fuse[0].bias.copy_(torch.arange(256) - 128)
    checkpoint = tmp_path / "net.pth"
    save_ego2s_checkpoint(checkpoint, net)
    local = np.zeros((2, 7, 31, 31), np.uint8)
    glob = np.zeros((2, 4, 37, 37), np.uint8)
    packed = pack_ref(local, glob, train_layout())
    fuse, q = forward_final(checkpoint, packed, np.zeros((2, 60), np.float32))
    assert fuse.shape == (2, 256) and q.shape == (2, 6)
    assert np.all(fuse >= 0)
    assert np.all(fuse[:, :128] == 0)
    assert np.any(fuse[:, 129:] > 0)


def test_saved_folds_reject_state_identity_mismatch(tmp_path) -> None:
    data = {"world": np.array(["w0", "w1", "w2", "w3", "w4"]), "state_id": np.arange(5)}
    path = tmp_path / "folds.npz"
    np.savez(path, world=data["world"], state_id=np.arange(1, 6), fold=np.arange(5))
    with pytest.raises(ValueError, match="do not align"):
        saved_folds(data, path)
