"""T1C capture and world-held-out analysis contracts (development diagnostic only)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.t1_fuse_probe_20261015 import collect
from src.model.ego2s_network import Ego2sNet


def test_nested_world_folds_never_leak_across_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """All outer and inner model fits hold every copy of the test world out."""
    from research.t1_fuse_probe_20261015 import analyze

    worlds = np.tile(np.repeat(np.arange(10), 4), 2)
    y = np.tile([0, 1], len(worlds) // 2)
    X = np.column_stack([worlds, y]).astype(np.float32)
    calls = []

    def fake_fit(
        train: np.ndarray,
        target: np.ndarray,
        test: np.ndarray,
        c_value: float,
    ) -> tuple[np.ndarray, None]:
        assert set(train[:, 0]).isdisjoint(set(test[:, 0]))
        calls.append(c_value)
        return test[:, 1] * 0.8 + 0.1, None

    monkeypatch.setattr(analyze, "_safe_probability", fake_fit)
    pred, chosen, notes = analyze.nested_oof(X, y, worlds)
    assert len(calls) == 5 * (3 * 3 + 1)
    assert len(chosen) == 5 and not notes
    assert np.array_equal(pred > 0.5, y.astype(bool))
    fold = analyze.folds_by_world(worlds)
    assert len(np.unique(fold)) == 5
    for world in np.unique(worlds):
        assert len(np.unique(fold[worlds == world])) == 1
    assert np.array_equal(fold, analyze.folds_by_world(worlds))


def test_bootstrap_auc_ties_and_reliability_endpoints() -> None:
    """Tie-aware AUC, deterministic world bootstrap, and probability-one binning."""
    from research.t1_fuse_probe_20261015 import analyze

    score = np.array([0.0, 0.5, 0.5, 1.0])
    y = np.array([0, 0, 1, 1])
    worlds = np.array([1, 1, 2, 2])
    result = analyze.auc_ci(score, y, worlds, boot=100)
    assert result["auc"] == 0.875
    assert result == analyze.auc_ci(score, y, worlds, boot=100)
    table = analyze.reliability(score, y)
    assert len(table) == 10
    assert sum(row["n"] for row in table) == 4
    assert table[-1]["n"] == 1 and table[-1]["observed_rate"] == 1


def test_resume_recomputes_partial_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint after H50 must not make a resume silently omit H100/H300."""
    import json

    from research.t1_fuse_probe_20261015 import analyze

    source = tmp_path / "B8_s0.npz"
    source.write_bytes(b"synthetic source")
    data = {"world": np.array([1, 2])}
    monkeypatch.setattr(analyze, "_load_capture", lambda path: data)
    monkeypatch.setattr(analyze, "_sklearn_dependencies", lambda: None)
    old = {
        "methods": {"ci": "7-sample world-cluster percentile bootstrap, no bootstrap refits"},
        "scopes": {"B8_s0": {"by_horizon": {"50": {"rows": 2}}}},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(old))
    for h in (50, 100, 300):
        (tmp_path / f"oof_B8_s0_h{h}.npz").touch()
    calls = []

    def fake_scope(name: str, arrays: dict, root: Path, boot: int, completed: object) -> dict:
        calls.append(name)
        return {"by_horizon": {str(h): {"rows": 2} for h in (50, 100, 300)}}

    monkeypatch.setattr(analyze, "_analyze_scope", fake_scope)
    assert analyze.main(["--root", str(tmp_path), "--boot", "7", "--resume"]) == 0
    assert calls == ["B8_s0"]
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert set(result["scopes"]["B8_s0"]["by_horizon"]) == {"50", "100", "300"}


def test_synthetic_capture_is_post_relu_and_256_dimensional() -> None:
    """A real fuse block emits its post-ReLU output, aligned to selected hero rows."""
    torch.set_num_threads(2)
    net = Ego2sNet(local_channels=7, n_scalars=60).eval()
    with torch.no_grad():
        net.fuse[0].weight.zero_()
        net.fuse[0].bias.copy_(torch.linspace(-2, 2, 256))
    x = torch.zeros(2, 444)
    with collect.fuse_hook(net) as captured:
        with torch.no_grad():
            before_relu = net.fuse[1](net.fuse[0](x))
            expected = net.fuse(x)
    features = np.concatenate(captured)
    assert (before_relu < 0).any()
    assert features.shape == (2, 256)
    assert features.dtype == np.float16
    assert np.array_equal(features, expected.numpy().astype(np.float16))
    assert (features >= 0).all() and (features == 0).any()
    assert len(net.fuse[2]._forward_hooks) == 0

    col = collect.Collector([101, 202])
    col.frame = 3
    col.record_fuse(
        SimpleNamespace(length=np.array([[900], [700]])),
        np.array([[0, 0], [1, 0]]),
        features,
        np.zeros((2, 6)),
        np.array([1, 2]),
    )
    col.death_frame[0], col.death_cause[0] = 50, 2
    data = col.output("synthetic", 5000)
    assert data["features"].shape == (1, 256)
    assert data["world"].tolist() == [101]
    assert data["y50"].tolist() == [True]
    assert data["keep50"].tolist() == [True]


def test_label_boundaries_competing_deaths_and_censoring() -> None:
    """Match M7's fatal-inclusive, strict distance<H and survivor horizon boundaries."""
    fd = np.array([149, 150, 149, -1, -1, 100])
    cause = np.array([2, 4, 3, 0, 0, 4])
    frame = np.array([100, 100, 100, 4950, 4951, 100])
    y, keep = collect.labels(fd, cause, frame, 50, 5000)
    assert y.tolist() == [True, False, False, False, False, True]
    assert keep.tolist() == [True, True, False, True, False, True]


def test_hook_removed_after_exception() -> None:
    """Instrumentation never leaves a hook attached after a failing capture."""
    net = Ego2sNet()
    with pytest.raises(RuntimeError):
        with collect.fuse_hook(net):
            raise RuntimeError("synthetic failure")
    assert len(net.fuse[2]._forward_hooks) == 0


@pytest.mark.slow
def test_capture_one_arm_two_worlds_two_hundred_frames(tmp_path: Path) -> None:
    """Exercise actual checkpoint/featurizer/eval wiring, including nonempty hook rows."""
    meta = collect.capture("B8_s0", tmp_path, worlds=2, frames=200, min_len=0)
    assert meta["checks"].get("observer_action_mismatch", 0) == 0
    assert meta["rows"] > 0
    with np.load(tmp_path / "runs" / "B8_s0.npz") as data:
        assert data["features"].shape == (meta["rows"], 256)
        assert data["features"].dtype == np.float16
        assert np.isfinite(data["features"]).all()
        assert len(np.unique(data["world"])) == 2
        with np.load(collect.M7_ROOT / "runs" / "B8_s0.npz") as old:
            sel = (old["env"] < 2) & (old["frame"] < 200) & (old["frame"] % 3 == 0)
            for key in ("env", "frame", "length", "q", "action"):
                assert np.array_equal(data[key], old[key][sel]), key
