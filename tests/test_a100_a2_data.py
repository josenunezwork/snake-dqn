from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from research.a100_a2 import data


def _normal(n: int = 2) -> dict[str, np.ndarray]:
    return {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "world": np.array(["w0"] * n, dtype="U64"),
        "fold": np.zeros(n, np.int16),
        "legal": np.ones((n, 6), bool),
        "died": np.zeros((n, 6), bool),
        "alive_steps": np.full((n, 6), 300, np.int16),
        "observed": np.ones((n, 6), bool),
        "decisive": np.zeros(n, bool),
        "a_star": np.full(n, -1, np.int16),
        "q": np.full((n, 6), np.nan, np.float32),
    }


def test_save_table_enforces_registered_shapes(tmp_path: Path) -> None:
    out = data._save_table(tmp_path / "t5.npz", _normal(), "t5")
    with np.load(out, allow_pickle=False) as archive:
        assert str(archive["schema"]) == data.SCHEMA
        assert archive["local"].shape == (2, 7, 31, 31)
        assert archive["q"].shape == (2, 6)
    bad = _normal()
    bad["observed"] = np.ones((2, 5), bool)
    with pytest.raises(ValueError, match="observed"):
        data._save_table(tmp_path / "bad.npz", bad, "t5")


def test_verify_bundle_refuses_mutated_bound_input(tmp_path: Path) -> None:
    payload = tmp_path / "x.bin"
    payload.write_bytes(b"bound")
    receipt = {
        "schema": "a2-input-bundle/v1",
        "files": [{"path": "x.bin", "sha256": data.sha256(payload), "bytes": 5}],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(receipt))
    assert data._verify_bundle(tmp_path)["schema"] == "a2-input-bundle/v1"
    payload.write_bytes(b"other")
    with pytest.raises(ValueError, match="hash mismatch"):
        data._verify_bundle(tmp_path)


def test_allowlisted_a5_discovery_requires_both_sidecars(tmp_path: Path) -> None:
    world = tmp_path / "exports/worlds/process-000/worlds/world-000000.json"
    world.parent.mkdir(parents=True)
    world.write_text("{}")
    world.with_name(world.name + ".complete").write_text("")
    with pytest.raises(ValueError, match="sidecar"):
        data._a5_world_paths(tmp_path)
    world.with_name(world.name + ".sha256").write_text("x")
    assert data._a5_world_paths(tmp_path) == [world]


def test_t5_event_intervals_keep_censoring_and_shift_deaths() -> None:
    horizon = np.array([[30, 60, 100, 200, 300], [30, 60, 100, 200, 300]], np.int16)
    died = np.array([[True, True, True, True, True], [False, False, False, False, False]])
    got = data._t5_alive_steps(died, horizon, np.ones_like(died, bool))
    assert got[0].tolist() == [29, 59, 99, 199, 299]
    assert got[1].tolist() == [30, 60, 100, 200, 300]


def test_t5_event_intervals_preserve_illegal_zero_sentinel() -> None:
    died = np.array([[False, True]], bool)
    horizon = np.array([[0, 30]], np.int16)
    legal = np.array([[False, True]], bool)
    assert data._t5_alive_steps(died, horizon, legal).tolist() == [[0, 29]]


def test_safe_extract_refuses_path_escape_and_accepts_root_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "labels.tar.gz"
    source = tmp_path / "source"
    source.mkdir()
    (source / "bundle.json").write_text("{}")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source / "bundle.json", arcname="bundle.json")
    destination = tmp_path / "unpack"
    destination.mkdir()
    assert data._safe_extract(archive, destination) == destination
