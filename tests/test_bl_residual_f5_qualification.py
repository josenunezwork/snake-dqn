"""Offline contracts for the F5 source-native CUDA qualification runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.bl_residual import f5_qualification as subject
from research.bl_residual.f5_r10a_annex import allocate


def _fixtures() -> dict:
    return {
        "schema": subject.FIXTURE_SCHEMA,
        "fixtures": [row for row in allocate(0) if row["timing_only"] is True],
    }


def test_fixture_loader_accepts_the_exact_sealed_disjoint_twelve(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(_fixtures()), encoding="utf-8")
    rows, sha = subject.load_fixtures(path)
    assert len(rows) == 12
    assert len(sha) == 64
    assert {row["group"] for row in rows} == {
        "timing/watch6",
        "timing/scripted",
        "timing/frozen",
        "timing/mixed",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["fixtures"][0].__setitem__("unknown", True),
        lambda value: value.__setitem__("outcomes", ["forbidden"]),
        lambda value: value["fixtures"][0].__setitem__("seed", 0),
        lambda value: value["fixtures"].pop(),
        lambda value: value["fixtures"][0].__setitem__("parent_id", "science/forbidden"),
    ],
)
def test_fixture_loader_refuses_payload_unknown_or_noncanonical_rows(
    tmp_path: Path, mutate
) -> None:
    value = copy.deepcopy(_fixtures())
    mutate(value)
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(subject.QualificationRefusal):
        subject.load_fixtures(path)


def test_producer_recovery_exports_are_create_only_and_unusable_as_fit(tmp_path: Path) -> None:
    exports = subject._producer_exports(tmp_path, fixture_sha256="a" * 64, count=12)
    assert {Path(row["path"]).relative_to(tmp_path).as_posix() for row in exports} == {
        "paired/fit-receipt.json",
        "paired/assignment-environment.json",
        "paired/fit-C.pth",
        "paired/fit-W.pth",
        "paired/f5-state-C.pt",
        "paired/f5-state-W.pt",
    }
    for row in exports:
        path = Path(row["path"])
        payload = (
            json.loads(path.read_text())
            if path.suffix == ".json"
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        assert payload["fit_usable"] is False
        assert Path(str(row["path"]) + ".sha256").is_file()
        assert Path(str(row["path"]) + ".complete").is_file()
    with pytest.raises(FileExistsError):
        subject._producer_exports(tmp_path, fixture_sha256="a" * 64, count=12)


def test_cpu_or_implicit_device_is_not_a_cuda_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(subject.QualificationRefusal, match="cuda:0"):
        subject._require_cuda("cuda:0")
    with pytest.raises(subject.QualificationRefusal, match="cuda:0"):
        subject._require_cuda("cpu")


def test_native_fixture_observation_uses_production_grid_and_bv8_features() -> None:
    observation, mask = subject._native_observation(_fixtures()["fixtures"][0]["seed"])
    assert observation["local"].shape == (1, 7, 31, 31)
    assert observation["global"].shape == (1, 4, 37, 37)
    assert observation["scalars"].shape == (1, 60)
    assert mask.shape == (6,)
    assert mask.any()


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires a real CUDA runtime"
)
def test_cuda_path_is_available_only_with_real_hardware() -> None:
    subject._require_cuda("cuda:0")


def test_calibration_schema_asset_is_present_and_digestable() -> None:
    assert len(subject._calibration_schema_sha256()) == 64
