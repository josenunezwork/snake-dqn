"""No-fit contract tests for the F4 D256 package preparer.

These fixtures exercise only synthetic JSON rows and mocked admission boundaries.
They must never call a residual fit, an optimizer, a reserve look, or a factory
payload outside the temporary fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from research.bl_residual import f4_d256_package as package


def test_job_cli_accepts_manifest_cuda_index_without_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.bl_residual import job

    calls = []

    def fake_run(inputs: Path, output: Path, *, mode: str, device: str) -> dict:
        calls.append((inputs, output, mode, device))
        return {"done": 1, "synthetic": False}

    monkeypatch.setattr(job, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job",
            "--input-dir",
            str(tmp_path / "inputs"),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "main",
            "--device",
            "cuda:0",
        ],
    )
    job.main()
    assert calls == [(tmp_path / "inputs", tmp_path / "out", "main", "cuda:0")]


def _row(*, ordinary: bool, status: list[str] | None = None, probability: float = 0.25) -> dict:
    return {
        "local": np.zeros((7, 31, 31), dtype=np.uint8).tolist(),
        "global": np.zeros((4, 37, 37), dtype=np.uint8).tolist(),
        "scalars": np.zeros(60, dtype=np.float32).tolist(),
        "q": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "legal": [True] * 6,
        "bare": 0,
        "died": [False] * 6,
        "status": status or ["EXACT"] * 6,
        "mass": [1.0] * 6,
        "sampling_kind": "ordinary" if ordinary else "doom",
        "sampling_prob": probability,
        "decision": 1 if ordinary else 0,
    }


def _life(path: Path, *, identity: str, parent: str, rescue: bool, rows: list[dict]) -> dict:
    path.write_text(json.dumps({"rows": rows}))
    return {
        "life_id": identity,
        "parent_world": parent,
        "rescue_bearing": rescue,
        "path": path,
    }


def test_timing_arm_refuses_before_publication_or_payload_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arm = tmp_path / "mac-timing-armed"
    arm.write_text("armed\n")
    monkeypatch.setattr(package, "TIMING_ARM", arm)
    monkeypatch.setattr(
        package, "require_publication", lambda *_: pytest.fail("publication opened")
    )
    monkeypatch.setattr(package, "_life_catalog", lambda *_: pytest.fail("labels opened"))
    with pytest.raises(ValueError, match="window remains armed"):
        package._prepare(
            tmp_path / "out",
            factory_ledger=tmp_path / "ledger",
            b8=tmp_path / "b8",
            judgment=tmp_path / "judgment",
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight",
        )


def test_publication_failure_precedes_factory_label_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(package, "require_window_closed", lambda: None)
    monkeypatch.setattr(
        package,
        "require_publication",
        lambda *_: (_ for _ in ()).throw(ValueError("unpublished ledger")),
    )
    monkeypatch.setattr(package, "_life_catalog", lambda *_: pytest.fail("labels opened"))
    with pytest.raises(ValueError, match="unpublished ledger"):
        package._prepare(
            tmp_path / "out",
            factory_ledger=tmp_path / "ledger",
            b8=tmp_path / "b8",
            judgment=tmp_path / "judgment",
            commit="HEAD",
            preflight_receipt=tmp_path / "preflight",
        )


def test_table_keeps_unknown_sentinels_masks_and_unclipped_probability_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(package, "advisory_from_row", lambda _row: np.array([True, False] * 3))
    selected = _life(
        tmp_path / "selected.json",
        identity="selected",
        parent="parent-a",
        rescue=True,
        rows=[_row(ordinary=False, probability=0.1)],
    )
    archive = _life(
        tmp_path / "archive.json",
        identity="archive",
        parent="parent-a",
        rescue=False,
        rows=[
            _row(
                ordinary=True,
                status=["EXACT", "UNKNOWN", "NOT_EVALUATED", "EXACT", "EXACT", "EXACT"],
                probability=0.5,
            )
        ],
    )
    unselected_rescue = _life(
        tmp_path / "unselected-rescue.json",
        identity="extra",
        parent="parent-a",
        rescue=True,
        rows=[_row(ordinary=True)],
    )
    table = package._table([selected, archive, unselected_rescue], {"selected"}, {"parent-a"})
    assert table["life_id"].tolist() == ["selected", "archive"]
    assert table["status"][1, 1] == "UNKNOWN"
    assert table["status"][1, 2] == "NOT_EVALUATED"
    assert table["action_mask"].tolist() == [[True, False, True, False, True, False]] * 2
    assert table["sampling_weight"].tolist() == pytest.approx([10.0, 2.0])
    assert table["ordinary"].tolist() == [False, True]


def test_stage_d1024_refuses_unpublished_fit_phase_before_checkpoint_or_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(package, "ROOT", tmp_path)
    monkeypatch.setattr(package, "require_window_closed", lambda: None)
    publication = tmp_path / package.FIT_PUBLICATION_PATH
    publication.parent.mkdir(parents=True)
    publication.write_text("local\n")
    monkeypatch.setattr(
        package.subprocess, "check_output", lambda *_args, **_kwargs: b"origin-main\n"
    )
    with pytest.raises(ValueError, match="match origin/main"):
        package.stage_d1024(tmp_path / "pack", publication, tmp_path / "out")


def test_canary_covers_main_transport_and_caps_leave_series_headroom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli.canary import job_device

    entrypoint = (package.ROOT / "research/bl_residual/f4_d256_colab.sh").read_text()
    monkeypatch.setattr("research.colab_cli.canary._entrypoint", lambda *_: entrypoint)
    inventory = [
        {"path": "/fixture/b8.pth", "dest": "b8.pth", "sha256": "a" * 64},
        {"path": "/fixture/metadata.tar", "dest": "metadata.tar", "sha256": "b" * 64},
    ]
    main, canary = package.manifests(tmp_path, "b" * 40, inventory)
    assert canary["inputs"] == main["inputs"] == inventory
    assert canary["env"]["METADATA_ARCHIVE_SHA256"] == "b" * 64
    assert canary["script"] == main["script"] == canary["canary"]["family"]
    assert canary["required_exports"] == main["required_exports"]
    assert main["output_dir"] != canary["output_dir"]
    liability = 0
    for manifest in (main, canary):
        assert manifest["device"] == manifest["env"]["F4_DEVICE"] == "cuda:0"
        # Exercise the qualified parser against the real shell bytes, including
        # its required-variable guard and manifest-bound --device argument.
        assert job_device(manifest) == "cuda"
        assert job_device(manifest, package.ROOT) == "cuda"
        limits, budget = manifest["f4"], manifest["budget"]
        assert budget["unit_cap"] == 2 * limits["base_units"]
        assert budget["reserve_units"] == 2 * limits["base_reserve_units"]
        assert manifest["timeout_minutes"] == 2 * limits["base_timeout_minutes"]
        assert manifest["requirements"]["allowed_gpus"] == ["L4"]
        assert (
            manifest["timeout_minutes"] * budget["rate_units_per_hour"] / 60 <= budget["unit_cap"]
        )
        assert manifest["resume"]["skip_verified"] is True
        liability += budget["unit_cap"] + budget["reserve_units"]
    assert liability == pytest.approx(6)
    assert liability < main["f4"]["fit_series_effective_units"] == 20
