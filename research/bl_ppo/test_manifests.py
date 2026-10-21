"""Focused immutable-manifest checks for the BL E4/E2 handoff."""

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_ppo.manifests import (
    ManifestRefused,
    e2_transitions,
    git_head,
    make_e2_manifests,
    make_e2_templates,
    make_e4_manifest,
)

COMMIT = git_head()


def _roster(tmp_path: Path) -> tuple[Path, ...]:
    rows = []
    for ordinal in range(5):
        path = tmp_path / f"roster-{ordinal}.pth"
        path.write_bytes(bytes([ordinal]))
        rows.append(path)
    return tuple(rows)


def _complete(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row))
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")


def test_e4_has_two_process_ruler_and_whole_session_cap(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    manifest = make_e4_manifest(
        commit=COMMIT,
        output_dir=tmp_path / "out",
        b8_checkpoint=b8,
        roster_checkpoints=_roster(tmp_path),
    )
    assert manifest["env"]["BL_PPO_PROCESS_COUNTS"] == "6,12"
    assert manifest["env"]["BL_PPO_WORLDS"] == "128"
    assert manifest["timeout_minutes"] == 30
    assert manifest["budget"]["unit_cap"] == 2.0


def test_e2_requires_complete_e4_and_uses_4m_below_kill_threshold(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    receipt = tmp_path / "e4.json"
    row = {
        "schema": "bl-ppo-e4-ruler/v1",
        "status": "COMPLETE",
        "scientific": True,
        "worlds": 128,
        "long_bodies": True,
        "selected_processes": 6,
        "configs": [
            {"processes": 6, "allocated_hero_transitions_per_second": 999.0},
            {"processes": 12, "allocated_hero_transitions_per_second": 800.0},
        ],
    }
    _complete(receipt, row)
    manifests = make_e2_manifests(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        e4_receipt=receipt,
        roster_checkpoints=_roster(tmp_path),
    )
    assert {m["env"]["BL_PPO_ARM"] for m in manifests} == {"R", "F"}
    assert {m["env"]["BL_PPO_TRANSITIONS"] for m in manifests} == {"4000000"}
    assert {m["env"]["BL_PPO_PROCESSES"] for m in manifests} == {"6"}
    assert {m["env"]["BL_PPO_CHECKPOINT_EVERY"] for m in manifests} == {"120"}
    assert all(m["budget"]["unit_cap"] == 9.0 for m in manifests)


def test_e2_requires_e4_and_explicit_budget_fitting_20m_switch(tmp_path: Path) -> None:
    receipt = tmp_path / "e4.json"
    receipt.write_text("{}")
    with pytest.raises(ManifestRefused):
        e2_transitions(receipt)
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    row = {
        "schema": "bl-ppo-e4-ruler/v1",
        "status": "COMPLETE",
        "scientific": True,
        "worlds": 128,
        "long_bodies": True,
        "selected_processes": 12,
        "configs": [
            {"processes": 6, "allocated_hero_transitions_per_second": 4100.0},
            {"processes": 12, "allocated_hero_transitions_per_second": 4200.0},
        ],
    }
    _complete(receipt, row)
    assert e2_transitions(receipt) == 8_000_000
    jobs = make_e2_manifests(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        e4_receipt=receipt,
        roster_checkpoints=_roster(tmp_path),
        request_scale=True,
    )
    assert {job["env"]["BL_PPO_TRANSITIONS"] for job in jobs} == {"20000000"}
    assert all(2 * job["requirements"]["est_minutes"] <= job["timeout_minutes"] for job in jobs)


def test_templates_have_no_invented_e4_input_and_runtime_demands_sidecars(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    templates = make_e2_templates(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        roster_checkpoints=_roster(tmp_path),
    )
    assert all(item["bl_ppo"]["template_only"] for item in templates)
    assert all(
        not any(row["dest"] == "e4-ruler.json" for row in item["inputs"]) for item in templates
    )
    script = (Path(__file__).with_name("colab.sh")).read_text()
    assert "${payload}.sha256" in script
    assert "${payload}.complete" in script


def test_materialized_e4_transport_includes_verifiable_sidecars(tmp_path: Path) -> None:
    import shutil

    from research.bl_ppo.exports import verify_export

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    receipt = tmp_path / "e4.json"
    _complete(
        receipt,
        {
            "schema": "bl-ppo-e4-ruler/v1",
            "status": "COMPLETE",
            "scientific": True,
            "worlds": 128,
            "long_bodies": True,
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": 2000},
                {"processes": 12, "allocated_hero_transitions_per_second": 1900},
            ],
        },
    )
    job = make_e2_manifests(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        e4_receipt=receipt,
        roster_checkpoints=_roster(tmp_path),
    )[0]
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    for row in job["inputs"]:
        target = uploaded / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    verify_export(uploaded / job["env"]["BL_PPO_E4_BASENAME"])
    assert job["env"]["BL_PPO_DEVICE"] == "cuda"
