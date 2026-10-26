"""Offline package invariants; no provider or real corpus fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from research.bl_residual import f5_package as package


def draft() -> dict:
    return {
        "scientific_admission": False,
        "commit": "a" * 40,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 60},
        "timeout_minutes": 200,
        "device": "cuda:0",
        "resume": {
            "unit_kind": "paired-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f5": {"transfer_seconds": 150, "staging_seconds": 300},
    }


def test_draft_is_reviewable_and_cannot_claim_admission() -> None:
    value = draft()
    assert len(package.validate_draft_manifest(value)) == 5
    value["scientific_admission"] = True
    with pytest.raises(ValueError, match="refuse operational"):
        package.validate_draft_manifest(value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeout_minutes", 480),
        ("device", "mps"),
        ("commit", "a" * 8),
        ("requirements", {"gpu": "A100"}),
        ("resume", {"skip_verified": False}),
        ("f5", {"transfer_seconds": 300, "staging_seconds": 300}),
    ],
)
def test_no_widened_timeout_placement_transfer_or_resume(key: str, value: object) -> None:
    candidate = copy.deepcopy(draft())
    candidate[key] = value
    with pytest.raises(ValueError):
        package.validate_draft_manifest(candidate)


def test_one_difference_and_original_denominator() -> None:
    record = package.one_difference()
    assert record["differences"] == ["qualifying_ordinary_B8_survivor_pair_weight"]
    assert record["C"]["qualifying_ordinary_B8_survivor_pair_weight"] == 1
    assert record["T"]["qualifying_ordinary_B8_survivor_pair_weight"] == 4
    assert record["T"]["recipe"]["updates"] == 2000
    assert record["C"]["survival_denominator"] == "original eligible pair count"


def test_prohibited_and_symlink_paths_refuse_before_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prohibited"):
        package.safe_metadata(tmp_path / "d4096" / "anything.json")
    with pytest.raises(ValueError, match="prohibited"):
        package.safe_metadata(tmp_path / "t34-round0-study-01" / "anything.json")
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        package.safe_metadata(link)


def test_bound_metadata_digest_mismatch_refuses(tmp_path: Path) -> None:
    path = tmp_path / "governance.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        package.binding(path, "0" * 64)


def test_final_canary_binds_every_science_byte_and_producer(tmp_path: Path) -> None:
    """The final transport probe must upload the eventual science files exactly."""
    names = {
        "control.json",
        "b8.pth",
        "selected-training.tar.xz",
        "f5-batch-plan.pt",
        "acceptance.json",
        "verified-ledger.json",
        "timing-fixtures.json",
    }
    from research.bl_ppo.exports import export_bytes

    immutable = {"selected-training.tar.xz", "f5-batch-plan.pt", "acceptance.json"}
    rows = []
    for name in sorted(names):
        path = tmp_path / name
        if name in immutable:
            export_bytes(path, b"synthetic inventory only: " + name.encode())
        else:
            path.write_bytes(b"synthetic inventory only: " + name.encode())
        rows.append({"path": str(path), "dest": name, "sha256": package.digest(path)})
    for name in sorted(immutable):
        for suffix in (".sha256", ".complete"):
            path = tmp_path / (name + suffix)
            rows.append({"path": str(path), "dest": name + suffix, "sha256": package.digest(path)})
    science = {
        "inputs": rows,
        "commit": "a" * 40,
        "source_paths": ["source.py"],
        "script": "research/bl_residual/f5_colab.sh",
        "required_exports": ["paired/fit-C.pth", "paired/fit-W.pth"],
    }
    probe = {**science, "inputs": []}
    result = package.bind_qualification_inventory(science, probe)
    assert result["inputs"] == rows
    assert set(result["required_exports"]) == {
        "paired/fit-C.pth",
        "paired/fit-W.pth",
        "qualification/qualification.json",
    }
    assert len(result["science_upload_inventory"]) == len(names) + 6
    assert science["inputs"] == rows
    science["inputs"] = rows[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        package.bind_qualification_inventory(science, probe)
    science["inputs"] = rows
    Path(rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        package.bind_qualification_inventory(science, probe)
