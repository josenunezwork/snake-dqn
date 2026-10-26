"""Synthetic tests for F5 collision metadata catalog; no real study source is opened."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_residual.f5_r10a_closure_20261026 import collision_catalog as catalog


def mirror(tmp_path: Path, name: str, rows: list[dict], classes: list[str]) -> dict:
    """Create a sealed lawful synthetic collision mirror."""
    path = tmp_path / f"{name}.json"
    raw = catalog.canonical({"schema": catalog.MIRROR_SCHEMA, "records": rows})
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_text(f"{catalog.sha(raw)}  {path.name}\n")
    return {
        "id": name,
        "path": str(path),
        "sha256": catalog.sha(raw),
        "classes": classes,
        "seal": "sidecar",
    }


def row(parent: str, seed: int, scopes: list[str]) -> dict:
    return {
        "parent_id": parent,
        "seed": seed,
        "aliases": [f"alias/{parent}"],
        "ancestry": [f"root/{parent}"],
        "scopes": scopes,
        "state": "failed",
    }


def test_missing_attestation_allows_only_provisional_whole_684_audit(tmp_path: Path) -> None:
    ref = mirror(
        tmp_path, "training", [row("p0", 1, ["training_development"])], ["training_development"]
    )
    report = catalog.build(
        {
            "schema": catalog.SCHEMA,
            "mirrors": [ref],
            "prior_annex": None,
            "repository_roots": [],
            "custodian_attestation": None,
        }
    )
    assert report["disposition"] == "HOLD"
    assert report["audit"]["all_684_executed"] is True
    assert report["audit"]["accepted_round"] is None
    assert "protected_allocations" in report["missing_classes"]


def test_complete_bound_attestation_runs_unchanged_whole_bank(tmp_path: Path) -> None:
    ref = mirror(
        tmp_path,
        "all",
        [row("p0", 1, list(catalog.REQUIRED_CLASSES))],
        list(catalog.REQUIRED_CLASSES),
    )
    plan = {
        "schema": catalog.SCHEMA,
        "mirrors": [ref],
        "prior_annex": None,
        "repository_roots": [],
        "custodian_attestation": None,
    }
    initial = catalog.build(plan)
    attestation = {
        "attestor": "synthetic custodian",
        "statement": "COMPLETE_F5_HISTORICAL_COLLISION_REGISTER",
        "catalog_sha256": initial["coverage_hash"],
        "covered_classes": list(catalog.REQUIRED_CLASSES),
    }
    report = catalog.build(
        {
            "schema": catalog.SCHEMA,
            "mirrors": [ref],
            "prior_annex": None,
            "repository_roots": [],
            "custodian_attestation": attestation,
        }
    )
    assert report["complete"] is True
    assert report["audit"]["all_684_executed"] is True
    assert report["audit"]["accepted_round"] is not None


@pytest.mark.parametrize("bad", ["D4096", "t34-round0-study-01"])
def test_protected_source_is_rejected_before_open(tmp_path: Path, bad: str) -> None:
    ref = {
        "id": "protected",
        "path": str(tmp_path / bad / "mirror.json"),
        "sha256": "0" * 64,
        "classes": ["protected_allocations"],
        "seal": "sidecar",
    }
    with pytest.raises(ValueError, match="protected"):
        catalog.read_mirror(ref)


def test_mirror_rejects_foreign_outcome_field(tmp_path: Path) -> None:
    ref = mirror(
        tmp_path,
        "foreign",
        [{**row("p0", 1, ["training_development"]), "return": 99}],
        ["training_development"],
    )
    with pytest.raises(ValueError, match="foreign"):
        catalog.read_mirror(ref)


def test_duplicate_ancestry_refuses_catalog(tmp_path: Path) -> None:
    ref = mirror(
        tmp_path,
        "dupe",
        [row("p0", 1, ["training_development"]), row("p1", 2, ["validation"])],
        ["training_development", "validation"],
    )
    payload = json.loads(Path(ref["path"]).read_text())
    payload["records"][1]["ancestry"] = ["root/p0"]
    raw = catalog.canonical(payload)
    Path(ref["path"]).write_bytes(raw)
    Path(ref["path"] + ".sha256").write_text(f"{catalog.sha(raw)}  dupe.json\n")
    ref["sha256"] = catalog.sha(raw)
    with pytest.raises(ValueError, match="alias or ancestry"):
        catalog.build(
            {
                "schema": catalog.SCHEMA,
                "mirrors": [ref],
                "prior_annex": None,
                "repository_roots": [],
                "custodian_attestation": None,
            }
        )


def test_structural_seed_projection_skips_all_forbidden_values_and_quoted_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = (
        b'{"seeds":[1,1101208186114],"outc\\u006fme":{"seeds":[99]},'
        b'"q":{"seed":88},"note":"escaped \\"seed\\":123"}'
    )
    right = b'{"seeds":[1,1101208186114],"outcome":"opaque arbitrary values","q":[777]}'
    decoded, original = [], catalog.json.loads

    def track(raw, *args, **kwargs):
        decoded.append(raw)
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(catalog.json, "loads", track)
    assert catalog._seed_values(left) == [1, 1101208186114]
    assert catalog._seed_values(right) == [1, 1101208186114]
    assert all(b"1101208186114" not in raw and b"99" not in raw for raw in decoded)


def test_protected_subdirectories_are_pruned_before_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    protected = tmp_path / "d4096-old-equiv-pack"
    protected.mkdir()
    (protected / "SEED_AUDIT.json").write_bytes(b"forbidden synthetic bytes")
    original, scanned = os.scandir, []

    def track(path):
        scanned.append(str(path))
        assert "4096" not in str(path)
        return original(path)

    monkeypatch.setattr(os, "scandir", track)
    register, rows = catalog.discover_repository([str(tmp_path)])
    assert register == [] and rows == []
    assert scanned


def test_symlinked_audit_root_is_refused_before_scan(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="root"):
        catalog.discover_repository([str(alias)])


def test_foreign_mirror_field_fails_before_value_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = mirror(
        tmp_path,
        "foreign-value",
        [{**row("p0", 1, ["validation"]), "outcome": "FORBIDDEN_SENTINEL"}],
        ["validation"],
    )
    original = catalog.json.loads

    def track(raw, *args, **kwargs):
        assert "FORBIDDEN_SENTINEL" not in str(raw)
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(catalog.json, "loads", track)
    with pytest.raises(ValueError, match="foreign"):
        catalog.read_mirror(ref)


def test_nested_object_in_identity_is_refused_without_decoding_its_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = row("p0", 1, ["validation"])
    item["parent_id"] = {"schema": "FORBIDDEN_SENTINEL"}
    ref = mirror(tmp_path, "nested-value", [item], ["validation"])
    original = catalog.json.loads

    def tracked(raw, *args, **kwargs):
        assert "FORBIDDEN_SENTINEL" not in str(raw)
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(catalog.json, "loads", tracked)
    with pytest.raises(ValueError, match="structure"):
        catalog.read_mirror(ref)


def test_catalog_publication_is_create_only_and_completion_last(tmp_path: Path) -> None:
    report = catalog.build(
        {
            "schema": catalog.SCHEMA,
            "mirrors": [],
            "prior_annex": None,
            "repository_roots": [],
            "custodian_attestation": None,
        }
    )
    target = tmp_path / "collision-catalog.json"
    catalog.write(target, report)
    original = target.read_bytes()
    assert Path(str(target) + ".complete.json").is_file()
    with pytest.raises(ValueError, match="create-only"):
        catalog.write(target, {**report, "disposition": "changed"})
    assert target.read_bytes() == original
