"""Metadata-only regressions for the pinned D4096 successor and access boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_residual.f5_adoption_collision_20261026 import published_metadata as published


@pytest.fixture(scope="module")
def metadata() -> tuple[dict, dict, dict]:
    values = []
    for name in ("d4096-control.json", "d4096-mac-hold.json", "d4096-seed-audit.json"):
        raw, _ = published.published_read(published.PACK / name)
        values.append(json.loads(raw))
    return tuple(values)


def test_published_array_uses_no_newline_digest_and_selected_count(metadata: tuple) -> None:
    control, hold, audit = metadata
    rows = published.validate_allocation(control, hold, audit)
    assert len(rows) == len({r["seed"] for r in rows}) == 7020
    assert {r["world_fold"] for r in rows} == {1, 2, 3, 4}
    assert [rows[0]["allocation_index"], rows[-1]["allocation_index"]] == [3553, 12344]
    assert 12344 - 3553 + 1 == 8792
    encoded = json.dumps(control["allocation"], sort_keys=True, separators=(",", ":")).encode()
    assert published.prior.digest(encoded) == published.ALLOCATION_SHA
    assert published.prior.digest(encoded + b"\n") != published.ALLOCATION_SHA
    assert all(r["aliases"] == [f'{control["namespace"]}/{r["parent_id"]}'] for r in rows)


@pytest.mark.parametrize("mutation", ["reorder", "remove", "seed", "fold", "index"])
def test_changed_published_allocation_is_refused(metadata: tuple, mutation: str) -> None:
    control, hold, audit = copy.deepcopy(metadata)
    allocation = control["allocation"]
    if mutation == "reorder":
        allocation.reverse()
    elif mutation == "remove":
        allocation.pop()
    else:
        key = {"seed": "seed", "fold": "world_fold", "index": "allocation_index"}[mutation]
        allocation[0][key] += 1
    with pytest.raises(ValueError, match="allocation array digest"):
        published.validate_allocation(control, hold, audit)


def test_ordered_audit_must_match_not_just_seed_set(metadata: tuple) -> None:
    control, hold, audit = copy.deepcopy(metadata)
    audit["seeds"].reverse()
    with pytest.raises(ValueError, match="ordered audit seeds"):
        published.validate_allocation(control, hold, audit)


@pytest.mark.parametrize(
    "path",
    [
        published.PACK / "d4096-state-mac-01/metadata.json",
        published.PACK / "d4096-verified-ledger.json",
        published.PACK / "lives/metadata.json",
        published.PACK / "d4096-control.json/../outcomes.json",
        Path("/never/open/t34-round0-study-01/metadata.json"),
        published.PACK.parent / "d4096-old-equiv-02-pack-01/d4096-control.json",
    ],
)
def test_allowlist_refuses_other_paths_before_any_access(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object) -> None:
        pytest.fail("unapproved filesystem access")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="not an expressly allowed"):
        published.published_read(path)


def test_original_protected_guard_is_not_weakened() -> None:
    with pytest.raises(ValueError, match="protected path"):
        published.prior.safe_path(published.PACK / "d4096-control.json")


def test_published_reader_refuses_hash_mismatch_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"{}")
    with pytest.raises(ValueError, match="published metadata hash mismatch"):
        published.published_read(published.PACK / "d4096-control.json")


def test_published_reader_refuses_ancestor_symlink_before_leaf_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = []
    path = published.PACK / "d4096-control.json"

    def symlink(candidate: Path) -> bool:
        checked.append(candidate)
        return candidate == published.PACK

    monkeypatch.setattr(Path, "is_symlink", symlink)
    with pytest.raises(ValueError, match="symlink"):
        published.published_read(path)
    assert path not in checked


def test_shard_reference_cannot_escape_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object) -> None:
        pytest.fail("escaped shard opened")

    monkeypatch.setattr(published.prior, "read", forbidden)
    with pytest.raises(ValueError, match="shard escape"):
        published.load_catalog(
            Path("/metadata"),
            {"alias_record_shards": [{"path": "../outcomes.json"}], "seed_union_shards": []},
        )


def test_whole_bank_is_clear_against_all_published_seeds(metadata: tuple) -> None:
    rows = published.validate_allocation(*metadata)
    result = published.prior.audit(rows, {r["seed"] for r in rows})
    assert result["candidate_round"] == 0
    assert result["rejected_rounds"] == []
    assert len(result["candidate_allocation"]) == 684
    assert result["accepted_round"] is None
    assert result["candidate_allocation_sha256"] == (
        "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
    )


@pytest.mark.parametrize(
    "mutation", ["none", "parent", "selected", "missing", "extra", "count", "overlap"]
)
def test_successor_verifier_binds_exact_ancestry_and_delta(metadata: tuple, mutation: str) -> None:
    expected = published.validate_allocation(*metadata)
    selected = {r["seed"] for r in expected}
    parent_rows = [{"parent_id": "prior", "seed": expected[0]["seed"], "aliases": []}]
    parent_seeds = {expected[0]["seed"], -123}
    rows, seeds = copy.deepcopy(parent_rows + expected), parent_seeds | selected
    report = {
        "selected_rows": 7020,
        "selected_distinct_seeds": 7020,
        "seeds_added": 7019,
        "already_excluded_seeds": [expected[0]["seed"]],
    }
    if mutation == "parent":
        rows[0]["aliases"] = ["forged-ancestry"]
    elif mutation == "selected":
        rows[-1]["seed"] = -456
    elif mutation == "missing":
        seeds.remove(-123)
    elif mutation == "extra":
        seeds.add(-456)
    elif mutation == "count":
        report["seeds_added"] = 7020
    elif mutation == "overlap":
        report["already_excluded_seeds"] = []
    if mutation == "none":
        published.verify_delta(parent_rows, parent_seeds, rows, seeds, expected, report)
    else:
        with pytest.raises(ValueError):
            published.verify_delta(parent_rows, parent_seeds, rows, seeds, expected, report)
