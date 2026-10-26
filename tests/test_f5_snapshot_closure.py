"""Outcome-free checks for the finite metadata reconciliation supplement."""

from __future__ import annotations

import copy
import json

import pytest

from research.bl_residual.f5_adoption_collision_20261026 import close_snapshot as closure


def test_complete_finite_snapshot_has_exactly_one_disposition_per_addition() -> None:
    snapshot, _ = closure.prior.read(
        {
            "path": str(closure.PACKAGE / "campaign-package-register-filename-inventory.json"),
            "sha256": closure.SNAPSHOT_SHA,
        }
    )
    additions = [
        i
        for i, row in enumerate(snapshot["sources"])
        if row["prior_catalog_crosswalk"] != "HASH_BOUND_SOURCE"
    ]
    assert len(additions) == 135
    assert all(closure.disposition(i) for i in additions)
    assert closure.OPAQUE_INDICES <= set(additions)
    assert len(closure.OPAQUE_INDICES) == 17
    with pytest.raises(ValueError, match="unclassified"):
        closure.disposition(999)


def test_p3_rejected_and_successor_rounds_reconstruct_physical_roots() -> None:
    assert [r["seed"] for r in closure.p3_rows(0)] == [
        1816274146,
        2687663043,
        1365248012,
        1425787744,
        3578048036,
    ]
    assert closure.seed_digest([r["seed"] for r in closure.p3_rows(1)]) == (
        "69930595a80cb549959f2068d045eeff61b75f03cbadfefb9cef2d86d31614ea"
    )
    assert not {r["seed"] for r in closure.p3_rows(0)} & {r["seed"] for r in closure.p3_rows(1)}


def test_p1_endpoint_is_complete_384_not_fit_rng_seeds() -> None:
    values = closure.p1_seeds()
    assert len(values) == len(set(values)) == 384
    assert closure.seed_digest(values) == (
        "0be2107f7a63dbe2bc1ab65ae6b342c21dffccfdf7605e89049371aa371674f9"
    )


@pytest.mark.parametrize(
    "suffix,count,expected",
    [
        ("main", 12, "c1adf89fb82d903ea84e4ea3fc60042e58df41bcf61d543084f42432a7f7a6d3"),
        ("canary", 2, "ddecb0c7597d8e86ba15e61237dd716df30a7f9e4bad1d4961505ac610219257"),
        ("local-proof", 3, "d0540f48d868950765878ac20ba21bedf85b582005601aaa1ec96434cf8a6b4a"),
    ],
)
def test_gpu_digest_binds_reconstructed_namespaces(suffix: str, count: int, expected: str) -> None:
    seeds = closure.prep_seeds(f"t34-gpu-identity-20261024/{suffix}/v1", count)
    assert closure.seed_digest(seeds) == expected


def test_life_aliases_require_original_parent_and_preserve_descendants() -> None:
    value = {
        "parents": {"old-namespace/life-7": "recorded-parent"},
        "catalog": [{"life_id": "old-namespace/life-7", "rescue_bearing": "NEVER_READ"}],
    }
    rows = closure.life_rows(
        value, {"recorded-parent": {"parent_id": "ultimate-root", "seed": 123}}
    )
    assert rows == [
        {
            "parent_id": "ultimate-root",
            "seed": 123,
            "aliases": ["recorded-parent", "old-namespace/life-7"],
        }
    ]
    with pytest.raises(ValueError, match="unresolved life ancestor"):
        closure.life_rows(value, {})
    bad = copy.deepcopy(value)
    bad["catalog"][0]["life_id"] = "foreign-life"
    with pytest.raises(ValueError, match="mismatch"):
        closure.life_rows(bad, {"recorded-parent": {"parent_id": "root", "seed": 123}})


def test_literal_projection_distinguishes_assay_fit_and_world_seeds() -> None:
    value = {
        "assay_seed": 1,
        "fit_seed": 2,
        "seed_count": 3,
        "world_seed": -8,
        "seeds": [2**63, 9],
        "seed_sha256": "ignore",
    }
    assert closure.literal_seeds(value) == {-8, 2**63, 9}


def test_missing_opaque_source_is_never_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    # A scientific output is reconciled through source fixtures. Even a read
    # attempt would violate this test; its old snapshot hash stays attributed.
    snapshot = {"sources": [{"prior_catalog_crosswalk": "HASH_BOUND_SOURCE"} for _ in range(301)]}
    for i in closure.OPAQUE_INDICES:
        snapshot["sources"][i] = {
            "prior_catalog_crosswalk": "NEEDS_RECONCILIATION",
            "path": "/never/open/discrepancy-register.json",
            "sha256": "0" * 64,
        }

    def refuse_read(*args: object) -> None:
        pytest.fail("opaque output opened")

    monkeypatch.setattr(closure.prior, "read", refuse_read)
    # Finite-set completeness is independently mandatory after projection.
    with pytest.raises(ValueError, match="finite snapshot coverage changed"):
        closure.crosswalk(snapshot, {})


def test_certificate_template_never_asserts_custodian_approval() -> None:
    manifest = {"register_snapshot_cutoff_utc": "2026-10-26T02:13:30Z"}
    evaluation = {
        "candidate_round": 0,
        "candidate_allocation_sha256": "1" * 64,
        "rejected_rounds": [],
    }
    certificate = closure.prior.certificate_template(manifest, evaluation)
    assert certificate["status"] == "UNATTESTED_TEMPLATE"
    assert certificate["omissions"] is None
    assert certificate["issuer"] is None
    assert all(value is None for value in certificate["assertions"].values())
    assert json.loads(closure.prior.canonical(certificate))["approval_record"] is None


def test_exhausted_allocator_cannot_be_sealed_or_verified_clear() -> None:
    with pytest.raises(ValueError, match="HOLD"):
        closure.require_clear({"candidate_round": None, "status": "HOLD_NO_CLEAR_ROUND"})
    closure.require_clear(
        {"candidate_round": 0, "status": "OBSERVED_METADATA_CLEAR_PENDING_CUSTODIAN"}
    )
