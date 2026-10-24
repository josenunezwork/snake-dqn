"""R33 proof dependency charter contracts; no operational acceptance is created here."""

from __future__ import annotations

import hashlib

import pytest

from tools.ops.proof_dependencies import (
    Refused,
    delta_input,
    dependency_inventory,
    validate_independent_acceptance,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def closure(destination: str = "one") -> dict:
    kinds = [
        "source",
        "input",
        "environment",
        "platform",
        "command",
        "fixture",
        "allocation",
        "contract",
    ]
    rows = [{"name": kind, "kind": kind, "sha256": sha(kind)} for kind in kinds]
    rows.append({"name": "destination:input", "kind": "location", "sha256": sha(destination)})
    return dependency_inventory("packet", rows)


def proof(inventory: dict) -> list[dict]:
    return [
        {
            "path": "old-proof.json",
            "sha256": sha("old"),
            "inventory_sha256": inventory["inventory_sha256"],
            "contract_sha256": sha("contract"),
            "reuse_permitted": True,
        }
    ]


def test_only_opted_in_destination_delta_reaches_pending_independent_review() -> None:
    old, new = closure(), closure("two")
    value = delta_input(
        previous=old,
        current=new,
        reused_proofs=proof(old),
        reviewer="codex-v",
        live_obligations=["native_canary", "live_admission"],
        charter_opt_in=True,
    )
    assert value["status"] == "PENDING_INDEPENDENT_ACCEPTANCE"
    assert value["reuse_class"] == "destination_or_transport_location"


@pytest.mark.parametrize("mutated", ["source", "environment", "platform", "allocation", "contract"])
def test_runtime_or_scientific_dependency_delta_requires_full_review(mutated: str) -> None:
    old = closure()
    rows = [dict(row) for row in old["dependencies"]]
    for row in rows:
        if row["kind"] == mutated:
            row["sha256"] = sha("changed-" + mutated)
    new = dependency_inventory("packet", rows)
    value = delta_input(
        previous=old,
        current=new,
        reused_proofs=proof(old),
        reviewer="codex-v",
        live_obligations=["native_canary", "live_admission"],
        charter_opt_in=True,
    )
    assert value["status"] == "FULL_REVIEW_REQUIRED"


def test_foreign_closure_missing_dependency_and_builder_self_certification_refuse() -> None:
    old = closure()
    foreign = proof(old)
    foreign[0]["inventory_sha256"] = sha("foreign")
    with pytest.raises(Refused):
        delta_input(
            previous=old,
            current=closure("two"),
            reused_proofs=foreign,
            reviewer="codex-v",
            live_obligations=["native_canary", "live_admission"],
            charter_opt_in=True,
        )
    with pytest.raises(Refused):
        dependency_inventory(
            "packet", [row for row in old["dependencies"] if row["kind"] != "fixture"]
        )
    with pytest.raises(Refused):
        delta_input(
            previous=old,
            current=closure("two"),
            reused_proofs=proof(old),
            reviewer="builder",
            live_obligations=["native_canary", "live_admission"],
            charter_opt_in=True,
        )
    stale = dict(old)
    stale["inventory_sha256"] = sha("stale")
    with pytest.raises(Refused):
        delta_input(
            previous=stale,
            current=closure("two"),
            reused_proofs=proof(old),
            reviewer="codex-v",
            live_obligations=["native_canary", "live_admission"],
            charter_opt_in=True,
        )


def test_only_codex_v_create_only_shape_can_accept_pending_input() -> None:
    old, new = closure(), closure("two")
    value = delta_input(
        previous=old,
        current=new,
        reused_proofs=proof(old),
        reviewer="codex-v",
        live_obligations=["native_canary", "live_admission"],
        charter_opt_in=True,
    )
    acceptance = {
        "schema": "r33-independent-delta-acceptance/v1",
        "input_sha256": value["input_sha256"],
        "reviewer": "codex-v",
        "decision": "ACCEPT",
        "checked": [
            "downloaded_bytes",
            "destination_semantics",
            "access_integrity",
            "runtime_environment",
            "consumer_behavior",
        ],
        "created_at": "external-create-only",
    }
    validate_independent_acceptance(acceptance, value)
