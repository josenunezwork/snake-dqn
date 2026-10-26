"""Label-blind F5 plan-successor and sole-stage-claim boundary proofs."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from research.bl_ppo.exports import export_torch
from research.bl_residual.f5_fit import (
    F5FitRefusal,
    F5Identity,
    materialize_batch_plan,
)
from research.bl_residual.f5_plan_successor import (
    create_successor,
    create_successor_from_export,
    export_runtime_plan,
    export_successor,
    load_successor,
    verify_successor,
)
from research.bl_residual.f5_protocol import (
    consume_operative_claim,
    issue_successor_claim,
    stage_claim,
    successor_claim_path,
    supersede_unconsumed_claim,
    verify_claim,
    verify_operative_claim,
)


def h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def metadata() -> dict[str, torch.Tensor]:
    """Minimal metadata-only two-stratum corpus; no features or labels exist here."""
    status = torch.ones((3, 6), dtype=torch.bool)
    legal = torch.ones((3, 6), dtype=torch.bool)
    action_mask = torch.zeros((3, 6), dtype=torch.bool)
    died = torch.zeros((3, 6), dtype=torch.bool)
    died[0, 0] = True
    ordinary = torch.tensor([False, False, True])
    return {
        "status": status,
        "legal": legal,
        "action_mask": action_mask,
        "died": died,
        "ordinary": ordinary,
        "life_id": torch.tensor([0, 0, 1]),
    }


def identity(prefix: str) -> F5Identity:
    return F5Identity(
        source_sha256=h(f"{prefix}-source"),
        loss_sha256=h("loss"),
        environment_sha256=h(f"{prefix}-environment"),
        allocation_sha256=h("allocation"),
        corpus_sha256=h("corpus"),
    )


def registered_source() -> str:
    return (Path(__file__).parents[1] / "research/bl_residual/fit.py").read_text()


def build_proof(predecessor: F5Identity, successor: F5Identity) -> dict[str, str]:
    return {
        "predecessor_torch_build": "testbuild-v6",
        "successor_torch_build": "testbuild-v6",
        "predecessor_source_sha256": predecessor.source_sha256,
        "successor_source_sha256": successor.source_sha256,
        "predecessor_environment_sha256": predecessor.environment_sha256,
        "successor_environment_sha256": successor.environment_sha256,
    }


def sealed_plan() -> tuple[object, dict[str, torch.Tensor]]:
    values = metadata()
    return materialize_batch_plan(values, identity("old")), values


def test_successor_carries_exact_plan_rng_and_replays_metadata_only(tmp_path: Path) -> None:
    predecessor, values = sealed_plan()
    final = identity("final-v6")
    successor = create_successor(
        predecessor,
        successor_identity=final,
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=build_proof(predecessor.identity, final),
    )
    verify_successor(
        successor,
        predecessor=predecessor,
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=build_proof(predecessor.identity, final),
    )
    assert successor.successor_plan.indices.shape == (2000, 256)
    assert successor.successor_plan.indices.numel() == 512_000
    assert successor.successor_plan.indices.equal(predecessor.indices)
    assert successor.successor_plan.initial_generator_state.equal(
        predecessor.initial_generator_state
    )
    output = tmp_path / "successor.pt"
    export_successor(output, successor)
    restored = load_successor(output)
    assert restored.successor_sha256 == successor.successor_sha256
    assert restored.successor_plan.indices.equal(successor.successor_plan.indices)
    with pytest.raises(FileExistsError):
        export_successor(output, successor)


@pytest.mark.parametrize("field", ["loss_sha256", "allocation_sha256", "corpus_sha256"])
def test_successor_refuses_frozen_identity_drift(field: str) -> None:
    predecessor, values = sealed_plan()
    final = replace(identity("final"), **{field: h("drift")})
    with pytest.raises(F5FitRefusal, match="recipe/loss/corpus/allocation"):
        create_successor(
            predecessor,
            successor_identity=final,
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=build_proof(predecessor.identity, final),
        )


def test_successor_refuses_row_order_rng_and_eligibility_drift() -> None:
    predecessor, values = sealed_plan()
    final = identity("final")
    successor = create_successor(
        predecessor,
        successor_identity=final,
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=build_proof(predecessor.identity, final),
    )
    changed_indices = successor.successor_plan.indices.clone()
    changed_indices[0, 0] = 1
    changed_plan = replace(successor.successor_plan, indices=changed_indices)
    with pytest.raises(F5FitRefusal, match="sampler bytes"):
        verify_successor(
            replace(successor, successor_plan=changed_plan),
            predecessor=predecessor,
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=build_proof(predecessor.identity, final),
        )
    changed_rng = replace(
        successor.successor_plan.initial_rng,
        torch_cpu_rng_state=torch.zeros_like(
            successor.successor_plan.initial_rng.torch_cpu_rng_state
        ),
    )
    with pytest.raises(F5FitRefusal, match="typed RNG"):
        verify_successor(
            replace(
                successor, successor_plan=replace(successor.successor_plan, initial_rng=changed_rng)
            ),
            predecessor=predecessor,
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=build_proof(predecessor.identity, final),
        )
    ineligible = {key: value.clone() for key, value in values.items()}
    ineligible["ordinary"][:] = False
    with pytest.raises(ValueError, match="no eligible lives"):
        verify_successor(
            successor,
            predecessor=predecessor,
            metadata=ineligible,
            registered_fit_source=registered_source(),
            build_proof=build_proof(predecessor.identity, final),
        )


def test_supersession_makes_the_old_unconsumed_claim_nonoperative(tmp_path: Path) -> None:
    predecessor_bindings = {
        "namespace": "f5",
        "allocation_sha256": h("allocation"),
        "source_sha256": h("old-source"),
        "selection_sha256": h("selection"),
        "batch_plan_sha256": h("old-plan"),
    }
    successor_bindings = {
        **predecessor_bindings,
        "source_sha256": h("final-source"),
        "batch_plan_sha256": h("successor-plan"),
    }
    old = tmp_path / "old-claim.json"
    stage_claim(old, **predecessor_bindings)
    supersede_unconsumed_claim(old, successor=successor_bindings)
    lineage = Path(str(old) + ".supersession.json")
    with pytest.raises(ValueError, match="superseded"):
        verify_claim(old, predecessor_bindings, supersession_path=lineage)
    with pytest.raises(ValueError, match="noncanonical"):
        verify_operative_claim(old, successor_bindings, supersession_path=lineage)
    current = successor_claim_path(old, successor_bindings)
    issue_successor_claim(
        predecessor_path=old, supersession_path=lineage, successor=successor_bindings
    )
    assert (
        verify_operative_claim(current, successor_bindings, supersession_path=lineage)["namespace"]
        == "f5"
    )


def test_production_successor_refuses_wrong_file_sha_and_build_proof(tmp_path: Path) -> None:
    predecessor, values = sealed_plan()
    plan_path = tmp_path / "old-plan.pt"
    export_torch(plan_path, predecessor)
    final = identity("final")
    with pytest.raises(F5FitRefusal, match="provenance pin"):
        create_successor_from_export(
            plan_path,
            expected_predecessor_export_sha256=h("wrong"),
            successor_identity=final,
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=build_proof(predecessor.identity, final),
        )


def test_runtime_export_is_trusted_ordinary_batch_plan(tmp_path: Path) -> None:
    predecessor, values = sealed_plan()
    predecessor_path = tmp_path / "old-plan.pt"
    old_receipt = export_torch(predecessor_path, predecessor)
    final = identity("final-runtime")
    proof = build_proof(predecessor.identity, final)
    successor = create_successor_from_export(
        predecessor_path,
        expected_predecessor_export_sha256=old_receipt["sha256"],
        successor_identity=final,
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=proof,
    )
    runtime_path = tmp_path / "runtime-plan.pt"
    receipt = export_runtime_plan(
        runtime_path,
        successor,
        predecessor_path=predecessor_path,
        expected_predecessor_sha256=old_receipt["sha256"],
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=proof,
    )
    from research.bl_residual.f5_job import _load_plan

    assert _load_plan(runtime_path).sha256 == successor.successor_plan.sha256
    assert receipt["rows"] == 512_000
    with pytest.raises(F5FitRefusal, match="provenance pin"):
        export_runtime_plan(
            tmp_path / "wrong-hash.pt",
            successor,
            predecessor_path=predecessor_path,
            expected_predecessor_sha256=h("wrong"),
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=proof,
        )
    untrusted = create_successor(
        predecessor,
        successor_identity=final,
        metadata=values,
        registered_fit_source=registered_source(),
        build_proof=proof,
    )
    with pytest.raises(F5FitRefusal, match="trusted predecessor"):
        export_runtime_plan(
            tmp_path / "untrusted.pt",
            untrusted,
            predecessor_path=predecessor_path,
            expected_predecessor_sha256=old_receipt["sha256"],
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=proof,
        )
    bad = build_proof(predecessor.identity, final)
    bad["successor_torch_build"] = "other-build"
    with pytest.raises(F5FitRefusal, match="build differs"):
        create_successor_from_export(
            predecessor_path,
            expected_predecessor_export_sha256=old_receipt["sha256"],
            successor_identity=final,
            metadata=values,
            registered_fit_source=registered_source(),
            build_proof=bad,
        )


def test_claim_copy_and_concurrent_issue_consume_have_one_winner(tmp_path: Path) -> None:
    predecessor = {
        "namespace": "f5",
        "allocation_sha256": h("allocation"),
        "source_sha256": h("old-source"),
        "selection_sha256": h("selection"),
        "batch_plan_sha256": h("old-plan"),
    }
    successor = {**predecessor, "source_sha256": h("new"), "batch_plan_sha256": h("plan")}
    old = tmp_path / "old.json"
    stage_claim(old, **predecessor)
    supersede_unconsumed_claim(old, successor=successor)
    lineage = Path(str(old) + ".supersession.json")
    outcomes: list[type[BaseException] | None] = []

    def issue() -> None:
        try:
            issue_successor_claim(
                predecessor_path=old, supersession_path=lineage, successor=successor
            )
            outcomes.append(None)
        except BaseException as exc:  # test records the create-only loser
            outcomes.append(type(exc))

    threads = [threading.Thread(target=issue), threading.Thread(target=issue)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count(None) == 1 and outcomes.count(FileExistsError) == 1
    current = successor_claim_path(old, successor)
    copied = tmp_path / "copied-current.json"
    stage_claim(copied, **successor)
    with pytest.raises(ValueError, match="copied or noncanonical"):
        consume_operative_claim(copied, successor, supersession_path=lineage)
    consumed: list[type[BaseException] | None] = []

    def consume() -> None:
        try:
            consume_operative_claim(current, successor, supersession_path=lineage)
            consumed.append(None)
        except BaseException as exc:  # test records the create-only loser
            consumed.append(type(exc))

    threads = [threading.Thread(target=consume), threading.Thread(target=consume)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert consumed.count(None) == 1 and consumed.count(ValueError) == 1
