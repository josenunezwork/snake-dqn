"""Create-only, label-blind successor for an already sealed F5 batch plan.

The successor is deliberately a narrow identity migration.  It carries the
sealed draw and every typed RNG boundary into a final source/environment
identity; it never draws rows, reads labels, or issues a stage claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from research.bl_ppo.exports import export_torch, verify_export
from research.bl_residual.f5_fit import (
    BatchPlan,
    F5FitRefusal,
    F5Identity,
    RngBoundary,
    materialize_batch_plan,
    require_registered_sampler_semantics,
    verify_batch_plan,
)

SCHEMA = "bl-f5-plan-identity-successor/v1"
ROWS = 512_000


@dataclass(frozen=True)
class F5PlanIdentitySuccessor:
    """A source/environment successor whose scientific plan bytes are unchanged."""

    schema: str
    predecessor_plan_sha256: str
    predecessor_export_sha256: str
    predecessor_identity: F5Identity
    successor_plan: BatchPlan
    scientific_sha256: str
    sampler_sha256: str
    sampler_build_sha256: str
    successor_sha256: str


def _hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"F5 {name} hash malformed")
    return value


def _typed_equal(left: object, right: object) -> bool:
    """Compare serialized RNG values while rejecting type-normalizing substitutions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if hasattr(left, "dtype") and hasattr(left, "shape") and hasattr(left, "tobytes"):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return set(left) == set(right) and all(_typed_equal(left[key], right[key]) for key in left)
    return left == right


def _rng_equal(left: RngBoundary, right: RngBoundary) -> bool:
    return (
        _typed_equal(left.python_rng_state, right.python_rng_state)
        and _typed_equal(left.numpy_rng_state, right.numpy_rng_state)
        and _typed_equal(left.torch_cpu_rng_state, right.torch_cpu_rng_state)
        and _typed_equal(left.torch_cuda_rng_states, right.torch_cuda_rng_states)
        and _typed_equal(left.named_generator_states, right.named_generator_states)
    )


def _scientific_payload(plan: BatchPlan) -> dict[str, Any]:
    """Return exactly the carried scientific content, excluding mutable identities."""
    return {
        "indices": plan.indices.detach().cpu().contiguous(),
        "initial_generator_state": plan.initial_generator_state.detach().cpu().contiguous(),
        "final_generator_state": plan.final_generator_state.detach().cpu().contiguous(),
        "initial_rng": plan.initial_rng,
        "final_rng": plan.final_rng,
        "recipe": asdict(plan.recipe),
    }


def _update_typed(hasher: "hashlib._Hash", value: object) -> None:
    """Hash typed RNG content without pickle/archive implementation bytes."""
    if isinstance(value, RngBoundary):
        hasher.update(b"RngBoundary")
        _update_typed(
            hasher,
            (
                value.python_rng_state,
                value.numpy_rng_state,
                value.torch_cpu_rng_state,
                value.torch_cuda_rng_states,
                value.named_generator_states,
            ),
        )
    elif value is None:
        hasher.update(b"none")
    elif isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        hasher.update(f"int:{value}".encode())
    elif isinstance(value, float):
        hasher.update(f"float:{value.hex()}".encode())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"str:" + str(len(encoded)).encode() + b":" + encoded)
    elif isinstance(value, bytes):
        hasher.update(b"bytes:" + str(len(value)).encode() + b":" + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(
            f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode() + tensor.numpy().tobytes()
        )
    elif isinstance(value, tuple):
        hasher.update(f"tuple:{len(value)}".encode())
        for item in value:
            _update_typed(hasher, item)
    elif isinstance(value, list):
        hasher.update(f"list:{len(value)}".encode())
        for item in value:
            _update_typed(hasher, item)
    elif isinstance(value, dict):
        hasher.update(f"dict:{len(value)}".encode())
        for key in sorted(value):
            _update_typed(hasher, key)
            _update_typed(hasher, value[key])
    elif hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        hasher.update(f"ndarray:{value.dtype}:{tuple(value.shape)}:".encode() + value.tobytes())
    else:
        raise TypeError(f"unsupported typed F5 RNG value: {type(value)!r}")


def _typed_sha256(value: object) -> str:
    hasher = hashlib.sha256()
    _update_typed(hasher, value)
    return hasher.hexdigest()


def scientific_sha256(plan: BatchPlan) -> str:
    """Hash the complete carried plan/RNG content without binding source metadata."""
    return _typed_sha256(_scientific_payload(plan))


def _successor_digest(value: F5PlanIdentitySuccessor) -> str:
    return _typed_sha256(
        {
            "schema": value.schema,
            "predecessor_plan_sha256": value.predecessor_plan_sha256,
            "predecessor_export_sha256": value.predecessor_export_sha256,
            "predecessor_identity": asdict(value.predecessor_identity),
            "successor_plan_sha256": value.successor_plan.sha256,
            "successor_identity": asdict(value.successor_plan.identity),
            "scientific_sha256": value.scientific_sha256,
            "sampler_sha256": value.sampler_sha256,
            "sampler_build_sha256": value.sampler_build_sha256,
        }
    )


def _copy_plan(plan: BatchPlan, identity: F5Identity) -> BatchPlan:
    return BatchPlan(
        indices=plan.indices.detach().cpu().clone(),
        initial_generator_state=plan.initial_generator_state.detach().cpu().clone(),
        final_generator_state=plan.final_generator_state.detach().cpu().clone(),
        initial_rng=copy.deepcopy(plan.initial_rng),
        final_rng=copy.deepcopy(plan.final_rng),
        identity=identity,
        recipe=plan.recipe,
        # Recompute through the public materialization verifier below; this
        # placeholder is never accepted as a sealed successor digest.
        sha256="",
    )


def _build_proof_sha256(
    proof: Mapping[str, str], predecessor: F5Identity, successor: F5Identity
) -> str:
    required = {
        "predecessor_torch_build",
        "successor_torch_build",
        "predecessor_source_sha256",
        "successor_source_sha256",
        "predecessor_environment_sha256",
        "successor_environment_sha256",
    }
    if set(proof) != required or any(
        not isinstance(value, str) or not value for value in proof.values()
    ):
        raise F5FitRefusal("F5 successor sampler build proof malformed")
    if proof["predecessor_torch_build"] != proof["successor_torch_build"]:
        raise F5FitRefusal("F5 successor sampler build differs between identities")
    if proof["successor_torch_build"] != str(torch.__version__):
        raise F5FitRefusal("F5 successor sampler build differs from the executing torch runtime")
    if any(
        proof[name] != value
        for name, value in (
            ("predecessor_source_sha256", predecessor.source_sha256),
            ("successor_source_sha256", successor.source_sha256),
            ("predecessor_environment_sha256", predecessor.environment_sha256),
            ("successor_environment_sha256", successor.environment_sha256),
        )
    ):
        raise F5FitRefusal("F5 successor sampler build proof identity differs")
    return hashlib.sha256(json.dumps(dict(proof), sort_keys=True).encode()).hexdigest()


def _require_registered_fit_source(
    registered_fit_source: str, predecessor_identity: F5Identity
) -> str:
    """Bind the supplied frozen sampler source to the registered loss identity."""
    if not isinstance(registered_fit_source, str):
        raise F5FitRefusal("F5 registered fit source must be text")
    observed = hashlib.sha256(registered_fit_source.encode("utf-8")).hexdigest()
    if observed != predecessor_identity.loss_sha256:
        raise F5FitRefusal("F5 registered fit source differs from the frozen loss identity")
    return require_registered_sampler_semantics(registered_fit_source)


def create_successor(
    predecessor: BatchPlan,
    *,
    successor_identity: F5Identity,
    metadata: Mapping[str, torch.Tensor],
    registered_fit_source: str,
    build_proof: Mapping[str, str],
    predecessor_export_sha256: str = "",
) -> F5PlanIdentitySuccessor:
    """Create a prospective successor after proving label-blind exact replay.

    Only source and environment identity fields may differ.  The caller must
    supply metadata already admitted by the package layer; neither this helper
    nor its verifier opens training labels or efficacy outputs.
    """
    if predecessor.indices.numel() != ROWS:
        raise F5FitRefusal("F5 predecessor must carry exactly 512,000 row indices")
    if any(
        getattr(predecessor.identity, name) != getattr(successor_identity, name)
        for name in ("loss_sha256", "allocation_sha256", "corpus_sha256")
    ):
        raise F5FitRefusal("F5 successor may not change recipe/loss/corpus/allocation identity")
    sampler = _require_registered_fit_source(registered_fit_source, predecessor.identity)
    build_sha = _build_proof_sha256(build_proof, predecessor.identity, successor_identity)
    # Deterministic replay supplies the identity-bound plan digest without a
    # duplicate implementation of f5_fit's internal digest.
    successor_plan = materialize_batch_plan(metadata, successor_identity)
    carried = _copy_plan(predecessor, successor_identity)
    carried = BatchPlan(
        carried.indices,
        carried.initial_generator_state,
        carried.final_generator_state,
        carried.initial_rng,
        carried.final_rng,
        carried.identity,
        carried.recipe,
        successor_plan.sha256,
    )
    result = F5PlanIdentitySuccessor(
        schema=SCHEMA,
        predecessor_plan_sha256=_hash(predecessor.sha256, "predecessor plan"),
        predecessor_export_sha256=predecessor_export_sha256,
        predecessor_identity=predecessor.identity,
        successor_plan=carried,
        scientific_sha256=scientific_sha256(predecessor),
        sampler_sha256=sampler,
        sampler_build_sha256=build_sha,
        successor_sha256="",
    )
    result = replace(result, successor_sha256=_successor_digest(result))
    verify_successor(
        result,
        predecessor=predecessor,
        metadata=metadata,
        registered_fit_source=registered_fit_source,
        build_proof=build_proof,
    )
    return result


def verify_successor(
    successor: F5PlanIdentitySuccessor,
    *,
    predecessor: BatchPlan,
    metadata: Mapping[str, torch.Tensor],
    registered_fit_source: str,
    build_proof: Mapping[str, str],
) -> None:
    """Label-blind proof of exact scientific carryover and admissible replay."""
    if not isinstance(successor, F5PlanIdentitySuccessor) or successor.schema != SCHEMA:
        raise F5FitRefusal("foreign F5 plan successor")
    if successor.predecessor_plan_sha256 != predecessor.sha256:
        raise F5FitRefusal("F5 successor predecessor plan digest differs")
    if successor.predecessor_identity != predecessor.identity:
        raise F5FitRefusal("F5 successor predecessor identity differs")
    plan = successor.successor_plan
    if plan.indices.numel() != ROWS or plan.indices.shape != (2000, 256):
        raise F5FitRefusal("F5 successor must carry exactly ordered 512,000 indices")
    if (
        any(
            getattr(predecessor.identity, name) != getattr(plan.identity, name)
            for name in ("loss_sha256", "allocation_sha256", "corpus_sha256")
        )
        or plan.recipe != predecessor.recipe
    ):
        raise F5FitRefusal("F5 successor changed recipe/loss/corpus/allocation")
    if (
        not torch.equal(plan.indices, predecessor.indices)
        or not torch.equal(plan.initial_generator_state, predecessor.initial_generator_state)
        or not torch.equal(plan.final_generator_state, predecessor.final_generator_state)
    ):
        raise F5FitRefusal("F5 successor changed sealed sampler bytes")
    if not _rng_equal(plan.initial_rng, predecessor.initial_rng) or not _rng_equal(
        plan.final_rng, predecessor.final_rng
    ):
        raise F5FitRefusal("F5 successor changed typed RNG bytes")
    sampler = _require_registered_fit_source(registered_fit_source, predecessor.identity)
    if successor.sampler_sha256 != sampler:
        raise F5FitRefusal("F5 successor sampler implementation/build differs")
    if successor.sampler_build_sha256 != _build_proof_sha256(
        build_proof, predecessor.identity, plan.identity
    ):
        raise F5FitRefusal("F5 successor sampler build proof differs")
    if successor.scientific_sha256 != scientific_sha256(
        predecessor
    ) or successor.scientific_sha256 != scientific_sha256(plan):
        raise F5FitRefusal("F5 successor scientific content digest differs")
    verify_batch_plan(predecessor, metadata)
    verify_batch_plan(plan, metadata)
    if successor.successor_sha256 != _successor_digest(
        F5PlanIdentitySuccessor(
            successor.schema,
            successor.predecessor_plan_sha256,
            successor.predecessor_export_sha256,
            successor.predecessor_identity,
            successor.successor_plan,
            successor.scientific_sha256,
            successor.sampler_sha256,
            successor.sampler_build_sha256,
            "",
        )
    ):
        raise F5FitRefusal("F5 successor digest mismatch")


def export_successor(path: Path, successor: F5PlanIdentitySuccessor) -> dict[str, Any]:
    """Publish the successor create-only with its normal immutable sidecars."""
    return export_torch(path, successor)


def load_successor(path: Path) -> F5PlanIdentitySuccessor:
    """Load only a completed immutable successor export."""
    verify_export(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, F5PlanIdentitySuccessor):
        raise F5FitRefusal("foreign F5 plan successor export")
    return value


def load_predecessor_export(path: Path, *, expected_sha256: str) -> tuple[BatchPlan, str]:
    """Load only the independently pinned old export for production migration."""
    observed = verify_export(path)
    if observed != _hash(expected_sha256, "predecessor export"):
        raise F5FitRefusal("F5 predecessor export differs from independent provenance pin")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, BatchPlan):
        raise F5FitRefusal("foreign F5 predecessor plan export")
    return value, observed


def create_successor_from_export(
    predecessor_path: Path,
    *,
    expected_predecessor_export_sha256: str,
    successor_identity: F5Identity,
    metadata: Mapping[str, torch.Tensor],
    registered_fit_source: str,
    build_proof: Mapping[str, str],
) -> F5PlanIdentitySuccessor:
    """Production-only successor entry point with file-bound old-plan provenance."""
    predecessor, observed = load_predecessor_export(
        predecessor_path, expected_sha256=expected_predecessor_export_sha256
    )
    return create_successor(
        predecessor,
        successor_identity=successor_identity,
        metadata=metadata,
        registered_fit_source=registered_fit_source,
        build_proof=build_proof,
        predecessor_export_sha256=observed,
    )


def export_runtime_plan(
    path: Path,
    successor: F5PlanIdentitySuccessor,
    *,
    predecessor_path: Path,
    expected_predecessor_sha256: str,
    metadata: Mapping[str, torch.Tensor],
    registered_fit_source: str,
    build_proof: Mapping[str, str],
) -> dict[str, str | int]:
    """Export the verified successor as the ordinary runtime ``BatchPlan``.

    This is the sole production bridge to the unchanged runtime loader.  It
    reloads the independently pinned predecessor instead of trusting an
    in-memory object supplied by an operator.
    """
    predecessor, observed = load_predecessor_export(
        predecessor_path, expected_sha256=expected_predecessor_sha256
    )
    if not successor.predecessor_export_sha256 or successor.predecessor_export_sha256 != observed:
        raise F5FitRefusal("F5 runtime successor lacks trusted predecessor export provenance")
    verify_successor(
        successor,
        predecessor=predecessor,
        metadata=metadata,
        registered_fit_source=registered_fit_source,
        build_proof=build_proof,
    )
    receipt = export_torch(path, successor.successor_plan)
    return {
        "schema": SCHEMA,
        "kind": "F5_RUNTIME_PLAN_SUCCESSOR_EXPORT",
        "runtime_plan_export_sha256": receipt["sha256"],
        "runtime_plan_sha256": successor.successor_plan.sha256,
        "predecessor_export_sha256": observed,
        "scientific_sha256": successor.scientific_sha256,
        "successor_sha256": successor.successor_sha256,
        "rows": ROWS,
    }
