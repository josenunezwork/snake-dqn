"""Verification for the amendment-06 hash-bound Y3 origin import."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from research.bl_ppo import p1_contract
from research.bl_ppo.exports import digest, verify_export
from research.bl_ppo.recovery import read_snapshot, snapshot_files

SCHEMA = "bl-p1-y3-origin-import/v1"
PREFIX_ACTIONS = 49_152
FAILED_PREFIX_SHA256 = "fb3c8c7b89bbe0f347ae89799fbb66cf9bf8373e987b1d8212fe286c7e95280c"
SOURCE_ALLOCATION = "source_latebody_06_e8_04_pack04"
ORIGINAL_NAMESPACE = "bl_p1_y3_source_latebody_06_e8_04_pack04"
ORIGIN_COMMIT = "5d877123ce8734537ef1f6924d5543be50dacdf7"
ORIGIN_SNAPSHOT_SHA256 = "dbdc10c61914cccd156e4a73de67b8004cacce133ae334686d393153fe4109ce"
ORIGIN_STATUS_SHA256 = "9b1f309a6264a23f150faff875d0b60ab7071bab3d0b2559e673bdad362feab7"
FAILED_RECEIPT_SHA256 = "aaa9f106c029b0235a174adbb38d617dad7c631af9dda49ce6fb01033f1db6eb"
OFFLINE_AUDIT_SHA256 = "738c4a5b959b81acfcbb2ea9cd0ebc881c1b090d70a9404454a53924a5427a33"


def canonical_sha256(value: Any) -> str:
    """Return the stable digest used for an immutable JSON subtree."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(root: Path, value: Any, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"origin import lacks {name} artifact")
    path = root / value["path"]
    if path.name != value["path"] or not path.is_file():
        raise ValueError(f"origin import {name} path is not a safe uploaded artifact")
    observed = verify_export(path)
    if observed != value.get("sha256"):
        raise ValueError(f"origin import {name} digest mismatch")
    return path, observed


def validate(control: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate an uploaded pre-actor origin without opening any outcome path."""
    value = control.get("y3_origin_import")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Y3 origin import declaration is malformed")
    if control.get("phase") == "y3-source":
        resume = control.get("resume", {})
        if (
            resume.get("allocation") != SOURCE_ALLOCATION
            or resume.get("namespace") == ORIGINAL_NAMESPACE
        ):
            raise ValueError("origin import requires the original allocation and a fresh namespace")
    snapshot, snapshot_sha = _artifact(root, value.get("origin_snapshot"), "origin snapshot")
    status_path, status_sha = _artifact(root, value.get("origin_status"), "origin status")
    _, failed_sha = _artifact(root, value.get("failed_receipt"), "failed receipt")
    _, audit_sha = _artifact(root, value.get("offline_audit"), "offline audit")
    if (snapshot_sha, status_sha, failed_sha, audit_sha) != (
        ORIGIN_SNAPSHOT_SHA256,
        ORIGIN_STATUS_SHA256,
        FAILED_RECEIPT_SHA256,
        OFFLINE_AUDIT_SHA256,
    ):
        raise ValueError("origin import does not name the ruled immutable evidence set")
    source_commit = value.get("source_commit")
    if source_commit != ORIGIN_COMMIT:
        raise ValueError("origin import lacks exact source commit")
    allowed_diff_paths = value.get("allowed_diff_paths")
    permitted = {
        "research/bl_ppo/p1_repair_admission.py",
        "research/bl_ppo/p1_qualification.py",
        "research/bl_ppo/e8_04_source_admission.py",
        "research/bl_ppo/p1_kernel_ambiguity_retry.py",
        "research/bl_ppo/p1_next_stage.py",
        "research/bl_ppo/p1_source_audit.py",
        "research/bl_ppo/run.py",
        "research/bl_ppo/p1_runtime.py",
        "research/bl_ppo/p1_source_repair.py",
        "research/bl_ppo/p1_repair_transport.py",
        "research/bl_ppo/p1_release_scan.py",
        "research/bl_ppo/test_p1_source_repair.py",
        "research/bl_ppo/p1_origin_import.py",
        "research/bl_ppo/test_p1_origin_import.py",
    }
    if (
        not isinstance(allowed_diff_paths, list)
        or not allowed_diff_paths
        or any(
            path not in permitted
            and not (
                Path(path).parent.as_posix() == "research/bl_ppo"
                and (
                    Path(path).name.startswith("test_p1_")
                    or Path(path).name == "test_e8_04_source_admission.py"
                )
            )
            for path in allowed_diff_paths
        )
    ):
        raise ValueError("origin import pin diff includes an out-of-scope path")
    scope = value.get("diff_scope")
    if (
        not isinstance(scope, dict)
        or scope.get("source_commit") != source_commit
        or scope.get("repair_commit") != control.get("source_commit")
    ):
        raise ValueError("origin import lacks a matching Mac-verified cross-pin diff scope")
    rows = scope.get("paths")
    if not isinstance(rows, list) or not rows:
        raise ValueError("origin import diff scope lacks path hashes")
    for row in rows:
        if not isinstance(row, dict) or row.get("path") not in allowed_diff_paths:
            raise ValueError("origin import diff scope includes an out-of-scope path")
        current = Path(__file__).parents[2] / row["path"]
        if not current.is_file() or digest(current) != row.get("repair_sha256"):
            raise ValueError("origin import current pin differs from Mac-verified diff scope")
    expected = value.get("expected_identity")
    if not isinstance(expected, dict):
        raise ValueError("origin import lacks frozen trajectory identity")
    required = {
        "arm": "F",
        "seed": 20261021,
        "target": p1_contract.SHAPE.target,
        "natural_collection_world_id_offset": 3_011_000,
    }
    if any(expected.get(key) != item for key, item in required.items()):
        raise ValueError("origin import identity is not the ruled P1 trajectory")
    for key in (
        "registration_sha256",
        "evaluation_bank_sha256",
        "e8_critic_convergence_amendment_04_sha256",
    ):
        if expected.get(key) != control.get(key):
            raise ValueError("origin import identity differs from repair control")
    if (
        control.get("natural_collection_world_id_offset")
        != expected["natural_collection_world_id_offset"]
    ):
        raise ValueError("origin import collection offset differs from repair control")
    status = json.loads(status_path.read_text())
    if status.get("critic_qualified") is not True or status.get("pause_at") != 0:
        raise ValueError("origin import status is not a qualified pre-actor pause")
    if status.get("snapshot", {}).get("sha256") != snapshot_sha:
        raise ValueError("origin import status does not bind origin snapshot")
    state = read_snapshot(snapshot)
    identity = state.get("identity", {})
    state_expected = {
        "p1_runtime": True,
        "arm": expected["arm"],
        "seed": expected["seed"],
        "target_transitions": expected["target"],
        "source_commit": source_commit,
        "p1_registration_sha256": expected["registration_sha256"],
        "p1_evaluation_bank_sha256": expected["evaluation_bank_sha256"],
    }
    if any(identity.get(key) != item for key, item in state_expected.items()):
        raise ValueError("origin import snapshot identity differs")
    counters = state.get("counters", {})
    if status.get("counters") != counters:
        raise ValueError("origin import status counters differ from snapshot")
    if counters.get("actor_transitions") != 0 or counters.get("actor_updates") != 0:
        raise ValueError("origin import snapshot is not pre-actor")
    expected_counters = value.get("expected_counters")
    frozen_counters = {
        key: item
        for key, item in counters.items()
        if key.startswith("critic_refit") or key.startswith("critic_fixed_holdout")
    }
    if (
        not isinstance(expected_counters, dict)
        or expected_counters != frozen_counters
        or any(counters.get(key) != item for key, item in expected_counters.items())
    ):
        raise ValueError("origin import refit counters differ")
    critic = status.get("critic_refit")
    if not isinstance(critic, dict) or canonical_sha256(critic) != value.get("critic_refit_sha256"):
        raise ValueError("origin import critic subtree digest mismatch")
    catalog = value.get("snapshot_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("origin import lacks transitive snapshot catalog")
    observed_catalog = [
        {"name": item.name, "sha256": verify_export(item), "size_bytes": item.stat().st_size}
        for item in snapshot_files(snapshot)
    ]
    if observed_catalog != catalog:
        raise ValueError("origin import transitive catalog differs")
    evidence_catalog = value.get("evidence_catalog")
    if not isinstance(evidence_catalog, list) or not evidence_catalog:
        raise ValueError("origin import lacks late-body rows/index evidence catalog")
    evidence_names = {row.get("name") for row in evidence_catalog if isinstance(row, dict)}
    if "late-body-rows-index.json" not in evidence_names:
        raise ValueError("origin import evidence catalog lacks late-body row index")
    index_path = root / "late-body-rows-index.json"
    index = json.loads(index_path.read_text())
    required_names = {"late-body-rows-index.json"} | {r["path"] for r in index["chunks"]}
    if evidence_names != required_names or len(evidence_names) != len(evidence_catalog):
        raise ValueError("origin import late-body evidence catalog differs from index")
    for row in evidence_catalog:
        path = root / row["name"]
        if (
            Path(row["name"]).name != row["name"]
            or verify_export(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise ValueError("origin import late-body evidence digest/catalog mismatch")
    for row in index["chunks"]:
        if verify_export(root / row["path"]) != row["sha256"]:
            raise ValueError("origin import late-body row differs from index digest")
    failed_receipt = json.loads((root / value["failed_receipt"]["path"]).read_text())
    offline_audit = json.loads((root / value["offline_audit"]["path"]).read_text())
    if (
        failed_sha not in offline_audit.get("evidence_sha256", {}).values()
        or offline_audit.get("critic_qualified") is not True
    ):
        raise ValueError("origin import offline audit does not reconcile failed receipt")
    receipt_exports = {
        (Path(row.get("path", "")).name, row.get("sha256"))
        for row in failed_receipt.get("exports", [])
        if isinstance(row, dict)
    }
    if any(
        (row["name"], row["sha256"]) not in receipt_exports for row in catalog + evidence_catalog
    ):
        raise ValueError("origin import catalog is not bound by failed receipt")
    audit_rows = offline_audit.get("evidence_sha256", {})
    for row in catalog:
        matches = [
            sha
            for name, sha in audit_rows.items()
            if Path(name).name == row["name"] and "/origin/" in name
        ]
        if matches and any(sha != row["sha256"] for sha in matches):
            raise ValueError("origin import catalog differs from offline audit")
    if (
        failed_receipt.get("stop_verified") is not True
        or failed_receipt.get("cleanup_error") is not None
    ):
        raise ValueError("origin import failed attempt has unverified release")
    prefix = value.get("prefix_witness")
    if (
        not isinstance(prefix, dict)
        or prefix.get("actor_actions") != PREFIX_ACTIONS
        or prefix.get("full_tape_sha256") != FAILED_PREFIX_SHA256
    ):
        raise ValueError("origin import prefix witness is malformed")
    return {
        "snapshot": snapshot,
        "origin_snapshot_sha256": snapshot_sha,
        "origin_status_sha256": status_sha,
        "failed_receipt_sha256": failed_sha,
        "offline_audit_sha256": audit_sha,
        "status": status,
        "critic_refit": critic,
        "critic_refit_sha256": canonical_sha256(critic),
        "counters": {key: counters[key] for key in expected_counters},
        "binding_sha256": canonical_sha256(value),
        "prefix_witness": prefix,
    }


@contextmanager
def guard_refit_entry() -> Iterator[None]:
    """Void imported execution on any refit collection, solve or gate entry.

    Actor collection and updates call their original implementations unchanged.
    A trapped refusal is re-raised on context exit, so the training loop cannot
    swallow it and publish an apparently valid continuation.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    from research.bl_ppo import collector, late_body, learner, refit

    entered = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        entered.append("refit/solve/gate")
        raise ValueError("VOID: imported origin entered refit, solve or critic gate")

    collect = collector.PersistentCollector.collect
    learner_type = getattr(learner.PPOLearner, "func", learner.PPOLearner)
    update = learner_type.update

    def actor_collect(self: Any, *args: Any, **kwargs: Any) -> Any:
        import sys

        counters = sys._getframe(1).f_locals.get("counters", {})
        if counters and counters.get("actor_enabled") is not True:
            return refuse()
        return collect(self, *args, **kwargs)

    def actor_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("actor_enabled") is not True:
            return refuse()
        return update(self, *args, **kwargs)

    with ExitStack() as stack:
        for name in ("collect_fixed_late_body_holdout", "collect_refit_late_supplement"):
            stack.enter_context(patch.object(late_body, name, refuse))
        for name in (
            "add_batch",
            "fit_complete_targets",
            "solve_e8_critic_convergence_amendment_03",
            "solve_e8_critic_convergence_amendment_04",
            "qualification",
            "score_fixture_holdout",
        ):
            stack.enter_context(patch.object(refit.CompleteLifeRefitBank, name, refuse))
        for name in (
            "critic_quality",
            "qualify_critic",
            "fit_completed_critic",
            "install_exact_return_critic_solution",
        ):
            stack.enter_context(patch.object(learner_type, name, refuse))
        stack.enter_context(patch.object(collector.PersistentCollector, "collect", actor_collect))
        stack.enter_context(patch.object(learner_type, "update", actor_update))
        try:
            yield
        finally:
            if entered:
                raise ValueError("VOID: imported origin entered refit, solve or critic gate")
