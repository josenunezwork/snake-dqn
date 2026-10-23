"""Prepare immutable, held P1 next-stage packs without allocating Colab."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export

QUALIFIED_CLI_TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"


def _commit_tree(repo: Path, commit: str) -> str:
    """Return the pinned Colab CLI tree, refusing a non-commit revision."""
    kind = subprocess.check_output(["git", "cat-file", "-t", commit], cwd=repo, text=True).strip()
    if kind != "commit":
        raise ValueError("next-stage pin must name a commit")
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=repo, text=True
    ).strip()


def prepare(*, root: Path, commit: str, repo: Path) -> dict[str, Any]:
    """Write held next-stage records for a qualified transport tree.

    No source continuation, replacement result, pricing result, or Colab call
    is consumed here.  Those facts must be bound later by Claude's review.
    """
    root = root.resolve()
    if root.exists():
        raise ValueError("next-stage root must be fresh")
    tree = _commit_tree(repo, commit)
    if tree != QUALIFIED_CLI_TREE:
        raise ValueError("commit does not carry the qualified Colab CLI tree")
    # Use the production pack writer so every held record has the same
    # control, manifest, upload preflight, receipt, and wrapper-shaped
    # DISPATCH text as a later admitted package.  The synthetic names are
    # deliberately unavailable: they preserve the dependency as a blocker.
    from research.bl_ppo.p1_manifests import prepare as prepare_manifest

    replacement_pack = root / "y3-replacement"
    qualification_pack = root / "qual-1m"
    replacement_receipt = prepare_manifest(
        root=replacement_pack,
        commit=commit,
        phase="y3-replacement",
        resume_name="UNVERIFIED-SOURCE-ORIGIN-SNAPSHOT.pt",
        expected_trace_name="UNVERIFIED-SOURCE-CONTINUATION-RECEIPT.json",
        hold_reason=(
            "new judge ruling required: post-science CHILD_NONZERO_EXIT and "
            "25.59u path exceeds 25.50u envelope"
        ),
    )
    qualification_receipt = prepare_manifest(
        root=qualification_pack,
        commit=commit,
        phase="qual-1m",
        resume_name="UNVERIFIED-Y3-ORIGIN-SNAPSHOT.pt",
        hold_reason="new judge must separately price and admit qualification after genuine Y3 PASS",
    )
    root.mkdir(parents=True, exist_ok=True)

    replacement_control = json.loads((replacement_pack / "p1-control.json").read_text())
    qualification_control = json.loads((qualification_pack / "p1-control.json").read_text())
    held_manifests: dict[str, Path] = {}
    for name, pack in (("replacement", replacement_pack), ("qualification", qualification_pack)):
        held = json.loads((pack / "manifest.json").read_text())
        held["scientific_admission"] = False
        held.setdefault("transport", {})["dispatch_ready"] = False
        held["held_reason"] = "requires a new admitted sibling after judge and Claude review"
        held["budget"]["rate_units_per_hour"] = 1.54
        held["budget"]["rate_tolerance_units_per_hour"] = 0.0
        held["pricing_status"] = "PROVISIONAL_NOT_RATIFIED"
        if name == "qualification":
            held["pricing_note"] = (
                "The inherited 4u/80min template values are not an admitted effective envelope; "
                "a judge must set the prospective 2x cap and timeout after genuine Y3 PASS."
            )
        path = pack / "held-manifest.json"
        export_json(path, held)
        held_manifests[name] = path
    replacement = {
        "schema": "bl-p1-next-stage-held/v1",
        "status": "HELD_NOT_ADMITTED",
        "phase": "y3-replacement",
        "commit": commit,
        "transport": {"qualified_colab_cli_tree": tree, "device": "cuda:0"},
        "resume": replacement_control["resume"],
        "pricing": {
            "status": "JUDGE_REQUIRED_NOT_FUNDED",
            "historical_replacement_cap_units": 3.86,
            "historical_timeout_minutes": 120,
            "rate_units_per_hour": 1.54,
            "reconciled_charged_units": 10.11,
            "remaining_units": 15.39,
            "exact_pin_canary_plus_source_plus_replacement_units": 25.59,
            "frozen_envelope_units": 25.50,
        },
        "required_before_materialization": [
            "immutable source receipt schema bl-p1-y3-source/v1 status "
            "COMPLETE_SOURCE_CONTINUATION",
            "source receipt has an actor-phase operational snapshot and hash-verified "
            "operational trace",
            "source fixed suffix has >=1000 actions, >=3 actor updates, and >=1 actor "
            "episode boundary",
            "Claude verifies a distinct released L4 assignment, fresh zero-other-assignment "
            "inventory, and rate <=1.54u/h",
            "judge/reconciliation confirms the replacement invoice is funded after the final "
            "source debit",
            "independent judge authorizes any repair after the post-science " "CHILD_NONZERO_EXIT",
            "Claude independently verifies the current pin and qualified tree before dispatch",
        ],
        "forbidden_substitutes": [
            "origin/refit-only snapshot",
            "boundary-only source result",
            "Y3_PASS_BOUNDARY_ONLY",
            "offline or synthetic replay",
        ],
        "dispatch": "NONE: held record only; Claude materializes a fresh manifest after "
        "every prerequisite.",
    }
    qualification = {
        "schema": "bl-p1-next-stage-held/v1",
        "status": "HELD_NOT_ADMITTED",
        "phase": "qual-1m",
        "commit": commit,
        "transport": {"qualified_colab_cli_tree": tree, "device": "cuda:0"},
        "fixed_boundary": {
            "requested_actor_transitions": 1_000_000,
            "whole_world_boundary": 1_000_064,
            "base_unit_cap": 4.0,
            "effective_cap_and_timeout": "JUDGE_REQUIRED_NOT_FROZEN",
        },
        "resume": qualification_control["resume"],
        "required_before_materialization": [
            "immutable Y3 result schema bl-p1-y3/v1 status PASS exactly",
            "Y3 operational_replay.status PASS; Y3_PASS_BOUNDARY_ONLY is not admission",
            "source and replacement supervisor terminal receipts prove distinct released "
            "assignments",
            "judge separately prices and admits the 1M qualification after Y3 PASS; no price "
            "is frozen here",
            "Claude verifies current qualified CLI tree, cuda:0, fresh inventory, rate, and "
            "all hash sidecars",
        ],
        "dispatch": "NONE: held record only; no 1M manifest or allocation is admitted by "
        "this preparation.",
    }
    export_json(root / "y3-replacement-held.json", replacement)
    export_json(root / "qual-1m-held.json", qualification)
    wrapper = root / "DISPATCH_NEXT_STAGE.txt"
    repo_text = str(repo.resolve())
    export_bytes(
        wrapper,
        (
            "P1 next stages are HELD_NOT_ADMITTED. Do not execute either command.\n"
            "Claude may use these wrapper-shaped forms only for a NEW admitted sibling after "
            "the judge and Claude discharge every held prerequisite.\n\n"
            f"{repo_text}/tools/ops/dispatch.sh p1-y3-replacement-next-stage "
            f"{held_manifests['replacement']} "
            f"--broker {repo_text} --cwd {repo_text} "
            f"--dispatch-txt {wrapper}\n"
            f"{repo_text}/tools/ops/dispatch.sh p1-qual-1m-next-stage "
            f"{held_manifests['qualification']} "
            f"--broker {repo_text} --cwd {repo_text} "
            f"--dispatch-txt {wrapper}\n"
        ).encode(),
    )
    result = {
        "schema": "bl-p1-next-stage-preparation/v1",
        "status": "PREPARED_HELD_NOT_DISPATCHED",
        "commit": commit,
        "qualified_colab_cli_tree": tree,
        "records": {
            "y3_replacement": {
                "path": "y3-replacement-held.json",
                "sha256": digest(root / "y3-replacement-held.json"),
            },
            "qual_1m": {"path": "qual-1m-held.json", "sha256": digest(root / "qual-1m-held.json")},
        },
        "held_phase_packs": {
            "y3_replacement": {
                "root": "y3-replacement",
                "manifest": "y3-replacement/manifest.json",
                "held_manifest": "y3-replacement/held-manifest.json",
                "control": "y3-replacement/p1-control.json",
                "receipt": "y3-replacement/receipt.json",
                "dispatch": "y3-replacement/DISPATCH.txt",
                "receipt_sha256": digest(replacement_pack / "receipt.json"),
            },
            "qual_1m": {
                "root": "qual-1m",
                "manifest": "qual-1m/manifest.json",
                "held_manifest": "qual-1m/held-manifest.json",
                "control": "qual-1m/p1-control.json",
                "receipt": "qual-1m/receipt.json",
                "dispatch": "qual-1m/DISPATCH.txt",
                "receipt_sha256": digest(qualification_pack / "receipt.json"),
            },
        },
        "held_phase_receipts": {
            "replacement": replacement_receipt,
            "qualification": qualification_receipt,
        },
        "dispatch_wrapper": {"path": wrapper.name, "sha256": digest(wrapper)},
        "colab_calls": 0,
        "dispatch_ready": False,
    }
    export_json(root / "preparation.json", result)
    return result


def bind_source(*, root: Path, source_receipt: Path) -> dict[str, Any]:
    """Bind an unverified source reference without admitting replacement.

    A held pack lacks the source control and its complete flat closure.  Claude
    must run ``p1_runtime._validate_operational_proof`` on those inputs before
    a manifest may be materialized.
    """
    # The preparation is immutable.  A later binding is a sibling receipt, so
    # its existence cannot rewrite the held pack Claude originally reviewed.
    verify_export(root / "preparation.json")
    verify_export(source_receipt)
    source = json.loads(source_receipt.read_text())
    if (
        source.get("schema") != "bl-p1-y3-source/v1"
        or source.get("status") != "COMPLETE_SOURCE_CONTINUATION"
    ):
        raise ValueError("replacement requires COMPLETE_SOURCE_CONTINUATION source evidence")
    snapshots, trace = source.get("operational_snapshots"), source.get("operational_trace")
    if not isinstance(snapshots, list) or not any(row.get("phase") == "actor" for row in snapshots):
        raise ValueError("source evidence lacks an actor-phase operational snapshot")
    if not isinstance(trace, dict) or not trace.get("path") or not trace.get("sha256"):
        raise ValueError("source evidence lacks an operational trace")
    row = {
        "schema": "bl-p1-next-stage-source-binding/v1",
        "status": "SOURCE_REFERENCE_BOUND_UNVERIFIED",
        "source_receipt": {"path": str(source_receipt.resolve()), "sha256": digest(source_receipt)},
        "replacement_still_requires": json.loads((root / "y3-replacement-held.json").read_text())[
            "required_before_materialization"
        ],
        "verification_required": (
            "Claude runs p1_runtime._validate_operational_proof against the source control "
            "and complete flat closure"
        ),
        "qualification_remains_held": True,
        "colab_calls": 0,
    }
    export_json(root / "source-binding.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staged = commands.add_parser("prepare")
    staged.add_argument("--root", type=Path, required=True)
    staged.add_argument("--commit", required=True)
    bound = commands.add_parser("bind-source")
    bound.add_argument("--root", type=Path, required=True)
    bound.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(prepare(root=args.root, commit=args.commit, repo=Path.cwd()), sort_keys=True)
        )
    else:
        print(
            json.dumps(
                bind_source(root=args.root, source_receipt=args.source_receipt), sort_keys=True
            )
        )


if __name__ == "__main__":
    main()
