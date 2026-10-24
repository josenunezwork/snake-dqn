"""Narrow §7 engineering admission, separate from the qualified transport tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCRIPT = "research/rpc_qualification/colab.sh"
AUTHORITY_SHA256 = "738a367bec2c3399ed8bd9383811d7fa643109a8aaba384a775f1c028a7285ea"
# Prospective fault definition and attempt-2 authority (gate change, pre-data).
FAULT_AUTHORITY_SHA256 = "fe8579c2aec5489ea25149b5d88036cb22daad0faa5310913231f3388716728e"
FAULT_DEFINITION = "l4-label-faults/v2"
WORKER_PIN = "2af48cd0b98a2063f46871cf4d320a5541b789af"
# Rebound to the v2 recovery-fix commit (judgment §4 step 1c); a18bf6b4 was attempt 1.
TRANSPORT_PIN = "4216927615fe80581daf4fb98d608a807c370ef3"
TRANSPORT_TREE = "4eefe6b10345f0a7b53c22f3158ce1c131093bfe"
# The independent reviewer's create-only ACCEPT lives outside the repository so
# recording it never moves the pack's pinned source commit.
REVIEW_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-24/"
    "CLAUDE_V_L4_REQUAL_02_20261024.json"
)


def require_review(transport_pin: str = TRANSPORT_PIN) -> dict[str, Any]:
    """Reviewer ACCEPT at this exact fix pin, required before any attempt-2 dispatch."""
    if not REVIEW_PATH.is_file():
        raise ValueError("attempt-2 package lacks its independent reviewer ACCEPT record")
    review = json.loads(Path(REVIEW_PATH).read_text())
    if (
        review.get("schema") != "l4-requal-02-review/v1"
        or review.get("verdict") != "ACCEPT"
        or review.get("transport_pin") != transport_pin
        or not isinstance(review.get("source_sha256"), dict)
        or not review["source_sha256"]
    ):
        raise ValueError("attempt-2 package lacks an independent reviewer ACCEPT at this pin")
    return review


BANK_SHA256 = "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"
FROZEN_ARCHIVE_SHA256 = "9ce2219ed1d6de0ccee84243d1b0ec3f71713431fe6e05334f0e073c958c6f45"
FROZEN_MANIFEST_SHA256 = "de492863ddb644e91f5bba3a7624377b1f4ad334e05694d86c38767db96962f5"
SCHEMA = "rpc-qualification-profile/v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelope(platform: str, mode: str) -> dict[str, Any]:
    """Freeze base/effective envelopes; reserves are inside the caps."""
    if platform not in {"cpu", "l4"} or mode not in {"loop", "injection"}:
        raise ValueError("foreign qualification platform/mode")
    if mode == "injection" and platform != "cpu":
        raise ValueError("only CPU injection is authorized")
    units, reserve, minutes = (
        (0.06, 0.012, 15)
        if mode == "injection"
        else (2.5, 0.5, 80) if platform == "l4" else (0.15, 0.03, 80)
    )
    return {
        "base": {"unit_cap": units, "reserve_units": reserve, "timeout_minutes": minutes},
        "effective": {
            "unit_cap": 2 * units,
            "reserve_units": 2 * reserve,
            "timeout_minutes": 2 * minutes,
        },
        "owner_multiplier": 2,
        "platform_series_effective_units": 15.0 if platform == "l4" else 0.90,
        "whole_package_effective_units": 16.02,
        "first_cap_wins": True,
    }


def prerequisite_arms(path: Path, repo: Path) -> dict[str, dict[str, Any]]:
    """v2 classification of each final-only acceptance arm at the fix pin.

    Deliberate acceptance faults are excluded only by each arm's create-only
    pre-dispatch injection manifest (bound by sha256 in the reviewed plan).
    """
    from research.bl_labels.l4_faults_v2 import classify_session, validate_injection_manifest
    from research.colab_cli.admission import require_final_only

    result = json.loads(path.read_text())
    require_final_only(result, repo)
    plan = result["final_plan"]
    if plan.get("commit") != TRANSPORT_PIN:
        raise ValueError("prerequisite is not the final-only acceptance at the v2 fix pin")
    planned = {job["name"]: job for job in plan.get("jobs", [])}
    arms: dict[str, dict[str, Any]] = {}
    for row in plan["execution"]:
        job = planned.get(row.get("job"))
        if job is None or not job.get("injection_manifest"):
            raise ValueError("prerequisite arm lacks its pre-dispatch injection manifest")
        manifest_path = Path(job["injection_manifest"])
        if sha(manifest_path) != job.get("injection_manifest_sha256"):
            raise ValueError("prerequisite injection manifest changed after planning")
        manifest = json.loads(manifest_path.read_text())
        receipt_path = Path(row["receipt"])
        if sha(receipt_path) != row.get("receipt_sha256"):
            raise ValueError("prerequisite receipt changed after acceptance")
        receipt = json.loads(receipt_path.read_text())
        validate_injection_manifest(manifest, job_id=receipt.get("job_id"))
        verdict = classify_session(receipt, manifest)
        arms["l4" if row["job"] == "l4" else "cpu"] = {
            **verdict,
            "harness_status": row.get("status"),
        }
    if set(arms) != {"cpu", "l4"}:
        raise ValueError("prerequisite lacks both acceptance arms")
    return arms


def require_prerequisite(path: Path, repo: Path, platform: str) -> dict[str, Any]:
    """The fix-pin final-only acceptance is necessary, never sufficient (§4 step 3).

    ``platform`` is the platform about to dispatch: L4 loops need the L4 arm
    v2 PASS; CPU loops and the CPU injection proof need the CPU arm v2 PASS.
    """
    if platform not in {"cpu", "l4"}:
        raise ValueError("unknown prerequisite platform")
    arms = prerequisite_arms(path, repo)
    arm = arms[platform]
    if arm["verdict"] != "PASS" or arm["harness_status"] != "PASS":
        raise ValueError(f"prerequisite {platform} arm has counted v2 faults or did not pass")
    return arm


def validate_manifest(manifest: dict[str, Any], repo: Path) -> None:
    """Authorize only the reviewed seven jobs under C1, without a family waiver."""
    rq = manifest.get("rq", {})
    platform, mode = rq.get("platform"), rq.get("mode")
    limits = envelope(platform, mode)
    expected = limits["effective"]
    if (
        manifest.get("script") != SCRIPT
        or rq.get("schema") != SCHEMA
        or rq.get("authority_sha256") != AUTHORITY_SHA256
        or rq.get("fault_authority_sha256") != FAULT_AUTHORITY_SHA256
        or rq.get("fault_definition") != FAULT_DEFINITION
        or rq.get("worker_pin") != WORKER_PIN
        or rq.get("bank_sha256") != BANK_SHA256
        or rq.get("frozen_source_sha256") != FROZEN_ARCHIVE_SHA256
        or rq.get("envelope") != limits
        or manifest.get("timeout_minutes") != expected["timeout_minutes"]
        or any(
            manifest.get("budget", {}).get(k) != expected[k] for k in ("unit_cap", "reserve_units")
        )
        or manifest.get("budget", {}).get("rate_units_per_hour")
        != (1.54 if platform == "l4" else 0.08)
        or manifest.get("requirements", {}).get("allowed_gpus")
        != (["L4"] if platform == "l4" else ["cpu"])
        or rq.get("workers") != (12 if platform == "l4" else 2)
        or rq.get("loop_seconds") != (3600 if mode == "loop" else 0)
        or manifest.get("canary")
        or manifest.get("resume_exemption")
    ):
        raise ValueError("qualification profile differs from frozen C1 envelope/identity")
    env = manifest.get("env", {})
    for key, value in {
        "RQ_MODE": mode,
        "RQ_PLATFORM": platform,
        "RQ_NAMESPACE": rq.get("namespace"),
        "RQ_ALLOCATION": rq.get("allocation"),
        "RQ_WORKERS": str(rq["workers"]),
        "RQ_LOOP_SECONDS": str(rq["loop_seconds"]),
        "RQ_SOURCE_COMMIT": WORKER_PIN,
        "RQ_SOURCE_CLOSURE_SHA256": rq.get("frozen_source_sha256"),
        "RQ_SOURCE_MANIFEST_SHA256": FROZEN_MANIFEST_SHA256,
    }.items():
        if not value or env.get(key) != value:
            raise ValueError(f"qualification environment differs: {key}")
    if manifest.get("resume") != {
        "unit_kind": "f1-root-repeat",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("qualification resume declaration differs")
    inputs = {row["dest"]: row for row in manifest.get("inputs", [])}
    for dest in (
        "inputs/f1-source.tar",
        "inputs/verified-ledger.json",
        "inputs/f1-root-bank.json.xz",
        "inputs/frozen-source.json",
    ):
        row = inputs.get(dest, {})
        if not row.get("path") or sha(Path(row["path"])) != row.get("sha256"):
            raise ValueError("qualification required input binding differs")
    if inputs["inputs/f1-source.tar"]["sha256"] != rq["frozen_source_sha256"]:
        raise ValueError("frozen worker source archive differs")
    if inputs["inputs/frozen-source.json"]["sha256"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen source manifest differs from reviewed F1 closure")
    from research.bl_labels.f1_ruler import bank_digest

    from .loop import _load_ledger

    if bank_digest(Path(inputs["inputs/f1-root-bank.json.xz"]["path"])) != BANK_SHA256:
        raise ValueError("qualification bank input differs")
    _load_ledger(
        Path(inputs["inputs/verified-ledger.json"]["path"]),
        rq["namespace"],
        rq["allocation"],
        BANK_SHA256,
        FROZEN_ARCHIVE_SHA256,
    )
    prerequisite = rq.get("prerequisite", {})
    path = Path(prerequisite.get("path", ""))
    if not path.is_file() or sha(path) != prerequisite.get("sha256"):
        raise ValueError("a18 acceptance prerequisite absent or changed")
    require_prerequisite(path, repo, "cpu" if mode == "injection" else platform)
    planned = rq.get("injection_manifest")
    if not isinstance(planned, dict) or sha(Path(planned.get("path", ""))) != planned.get("sha256"):
        raise ValueError("session lacks its create-only pre-dispatch injection manifest")
    from research.bl_labels.l4_faults_v2 import load_injection_manifest

    injections = load_injection_manifest(Path(planned["path"]), job_id=manifest.get("job_id"))
    request_ids = [entry.get("request_id") for entry in injections["entries"]]
    if mode == "injection":
        if (
            len(request_ids) != 3
            or len(set(request_ids)) != 3
            or planned.get("request_ids") != request_ids
            or any(e.get("type") != "post_send_response_loss" for e in injections["entries"])
        ):
            raise ValueError("injection session needs exactly three planned response losses")
    elif injections["entries"]:
        raise ValueError("loop sessions have no deliberate faults")
    review = require_review()
    for relative, digest in review["source_sha256"].items():
        if sha(repo / relative) != digest:
            raise ValueError(f"reviewed source differs: {relative}")
        pinned = subprocess.check_output(
            ["git", "show", f"{manifest['commit']}:{relative}"], cwd=repo, timeout=30
        )
        if hashlib.sha256(pinned).hexdigest() != digest:
            raise ValueError(f"manifest does not pin reviewed source: {relative}")
