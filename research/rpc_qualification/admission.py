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
# Evidence-only successor authorized by the prerequisite-refusal ruling, steps 2-4.
# The previous recovery/transport behavior remains fixed by differential proof.
TRANSPORT_PIN = "d68ba07c350d3f642879ee6df2cf9af9d34f0601"
TRANSPORT_TREE = "6415984f6cedc17a828a6a0d8fffc4519f7cb273"
# The independent reviewer's create-only ACCEPT lives outside the repository so
# recording it never moves the pack's pinned source commit.
REVIEW_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-24/"
    "ASTRA_L4_V2_REPAIR_REVIEW07_ACCEPT_20261024.json"
)
ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PACK_ROOT = ARTIFACT_ROOT / "big-lever/round-24/rpc-qualification-09"
PROOF_RELATIVE = "research/rpc_qualification/FAKE_CLI_PROOF_20261024_V2_09.json"
AUTHORITY_FILES = {
    "prerequisite_refusal": (
        ARTIFACT_ROOT / "big-lever/exp/astra-20261024/JUDGMENT_L4_V2_PREREQUISITE_REFUSAL.md",
        "b8ffb06bc9d91a2e6dd57720a08cdb880141b50debb9c051d188e51db82d7261",
    ),
    "R": (
        ARTIFACT_ROOT / "bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md",
        FAULT_AUTHORITY_SHA256,
    ),
    "J": (
        ARTIFACT_ROOT / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REQUAL_ATTEMPT2.md",
        "529f2b1aa069d1e97518ea30624ee807a153063303eaf6241c00a10d1a942d57",
    ),
    "H": (
        ARTIFACT_ROOT / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_HOLD_RESOLUTION.md",
        "adddd711494b8d855d70425e028d3606531d7b8527e4138ca8419baf217299ed",
    ),
    "review03": (
        ARTIFACT_ROOT / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW03.md",
        "77fd5e294fdf183349a4b0adbd6e2cc36ccbb8c458e4d1dd26af474ad39236f6",
    ),
    "review04": (
        ARTIFACT_ROOT / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW04.md",
        "75bcbbb9e6ae72ff6bcf728d9bab61a25127e43d61d2c5f83a73da73e65c3f22",
    ),
    "review05": (
        ARTIFACT_ROOT / "big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW05.md",
        "d24efca6dac518e0643c51cf6540bd726b5423328a7b5123681ffa448cfc5b07",
    ),
}


def git_commit(repo: Path) -> str:
    """Return the exact commit a review is allowed to bind."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=30
    ).strip()


def review_source_inventory(repo: Path, package_commit: str | None = None) -> dict[str, str]:
    """Return the complete reviewed gate/transport source inventory.

    Keep this set mechanical.  A reviewer may not select a convenient subset of
    the transport, classifier, admission wrapper, or the tests that establish
    the recovery proof.
    """
    package_commit = package_commit or git_commit(repo)
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", package_commit], cwd=repo, text=True, timeout=30
    ).splitlines()
    explicit = {
        "research/bl_labels/l4_faults_v2.py",
        "tools/ops/requalify.py",
        "tools/ops/dispatch.py",
        "tools/ops/qualification.py",
        "tools/ops/dispatch.sh",
        "tools/ops/watchd.sh",
        "tests/test_l4_cpu_acceptance_classifier.py",
        "tests/test_l4_prerequisite_evidence.py",
        "tests/test_colab_cli_acceptance_stall_diagnosis.py",
        "tests/test_colab_cli_rpc_recovery.py",
        "tests/test_l4_faults_v2.py",
        "tests/test_l4_lane_admission.py",
        "tests/test_ops_qualification.py",
        "tests/test_colab_cli_evidence.py",
        "tests/test_colab_cli_evidence_differential.py",
        "tests/test_colab_cli_acceptance.py",
        "tests/test_colab_cli_latency_policy.py",
        "tests/test_colab_cli_transfers.py",
        "tests/test_rpc_qualification_matrix.py",
        "tests/test_rpc_qualification_injection.py",
        "tests/test_rpc_qualification_loop.py",
        "tests/test_rpc_qualification_profile.py",
        "tests/test_rpc_qualification_probe_policy.py",
        "tests/test_rpc_qualification_checker.py",
        "tests/fixtures/make_l4_faults_v2_fixture.py",
        "tests/fixtures/fake_colab.py",
        "tests/test_colab_cli_dispatch_warmup.py",
    }
    paths = [
        path
        for path in tracked
        if (
            path in explicit
            or path.startswith("tests/fixtures/l4_prerequisite_refusal/")
            or (path.startswith("research/colab_cli/") and path.endswith((".py", ".sh")))
            or (path.startswith("research/rpc_qualification/") and path.endswith((".py", ".sh")))
        )
    ]
    return {path: sha(repo / path) for path in paths}


def verify_review_sources(review: dict[str, Any], repo: Path, package_commit: str) -> None:
    """Verify the complete ACCEPT inventory against working and pinned bytes."""
    if review.get("package_commit") != package_commit:
        raise ValueError("reviewed package commit differs from the dispatch package")
    expected = review_source_inventory(repo, package_commit)
    supplied = review.get("source_sha256")
    if supplied != expected:
        raise ValueError("reviewer ACCEPT lacks the complete source inventory")
    for relative, digest in supplied.items():
        if sha(repo / relative) != digest:
            raise ValueError(f"reviewed source differs: {relative}")
        pinned = subprocess.check_output(
            ["git", "show", f"{package_commit}:{relative}"], cwd=repo, timeout=30
        )
        if hashlib.sha256(pinned).hexdigest() != digest:
            raise ValueError(f"reviewed source differs from package commit: {relative}")


def require_review(
    transport_pin: str = TRANSPORT_PIN,
    *,
    repo: Path | None = None,
    package_commit: str | None = None,
) -> dict[str, Any]:
    """Reviewer ACCEPT at this exact fix pin, required before any attempt-2 dispatch."""
    repo = repo or Path(__file__).resolve().parents[2]
    package_commit = package_commit or git_commit(repo)
    if not REVIEW_PATH.is_file():
        raise ValueError("attempt-2 package lacks its independent reviewer ACCEPT record")
    review = json.loads(Path(REVIEW_PATH).read_text())
    if (
        review.get("schema") != "l4-requal-02-review/v1"
        or review.get("verdict") != "ACCEPT"
        or review.get("transport_pin") != transport_pin
        or not isinstance(review.get("package_commit"), str)
        or len(review["package_commit"]) != 40
        or not isinstance(review.get("source_sha256"), dict)
    ):
        raise ValueError("attempt-2 package lacks an independent reviewer ACCEPT at this pin")
    verify_review_sources(review, repo, package_commit)
    verify_review_artifacts(review, repo, package_commit)
    return review


def review_artifact_bindings(repo: Path, package_commit: str) -> dict[str, Any]:
    """Compute exact pre-data pack identities for the independent review record."""
    from research.bl_labels.exports import _verify

    paths = {
        "plan": PACK_ROOT / "plan.json",
        "proof": repo / PROOF_RELATIVE,
        "trace": PACK_ROOT / "DIFFERENTIAL_TRACE.json",
        "helper": PACK_ROOT / "GUARDED_OPERATOR_COMMANDS_V6.sh",
    }
    for name in ("plan", "proof", "trace"):
        _verify(paths[name])
    plan = json.loads(paths["plan"].read_text())
    proof = json.loads(paths["proof"].read_text())
    if (
        plan.get("source_commit") != package_commit
        or plan.get("transport_pin") != TRANSPORT_PIN
        or plan.get("transport_tree") != TRANSPORT_TREE
        or plan.get("fake_proof") != {"path": str(paths["proof"]), "sha256": sha(paths["proof"])}
        or sha(paths["trace"]) != proof.get("differential_trace_sha256")
        or json.loads(paths["trace"].read_text()) != proof.get("differential_trace")
    ):
        raise ValueError("review artifacts differ from the frozen package bindings")
    authorities = {name: sha(path) for name, (path, _digest) in AUTHORITY_FILES.items()}
    if authorities != {name: digest for name, (_path, digest) in AUTHORITY_FILES.items()}:
        raise ValueError("review authority bytes differ from the ratified ruling")
    return {
        "artifact_sha256": {name: sha(path) for name, path in paths.items()},
        "authority_sha256": authorities,
        "transport_tree": TRANSPORT_TREE,
    }


def verify_review_artifacts(review: dict[str, Any], repo: Path, package_commit: str) -> None:
    """Refuse copied ACCEPTs whose plan, proof, trace, helper or authority changed."""
    expected = review_artifact_bindings(repo, package_commit)
    if any(review.get(key) != value for key, value in expected.items()):
        raise ValueError("reviewer ACCEPT artifact/authority digests or transport tree differ")


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


def prerequisite_arms(
    path: Path, repo: Path, package_commit: str | None = None
) -> dict[str, dict[str, Any]]:
    """v2 classification of each final-only acceptance arm at the fix pin.

    Deliberate acceptance faults are excluded only by each arm's create-only
    pre-dispatch injection manifest (bound by sha256 in the reviewed plan).
    """
    from research.bl_labels.l4_faults_v2 import classify_session, validate_injection_manifest
    from research.colab_cli.admission import require_final_only

    result = json.loads(path.read_text())
    require_final_only(result, repo)
    plan = result["final_plan"]
    package_commit = package_commit or git_commit(repo)
    if plan.get("commit") != package_commit:
        raise ValueError(
            "prerequisite is not the final-only acceptance at the reviewed package commit"
        )
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{package_commit}:research/colab_cli"],
        cwd=repo,
        text=True,
        timeout=30,
    ).strip()
    if tree != TRANSPORT_TREE:
        raise ValueError("prerequisite package transport tree differs from the v2 fix pin")
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
        observer: dict[str, Any] | None = None
        # The observer is created after the immutable receipt, so it is a
        # sibling artifact. Bind its raw bytes to this exact receipt/job and
        # endpoint before letting it excuse a designed watchdog release.
        observer_path = receipt_path.with_name("acceptance-observer.json")
        terminal_entries = [
            entry for entry in manifest["entries"] if entry["command"] == "terminal"
        ]
        if terminal_entries:
            if not observer_path.is_file():
                raise ValueError("prerequisite watchdog arm lacks acceptance observer")
            raw_observer = observer_path.read_bytes()
            observer = json.loads(raw_observer)
            if not isinstance(observer, dict):
                raise ValueError("acceptance observer is not an object")
            gate_id = terminal_entries[0].get("watchdog_event_id")
            gate = (observer.get("gates") or {}).get(gate_id)
            planned_gate = next(
                (
                    candidate
                    for candidate in plan.get("gates", [])
                    if candidate.get("id") == gate_id
                ),
                None,
            )
            endpoint = receipt.get("endpoint", receipt.get("accounting_endpoint"))
            if (
                not isinstance(gate, dict)
                or not isinstance(planned_gate, dict)
                or gate.get("planned_event_id") != gate_id
                or planned_gate.get("evidence_record") != gate
                or gate.get("job_id") != receipt.get("job_id")
                or not isinstance(endpoint, str)
                or not endpoint
                or gate.get("endpoint") != endpoint
            ):
                raise ValueError("acceptance observer is not bound to the receipt job/endpoint")
            expected_observer = row.get("acceptance_observer")
            if expected_observer is not None and expected_observer != str(observer_path):
                raise ValueError("acceptance observer path differs from final plan binding")
            expected_digest = row.get("acceptance_observer_sha256")
            observed_digest = hashlib.sha256(raw_observer).hexdigest()
            if expected_digest is not None and expected_digest != observed_digest:
                raise ValueError("acceptance observer hash differs from final plan binding")
            # Record these immutable bindings in the return value, while the
            # classifier consumes only the exact parsed observer bytes.
            observer = {
                **observer,
                "_binding": {
                    "path": str(observer_path),
                    "sha256": observed_digest,
                    "receipt_sha256": sha(receipt_path),
                    "injection_manifest_sha256": sha(manifest_path),
                    "plan_sha256": hashlib.sha256(
                        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                },
            }
        verdict = (
            classify_session(receipt, manifest, acceptance_observer=observer)
            if observer is not None
            else classify_session(receipt, manifest)
        )
        arms["l4" if row["job"] == "l4" else "cpu"] = {
            **verdict,
            "harness_status": row.get("status"),
            **({"acceptance_observer": observer["_binding"]} if observer is not None else {}),
        }
    if set(arms) != {"cpu", "l4"}:
        raise ValueError("prerequisite lacks both acceptance arms")
    return arms


def require_prerequisite(
    path: Path, repo: Path, platform: str, *, package_commit: str | None = None
) -> dict[str, Any]:
    """The fix-pin final-only acceptance is necessary, never sufficient (§4 step 3).

    ``platform`` is the platform about to dispatch: L4 loops need the L4 arm
    v2 PASS; CPU loops and the CPU injection proof need the CPU arm v2 PASS.
    """
    if platform not in {"cpu", "l4"}:
        raise ValueError("unknown prerequisite platform")
    arms = prerequisite_arms(path, repo, package_commit)
    failed = [
        name
        for name, arm in arms.items()
        if arm["verdict"] != "PASS" or arm["harness_status"] != "PASS"
    ]
    if failed:
        raise ValueError(
            "required " + ", ".join(sorted(failed)) + " arm has counted v2 faults or did not pass"
        )
    return arms[platform]


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
    require_review(repo=repo, package_commit=manifest["commit"])
