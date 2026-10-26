"""Offline, create-only F5 W review/dispatch preparation; never a dispatcher.

Only explicit governance and existing status metadata are read. No provider,
training data, checkpoint, simulator or fit API is imported by this module.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024")
LEDGER = "research/bl_residual/ledger/F5_REGISTRATION_20261025"
ENV = "research/bl_residual/f5_adoption_environment_20261026_v6"
COLLISION = "research/bl_residual/f5_adoption_collision_20261026/final-v4"
ADOPTION = "F5_REGISTRATION_ADOPTED_20261025.md"
SELECTION = "F5_SELECTION_W_20261025_20261026T032844Z.json"
ADOPTION_SHA = "4e218263f169334415968f02eb2d767e1ff136dfc3ccf470090dd474b31dd6a8"
SELECTION_SHA = "b4a880816bcd050cc36aebed7f73547a37a5fc6890889fff7203771a6449d3c3"
ASSIGNMENT_AMENDMENT = "F5_L4_ASSIGNMENT_IDENTITY_AMENDMENT_ADOPTED_20261025.md"
ASSIGNMENT_AMENDMENT_SHA = "a1dc92ffb8fa341fb51738a56fe0097d75e15b0aff7194940c12979adbb04d62"
ASSIGNMENT_DRAFT_SHA = "7eda19efb72f89a7e2269d40863288f9295358f2566f3a02d3ca4d3c11933359"
REJECTION_SHA = "b672b7b62bd7744e3226aabbd516a8a3a562b6d57f97c128b16c7f130d1e5aed"
CAPTURE_REJECTION_SHA = "f1b0244d36f9cef541a3293a4749fa0cd7f6ef49b7de31fc056ca9c0ad8bd6e8"
OWNER_RULING_CAPTURE_PATH = ART / "OWNER_RULING_COLAB_BUDGET_20261025.txt"
OWNER_RULING_CAPTURE_SHA = "48df0ca878dea69a6e0608946f8770cd56e7d017d79890b9566ab31debba767d"
ALLOCATION_SHA = "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
SEMANTIC_COMMIT = "124dd22cd560a74a210a61f0ba473461328e82f8"
FIT_SHA = "85c7e77d34e14a3b5036984cc1af92df4e3724f4c9fe453d813a56c1963822a7"
THREADS = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}
INTERPRETATION = {
    "interpretation": "EXPLORATORY_SCREENING",
    "confirmatory_efficacy_claim": False,
    "release_authorized": False,
}


def canonical(value: Any) -> bytes:
    """Match finite canonical export encoding."""
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def hold_timing_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove transport-only sizing from a full timing proposal.

    Unknown numeric fields deliberately fail broker validation. A HOLD label
    alone would not prevent a copied transport canary from being dispatched.
    A reviewed successor must supply its own forecast and resume contract.
    """
    proposal = {
        **manifest,
        "requirements": {**manifest["requirements"], "est_minutes": None},
        "timeout_minutes": None,
        "timing_envelope_status": "HOLD_REQUIRES_REVIEWED_FORECAST_TIMEOUT_AND_RESUME",
    }
    proposal.pop("resume", None)
    proposal.pop("resume_exemption", None)
    return proposal


def safe_metadata(path: Path) -> bytes:
    """Read only an explicit non-protected metadata file with no symlink ancestry."""
    if any("d4096" in part.lower() or part == "t34-round0-study-01" for part in path.parts):
        raise ValueError("prohibited path")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink metadata path")
    if path.suffix not in {".md", ".json", ".lock", ".sha256"}:
        raise ValueError("not metadata")
    return path.read_bytes()


def binding(path: Path, expected: str | None = None) -> dict[str, Any]:
    """Bind explicit metadata, checking a declared digest when supplied."""
    payload = safe_metadata(path)
    sha = hashlib.sha256(payload).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"metadata digest mismatch: {path.name}")
    return {"path": str(path), "sha256": sha, "size": len(payload)}


def source_review(repo: Path = ROOT) -> dict[str, Any]:
    """Compare the actual reused F4 functions to the registered Git object."""
    frozen = subprocess.check_output(
        ["git", "show", f"{SEMANTIC_COMMIT}:research/bl_residual/fit.py"], cwd=repo
    )
    if hashlib.sha256(frozen).hexdigest() != FIT_SHA:
        raise ValueError("registered F4 source object mismatch")
    current = (repo / "research/bl_residual/fit.py").read_bytes()
    names = [
        "_stratum_groups",
        "_life_uniform_indices",
        "_sample_batch_indices",
        "_preference_loss",
        "_ordinary_growth_loss",
        "_resolved_action_mask",
        "_served_argmax",
    ]

    def symbols(source: bytes) -> dict[str, str]:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }

    a, b = symbols(frozen), symbols(current)
    matches = {name: name in a and a[name] == b.get(name) for name in names}
    if not all(matches.values()):
        raise ValueError("registered F4 symbol drift")
    return {
        "semantic_commit": SEMANTIC_COMMIT,
        "frozen_fit_sha256": FIT_SHA,
        "current_fit_sha256": hashlib.sha256(current).hexdigest(),
        "symbols_equal": matches,
    }


def governance(repo: Path = ROOT, artifacts: Path = ART) -> dict[str, Any]:
    """Bind adoption's exact governance schedule without following payload references."""
    adopted = repo / LEDGER / ADOPTION
    selection = repo / LEDGER / SELECTION
    amendment = repo / LEDGER / ASSIGNMENT_AMENDMENT
    records = [
        binding(adopted, ADOPTION_SHA),
        binding(selection, SELECTION_SHA),
        binding(amendment, ASSIGNMENT_AMENDMENT_SHA),
    ]
    published_amendment = subprocess.check_output(
        ["git", "show", f"origin/main:{LEDGER}/{ASSIGNMENT_AMENDMENT}"], cwd=repo
    )
    if published_amendment != amendment.read_bytes():
        raise ValueError("assignment amendment not published on main")
    records.extend(
        [
            binding(
                artifacts
                / "F5_W_PREFIT_20261025/final-03/F5_L4_ASSIGNMENT_IDENTITY_AMENDMENT_DRAFT.md",
                ASSIGNMENT_DRAFT_SHA,
            ),
            binding(artifacts / "REJECT_F5_W_IMPLEMENTATION_36561254.md", REJECTION_SHA),
            binding(artifacts / "REJECT_F5_W_IMPLEMENTATION_22e86986.md", CAPTURE_REJECTION_SHA),
        ]
    )
    # Adoption explicitly incorporates these exact metadata references. D4096 is
    # deliberately never followed, including its allocation metadata.
    schedule = re.findall(r"- `([^`]+)` — SHA256 `([0-9a-f]{64})`", adopted.read_text())
    skipped = []
    for name, sha in schedule:
        original = Path(name)
        if any("d4096" in p.lower() or p == "t34-round0-study-01" for p in original.parts):
            skipped.append({"path": name, "sha256": sha, "read": False})
            continue
        if "/snake-dqn/" in name:
            path = repo / name.split("/snake-dqn/", 1)[1]
        elif original.is_relative_to(ART):
            path = artifacts / original.relative_to(ART)
        else:
            # External expert transcript is governance explicitly bound by adoption.
            path = original
        records.append(binding(path, sha))
    closure = json.loads(safe_metadata(repo / COLLISION / "CLOSURE_PACKET_v4.json"))
    for name in ("CLOSURE_PACKET_v4.json", "allocation-evaluation.json", "omissions.json"):
        path = repo / COLLISION / name
        expected = (
            "74aae60adbe3658addb59868d4d0a3289252dd7a0ea77d5338b5a053079b3262"
            if name == "CLOSURE_PACKET_v4.json"
            else closure["file_hashes"][name]
        )
        records.append(binding(path, expected))
        published = subprocess.check_output(
            ["git", "show", f"origin/main:{COLLISION}/{name}"], cwd=repo
        )
        if published != path.read_bytes():
            raise ValueError("unpublished final-v4 closure metadata")
    value = json.loads(selection.read_bytes())
    if (value.get("decision"), value.get("selected_id"), value.get("interpretation")) != (
        "SELECT_W",
        "W",
        "EXPLORATORY_SCREENING",
    ):
        raise ValueError("not the published selected W design")
    for name in (ADOPTION, SELECTION):
        relative = f"{LEDGER}/{name}"
        published = subprocess.check_output(["git", "show", f"origin/main:{relative}"], cwd=repo)
        if published != (repo / relative).read_bytes():
            raise ValueError("selection/adoption not published on origin/main")
    return {
        "bindings": records,
        "prohibited_references_not_followed": skipped,
        "selection_publication_verified": True,
        "historical_collision_coverage": "UNRESOLVED_RISK_ACCEPTED_FOR_BOTH_CONTRASTS",
    }


def one_difference() -> dict[str, Any]:
    """Machine-comparable common recipe and sole intervention."""
    from research.bl_residual.fit import FitRecipe

    common = {
        "recipe": dataclasses.asdict(FitRecipe()),
        "initializer": "frozen B8 + zero Linear(256,6)",
        "corpus": "selected960; exclude sibling9 and original fold0/census descendants",
        "batch_plan": "one sealed ordered 2000x256 plan; same identities C/T",
        "environment": f"{ENV}/f5-common-environment-v6.json",
        "export": "research.bl_residual.model.fuse_residual",
        "survival_denominator": "original eligible pair count",
    }
    c = {**common, "qualifying_ordinary_B8_survivor_pair_weight": 1}
    t = {**common, "qualifying_ordinary_B8_survivor_pair_weight": 4}
    differences = [key for key in c if c[key] != t[key]]
    if differences != ["qualifying_ordinary_B8_survivor_pair_weight"]:
        raise ValueError("more than the registered single difference")
    return {
        "C": c,
        "T": t,
        "differences": differences,
        "qualifier": (
            "ordinary & B8 exact/legal/served survivor & "
            "pair survivor=B8 & alternative exact dier"
        ),
    }


def corpus_review(artifacts: Path = ART) -> dict[str, Any]:
    """Validate the published status-only projection; never open original NPZ data."""
    from research.bl_residual.f5_data import metadata_cache

    folder = artifacts / "f5-selected960-49fd1dfa"
    scan = folder / "scan.json"
    inventory = folder / "extractor-input-inventory.json"
    binding(scan, "8b4910213475266038c8a6a65ad9aea00cd301eba08311f44bb59bc23c312ac4")
    binding(inventory, "cc20e1e7f5c321de5a1f63b94945c92c15c852a8499e88ab56659d1d2d7c1615")
    metadata = metadata_cache(json.loads(scan.read_bytes()), json.loads(inventory.read_bytes()))
    row_projection = (
        json.dumps(metadata.rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return {
        "selected_lives": len(metadata.selected_lives),
        "eligible_lives": len(metadata.eligible_lives),
        "eligible_parents": len({row["parent_world"] for row in metadata.rows}),
        "eligible_rows": len(metadata.rows),
        "doom_rows": sum(not row["ordinary"] for row in metadata.rows),
        "ordinary_rows": sum(row["ordinary"] for row in metadata.rows),
        "ordered_metadata_rows_sha256": hashlib.sha256(row_projection).hexdigest(),
        "excluded_parent_count": len(metadata.excluded_parents),
        "selected_sibling_policy": "all nine ordinary-only siblings excluded from both arms",
        "source": dataclasses.asdict(metadata.source),
        "status_inventory_sha256": metadata.scan_sha256,
        "labels_opened": False,
        "actual_plan": "requires selected training labels and frozen L4 build",
    }


def source_inventory(repo: Path = ROOT) -> dict[str, str]:
    """Freeze local executable dependencies without importing their runtime code."""
    pending = [
        repo / "research/bl_residual" / name
        for name in (
            "f5_package.py",
            "f5_fit.py",
            "f5_data.py",
            "f5_job.py",
            "f5_timing.py",
            "f5_rules.py",
            "f5_protocol.py",
            "f5_environment.py",
            "f5_qualification.py",
            "f5_progress.py",
        )
    ]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if module.split(".")[0] not in {"src", "research"}:
                    continue
                candidate = repo.joinpath(*module.split("."))
                for source in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if source.is_file() and source not in seen:
                        pending.append(source)
    seen.add(repo / "research/bl_residual/f5_colab.sh")
    seen.add(repo / "research/bl_residual/f5_environment_schema_v5.json")
    seen.add(repo / "research/bl_residual/f5_calibration_schema_v1.json")
    seen.add(repo / "research/apex_safety_20260926/deployment.yaml")
    seen.update((repo / "configs").glob("*.yaml"))
    return {str(path.relative_to(repo)): digest(path) for path in sorted(seen)}


def validate_draft_manifest(value: dict[str, Any]) -> list[str]:
    """Offline static validation; no CLI imports, account access or allocation."""
    if value.get("scientific_admission") is not False:
        raise ValueError("unreviewed draft must refuse operational dispatch")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("commit", "")):
        raise ValueError("full source commit required")
    if value["requirements"] != {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 6,
        "est_minutes": 60,
    }:
        raise ValueError("registered L4 placement mismatch")
    if value["timeout_minutes"] != 200 or value["device"] != "cuda:0":
        raise ValueError("paired-fit effective envelope/device mismatch")
    if value["resume"] != {
        "unit_kind": "paired-fit",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume contract mismatch")
    if value["f5"]["transfer_seconds"] != 150 or value["f5"]["staging_seconds"] != 300:
        raise ValueError("standard transfer/staging bars required")
    return [
        "independent source/one-difference/batch-plan review",
        "sealed actual L4 environment and full batch plan",
        "current-source timing/parity and complete measured forecasts",
        "qualified CLI transport and source-native canary/packet acceptance",
        "hash-bound selected training payload and verified empty ledger upload",
    ]


def bind_qualification_inventory(
    science: dict[str, Any], qualification: dict[str, Any]
) -> dict[str, Any]:
    """Bind the final exact upload set; no placeholder or size-only coverage.

    Call after actual plan/timing review and final acceptance have been sealed.
    A prior timing run cannot qualify files that did not yet exist. The final
    transport canary repeats only the sealed timing workload with these exact
    science bytes; it never opens the selected training package.
    """
    from research.colab_cli import canary

    immutable_names = {"selected-training.tar.xz", "f5-batch-plan.pt", "acceptance.json"}
    required = {
        "control.json",
        "b8.pth",
        "selected-training.tar.xz",
        "f5-batch-plan.pt",
        "acceptance.json",
        "verified-ledger.json",
        "timing-fixtures.json",
    }
    required |= {name + suffix for name in immutable_names for suffix in (".sha256", ".complete")}
    rows = science.get("inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("complete actual science upload inventory required")
    destinations: set[str] = set()
    private_total = 0
    for row in rows:
        if set(row) != {"path", "dest", "sha256"}:
            raise ValueError("F5 qualification requires actual uploaded local files")
        name, path = row["dest"], Path(row["path"])
        if not isinstance(name, str) or Path(name).name != name or name in destinations:
            raise ValueError("unsafe or duplicate science upload destination")
        destinations.add(name)
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError("actual science input missing or unsafe")
        if digest(path) != row["sha256"]:
            raise ValueError("science input digest mismatch")
        private_total += path.stat().st_size
    if not required <= destinations or private_total > 45_397_643:
        raise ValueError("incomplete or oversized actual private science inventory")
    from research.bl_ppo.exports import verify_export

    uploaded = {row["dest"]: Path(row["path"]) for row in rows}
    for name in immutable_names:
        path = uploaded[name]
        if path.name != name:
            raise ValueError("immutable upload filename must match runtime destination")
        verify_export(path)
        for suffix in (".sha256", ".complete"):
            if uploaded[name + suffix].read_bytes() != Path(str(path) + suffix).read_bytes():
                raise ValueError("immutable uploaded sidecar differs from verified export")
    if (
        science["commit"] != qualification["commit"]
        or science["source_paths"] != qualification["source_paths"]
    ):
        raise ValueError("qualification source differs from science")
    if canary.family(science) != canary.family(qualification):
        raise ValueError("qualification entrypoint differs from science")
    result = json.loads(json.dumps(qualification))
    result["inputs"] = json.loads(json.dumps(rows))
    result["required_exports"] = sorted(
        set(science["required_exports"]) | {"qualification/qualification.json"}
    )
    result["inventory_status"] = "BOUND_EXACT_ACTUAL_SCIENCE_UPLOADS"
    result["science_upload_inventory"] = canary.input_inventory(science)
    return result


def capture_remainder(prior_receipt: Path | None) -> dict[str, Any]:
    """Carry the failed capture debit forward without approving a new envelope."""
    elapsed = 0.0
    prior = None
    debit = None
    if prior_receipt is not None:
        prior_receipt = prior_receipt.absolute()
        prior = binding(prior_receipt)
        receipt = json.loads(safe_metadata(prior_receipt))
        elapsed = receipt.get("elapsed_seconds")
        if (
            receipt.get("job_id") != "bl-f5-w-capture-repair-22e86986-01"
            or receipt.get("stop_verified") is not True
            or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
            or receipt.get("exports") != []
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or not isinstance(receipt.get("accounting"), dict)
            or not receipt["accounting"]
        ):
            raise ValueError("prior capture must be a stopped, failed attempt-01 receipt")
        debit = receipt.get("accounting")
    remaining = max(0.0, 600.0 - elapsed)
    reasons = []
    if remaining < 180:
        reasons.append("remaining capture allowance is below supervisor 180-second minimum")
    if remaining < 600:
        reasons.append("remaining capture allowance is below twice inherited 300-second forecast")
    return {
        "schema": "bl-f5-capture-carry-forward/v1",
        "original_capture_allowance_seconds": 600,
        "prior_capture_receipt": prior,
        "prior_elapsed_seconds": elapsed,
        "prior_accounting": debit,
        "remaining_capture_seconds": remaining,
        "remaining_aggregate_seconds": max(0.0, 14400.0 - elapsed),
        "inherited_estimate_seconds": 300,
        "allowance_reset": False,
        "automatic_retries": False,
        "admission": "HOLD" if reasons else "DRAFT_PENDING_INDEPENDENT_REVIEW",
        "hold_reasons": reasons,
    }


def _diagnostic_only_capture_exports(exports: Any) -> bool:
    """Permit attempt-03 refusal diagnostics, never admitted/scientific exports."""
    if not isinstance(exports, list) or len(exports) != 2:
        return False
    expected = {"environment-raw.json", "environment-diagnostic.json"}
    paths = set()
    for row in exports:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or not isinstance(row["path"], str)
            or row["path"] not in expected
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            or type(row["size"]) is not int
            or row["size"] <= 0
        ):
            return False
        paths.add(row["path"])
    return paths == expected


def capture_history(prior_receipts: list[Path]) -> dict[str, Any]:
    """Bind all stopped failed captures and retain their raw accounting evidence.

    This is disclosure only.  It does not treat historical unit debits or the
    superseded 600-second capture allowance as an operational admission rule.
    """
    expected_jobs = (
        "bl-f5-w-capture-repair-22e86986-01",
        "bl-f5-w-capture-repair-22e86986-02",
        "bl-f5-w-capture-repair-22e86986-03",
    )
    if len(prior_receipts) not in {2, 3}:
        raise ValueError("capture history requires two or three published stopped attempts")
    expected_jobs = expected_jobs[: len(prior_receipts)]
    rows = []
    for expected_job, path in zip(expected_jobs, prior_receipts, strict=True):
        absolute = path.absolute()
        receipt = json.loads(safe_metadata(absolute))
        elapsed = receipt.get("elapsed_seconds")
        accounting = receipt.get("accounting")
        if (
            receipt.get("job_id") != expected_job
            or receipt.get("stop_verified") is not True
            or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
            or (
                receipt.get("exports") != []
                if expected_job != "bl-f5-w-capture-repair-22e86986-03"
                else not _diagnostic_only_capture_exports(receipt.get("exports"))
            )
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or not isinstance(accounting, dict)
            or not isinstance(accounting.get("last_crosscheck"), dict)
        ):
            raise ValueError("prior capture history must contain stopped non-science failures")
        crosscheck = accounting["last_crosscheck"]
        debit = crosscheck.get("attributable_account_debit_units")
        observed = crosscheck.get("observed_account_debit_units")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (debit, observed)
        ):
            raise ValueError("prior capture accounting debit is malformed")
        rows.append(
            {
                "receipt": binding(absolute),
                "job_id": expected_job,
                "actual_elapsed_seconds": elapsed,
                "accounting_debit_units": debit,
                "observed_account_debit_units": observed,
                "raw_accounting": accounting,
            }
        )
    return {
        "schema": "bl-f5-capture-history/v1",
        "attempts": rows,
        "cumulative_actual_elapsed_seconds": sum(row["actual_elapsed_seconds"] for row in rows),
        "cumulative_accounting_debit_units": sum(row["accounting_debit_units"] for row in rows),
        "cumulative_observed_account_debit_units": sum(
            row["observed_account_debit_units"] for row in rows
        ),
        "budget_status": "DISCLOSED_NOT_AN_ADMISSION_BLOCKER",
        "rate_status": "STALE_HISTORICAL_OBSERVATIONS_REQUIRE_CURRENT_ACCOUNT_REFRESH",
    }


def capture_owner_override(ruling: Path | None) -> dict[str, Any] | None:
    """Bind the only owner ruling that permits the generous capture timeout."""
    if ruling is None:
        return None
    absolute = ruling.absolute()
    if (
        absolute != OWNER_RULING_CAPTURE_PATH
        or not absolute.is_file()
        or any(part.is_symlink() for part in (absolute, *absolute.parents))
    ):
        raise ValueError("capture owner ruling must be the exact known owner file")
    payload = absolute.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != OWNER_RULING_CAPTURE_SHA:
        raise ValueError("capture owner ruling digest mismatch")
    bound = {"path": str(absolute), "sha256": sha, "size": len(payload)}
    return {
        "binding": bound,
        "capture_runaway_timeout_minutes": 30,
        "technical_budget": "generous runaway timeout only; no unit-budget admission blocker",
        "rate_status": "historical pricing is stale; refresh the current account before operation",
    }


def build(
    output: Path,
    repo: Path = ROOT,
    artifacts: Path = ART,
    selected_package: Path | None = None,
    prior_capture_receipt: Path | None = None,
    prior_capture_receipts: list[Path] | None = None,
    capture_owner_ruling: Path | None = None,
) -> dict[str, Any]:
    """Seal a reviewable preparation packet; scientific manifests stay denied."""
    from research.bl_residual.f5_timing import qualification_plan

    output = output.absolute()
    if output.exists():
        raise FileExistsError("create-only packet output already exists")
    published_failure = repo / (
        "research/bl_residual/ledger/F5_W_CAPTURE_AFTER_22e86986_REPAIR_01/"
        "output-01/receipt.json"
    )
    if (
        published_failure.exists()
        and prior_capture_receipt is None
        and prior_capture_receipts is None
    ):
        raise ValueError(
            "published attempt-01 requires explicit --prior-capture-receipt accounting"
        )
    authority = governance(repo, artifacts)
    if prior_capture_receipt is not None and prior_capture_receipts is not None:
        raise ValueError("use either singular or plural prior capture receipts")
    capture_accounting = capture_remainder(prior_capture_receipt)
    capture_history_value = (
        None if prior_capture_receipts is None else capture_history(prior_capture_receipts)
    )
    capture_override = capture_owner_override(capture_owner_ruling)
    source = source_review(repo)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    evaluation = json.loads(safe_metadata(repo / COLLISION / "allocation-evaluation.json"))
    allocation = evaluation["candidate_allocation"]
    from research.bl_residual.f5_r10a_annex import allocate, encoded

    if (
        evaluation["sealed_allocation_sha256"] != ALLOCATION_SHA
        or evaluation["sealed_round"] != 0
        or len(allocation) != 684
    ):
        raise ValueError("foreign exploratory allocation")
    if hashlib.sha256(encoded(allocation)).hexdigest() != ALLOCATION_SHA:
        raise ValueError("changed full 684 allocation digest")
    if allocation != allocate(0):
        raise ValueError("allocation differs from frozen physical-seed/phase/order constructor")
    corpus = corpus_review(artifacts)
    hashes = source_inventory(repo)
    for name, expected in hashes.items():
        pinned = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        if hashlib.sha256(pinned).hexdigest() != expected:
            raise ValueError("review packet requires a committed exact source closure")
    source_paths = sorted(hashes) + [
        f"{ENV}/requirements-f5-l4-cu126.lock",
        f"{ENV}/f5-common-environment-v6.json",
    ]
    prepared_input = None
    if selected_package is not None:
        from research.bl_ppo.exports import verify_export

        selected_package = selected_package.absolute()
        verify_export(selected_package)
        receipt_path = selected_package.parent / "packaging-receipt.json"
        verify_export(receipt_path)
        prepared_input = json.loads(safe_metadata(receipt_path))
        if (
            prepared_input["package_sha256"] != digest(selected_package)
            or prepared_input["package_bytes"] != selected_package.stat().st_size
            or selected_package.stat().st_size > 45397643
        ):
            raise ValueError("selected private package digest/size/cap mismatch")
    ledger = {
        "schema": "bl-f5-verified-ledger/v1",
        "namespace": "bl-residual-f5-w-20261025-v1",
        "allocation": None,
        "allocation_sha256": ALLOCATION_SHA,
        "units": [],
        "status": "DRAFT_BIND_RUNTIME_CONTROL_ALLOCATION_BEFORE_UPLOAD",
    }
    manifest = {
        "job_id": f"bl-f5-w-paired-{commit[:8]}-01",
        "commit": commit,
        "script": "research/bl_residual/f5_colab.sh",
        "source_paths": source_paths,
        "device": "cuda:0",
        "scientific_admission": False,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 60},
        "timeout_minutes": 200,
        "output_dir": str(output / "paired-output-01"),
        "inputs": (
            []
            if selected_package is None
            else [
                {
                    "path": str(Path(str(selected_package) + suffix)),
                    "dest": "selected-training.tar.xz" + suffix,
                    "sha256": digest(Path(str(selected_package) + suffix)),
                }
                for suffix in ("", ".sha256", ".complete")
            ]
        ),
        "required_exports": [
            "paired/assignment-environment.json",
            "paired/fit-receipt.json",
            "paired/fit-C.pth",
            "paired/fit-W.pth",
            "paired/f5-state-C.pt",
            "paired/f5-state-W.pt",
        ],
        "env": {
            **THREADS,
            "F5_MODE": "fit",
            "F5_DEVICE": "cuda:0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "resume": {
            "unit_kind": "paired-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 2 * 1024**3,
        "f5": {
            **INTERPRETATION,
            "stage": "S1",
            "transfer_seconds": 150,
            "staging_seconds": 300,
            "private_upload_bytes_max": 45397643,
            "base_hours": 2,
            "effective_hours": 4,
            "canary_included_in_fit_envelope": True,
            "subphase_wall_minutes": {"capture": 10, "prepare_canary": 30, "paired_fit": 200},
            "aggregate_effective_wall_minutes": 240,
            "automatic_retries": False,
            "technical_budget": "operator binds verified current rate; no scientific unit ceiling",
            "inputs_status": "HOLD_UNTIL_PRE_FIT_GATES",
        },
    }
    if selected_package is not None:
        b8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
        b8_sha = digest(b8)
        if b8_sha != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
            raise ValueError("fixed B8 input digest mismatch")
        manifest["inputs"].append({"path": str(b8), "dest": "b8.pth", "sha256": b8_sha})
        private_known = sum(Path(item["path"]).stat().st_size for item in manifest["inputs"])
        if private_known > 45397643:
            raise ValueError("private training/B8 inventory exceeds standard cap")
        manifest["f5"]["known_private_input_bytes"] = private_known
        manifest["f5"]["remaining_private_bytes_for_plan_and_gate_metadata"] = (
            45397643 - private_known
        )
    blocked = validate_draft_manifest(manifest)
    if capture_override is None:
        blocked.extend(capture_accounting["hold_reasons"])
    timing = qualification_plan(allocation)
    output.mkdir(parents=True)
    files = {}
    fixture_rows = [row for row in allocation if row.get("timing_only") is True]
    files["timing-fixtures.json"] = export_json(
        output / "timing-fixtures.json",
        {
            "schema": "bl-f5-sealed-timing-fixtures/v1",
            "fixtures": fixture_rows,
        },
    )
    from research.bl_residual.f5_environment import policy_binding

    policy = policy_binding()
    files["environment-policy.json"] = export_json(output / "environment-policy.json", policy)
    files["bootstrap-control.json"] = export_json(
        output / "bootstrap-control.json",
        {
            "schema": "bl-f5-bootstrap-qualification/v1",
            "source_hashes": hashes,
            "environment_policy": policy,
            "qualification_fixture_sha256": files["timing-fixtures.json"]["sha256"],
            "b8_sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
        },
    )
    fixture_input = {
        "path": str(output / "timing-fixtures.json"),
        "dest": "timing-fixtures.json",
        "sha256": files["timing-fixtures.json"]["sha256"],
    }
    for name, value in (
        ("authority.json", authority),
        ("source-review.json", source),
        ("corpus-review.json", corpus),
        ("one-difference.json", one_difference()),
        ("allocation.json", allocation),
        ("verified-ledger.json", ledger),
        ("timing-plan.json", timing),
        ("paired-fit.draft.json", manifest),
    ):
        files[name] = export_json(output / name, value)
    canary = {
        **manifest,
        "job_id": f"bl-f5-w-prepare-{commit[:8]}-01",
        "timeout_minutes": 30,
        "output_dir": str(output / "prepare-output-01"),
        "env": {
            **manifest["env"],
            "F5_MODE": "prepare-qualification",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
        },
        "required_exports": ["prepare/prepare-receipt.json", "prepare/f5-batch-plan.pt"],
        "requirements": {**manifest["requirements"], "est_minutes": 15},
        "resume_exemption": {
            "kind": "short-canary",
            "reason": "sealed exact-input plan/environment qualification; no scientific outcomes",
            "timeout_minutes": 30,
        },
    }
    canary["inputs"] = [*manifest["inputs"], fixture_input]
    canary["required_exports"] += [
        *manifest["required_exports"],
        "qualification/qualification.json",
    ]
    canary["substeps"] = ["CPU_PLAN_MATERIALIZATION", "DISJOINT_CUDA_TRANSPORT_QUALIFICATION"]
    del canary["resume"]
    files["prepare.draft.json"] = export_json(output / "prepare.draft.json", canary)
    capture_timeout_minutes = (
        30 if capture_override is not None else capture_accounting["remaining_capture_seconds"] / 60
    )
    capture_attempt = (
        f"{len(capture_history_value['attempts']) + 1:02d}"
        if capture_history_value is not None
        else ("01" if prior_capture_receipt is None else "02")
    )
    capture = {
        **canary,
        "job_id": (
            f"bl-f5-w-capture-{commit[:8]}-01"
            if capture_attempt == "01"
            else f"bl-f5-w-capture-repair-22e86986-{capture_attempt}"
        ),
        "output_dir": str(output / f"capture-output-{capture_attempt}"),
        "inputs": [
            *[item for item in manifest["inputs"] if item["dest"] == "b8.pth"],
            fixture_input,
            {
                "path": str(output / "bootstrap-control.json"),
                "dest": "bootstrap-control.json",
                "sha256": files["bootstrap-control.json"]["sha256"],
            },
        ],
        "env": {**canary["env"], "F5_MODE": "capture-qualification"},
        "required_exports": [
            "environment.json",
            "environment-raw.json",
            "environment-diagnostic.json",
            "qualification/qualification.json",
            *manifest["required_exports"],
        ],
        "substeps": ["RAW_ENVIRONMENT_CAPTURE", "DISJOINT_CUDA_TRANSPORT_QUALIFICATION"],
        "timeout_minutes": capture_timeout_minutes,
        "capture_accounting": capture_accounting,
        "prior_capture_history": capture_history_value,
        "capture_owner_override": capture_override,
        "requirements": {**canary["requirements"], "est_minutes": 5},
        "resume_exemption": {
            "kind": "short-canary",
            "reason": (
                "Environment capture and disjoint transport only; prior stopped attempts are "
                "disclosed as accounting evidence and do not block owner-approved runaway timeout"
            ),
            "timeout_minutes": capture_timeout_minutes,
        },
    }
    files["environment-capture.draft.json"] = export_json(
        output / "environment-capture.draft.json", capture
    )
    if prepared_input is not None:
        files["training-package.json"] = export_json(
            output / "training-package.json", prepared_input
        )
    qualification = {
        **manifest,
        "job_id": f"bl-f5-w-qualification-{commit[:8]}-01",
        "output_dir": str(output / "qualification-output-01"),
        "timeout_minutes": 30,
        "env": {
            **manifest["env"],
            "F5_MODE": "qualification-transport",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
        },
        "requirements": {**manifest["requirements"], "est_minutes": 15},
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_colab.sh",
        },
        "required_exports": [*manifest["required_exports"], "qualification/qualification.json"],
        "resume_exemption": {
            "kind": "short-canary",
            "timeout_minutes": 30,
            "reason": "twelve sealed timing-only fixtures; no scientific fits or outcomes",
        },
        "inventory_status": (
            "HOLD: bind final full science inventory unchanged before family canary"
        ),
    }
    qualification.pop("resume", None)
    timing_qualification = {
        **qualification,
        "job_id": f"bl-f5-w-timing-{commit[:8]}-01",
        "output_dir": str(output / "timing-output-01"),
        "env": {**qualification["env"], "F5_MODE": "qualification"},
        "inputs": [item for item in manifest["inputs"] if item["dest"] == "b8.pth"]
        + [fixture_input],
        "inventory_status": "HOLD: bind actual timing control; no selected-training upload",
    }
    # These are the immutable source-native evaluator pool, not training or
    # candidate inputs. Full timing uses a separate packet so this pool never
    # expands the paired-fit private-upload ceiling.
    calibration_checkpoints = (
        (
            "champion_a5_freespace_20260621.pth",
            "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
        ),
        ("best_apex_fs.pth", "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195"),
        (
            "best_apex_stage1_fs.pth",
            "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
        ),
        (
            "best_apex_pre_fs.pth",
            "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
        ),
        (
            "frp3_m3_s12_u60000_20261005.pth",
            "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
        ),
    )
    if selected_package is not None:
        for name, expected in calibration_checkpoints:
            path = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes") / name
            if digest(path) != expected:
                raise ValueError("timing evaluator checkpoint bytes differ from source pins")
            timing_qualification["inputs"].append(
                {"path": str(path), "dest": name, "sha256": expected}
            )
        timing_bytes = sum(
            Path(item["path"]).stat().st_size for item in timing_qualification["inputs"]
        )
        if timing_bytes > 45_397_643:
            raise ValueError("timing qualification inventory exceeds unchanged private cap")
        timing_qualification["known_private_input_bytes"] = timing_bytes
    timing_qualification = hold_timing_envelope(timing_qualification)
    files["timing-qualification.draft.json"] = export_json(
        output / "timing-qualification.draft.json", timing_qualification
    )
    files["qualification-canary.draft.json"] = export_json(
        output / "qualification-canary.draft.json", qualification
    )
    control_template = {
        "schema": "bl-residual-f5-runtime/v1",
        "namespace": "bl-residual-f5-w-20261025-v1",
        "identity": {
            "source_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()
            ).hexdigest(),
            "loss_sha256": hashes["research/bl_residual/f5_fit.py"],
            "environment_sha256": (
                "bc05167c188680478663235f2b3dcfb70d" "7959e9d392071384409ef958b0f0f5"
            ),
            "allocation_sha256": ALLOCATION_SHA,
            "corpus_sha256": None if prepared_input is None else prepared_input["corpus_sha256"],
        },
        "source_hashes": hashes,
        "qualification_fixture_sha256": files["timing-fixtures.json"]["sha256"],
        "qualification_b8_sha256": (
            "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
        ),
        "environment_policy": policy,
        "environment_capture": None,
        "input_hashes": (
            {}
            if prepared_input is None
            else {"selected-training.tar.xz": prepared_input["package_sha256"]}
        ),
        "allocation": None,
        "ledger_sha256": None,
        "status": "TEMPLATE_NEEDS_ACTUAL_L4_CAPTURE_THEN_ALLOCATION_AND_LEDGER_SEAL",
    }
    files["control.template.json"] = export_json(output / "control.template.json", control_template)
    main_control_template = {
        **control_template,
        "input_hashes": dict(control_template["input_hashes"]),
    }
    if prepared_input is not None:
        main_control_template["input_hashes"]["b8.pth"] = b8_sha
    files["paired-control.template.json"] = export_json(
        output / "paired-control.template.json", main_control_template
    )
    timing_control = {
        **control_template,
        "input_hashes": {
            "b8.pth": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
        },
        "qualification_checkpoint_pool": dict(calibration_checkpoints),
    }
    files["timing-control.template.json"] = export_json(
        output / "timing-control.template.json", timing_control
    )
    from research.colab_cli import canary as broker_canary

    broker_shapes = {}
    for name, probe in (
        ("capture", capture),
        ("prepare", canary),
        ("final_transport", qualification),
    ):
        broker_canary.check(probe, repo)
        broker_shapes[name] = "PASS_REAL_BROKER_PROBE_SHAPE_ONLY_NOT_EXECUTION"
    broker_shapes["timing"] = timing_qualification["timing_envelope_status"]
    dry_run = {
        "schema": "bl-f5-offline-dry-run/v1",
        "status": "PASS_OFFLINE_PREPARATION",
        "operational_admission": "HOLD",
        "allocations": 0,
        "provider_calls": 0,
        "fits": 0,
        "source_commit": commit,
        "blockers": blocked,
        "dispatch_wrapper": "tools/ops/dispatch.sh; NOT invoked",
        "draft_manifest_refuses_dispatch": True,
        "real_broker_probe_checks": broker_shapes,
        "actual_cuda_observer_evidence": "NOT_RUN_NO_CUDA_NO_DISPATCH",
        "actual_timing_feasibility": "HOLD_PENDING_MEASURED_L4_AND_MAC_RECEIPTS",
        "qualification_envelope": {
            "capture_minutes": 10,
            "transport_only_qualification_minutes": 30,
            "full_timing_minutes": None,
            "paired_fit_minutes": 200,
            "historical_aggregate_minutes": 240,
            "aggregate_minutes": None,
            "capture_accounting": capture_accounting,
            "rule": (
                "all elapsed and accounting evidence is retained; the exact owner ruling "
                "sets the capture runaway timeout and unit history never blocks admission"
                if capture_override is not None
                else (
                    "all attempts/substeps share original cumulative allowance; "
                    "never reset by a fresh job"
                )
            ),
            "final_operational_gate": (
                "bind all actual elapsed receipts and remaining timeout before execution"
            ),
        },
        "selected_training_package": prepared_input,
        "complete_private_fit_inventory": "HOLD: add bound control, plan, acceptance and ledger",
        **INTERPRETATION,
    }
    files["dry-run.json"] = export_json(output / "dry-run.json", dry_run)
    packet = {
        "schema": "bl-f5-w-prefit-review/v1",
        "status": "PREPARED_NOT_ACCEPTED",
        **INTERPRETATION,
        "source_commit": commit,
        "source_hashes": hashes,
        "files": files,
        "next_gate": blocked,
        "actual_batch_plan": None,
        "batch_plan_status": "requires selected training packaging + pinned L4 environment",
        "payload_access_in_this_build": False,
        "training_packaging_predecessor": (
            None
            if prepared_input is None
            else {
                "scope": "selected training packaging only; no aggregate/Q/loss inspection",
                "package_sha256": prepared_input["package_sha256"],
            }
        ),
        "dispatch_or_fit": False,
    }
    return export_json(output / "REVIEW_PACKET.json", packet)


def main() -> None:
    """Expose only offline build/dry-run, with no run/dispatch subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--selected-package", type=Path)
    parser.add_argument("--prior-capture-receipt", type=Path)
    parser.add_argument("--prior-capture-receipts", type=Path, nargs="+")
    parser.add_argument("--capture-owner-ruling", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output,
                selected_package=args.selected_package,
                prior_capture_receipt=args.prior_capture_receipt,
                prior_capture_receipts=args.prior_capture_receipts,
                capture_owner_ruling=args.capture_owner_ruling,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
