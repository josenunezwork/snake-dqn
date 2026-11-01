"""Host-side admission adapter for Watch6 V3 (Astra Ruling Points 2, 3, 4, 6).

Preserves every Watch6 V3 admission audit, 2x envelope, <=12/<=90s upload cap, L4-only W4
placement, canary check, and runtime closure verification while adapting only the pre-start
failure predicate to accept a PROVEN never-allocated attempt per Ruling Point 4.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import admission as v3_admission
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.costs import verify_report_evidence
from research.watch6_search_efficacy_v3.uploads import METADATA, check_uploads, stage_metadata
from tools.ops.package_spec import digest
from tools.watch6_v3_accounting import (
    ADAPTER_ROOT,
    BASELINE_COMMIT,
    BASELINE_SCIENCE,
    verify_baseline_closure,
)
from tools.watch6_v3_accounting.proof import (
    has_forbidden_or_contradictory_output_artifacts,
    is_proven_never_allocated_attempt,
)

EXTERNAL_TRANSPORT_CLASSES = v3_admission.EXTERNAL_TRANSPORT_CLASSES
FORBIDDEN_TIMEOUT_SUBSTRINGS = v3_admission.FORBIDDEN_TIMEOUT_SUBSTRINGS
has_non_transport_fault = v3_admission.has_non_transport_fault
has_verified_transport_traceback = v3_admission.has_verified_transport_traceback
completed_ledger = v3_admission.completed_ledger
verify_runtime = v3_admission.verify_runtime
runtime_admit = v3_admission.runtime_admit


def is_prestart_transport_failure(
    receipt: dict,
    output_dir: Path | None,
    *,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
) -> bool:
    """True iff baseline V3 pre-start proof or Ruling Point 4 never-allocated proof passes."""
    if isinstance(output_dir, Path):
        for fault_file in ("runtime-measurement.json", "scientific-fault.json"):
            if (output_dir / fault_file).exists() or (output_dir / fault_file).is_symlink():
                return False
        if output_dir.is_dir() and (
            any(output_dir.glob("runtime-measurement*"))
            or any(output_dir.glob("scientific-fault*"))
        ):
            return False
        if receipt.get("endpoint") is None and has_forbidden_or_contradictory_output_artifacts(
            output_dir,
            job_id=receipt.get("job_id") if isinstance(receipt.get("job_id"), str) else None,
            session=receipt.get("session") if isinstance(receipt.get("session"), str) else None,
        ):
            return False
    if v3_admission.is_prestart_transport_failure(receipt, output_dir):
        return True
    return is_proven_never_allocated_attempt(
        receipt,
        output_dir,
        ops_dir=ops_dir,
        ops_name=ops_name,
    )


def is_external_transport_compute_failure(
    receipt: dict,
    output_dir: Path | None = None,
) -> bool:
    """Allocated compute failures continue to require verified stop and clean telemetry."""
    if isinstance(output_dir, Path):
        if (output_dir / "scientific-fault.json").exists() or (
            output_dir / "scientific-fault.json"
        ).is_symlink():
            return False
        if output_dir.is_dir() and any(output_dir.glob("scientific-fault*")):
            return False
    return v3_admission.is_external_transport_compute_failure(receipt, output_dir)


def verify_prior_receipts_lineage(
    shard: int,
    prior_receipts: list[dict],
    prestart_used: int,
    compute_attempt: int,
    total_dispatches: int,
    *,
    ops_dir: Path | None = None,
) -> None:
    """Verify declared prior receipts match retry counters and baseline V3 science/commit."""
    if not isinstance(prior_receipts, list):
        raise ValueError("prior_receipts must be a list")
    if total_dispatches != 1 + prestart_used + compute_attempt or len(prior_receipts) != (
        prestart_used + compute_attempt
    ):
        raise ValueError("retry-cap counters inconsistent with prior_receipts length")
    seen_paths: set[Path] = set()
    actual_prestart = 0
    actual_compute = 0
    for entry in prior_receipts:
        if not isinstance(entry, dict):
            raise ValueError("malformed prior_receipts entry")
        r_path = Path(entry.get("receipt_path", ""))
        if not r_path.is_absolute() or r_path.is_symlink() or not r_path.is_file():
            raise ValueError(f"missing prior receipt in lineage: {r_path}")
        resolved = r_path.resolve()
        if resolved in seen_paths or digest(resolved) != entry.get("receipt_sha256"):
            raise ValueError(f"duplicate or hash-mismatched prior receipt: {r_path}")
        seen_paths.add(resolved)
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
        m_path = resolved.parent / "manifest.json"
        if (
            m_path.is_symlink()
            or not m_path.is_file()
            or digest(m_path) != receipt.get("manifest_sha256")
        ):
            raise ValueError(f"prior receipt manifest_sha256 mismatch: {r_path}")
        m_data = json.loads(m_path.read_text(encoding="utf-8"))
        m_bind = m_data.get("watch6_v3", {})
        m_req = m_data.get("requirements", {})
        if (
            m_data.get("commit") != BASELINE_COMMIT
            or receipt.get("commit") != BASELINE_COMMIT
            or m_bind.get("namespace") != p.NAMESPACE
            or m_bind.get("science") != p.identity()
            or m_bind.get("science") != BASELINE_SCIENCE
            or m_bind.get("primary_allocation") != p.allocation()
            or any(k in m_bind for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
            or m_req.get("gpu") != "L4"
            or m_req.get("allowed_gpus") != ["L4"]
            or "prefer" in m_req
            or m_data.get("env", {}).get("WATCH6_FIXTURE") != "0"
            or int(m_data.get("env", {}).get("WATCH6_SHARD", -1)) != shard
        ):
            raise ValueError(
                f"foreign shard/science/commit/placement in prior receipt lineage: {r_path}"
            )
        declared_kind = entry.get("classification")
        if is_prestart_transport_failure(receipt, resolved.parent, ops_dir=ops_dir):
            if declared_kind != "prestart":
                raise ValueError(
                    f"prior receipt classification mismatch (expected prestart): {r_path}"
                )
            actual_prestart += 1
        elif is_external_transport_compute_failure(receipt, resolved.parent):
            if declared_kind != "external_transport_compute":
                raise ValueError(
                    "prior receipt classification mismatch "
                    f"(expected external_transport_compute): {r_path}"
                )
            actual_compute += 1
        else:
            raise ValueError(f"forbidden failure class in prior_receipts lineage: {r_path}")
    if actual_prestart != prestart_used or actual_compute != compute_attempt:
        raise ValueError("prior_receipts classifications do not match retry counters")


def admit(
    manifest: dict,
    *,
    pending_qualification: bool = False,
    ops_dir: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Verify baseline closure equality, controller, V3 allocation, L4 W4, uploads, and 2x caps."""
    root = (repo or ADAPTER_ROOT).resolve()
    verify_baseline_closure(root)

    if manifest.get("commit") != BASELINE_COMMIT:
        got_commit = manifest.get("commit")
        raise ValueError(
            f"Watch6 V3 runtime commit must remain {BASELINE_COMMIT}, got {got_commit!r}"
        )
    if manifest.get("source_paths") != p.source_paths():
        raise ValueError("manifest source_paths differs from frozen protocol.source_paths()")

    binding = manifest["watch6_v3"]
    if manifest.get("pending_qualification") and not pending_qualification:
        raise ValueError("main packet awaits genuine passing same-pin canary; finalize first")
    if pending_qualification and manifest.get("pending_qualification") is not True:
        raise ValueError("pending validation only accepts explicitly pending packets")
    if any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")):
        raise ValueError("reserve seed substitution is forbidden in Watch6 V3")
    if (
        binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("science") != BASELINE_SCIENCE
        or binding.get("primary_allocation") != p.allocation()
    ):
        raise ValueError("foreign V3 source/namespace/allocation")
    if manifest.get("script") != p.SCRIPT or manifest.get("device") != "cpu":
        raise ValueError("wrong V3 entrypoint or device")
    req = manifest.get("requirements", {})
    if (
        req.get("gpu") != "L4"
        or req.get("allowed_gpus") != ["L4"]
        or "prefer" in req
        or req.get("min_vcpu", 0) < 12
    ):
        raise ValueError(
            "only registered >=12 vCPU L4 CPU placement (no A100 substitution) is admitted"
        )
    base_minutes = binding.get("base_minutes")
    if type(base_minutes) is not int or not 0 < base_minutes <= 660:
        raise ValueError("invalid base_minutes envelope")
    if (
        manifest.get("timeout_minutes") != 2 * base_minutes
        or req.get("est_minutes") != base_minutes
    ):
        raise ValueError("requires pre-data 2x wall guard matching est_minutes")
    budget = manifest.get("budget", {})
    rate = budget.get("rate_units_per_hour")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("budget requires positive finite rate_units_per_hour")
    expected_base_units = base_minutes / 60 * float(rate)
    if not math.isclose(
        budget.get("unit_cap", -1.0), 2 * expected_base_units, rel_tol=1e-9
    ) or not math.isclose(
        budget.get("reserve_units", -1.0), 2 * expected_base_units * 0.2, rel_tol=1e-9
    ):
        raise ValueError("requires pre-data 2x pricing envelope")

    probe = manifest.get("canary", {}).get("mode") == "probe"
    prestart_used = binding.get("prestart_retries_used", 0)
    compute_attempt = binding.get("compute_completion_attempt", 0)
    total_dispatches = binding.get("total_dispatches_for_shard", 1)
    top_completion = manifest.get("completion_attempt", 0)
    phase = binding.get("phase")
    expected_phase = "canary" if probe else "primary"
    if (
        phase != expected_phase
        or type(prestart_used) is not int
        or prestart_used < 0
        or type(compute_attempt) is not int
        or not 0 <= compute_attempt <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(top_completion) is not int
        or not 0 <= top_completion <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or compute_attempt + top_completion > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(total_dispatches) is not int
        or total_dispatches != 1 + prestart_used + compute_attempt
    ):
        raise ValueError("retry or phase counters violate pre-registered V3 caps")
    if probe and (
        prestart_used != 0 or compute_attempt != 0 or total_dispatches != 1 or top_completion != 0
    ):
        raise ValueError("canary probe must have zero shard retry counters")

    env = manifest.get("env", {})
    if env.get("WATCH6_FIXTURE") != ("1" if probe else "0"):
        raise ValueError("canary/main fixture isolation mismatch")
    if env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("requires explicit SNAKE_WATCH_SOURCE=live")
    if int(env.get("WATCH6_BASE_MINUTES", -1)) != base_minutes:
        raise ValueError("runtime base_minutes env mismatch")
    if manifest.get("required_exports") != ["efficacy/operational.json", "performance.json"]:
        raise ValueError("required_exports mismatch")
    if manifest.get("resume") != {
        "unit_kind": "paired-session",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume policy mismatch")

    prior_receipts = binding.get("prior_receipts", [])
    if probe:
        if prior_receipts != []:
            raise ValueError("canary probe cannot carry shard prior_receipts")
    else:
        verify_prior_receipts_lineage(
            int(env["WATCH6_SHARD"]),
            prior_receipts,
            prestart_used,
            compute_attempt,
            total_dispatches,
            ops_dir=ops_dir,
        )

    for item in manifest["inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input digest mismatch: {item['dest']}")
    upload_report = check_uploads(manifest)
    meta_item = next(i for i in manifest["inputs"] if i["dest"] == METADATA)
    if meta_item["sha256"] != env.get("WATCH6_METADATA_SHA256"):
        raise ValueError("runtime metadata SHA differs from uploaded manifest input")

    with tempfile.TemporaryDirectory(prefix="watch6-v3-acct-admission-") as temporary:
        stage_root = Path(temporary)
        shutil.copyfile(meta_item["path"], stage_root / METADATA)
        stage_metadata(stage_root, env["WATCH6_METADATA_SHA256"])
        verify_runtime(stage_root, env)

    if not probe and not pending_qualification:
        canary.check(manifest, root)
    if not probe:
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_report_evidence(report)
    return upload_report
