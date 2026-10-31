"""Fail-closed admission checks for Watch6 V3 on Mac host and remote L4 runtime."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy.uploads import METADATA, check_uploads, stage_metadata
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.costs import empirical_forecast, verify_report_evidence
from research.watch6_search_efficacy_v3.mirror import validate as validate_mirror
from tools.ops.package_spec import digest, read_ledger, verify_immutable

EXTERNAL_TRANSPORT_CLASSES = {
    "RPC_EXEC_POLL_TIMEOUT",
    "TRANSPORT_TIMEOUT",
    "RUNTIME_LOST",
}
FORBIDDEN_TIMEOUT_SUBSTRINGS = (
    "WALL_TIMEOUT",
    "WHOLE-SESSION WALL CAP",
    "WALL CAP REACHED",
    "TIMEOUT_MINUTES",
    "EXCEEDED WALL",
    "EXCEEDED TIMEOUT",
    "WATCHDOG",
    "FAULTHANDLER",
)


def is_external_transport_compute_failure(receipt: dict, output_dir: Path | None = None) -> bool:
    """True iff a stopped compute-started attempt failed purely on external transport/VM loss.

    Explicitly excludes CHILD_NONZERO_EXIT, mirror/Q/scientific errors, shard
    wall-clock timeouts, and 900s faulthandler stall watchdog exits so duration-
    or survival-correlated failures can never trigger MCAR reserve substitution.
    """
    if (
        receipt.get("status") == "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        return False
    failure_class = str(receipt.get("failure_class") or "")
    if failure_class == "CHILD_NONZERO_EXIT":
        return False
    terminal = receipt.get("terminal")
    if isinstance(terminal, dict):
        return False
    if output_dir is not None and (output_dir / "terminal.json").exists():
        return False
    error_upper = str(receipt.get("error") or "").upper()
    if any(
        token in failure_class.upper() or token in error_upper
        for token in FORBIDDEN_TIMEOUT_SUBSTRINGS
    ):
        return False
    if failure_class in EXTERNAL_TRANSPORT_CLASSES:
        return True
    if failure_class == "SUPERVISOR":
        return any(
            token in error_upper
            for token in (
                "POLL",
                "RPC",
                "TRANSPORT",
                "SESSION",
                "RUNTIME_LOST",
                "DISCONNECT",
                "EOF",
                "CONNECTION",
                "LEASE",
            )
        )
    return False


def verify_reserve_lineage(reserve_slots: list[int], lineage: dict) -> None:
    """Verify 3 stopped external-transport primary compute attempts per reserve shard."""
    if not reserve_slots:
        if lineage:
            raise ValueError("unexpected reserve_lineage when reserve_slots is empty")
        return
    if len(reserve_slots) > p.MAX_RESERVE_SUBSTITUTIONS:
        raise ValueError(
            f"reserve substitutions {len(reserve_slots)} exceed cap {p.MAX_RESERVE_SUBSTITUTIONS}"
        )
    if sorted(set(reserve_slots)) != reserve_slots or any(
        i not in range(p.COUNT) for i in reserve_slots
    ):
        raise ValueError("invalid or duplicate reserve_slots")
    shards_needed = sorted({i % p.SHARDS for i in reserve_slots})
    shard_entries = lineage.get("shards") if isinstance(lineage, dict) else None
    if not isinstance(shard_entries, dict) or sorted(map(int, shard_entries)) != shards_needed:
        raise ValueError("reserve_lineage missing shard receipt bindings")
    for shard in shards_needed:
        records = shard_entries[str(shard)] if str(shard) in shard_entries else shard_entries[shard]
        if not isinstance(records, list) or len(records) != 1 + p.MAX_COMPUTE_COMPLETIONS_PER_SHARD:
            raise ValueError(
                f"reserve activation on shard {shard} requires exactly 3 compute receipts"
            )
        seen_receipts = set()
        for item in records:
            receipt_path = Path(item["receipt_path"])
            if not receipt_path.is_absolute() or not receipt_path.is_file():
                raise ValueError(f"missing reserve-lineage receipt: {receipt_path}")
            r_sha = digest(receipt_path)
            if r_sha != item.get("receipt_sha256") or r_sha in seen_receipts:
                raise ValueError("duplicate or mismatched reserve-lineage receipt SHA256")
            seen_receipts.add(r_sha)
            receipt = json.loads(receipt_path.read_text())
            manifest_path = receipt_path.parent / "manifest.json"
            if not manifest_path.is_file() or digest(manifest_path) != receipt.get(
                "manifest_sha256"
            ):
                raise ValueError("reserve-lineage receipt manifest binding mismatch")
            manifest = json.loads(manifest_path.read_text())
            binding = manifest.get("watch6_v3", {})
            if (
                binding.get("science") != p.identity()
                or int(manifest.get("env", {}).get("WATCH6_SHARD", -1)) != shard
                or manifest.get("env", {}).get("WATCH6_FIXTURE") != "0"
                or binding.get("phase", "primary") != "primary"
            ):
                raise ValueError(
                    "reserve-lineage receipt is not a primary main attempt of this shard"
                )
            if not is_external_transport_compute_failure(receipt, receipt_path.parent):
                raise ValueError(
                    f"shard {shard} receipt {receipt_path} is not an external-transport failure"
                )
            for slot in reserve_slots:
                if slot % p.SHARDS == shard:
                    exported_names = {row.get("path") for row in receipt.get("exports", [])}
                    if (
                        f"efficacy/pair-{slot:02d}.json" in exported_names
                        or (receipt_path.parent / f"exports/efficacy/pair-{slot:02d}.json").exists()
                    ):
                        raise ValueError(
                            f"slot {slot} already exported; reserve substitution forbidden"
                        )


def completed_ledger(path: Path) -> list[dict]:
    """Read only identity/size/hash metadata from the Mac-verified V3 ledger."""
    rows = read_ledger(path, p.NAMESPACE, "watch6", {"science": p.identity()})
    if any(row["index"] not in range(p.COUNT) for row in rows):
        raise ValueError("foreign V3 allocation index in verified ledger")
    return rows


def admit(manifest: dict, *, pending_qualification: bool = False) -> dict:
    """Check controller, V3 allocations, 24x4 W4 empirical forecast, and 2x envelopes."""
    binding = manifest["watch6_v3"]
    if manifest.get("pending_qualification") and not pending_qualification:
        raise ValueError("main packet awaits genuine passing same-pin canary; finalize first")
    if pending_qualification and manifest.get("pending_qualification") is not True:
        raise ValueError("pending validation only accepts explicitly pending packets")
    if (
        binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
        or binding.get("reserve_allocation") != p.reserve_allocation()
    ):
        raise ValueError("foreign V3 source/namespace/allocation")
    if manifest.get("script") != p.SCRIPT or manifest.get("device") != "cpu":
        raise ValueError("wrong V3 entrypoint or device")
    req = manifest.get("requirements", {})
    if req.get("gpu") != "L4" or req.get("allowed_gpus") != ["L4"] or req.get("min_vcpu", 0) < 12:
        raise ValueError("only registered >=12 vCPU L4 CPU placement is admitted")
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

    prestart_used = binding.get("prestart_retries_used", 0)
    compute_attempt = binding.get("compute_completion_attempt", 0)
    total_dispatches = binding.get("total_dispatches_for_shard", 1)
    phase = binding.get("phase", "primary")
    if (
        type(prestart_used) is not int
        or not 0 <= prestart_used <= p.MAX_PRESTART_RETRIES_PER_SHARD
        or type(compute_attempt) is not int
        or not 0 <= compute_attempt <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(total_dispatches) is not int
        or not 1 <= total_dispatches <= p.MAX_TOTAL_DISPATCHES_PER_SHARD
        or phase not in ("canary", "primary", "reserve")
    ):
        raise ValueError("retry or phase counters exceed pre-registered V3 caps")
    if phase == "reserve" and compute_attempt != 0:
        raise ValueError(
            "reserve seed allows at most 1 compute-started attempt (compute_completion_attempt=0)"
        )

    probe = manifest.get("canary", {}).get("mode") == "probe"
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

    for item in manifest["inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input digest mismatch: {item['dest']}")
    upload_report = check_uploads(manifest)
    meta_item = next(i for i in manifest["inputs"] if i["dest"] == METADATA)
    if meta_item["sha256"] != env.get("WATCH6_METADATA_SHA256"):
        raise ValueError("runtime metadata SHA differs from uploaded manifest input")

    reserve_slots = binding.get("reserve_slots", [])
    verify_reserve_lineage(reserve_slots, binding.get("reserve_lineage", {}))

    with tempfile.TemporaryDirectory(prefix="watch6-v3-admission-") as temporary:
        root = Path(temporary)
        shutil.copyfile(meta_item["path"], root / METADATA)
        for cost in manifest["inputs"]:
            if cost["dest"] in {
                "cost-evidence.json",
                "cost-evidence.json.sha256",
                "cost-evidence.json.complete",
            }:
                shutil.copyfile(cost["path"], root / cost["dest"])
        stage_metadata(root, env["WATCH6_METADATA_SHA256"])
        contract = verify_runtime(root, env)
        if contract["reserve_slots"] != reserve_slots:
            raise ValueError("manifest reserve_slots differs from uploaded runtime-contract.json")
        if phase == "reserve" and not reserve_slots:
            raise ValueError("reserve phase requires nonempty reserve_slots")
        if phase == "primary" and not probe:
            shard = int(env["WATCH6_SHARD"])
            if any(slot % p.SHARDS == shard for slot in reserve_slots):
                raise ValueError("primary phase cannot run a shard with active reserve_slots")

    if not probe and not pending_qualification:
        canary.check(manifest, p.ROOT)
    if not probe:
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_report_evidence(report)
    return upload_report


def verify_runtime(root: Path, env: dict) -> dict:
    """Execute-time guard validating contract, registration hash, mirror, and 24x4 W4 sizing."""
    contract = json.loads((root / "runtime-contract.json").read_text())
    reserve_slots = contract.get("reserve_slots", [])
    if (
        not isinstance(reserve_slots, list)
        or sorted(set(reserve_slots)) != reserve_slots
        or any(type(i) is not int or i not in range(p.COUNT) for i in reserve_slots)
        or len(reserve_slots) > p.MAX_RESERVE_SUBSTITUTIONS
    ):
        raise ValueError("invalid reserve_slots in runtime-contract.json")
    expected_contract = {
        "schema": "watch6-v3-runtime-contract/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "primary_allocation": p.allocation(),
        "reserve_allocation": p.reserve_allocation(),
        "reserve_slots": reserve_slots,
    }
    if contract != expected_contract:
        raise ValueError("runtime contract source/allocation mismatch")
    if env.get("WATCH6_SCIENCE") != p.identity() or env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("runtime science or SNAKE_WATCH_SOURCE env mismatch")

    shards = int(env.get("WATCH6_SHARDS", "0"))
    workers = int(env.get("WATCH6_WORKERS", "0"))
    if (
        env.get("WATCH6_FIXTURE") not in ("0", "1")
        or shards != p.SHARDS
        or int(env.get("WATCH6_SHARD", "-1")) not in range(shards)
        or workers != p.WORKERS
    ):
        raise ValueError("runtime shard/fixture/workers mismatch (requires 24 shards, W4)")

    verified_rows = completed_ledger(root / "verified-ledger.json")
    verified_indices = {row["index"] for row in verified_rows}
    if verified_indices & set(reserve_slots):
        raise ValueError("verified primary slot cannot also appear in reserve_slots")

    verify_immutable(root / "mac-mirror.json")
    validate_mirror(json.loads((root / "mac-mirror.json").read_text()))

    protocol_meta = json.loads((root / "judge-approval.json").read_text())
    if protocol_meta != {
        "schema": "watch6-v3-protocol-binding/v1",
        "implementer": "Argon",
        "judge": "Astra",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "registration_sha256": digest(p.registration()),
    }:
        raise ValueError("uploaded protocol/registration binding mismatch")

    if env["WATCH6_FIXTURE"] == "0":
        if "WATCH6_MEASUREMENTS" not in env:
            raise ValueError("main shard requires empirical cost measurements")
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_immutable(root / "cost-evidence.json")
        if json.loads((root / "cost-evidence.json").read_text()) != report:
            raise ValueError("WATCH6_MEASUREMENTS differs from uploaded cost-evidence.json")
        selected = empirical_forecast(
            report,
            json.loads(env["WATCH6_FROZEN_LAYOUT"]) if "WATCH6_FROZEN_LAYOUT" in env else None,
        )
        if (
            workers != selected["workers"]
            or shards != selected["shards"]
            or int(env.get("WATCH6_BASE_MINUTES", "0")) != selected["base_minutes_per_shard"]
        ):
            raise ValueError("runtime sizing differs from empirical 24x4 W4 forecast")
    return contract


def runtime_admit(
    *, workers: int | None = None, shards: int | None = None, shard: int | None = None
) -> dict:
    """Refuse direct CLI controls that differ from admitted manifest controls."""
    env = dict(os.environ)
    contract = verify_runtime(Path(env["COLAB_INPUT_DIR"]), env)
    if env["WATCH6_FIXTURE"] == "0" and (os.cpu_count() or 0) < 12:
        raise ValueError("main shards require >=12 vCPU at execution time")
    for value, key in (
        (workers, "WATCH6_WORKERS"),
        (shards, "WATCH6_SHARDS"),
        (shard, "WATCH6_SHARD"),
    ):
        if value is not None and value != int(env[key]):
            raise ValueError("runtime CLI differs from admitted manifest")
    return contract
