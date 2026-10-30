"""Fail-closed checks used by dispatch dry-run and the real runtime entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.mirror import validate as validate_mirror
from research.watch6_search_efficacy.performance import forecast
from research.watch6_search_efficacy.run import completed
from research.watch6_search_efficacy.uploads import METADATA, check_uploads, stage_metadata
from tools.ops.package_spec import digest, verify_immutable


def admit(manifest: dict, *, pending_qualification: bool = False) -> dict:
    """Check controller, allocation, measured forecast and pre-data 2x envelopes."""
    binding = manifest["watch6_efficacy"]
    if manifest.get("pending_qualification") and not pending_qualification:
        raise ValueError("main packet awaits genuine passing same-pin canary; finalize first")
    if pending_qualification and manifest.get("pending_qualification") is not True:
        raise ValueError("pending validation only accepts explicitly pending packets")
    if binding["science"] != p.identity() or binding["allocation"] != p.allocation():
        raise ValueError("foreign source/allocation")
    if manifest["script"] != p.SCRIPT or manifest["device"] != "cpu":
        raise ValueError("wrong entrypoint/device")
    if (
        manifest["requirements"]["gpu"] != "L4"
        or manifest["requirements"]["allowed_gpus"] != ["L4"]
        or manifest["requirements"]["min_vcpu"] < 12
    ):
        raise ValueError("only registered >=12vCPU L4 CPU placement")
    if manifest["timeout_minutes"] != 2 * binding["base_minutes"]:
        raise ValueError("requires pre-data 2x wall guard")
    if (
        binding["base_minutes"] > 660
        or manifest["requirements"]["est_minutes"] != binding["base_minutes"]
    ):
        raise ValueError("forecast exceeds safe provider session length")
    for key in ("unit_cap", "reserve_units"):
        if manifest["budget"][key] != 2 * binding["base_" + key]:
            raise ValueError("requires pre-data 2x pricing envelope")
    probe = manifest.get("canary", {}).get("mode") == "probe"
    env = manifest["env"]
    if env["WATCH6_FIXTURE"] != ("1" if probe else "0"):
        raise ValueError("canary/main fixture isolation mismatch")
    if int(env["WATCH6_BASE_MINUTES"]) != binding["base_minutes"]:
        raise ValueError("runtime envelope mismatch")
    if manifest["resume"] != {
        "unit_kind": "paired-session",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume policy mismatch")
    # Host dry-run reads precisely the bytes the execute-time guard receives.
    for item in manifest["inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("input digest mismatch")
    upload_report = check_uploads(manifest)
    item = next(i for i in manifest["inputs"] if i["dest"] == METADATA)
    if item["sha256"] != env.get("WATCH6_METADATA_SHA256"):
        raise ValueError("runtime metadata SHA differs from uploaded manifest input")
    # Exercise precisely the remote unpack/admission path, without trusting local
    # unuploaded staging files or mutating the immutable package.
    with tempfile.TemporaryDirectory(prefix="watch6-admission-") as temporary:
        root = Path(temporary)
        shutil.copyfile(item["path"], root / METADATA)
        for cost in manifest["inputs"]:
            if cost["dest"] in {
                "cost-evidence.json",
                "cost-evidence.json.sha256",
                "cost-evidence.json.complete",
            }:
                shutil.copyfile(cost["path"], root / cost["dest"])
        stage_metadata(root, env["WATCH6_METADATA_SHA256"])
        verify_runtime(root, env)
    if not probe and not pending_qualification:
        canary.check(manifest, p.ROOT)
    if not probe:
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        if report.get("schema") == "watch6-empirical-costs/v1":
            from research.watch6_search_efficacy.costs import verify_report_evidence

            verify_report_evidence(report)
        else:
            receipt_path = Path(manifest["canary"]["receipt"]["path"])
            recovered = json.loads((receipt_path.parent / "exports/performance.json").read_text())
            if recovered != report:
                raise ValueError("runtime measurements differ from verified canary export")
    return upload_report


def verify_runtime(root: Path, env: dict) -> None:
    """Execute-time guard also validates measured sizing, worker count and ruling."""
    contract = json.loads((root / "runtime-contract.json").read_text())
    if contract != {"science": p.identity(), "allocation": p.allocation()}:
        raise ValueError("runtime source/allocation mismatch")
    if env.get("WATCH6_SCIENCE") != p.identity():
        raise ValueError("runtime science env mismatch")
    shards = int(env.get("WATCH6_SHARDS", "0"))
    workers = int(env.get("WATCH6_WORKERS", "0"))
    if (
        env.get("WATCH6_FIXTURE") not in ("0", "1")
        or shards not in (3, 6, 12, 24, 48, 96)
        or int(env["WATCH6_SHARD"]) not in range(shards)
        or workers not in p.WORKERS
    ):
        raise ValueError("runtime shard/fixture/workers mismatch")
    completed(root / "verified-ledger.json")
    verify_immutable(root / "mac-mirror.json")
    validate_mirror(json.loads((root / "mac-mirror.json").read_text()))
    approval = json.loads((root / "judge-approval.json").read_text())
    if (
        approval.get("status") != "RATIFIED"
        or approval.get("judge") not in ("Astra", "Fable")
        or approval.get("science") != p.identity()
        or approval.get("registration_sha256") != digest(p.registration())
        or approval.get("ruling_sha256") != digest(Path(__file__).with_name("RULING.md"))
    ):
        raise ValueError("recorded judge ratification absent or different")
    if env["WATCH6_FIXTURE"] == "0":
        if "WATCH6_MEASUREMENTS" not in env:
            raise ValueError("main requires measured canary forecast")
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        if report.get("schema") == "watch6-empirical-costs/v1":
            verify_immutable(root / "cost-evidence.json")
            if json.loads((root / "cost-evidence.json").read_text()) != report:
                raise ValueError("empirical costs differ from uploaded evidence")
        selected = forecast(
            report,
            json.loads(env["WATCH6_FROZEN_LAYOUT"]) if "WATCH6_FROZEN_LAYOUT" in env else None,
        )
        if (
            workers != selected["workers"]
            or shards != selected["shards"]
            or int(env.get("WATCH6_BASE_MINUTES", "0")) != selected["base_minutes_per_shard"]
        ):
            raise ValueError("runtime sizing differs from measured forecast with margin")


def runtime_admit(
    *, workers: int | None = None, shards: int | None = None, shard: int | None = None
) -> None:
    """Refuse direct CLI controls that differ from admitted manifest controls."""
    env = dict(os.environ)
    verify_runtime(Path(env["COLAB_INPUT_DIR"]), env)
    if env["WATCH6_FIXTURE"] == "0" and (os.cpu_count() or 0) < 12:
        raise ValueError("main shards require >=12 vCPU at execution time")
    for value, key in (
        (workers, "WATCH6_WORKERS"),
        (shards, "WATCH6_SHARDS"),
        (shard, "WATCH6_SHARD"),
    ):
        if value is not None and value != int(env[key]):
            raise ValueError("runtime CLI differs from admitted manifest")
