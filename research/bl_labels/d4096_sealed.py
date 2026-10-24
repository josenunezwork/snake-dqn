"""Metadata-only seals for discardable, outcome-blind D4096 collection.

This module intentionally accepts only receipts and terminal accounting.  It
does not open a D4096 label, life export, ledger, or result payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .d4096_design import collection_disclosure
from .exports import _verify, publish_json

JUDGMENT_SHA256 = "ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb"
READINESS_SCHEMA = "bl-labels-d4096-sealed-readiness/v1"
PROJECTION_SCHEMA = "bl-labels-d4096-mac-h-projection/v1"
DISCARD_SCHEMA = "d4096-not-run-by-rule/v1"
FIXED1024_DENOMINATOR = 1024
TARGET_SELECTED_COUNT = 3072
MAC_H_MULTIPLIER = 1.2
MAC_H_CAP = 130.0
_DISCARD_LABELS = frozenset({"KILL_FIT", "INSUFFICIENT", "STOP_FUNDING", "INVALID", "HOLD"})
CANONICAL_PACK = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/"
    "d4096-old-equiv-02-pack-01"
)


def _present(path: Path) -> bool:
    """Treat dangling symlinks as present evidence, never as permission to create."""
    return path.exists() or path.is_symlink()


def _fresh_receipt(path: Path) -> None:
    if any(_present(Path(str(path) + suffix)) for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError("sealed D4096 receipt is create-only")


def _metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read an immutable JSON receipt and return its path/hash binding."""
    sha = _verify(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("sealed metadata is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError("sealed metadata must be an object")
    return value, {"path": str(path.resolve()), "sha256": sha}


def _bound_metadata(binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("sealed metadata binding is malformed")
    path, sha = binding["path"], binding["sha256"]
    if not isinstance(path, str) or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("sealed metadata binding is malformed")
    value, actual = _metadata(Path(path))
    if actual["sha256"] != sha:
        raise ValueError("sealed metadata binding hash differs")
    return value


def _terminal_wall(binding: Any, stage: str) -> float:
    value = _bound_metadata(binding)
    elapsed = value.get("cumulative_elapsed_seconds")
    if (
        value.get("schema") != "bl-labels-f0-mac-driver/v1"
        or value.get("status") != "COMPLETE"
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not isinstance(value.get("namespace"), str)
        or not value["namespace"].endswith(f"-mac-{stage.lower()}")
    ):
        raise ValueError(f"{stage} terminal metadata is not a clean completed wall receipt")
    if stage == "D1024":
        from .frozen_factory import OLD_PIN

        if value.get("workers") != 6 or value.get("collector_source_commit") != OLD_PIN:
            raise ValueError("D1024 terminal is not the admitted OLD W6 collector")
    return float(elapsed)


def _supervisor_binding(path: Path, job_id: str) -> dict[str, Any]:
    """Hash plain supervisor terminal metadata and its bound manifest."""
    raw, manifest_raw = path.read_bytes(), (path.parent / "manifest.json").read_bytes()
    try:
        receipt, manifest = json.loads(raw), json.loads(manifest_raw)
    except json.JSONDecodeError as error:
        raise ValueError("D1024 supervisor metadata is not JSON") from error
    receipt_sha, manifest_sha = (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(manifest_raw).hexdigest(),
    )
    if not isinstance(receipt, dict) or not isinstance(manifest, dict):
        raise ValueError("D1024 supervisor metadata must be objects")
    state = (
        "TERMINAL"
        if receipt.get("status") == "PASS"
        else "HOLD" if receipt.get("status") == "INCOMPLETE" else None
    )
    never_allocated = receipt.get("status") == "INCOMPLETE" and all(
        key in receipt and receipt[key] is None
        for key in ("generation", "session", "endpoint", "terminal")
    )
    mode = "canary" if job_id.startswith("f4-d1024-canary-") else "main"
    if (
        not job_id.startswith("f4-d1024-")
        or state is None
        or receipt.get("job_id") != job_id
        or manifest.get("job_id") != job_id
        or receipt.get("manifest_sha256") != manifest_sha
        or (receipt.get("stop_verified") is not True and not never_allocated)
        or manifest.get("script") != "research/bl_residual/f4_d256_colab.sh"
        or manifest.get("env", {}).get("F4_DOSE") != "1024"
        or manifest.get("env", {}).get("F4_MODE") != mode
        or receipt.get("commit") != manifest.get("commit")
        or not isinstance(manifest.get("commit"), str)
        or len(manifest["commit"]) != 40
    ):
        raise ValueError("D1024 fit receipt is not a sealed terminal/HOLD supervisor receipt")
    return {
        "job_id": job_id,
        "state": state,
        "receipt": {"path": str(path.resolve()), "sha256": receipt_sha},
        "manifest": {
            "path": str((path.parent / "manifest.json").resolve()),
            "sha256": manifest_sha,
        },
    }


def _validate_w6_throughput(path: Path) -> dict[str, Any]:
    from .w12_runtime import verify_d1024_old_w6_throughput

    value = verify_d1024_old_w6_throughput(path)
    if (
        value.get("schema") != "bl-labels-f0-d1024-old-w6-throughput/v1"
        or value.get("stage") != "d1024"
        or value.get("workers") != 6
        or not isinstance(value.get("source_commit"), str)
        or len(value["source_commit"]) != 40
    ):
        raise ValueError("D1024 W6 throughput metadata identity differs")
    _terminal_wall(value.get("d1024_terminal"), "D1024")
    streams = value.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("D1024 W6 throughput stream inventory is missing")
    for stream in streams:
        # Reuse the producer's validation without writing another receipt.
        if not isinstance(stream, str) or not stream:
            raise ValueError("D1024 W6 throughput stream name is malformed")
    return value


def seal_readiness(
    output: Path,
    d1024_terminal: Path,
    d1024_fit_receipts: dict[str, Path],
    d1024_w6_metadata: Path,
    *,
    factory_exited: bool,
    mac_lease_released: bool,
    all_fit_jobs_listed: bool,
) -> dict[str, Any]:
    """Create the B1 metadata attestation after D1024 has stopped.

    The terminal and every fit state are rehashed and copied as path/hash
    bindings.  The caller attests only factory exit and lease release, facts
    unavailable in the driver's terminal JSON after the lease context exits.
    """
    _fresh_receipt(output)
    if not (factory_exited is True and mac_lease_released is True and all_fit_jobs_listed is True):
        raise ValueError(
            "B1 needs explicit factory-exited, lease-released, and complete-job attestations"
        )
    terminal, terminal_binding = _metadata(d1024_terminal)
    if (
        terminal.get("schema") != "bl-labels-f0-mac-driver/v1"
        or terminal.get("status") != "COMPLETE"
    ):
        raise ValueError("D1024 factory is not terminal COMPLETE")
    if not isinstance(d1024_fit_receipts, dict) or not d1024_fit_receipts:
        raise ValueError("B1 needs every D1024 fit terminal/HOLD receipt")
    jobs = []
    for job_id, path in sorted(d1024_fit_receipts.items()):
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("D1024 fit-job inventory is malformed")
        jobs.append(_supervisor_binding(path, job_id))
    throughput = _validate_w6_throughput(d1024_w6_metadata)
    if throughput["d1024_terminal"] != terminal_binding:
        raise ValueError("D1024 W6 throughput binds another terminal")
    result = {
        "schema": READINESS_SCHEMA,
        **collection_disclosure(actual_execution_placement={"d4096": "NOT_STARTED"}),
        "judgment_sha256": JUDGMENT_SHA256,
        "d1024_terminal": terminal_binding,
        "factory_exited": factory_exited,
        "mac_lease_released": mac_lease_released,
        "all_fit_jobs_listed": all_fit_jobs_listed,
        "d1024_fit_jobs": jobs,
        "d1024_w6_metadata": {
            "path": str(d1024_w6_metadata.resolve()),
            "sha256": _verify(d1024_w6_metadata),
        },
    }
    publish_json(output, result)
    return result


def validate_readiness(path: Path) -> dict[str, Any]:
    """Fail closed unless B1's exact metadata bindings still verify."""
    value, _ = _metadata(path)
    if (
        value.get("schema") != READINESS_SCHEMA
        or value.get("judgment_sha256") != JUDGMENT_SHA256
        or value.get("factory_exited") is not True
        or value.get("mac_lease_released") is not True
        or value.get("all_fit_jobs_listed") is not True
    ):
        raise ValueError("D4096 sealed readiness identity differs")
    terminal = _bound_metadata(value.get("d1024_terminal"))
    if (
        terminal.get("schema") != "bl-labels-f0-mac-driver/v1"
        or terminal.get("status") != "COMPLETE"
    ):
        raise ValueError("D1024 terminal metadata is not COMPLETE")
    jobs = value.get("d1024_fit_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("D4096 sealed readiness has no D1024 fit jobs")
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"job_id", "state", "receipt", "manifest"}:
            raise ValueError("D4096 fit-job binding is malformed")
        receipt = job.get("receipt")
        if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
            raise ValueError("D1024 fit-job state binding differs")
        if _supervisor_binding(Path(receipt["path"]), job.get("job_id", "")) != job:
            raise ValueError("D1024 fit-job state binding differs")
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("B1 refuses duplicate D1024 fit-job identities")
    w6_binding = value.get("d1024_w6_metadata")
    if not isinstance(w6_binding, dict) or set(w6_binding) != {"path", "sha256"}:
        raise ValueError("D1024 W6 throughput binding is malformed")
    _bound_metadata(w6_binding)
    throughput = _validate_w6_throughput(Path(w6_binding["path"]))
    if throughput["d1024_terminal"] != value["d1024_terminal"]:
        raise ValueError("D1024 W6 throughput binds another terminal")
    return value


def seal_mac_h_projection(
    output: Path, readiness: Path, d256_terminal: Path, d1024_terminal: Path
) -> dict[str, Any]:
    """Create B2 from completed terminal metadata, using its frozen denominator."""
    _fresh_receipt(output)
    validate_readiness(readiness)
    _d256, d256_binding = _metadata(d256_terminal)
    _d1024, d1024_binding = _metadata(d1024_terminal)
    d256_seconds = _terminal_wall(d256_binding, "D256")
    d1024_seconds = _terminal_wall(d1024_binding, "D1024")
    readiness_value = validate_readiness(readiness)
    if d1024_binding != readiness_value["d1024_terminal"]:
        raise ValueError("B2 D1024 terminal differs from the B1 readiness terminal")
    projected = (
        (d256_seconds + d1024_seconds)
        / 3600
        / FIXED1024_DENOMINATOR
        * TARGET_SELECTED_COUNT
        * MAC_H_MULTIPLIER
    )
    if projected > MAC_H_CAP:
        raise ValueError("HOLD: D4096 Mac-h projection exceeds 130 h")
    result = {
        "schema": PROJECTION_SCHEMA,
        **collection_disclosure(
            actual_execution_placement={"d4096": "NOT_STARTED", "wall_time_inputs": "Mac"}
        ),
        "judgment_sha256": JUDGMENT_SHA256,
        "readiness": {"path": str(readiness.resolve()), "sha256": _verify(readiness)},
        "d256_terminal": d256_binding,
        "d1024_terminal": d1024_binding,
        "fixed1024_denominator": FIXED1024_DENOMINATOR,
        "selected_count": TARGET_SELECTED_COUNT,
        "multiplier": MAC_H_MULTIPLIER,
        "d256_cumulative_wall_seconds": d256_seconds,
        "d1024_cumulative_wall_seconds": d1024_seconds,
        "projected_mac_hours": projected,
        "verdict": "PASS",
    }
    publish_json(output, result)
    return result


def validate_projection(path: Path, readiness: Path | None = None) -> dict[str, Any]:
    """Verify B2 and recompute the projection only from its bound terminals."""
    value, _ = _metadata(path)
    if value.get("schema") != PROJECTION_SCHEMA or value.get("judgment_sha256") != JUDGMENT_SHA256:
        raise ValueError("D4096 Mac-h projection identity differs")
    readiness_binding = value.get("readiness")
    ready = _bound_metadata(readiness_binding)
    if readiness is not None and (
        readiness_binding.get("path") != str(readiness.resolve())
        or readiness_binding.get("sha256") != _verify(readiness)
    ):
        raise ValueError("D4096 Mac-h projection binds another readiness receipt")
    if ready.get("schema") != READINESS_SCHEMA:
        raise ValueError("D4096 Mac-h projection readiness is foreign")
    validate_readiness(Path(readiness_binding["path"]))
    d256 = _terminal_wall(value.get("d256_terminal"), "D256")
    d1024 = _terminal_wall(value.get("d1024_terminal"), "D1024")
    if value.get("d1024_terminal") != ready.get("d1024_terminal"):
        raise ValueError("B2 D1024 terminal differs from the B1 readiness terminal")
    expected = (
        (d256 + d1024) / 3600 / FIXED1024_DENOMINATOR * TARGET_SELECTED_COUNT * MAC_H_MULTIPLIER
    )
    if (
        value.get("fixed1024_denominator") != FIXED1024_DENOMINATOR
        or value.get("selected_count") != TARGET_SELECTED_COUNT
        or value.get("multiplier") != MAC_H_MULTIPLIER
        or value.get("d256_cumulative_wall_seconds") != d256
        or value.get("d1024_cumulative_wall_seconds") != d1024
        or value.get("verdict") != "PASS"
        or value.get("projected_mac_hours") != expected
        or expected > MAC_H_CAP
    ):
        raise ValueError("D4096 Mac-h projection does not satisfy the frozen rule")
    return value


def _stop_label(label: object, stage: object, unrecoverable: object) -> str:
    allowed = {
        "d1024-fit": {"KILL_FIT", "INSUFFICIENT"},
        "r-a": {"STOP_FUNDING", "INVALID", "HOLD"},
    }
    if stage not in allowed or label not in allowed[stage]:
        raise ValueError("discard seal needs a terminal D1024/R-A stop label or receipt")
    if label == "HOLD" and unrecoverable is not True:
        raise ValueError("R-A HOLD must be explicitly unrecoverable")
    return str(label)


def _receipt_stop(value: dict[str, Any], unrecoverable: object) -> tuple[str, str]:
    """Infer stage from the receipt schema and apply the explicit-route rule."""
    schema = value.get("schema")
    if schema == "bl-residual-fit-fault/v1":
        from research.bl_residual.protocol import NAMESPACE

        label = (
            value.get("kind")
            if value.get("dose") == 1024 and value.get("namespace") == NAMESPACE
            else None
        )
        stage = "d1024-fit"
    elif schema in {
        "bl-residual-funding-look/v1",
        "bl-labels-d4096-unrecoverable-ra-hold/v1",
    }:
        label, stage = value.get("verdict"), "r-a"
    else:
        label, stage = None, None
    return _stop_label(label, stage, unrecoverable), stage


def seal_not_run_by_rule(
    output: Path,
    readiness: Path,
    terminal: Path | None = None,
    *,
    terminal_label: str | None = None,
    terminal_stage: str | None = None,
    unrecoverable: bool = False,
) -> dict[str, Any]:
    """Write B6's irreversible discard seal from a terminal label/receipt."""
    if output.resolve() != (CANONICAL_PACK / "d4096-not-run-by-rule.json").resolve():
        raise ValueError("discard seal must use the canonical D4096 pack path")
    _fresh_receipt(output)
    validate_readiness(readiness)
    if (terminal is None) == (terminal_label is None):
        raise ValueError("bind exactly one terminal receipt or explicit terminal label")
    terminal_binding = None
    if terminal is None:
        label = _stop_label(terminal_label, terminal_stage, unrecoverable)
    else:
        value, terminal_binding = _metadata(terminal)
        label, inferred_stage = _receipt_stop(value, unrecoverable)
        if terminal_stage is not None and terminal_stage != inferred_stage:
            raise ValueError("discard terminal stage differs from its receipt")
        terminal_stage = inferred_stage
    result = {
        "schema": DISCARD_SCHEMA,
        **collection_disclosure(
            actual_execution_placement="See bound collection attempts; discarded by rule"
        ),
        "judgment_sha256": JUDGMENT_SHA256,
        "readiness": {"path": str(readiness.resolve()), "sha256": _verify(readiness)},
        "terminal": terminal_binding,
        "terminal_label": label,
        "terminal_stage": terminal_stage,
        "unrecoverable": unrecoverable,
    }
    publish_json(output, result)
    return result


def require_not_discarded(path: Path | None) -> None:
    """Raise if a valid B6 seal exists; malformed candidate seals also HOLD."""
    canonical = CANONICAL_PACK / "d4096-not-run-by-rule.json"
    if path is None or path != canonical:
        if any(_present(Path(str(canonical) + suffix)) for suffix in ("", ".sha256", ".complete")):
            require_not_discarded(canonical)
        if path is None:
            return
    companions = [
        path,
        path.with_name(path.name + ".sha256"),
        path.with_name(path.name + ".complete"),
    ]
    if not _present(path):
        if any(_present(item) for item in companions[1:]):
            raise ValueError("D4096 discard seal is partial; HOLD")
        return
    if any(item.is_symlink() for item in companions):
        raise ValueError("D4096 discard seal contains a symlink; HOLD")
    value, _ = _metadata(path)
    if value.get("schema") != DISCARD_SCHEMA or value.get("judgment_sha256") != JUDGMENT_SHA256:
        raise ValueError("D4096 discard seal is malformed; HOLD")
    validate_readiness(Path(value["readiness"]["path"]))
    if value.get("terminal") is None:
        _stop_label(
            value.get("terminal_label"), value.get("terminal_stage"), value.get("unrecoverable")
        )
        raise ValueError("NOT_RUN_BY_RULE: D4096 has a sealed discard receipt")
    terminal = _bound_metadata(value.get("terminal"))
    label, stage = _receipt_stop(terminal, value.get("unrecoverable"))
    if value.get("terminal_label") != label or value.get("terminal_stage") != stage:
        raise ValueError("D4096 discard seal is malformed; HOLD")
    raise ValueError("NOT_RUN_BY_RULE: D4096 has a sealed discard receipt")


def require_funding_receipt(path: Path | None) -> dict[str, Any]:
    """Require the later independent PASS before fitting or publishing D4096."""
    if path is None:
        raise ValueError("STOP_FUNDING: D4096 requires a PASS funding receipt")
    value, _ = _metadata(path)
    from dataclasses import asdict

    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import F4_REGISTRATION_SHA, NAMESPACE, canonical_sha

    inputs = value.get("inputs")
    digests = (
        value.get("reserve_allocation_sha256"),
        value.get("d256_fit_sha256"),
        value.get("d1024_fit_sha256"),
    )
    valid_digest = (
        lambda item: isinstance(item, str)
        and len(item) == 64
        and all(char in "0123456789abcdef" for char in item)
    )
    if (
        value.get("schema") != "bl-residual-funding-look/v1"
        or value.get("verdict") != "PASS"
        or value.get("series_id") != NAMESPACE
        or value.get("registration_sha256") != F4_REGISTRATION_SHA
        or value.get("recipe_sha256") != canonical_sha(asdict(FitRecipe()))
        or value.get("doses") != [256, 1024]
        or not all(valid_digest(item) for item in digests)
        or not isinstance(inputs, dict)
        or inputs.get("d256_checkpoint_sha256") != value.get("d256_fit_sha256")
        or inputs.get("d1024_checkpoint_sha256") != value.get("d1024_fit_sha256")
    ):
        raise ValueError("STOP_FUNDING: D4096 requires a PASS funding receipt")
    return value


def seal_i3_mac_projection(
    output: Path, mac_projection: Path, d256_collection: Path, d1024_collection: Path
) -> dict[str, Any]:
    """Bind B2 to I3's strict ledger-based interface without changing either rule."""
    from research.bl_residual.collection_receipts import (
        COLLECTION_SCHEMA,
        MAC_PROJECTION_SCHEMA,
        validate_mac_projection,
    )

    _fresh_receipt(output)
    original = validate_projection(mac_projection)
    collections = [_metadata(path)[0] for path in (d256_collection, d1024_collection)]
    walls = {
        "256": original["d256_cumulative_wall_seconds"],
        "1024": original["d1024_cumulative_wall_seconds"],
    }
    for value, dose in zip(collections, (256, 1024)):
        if (
            value.get("schema") != COLLECTION_SCHEMA
            or value.get("dose") != dose
            or value.get("cumulative_wall_seconds") != walls[str(dose)]
            or value.get("placement") != "mac"
            or value.get("charged_units") != 0
        ):
            raise ValueError("I3 collection wall metadata differs from the pre-dispatch B2 seal")
    result = {
        "schema": MAC_PROJECTION_SCHEMA,
        **collection_disclosure(actual_execution_placement={"wall_time_inputs": "Mac"}),
        "judgment_sha256": JUDGMENT_SHA256,
        "placement": "mac",
        "verdict": "PASS",
        "selected_lives": 1024,
        "increment_lives": 3072,
        "margin": 1.2,
        "wall_cap_hours": 130,
        "collection_ledger_sha256": {
            str(c["dose"]): c["collection_ledger_sha256"] for c in collections
        },
        "cumulative_wall_seconds": walls,
        "projected_mac_hours": original["projected_mac_hours"],
        "pre_dispatch_b2": {
            "path": str(mac_projection.resolve()),
            "sha256": _verify(mac_projection),
        },
        "collection_receipts": [_metadata(path)[1] for path in (d256_collection, d1024_collection)],
    }
    validate_mac_projection(result, collections)
    publish_json(output, result)
    return result


def _release_l4_accounting(
    ledger: Path | None, inventory: Path | None, *, series_id: str, no_l4_attempts: bool
) -> dict[str, Any]:
    """Compare charges with a separately sealed, operator-complete attempt inventory.

    The inventory lists every supervisor receipt, including setup/identity/failure
    attempts, whether or not any labels were imported. Completeness of that list
    remains an explicit operator attestation, as with B1's fit-job inventory.
    """
    from research.bl_residual.collection_receipts import l4_accounting
    from research.bl_residual.protocol import F4_REGISTRATION_SHA

    if ledger is None and inventory is None and no_l4_attempts is True:
        return {"status": "ABSENT_NO_L4_ATTEMPTS_ATTESTED", "ledger": None, "inventory": None}
    if ledger is None or inventory is None or no_l4_attempts:
        raise ValueError(
            "release needs L4 ledger and complete attempt inventory, or no-L4 attestation"
        )
    charges, ledger_binding = _metadata(ledger)
    attempts, inventory_binding = _metadata(inventory)
    accounting = l4_accounting(charges, series_id)
    if (
        attempts.get("schema") != "bl-labels-d4096-l4-attempt-inventory/v1"
        or attempts.get("series_id") != series_id
        or attempts.get("registration_sha256") != F4_REGISTRATION_SHA
        or attempts.get("dose") != 4096
        or attempts.get("all_attempts_listed") is not True
        or not isinstance(attempts.get("attempts"), list)
    ):
        raise ValueError("release requires a complete D4096 L4 attempt inventory")
    digests = []
    for row in attempts["attempts"]:
        binding = row.get("receipt") if isinstance(row, dict) else None
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("L4 attempt receipt binding is malformed")
        path = Path(binding["path"])
        raw = path.read_bytes()
        receipt = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if not path.is_absolute() or digest != binding["sha256"] or not isinstance(receipt, dict):
            raise ValueError("L4 attempt receipt binding differs")
        digests.append(digest)
    if len(digests) != len(set(digests)) or set(digests) != {
        row["receipt_sha256"] for row in accounting["attempts"]
    }:
        raise ValueError("L4 accounting omits or duplicates an inventoried attempt")
    return {
        "status": "BOUND_COMPLETE_INVENTORY",
        "ledger": ledger_binding,
        "inventory": inventory_binding,
        "attempt_count": len(digests),
        "charged_units": accounting["charged_units"],
    }


def seal_collection_release(
    output: Path,
    ledger: Path,
    funding_receipt: Path,
    *,
    pack: Path,
    code_commit: str,
    l4_ledger: Path | None = None,
    l4_attempt_inventory: Path | None = None,
    no_l4_attempts: bool = False,
) -> dict[str, Any]:
    """Release exactly one complete funded union to I3, after canonical B6 checks."""
    from . import d4096_final, d4096_resume, mac_driver

    _fresh_receipt(output)
    require_not_discarded(pack / "d4096-not-run-by-rule.json")
    funding = require_funding_receipt(funding_receipt)
    commit = d4096_final.require_pinned_source(code_commit)
    l4_binding = _release_l4_accounting(
        l4_ledger,
        l4_attempt_inventory,
        series_id=funding["series_id"],
        no_l4_attempts=no_l4_attempts,
    )
    control = d4096_resume.original_control(pack)
    attempts, elapsed = d4096_resume.attempt_chain(pack, control, final=True)
    value, binding = _metadata(ledger)
    if value.get("attempt_exports") is not None:
        d4096_resume.verify_union(ledger, control, attempts)
    else:
        if no_l4_attempts or l4_binding.get("attempt_count", 0) == 0:
            raise ValueError(
                "mixed-platform union release requires nonempty L4 accounting and inventory"
            )
        closed, _ = _metadata(ledger.parent / "union-close.json")
        if (
            closed.get("schema") != "bl-labels-f0-union-close/v1"
            or closed.get("verdict") != "CLEAN"
            or closed.get("declared_pending") != []
            or closed.get("prefix_length") != len(control["allocation"])
            or closed.get("prefix_ledger_sha256") != binding["sha256"]
            or closed.get("mac_attempt_exports") != [str(p.resolve()) for p in attempts]
            or closed.get("cumulative_mac_elapsed_seconds") != elapsed
            or closed.get("funding_receipt_sha256") != _verify(funding_receipt)
        ):
            raise ValueError("D4096 release requires its verified complete Mac/L4 union close")
    _, _, pending = mac_driver._prepare_jobs(control, ledger)
    if pending:
        raise ValueError("D4096 release cannot open an incomplete union for fitting")
    result = {
        "schema": "bl-labels-d4096-collection-disposition/v1",
        **collection_disclosure(
            actual_execution_placement=(
                "Mac" if value.get("attempt_exports") is not None else "Mac + Colab L4 CPU"
            )
        ),
        "disposition": "RELEASED",
        "judgment_sha256": JUDGMENT_SHA256,
        **{
            key: funding[key]
            for key in (
                "series_id",
                "registration_sha256",
                "reserve_allocation_sha256",
                "recipe_sha256",
            )
        },
        "l4_accounting": l4_binding,
        "collection_ledger_sha256": binding["sha256"],
        "funding_receipt_sha256": _verify(funding_receipt),
        "collection_ledger": binding,
        "code_commit": commit,
    }
    publish_json(output, result)
    return result


def _fit_receipts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in result or not path:
            raise ValueError("--fit-receipt must be JOB_ID=PATH with unique JOB_ID")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("seal-readiness")
    readiness.add_argument("--out", type=Path, required=True)
    readiness.add_argument("--d1024-terminal", type=Path, required=True)
    readiness.add_argument("--d1024-w6-metadata", type=Path, required=True)
    readiness.add_argument("--fit-receipt", action="append", default=[])
    readiness.add_argument("--factory-exited", action="store_true")
    readiness.add_argument("--mac-lease-released", action="store_true")
    readiness.add_argument("--all-fit-jobs-listed", action="store_true")
    projection = commands.add_parser("seal-projection")
    projection.add_argument("--out", type=Path, required=True)
    projection.add_argument("--readiness", type=Path, required=True)
    projection.add_argument("--d256-terminal", type=Path, required=True)
    projection.add_argument("--d1024-terminal", type=Path, required=True)
    adapter = commands.add_parser("seal-i3-projection")
    adapter.add_argument("--out", type=Path, required=True)
    adapter.add_argument("--mac-projection", type=Path, required=True)
    adapter.add_argument("--d256-collection", type=Path, required=True)
    adapter.add_argument("--d1024-collection", type=Path, required=True)
    release = commands.add_parser("release-collection")
    release.add_argument("--out", type=Path, required=True)
    release.add_argument("--ledger", type=Path, required=True)
    release.add_argument("--funding-receipt", type=Path, required=True)
    release.add_argument("--pack", type=Path, default=CANONICAL_PACK)
    release.add_argument("--code-commit", required=True)
    release.add_argument("--l4-ledger", type=Path)
    release.add_argument("--l4-attempt-inventory", type=Path)
    release.add_argument("--no-l4-attempts", action="store_true")
    discard = commands.add_parser("seal-not-run")
    discard.add_argument("--out", type=Path, required=True)
    discard.add_argument("--readiness", type=Path, required=True)
    terminal_choice = discard.add_mutually_exclusive_group(required=True)
    terminal_choice.add_argument("--terminal", type=Path)
    terminal_choice.add_argument("--terminal-label", choices=sorted(_DISCARD_LABELS))
    discard.add_argument("--terminal-stage", choices=("d1024-fit", "r-a"))
    discard.add_argument("--unrecoverable", action="store_true")
    args = parser.parse_args()
    if args.command == "seal-readiness":
        result = seal_readiness(
            args.out,
            args.d1024_terminal,
            _fit_receipts(args.fit_receipt),
            args.d1024_w6_metadata,
            factory_exited=args.factory_exited,
            mac_lease_released=args.mac_lease_released,
            all_fit_jobs_listed=args.all_fit_jobs_listed,
        )
    elif args.command == "seal-projection":
        result = seal_mac_h_projection(
            args.out, args.readiness, args.d256_terminal, args.d1024_terminal
        )
    elif args.command == "seal-i3-projection":
        result = seal_i3_mac_projection(
            args.out, args.mac_projection, args.d256_collection, args.d1024_collection
        )
    elif args.command == "release-collection":
        result = seal_collection_release(
            args.out,
            args.ledger,
            args.funding_receipt,
            pack=args.pack,
            code_commit=args.code_commit,
            l4_ledger=args.l4_ledger,
            l4_attempt_inventory=args.l4_attempt_inventory,
            no_l4_attempts=args.no_l4_attempts,
        )
    else:
        result = seal_not_run_by_rule(
            args.out,
            args.readiness,
            args.terminal,
            terminal_label=args.terminal_label,
            terminal_stage=args.terminal_stage,
            unrecoverable=args.unrecoverable,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
