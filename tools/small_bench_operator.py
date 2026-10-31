"""Local operator transaction for approved small engineering benchmarks.

No allocation API is exposed here. Dispatch uses the existing governed wrapper.
The register and account lock are shared across packages, never per-job files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.colab_cli.accounting import ACCOUNT_ASSIGNMENT_CAP
from tools import small_bench_register as register


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def bound(value: dict[str, str]) -> Path:
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink() or sha(path) != value["sha256"]:
        raise ValueError(f"changed or invalid binding: {path}")
    return path


def publish(path: Path, payload: bytes) -> None:
    """Create immutable payload, SHA sidecar, then completion marker last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, data in (
        (path, payload),
        (
            Path(str(path) + ".sha256"),
            f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n".encode(),
        ),
        (Path(str(path) + ".complete"), b"complete\n"),
    ):
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            os.unlink(temporary)


def publish_json(path: Path, value: Any) -> None:
    publish(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def verified(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or Path(str(path) + ".complete").read_text() != "complete\n"
        or Path(str(path) + ".sha256").read_text().split()[0] != sha(path)
    ):
        raise ValueError(f"incomplete or changed record: {path}")
    return json.loads(path.read_text())


def configuration(path: Path, *, review: bool = True) -> dict[str, Any]:
    config = verified(path)
    for source in config["bindings"]:
        bound(source)
    if "outside_tail_source" in config:
        bound(config["outside_tail_source"])
    manifest = json.loads(bound(config["manifest"]).read_text())
    if (
        manifest["job_id"] != config["job_id"]
        or manifest["requirements"]["gpu"] != "A100"
        or manifest["timeout_minutes"] != config["cap"]["wall_minutes"]
        or manifest["budget"]["unit_cap"] != config["cap"]["units"]
        or not 0 < manifest["budget"]["rate_units_per_hour"] <= 10
        or config["packet_ceiling"]["units"] > 25
        or config["purpose"] not in ("timing", "profiling", "correctness")
    ):
        raise ValueError("wrong job, caps, scope, or A100 rate identity")
    if review:
        accepted = verified(Path(config["adapter_review"]))
        if accepted.get("status") != "ACCEPT" or accepted.get("config_sha256") != sha(path):
            raise ValueError("adapter requires its one independent offline acceptance")
        for source in accepted["implementation"]:
            bound(source)
    return config


def run_check(config: dict[str, Any]) -> None:
    """Run the package's existing offline acceptance and governed wrapper dry-run."""
    subprocess.run(config["offline_check"], cwd=config["cwd"], check=True, timeout=300)


def fingerprint(config: dict[str, Any]) -> dict[str, str]:
    """Operational identity only: do not read weights or scientific payloads."""
    files: set[Path] = set()
    for item in config["watch_globs"]:
        files.update(Path(item["root"]).glob(item["pattern"]))
    return {str(p): sha(p) for p in sorted(files) if p.is_file()}


def capture(config: dict[str, Any], folder: Path) -> dict[str, Any]:
    folder.mkdir(parents=True)
    started = utc_now()
    sources = {}
    for command in ("usage", "sessions"):
        result = subprocess.run(
            [config["colab"], "--auth", "oauth2", command],
            capture_output=True,
            timeout=60,
            check=False,
        )
        target = folder / f"{command}.txt"
        publish(target, result.stdout + result.stderr)
        sources[command] = binding(target)
        if result.returncode:
            raise ValueError(f"live {command} capture failed; evidence retained")
    value = {"started_utc": stamp(started), "completed_utc": stamp(utc_now()), **sources}
    publish_json(folder / "capture.json", value)
    return value


def live_checks(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    usage_raw = bound(evidence["usage"]).read_text()
    sessions_raw = bound(evidence["sessions"]).read_text()
    usage = {}
    for key, label in (
        ("balance", "Current balance"),
        ("rate", "Usage rate"),
        ("assignments", "Active assignments"),
    ):
        matches = re.findall(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", usage_raw)
        if len(matches) != 1 or not math.isfinite(float(matches[0])):
            raise ValueError("unrecognized live usage")
        usage[key] = float(matches[0])
    sessions = {}
    for line in sessions_raw.splitlines():
        if line.startswith("[") and not line.startswith("[colab]"):
            match = re.fullmatch(
                r"\[([^\]]+)\]\s+(\S+)\s+\| Hardware:\s*([^|]+?)(?:\s+\|.*)?", line
            )
            if match is None or match[2] in sessions:
                raise ValueError("unrecognized or duplicate allocation")
            sessions[match[2]] = {"session": match[1], "hardware": match[3].strip().upper()}
    empty = "No active sessions found on server." in sessions_raw
    if (not sessions and not empty) or (sessions and empty):
        raise ValueError("unknown or contradictory allocation")
    if len(sessions) >= ACCOUNT_ASSIGNMENT_CAP or usage["assignments"] != len(sessions):
        raise ValueError("account assignment cap/count mismatch")
    account = Path(config["account_root"])
    if list(account.glob("pending-*.json")):
        raise ValueError("unknown/pending allocation")
    records = {}
    for path in account.glob("active-*.json"):
        record = json.loads(path.read_text())
        endpoint = record["endpoint"]
        if endpoint in records or path.name != f"active-{record['session']}.json":
            raise ValueError("ambiguous broker ownership")
        records[endpoint] = (record, binding(path))
    rates, owned = {}, []
    for endpoint, session in sessions.items():
        if session["hardware"] == "A100" or endpoint not in records:
            raise ValueError("competing A100 or unknown allocation")
        record, source = records[endpoint]
        if record["session"] != session["session"]:
            raise ValueError("wrong broker session identity")
        rate = record.get("observed_rate_units_per_hour")
        cap = record.get("unit_cap")
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("assignment lacks finite full remaining cap")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("missing positive calibrated rate")
        if (
            session["hardware"] not in {"L4", "CPU"}
            or rate != {"L4": 1.54, "CPU": 0.08}[session["hardware"]]
        ):
            raise ValueError("uncalibrated hardware/rate")
        if record.get("accounting_exclusive") or record.get("exclusive_accounting"):
            raise ValueError("conflicting accounting-exclusive job")
        rates[endpoint] = rate
        owned.append({"endpoint": endpoint, "source": source, "full_cap": record["unit_cap"]})
    if not math.isclose(sum(rates.values()), usage["rate"], abs_tol=1e-8):
        raise ValueError("unexplained account rate")
    for source in config["previous_releases"]:
        receipt = json.loads(bound(source).read_text())
        if receipt.get("stop_verified") is not True or receipt.get("endpoint") in sessions:
            raise ValueError("prior A100 release/disappearance unproved")
    return {
        "balance": usage["balance"],
        "assignments": owned,
        "rates": rates,
        "prior_release_and_disappearance": True,
        "no_pending_or_competing_A100": True,
    }


def current_admission(config: dict[str, Any]) -> dict[str, Any]:
    """The single pointer switches atomically; each prior trio stays immutable."""
    pointer = Path(config["admission_pointer"])
    reference = json.loads(pointer.read_text())
    admission = verified(bound(reference))
    evidence = admission["capture"]
    usage = bound(evidence["usage"])
    bound(evidence["sessions"])
    if verified(usage.parent / "capture.json") != evidence:
        raise ValueError("capture provenance changed after admission")
    return admission


def assignment_coverage(finances: dict[str, Any], live: dict[str, Any]) -> None:
    """Keep each concurrent endpoint's full remaining cap financially covered."""
    if finances["external_commitments_units"] + 1e-9 < sum(
        assignment["full_cap"] for assignment in live["assignments"]
    ):
        raise ValueError("live assignment full caps lack external register coverage")


def financial_snapshot(
    finances: dict[str, Any], config: dict[str, Any], balance: float
) -> dict[str, Any]:
    """Make every admission row's scope/account arithmetic directly reviewable."""
    prior = {
        key: value
        for key, value in finances["outstanding"].items()
        if key != config["reservation_id"]
    }
    scopes = {}
    for name in config["scopes"]:
        scope = finances["scopes"][name]
        held = {
            key: sum(hold["pair"][key] for hold in prior.values() if name in hold["scopes"])
            for key in ("units", "wall_minutes")
        }
        scopes[name] = {
            **scope,
            "outstanding_holds": held,
            "available_before_reservation": {
                key: scope["ceiling"][key] - scope["spent"][key] - held[key]
                for key in ("units", "wall_minutes")
            },
        }
    account_holds = sum(hold["account_units"] for hold in prior.values())
    return {
        "scopes": scopes,
        "holds": prior,
        "live_balance_units": balance,
        "external_commitments_units": finances["external_commitments_units"],
        "unsettled_debit_holds_units": finances["unsettled_debit_holds_units"],
        "account_holds_units": account_holds,
        "live_available_before_reservation_units": balance
        - account_holds
        - finances["external_commitments_units"]
        - finances["unsettled_debit_holds_units"],
        "full_new_reservation": {
            key: config["cap"][key] + config["outside_tail"][key]
            for key in ("units", "wall_minutes")
        },
        "same_reservation_excluded_once": config["reservation_id"] in finances["reservations"],
    }


def admit(config_path: Path) -> dict[str, Any]:
    config = configuration(config_path)
    run_check(config)
    with (
        register.reservation_lock(Path(config["reservation_lock"])),
        register.reservation_lock(Path(config["account_root"]) / "broker.lock"),
    ):
        rows = register.read(Path(config["register"]))
        # A failed refresh must never leave an older affordable observation
        # dispatchable. Only this mutable pointer is invalidated; immutable
        # captures/admissions and the existing full reservation are preserved.
        pointer = Path(config["admission_pointer"])
        pointer.unlink(missing_ok=True)
        if pointer.parent.is_dir():
            directory_fd = os.open(pointer.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        before = fingerprint(config)
        baseline = rows[0].get("operational_baseline")
        # Later supported operational deltas are register events, never silent
        # replacements of the initial completeness attestation.
        for delta in rows[1:]:
            if "operational_baseline" in delta:
                baseline = delta["operational_baseline"]
        if baseline is not None and before != baseline:
            raise ValueError(
                "unrecorded operational liability delta; append supported register delta"
            )
        evidence = capture(config, Path(config["admissions_dir"]) / uuid.uuid4().hex)
        live = live_checks(config, evidence)
        finances = register.state(rows)
        assignment_coverage(finances, live)
        if before != fingerprint(config):
            raise ValueError("assignment or liability changed during capture; recapture")
        package = dict(
            config,
            manifest_sha256=config["manifest"]["sha256"],
            capture_utc=evidence["started_utc"],
            live_balance_units=live["balance"],
            operator=os.environ.get("USER", "unknown"),
        )
        event = register.reserve(rows, package, config["reservation_id"], utc_now())
        event.update(
            capture=evidence,
            live_checks=live,
            config=binding(config_path),
            operational_fingerprint=before,
            financial_before=financial_snapshot(finances, config, live["balance"]),
        )
        row = register.append(Path(config["register"]), event)
        financial = register.state(register.read(Path(config["register"])))
        admission = {
            "status": "ADMIT_INITIAL_MAIN",
            "reservation_id": config["reservation_id"],
            "register_revision": row["revision"],
            "register_row_hash": row["row_hash"],
            "config": binding(config_path),
            "capture": evidence,
            "live_checks": live,
            "operational_fingerprint": before,
            "row": row,
        }
        directory = Path(evidence["usage"]["path"]).parent
        target = directory / "OPERATOR_ADMISSION.json"
        publish_json(target, admission)
        first = financial["first_unaudited_admission"]
        publish_json(
            directory / "BATCH_AUDIT.json",
            {
                "status": "PENDING_INDEPENDENT_BATCH_AUDIT",
                "first_unaudited_admission_utc": stamp(first) if first else None,
                "unaudited_gross_units": financial["unaudited_gross_units"],
                "prefix_revision": row["revision"],
                "prefix_hash": row["row_hash"],
                "due_within_hours": 24,
                "maximum_gross_reservations_before_audit_units": 100,
                "register": config["register"],
                "no_independent_acceptance_claimed": True,
            },
        )
        # A mutable pointer, not mutable evidence. A crash leaves a conservative hold.
        pointer = Path(config["admission_pointer"])
        fd, temporary = tempfile.mkstemp(dir=pointer.parent, prefix=".admission-pointer-")
        with os.fdopen(fd, "w") as stream:
            json.dump(binding(target), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)
        log = Path(config["campaign_log"])
        with log.open("a") as stream:
            stream.write(
                f"\nSmall-bench admission `{config['job_id']}`: "
                f"ruling `{config['ruling']['sha256']}`, "
                f"reservation `{config['reservation_id']}`, register revision {row['revision']} "
                f"SHA256 `{row['row_hash']}`, admission `{target}` SHA256 `{sha(target)}`. "
                "Reserved once; dispatch not implied. Batch audit remains asynchronous.\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "status": admission["status"],
            "admission": str(target),
            "allocations": 0,
            "reservation_id": config["reservation_id"],
        }


def check_admission(config_path: Path, now: datetime | None = None) -> dict[str, Any]:
    config = configuration(config_path)
    admission = current_admission(config)
    rows = register.read(Path(config["register"]))
    if (
        admission["config"] != binding(config_path)
        or not rows
        or (rows[-1]["row_hash"] != admission["register_row_hash"])
    ):
        raise ValueError("reservation/liability revision changed; repeat admission delta check")
    moment = now or utc_now()
    age = (moment - datetime.fromisoformat(admission["capture"]["started_utc"])).total_seconds()
    if not 0 <= age < 600:
        raise ValueError("stale capture; rerun admit with the same reservation ID")
    if fingerprint(config) != admission["operational_fingerprint"]:
        raise ValueError("assignment/liability identity changed; repeat admission")
    # Re-run financial/audit/deadline checks against the same full reservation.
    package = dict(
        config,
        manifest_sha256=config["manifest"]["sha256"],
        capture_utc=admission["capture"]["started_utc"],
        live_balance_units=admission["live_checks"]["balance"],
        operator=admission["row"]["package"]["operator"],
    )
    register.reserve(rows, package, config["reservation_id"], moment)
    return admission


def dispatch(config_path: Path, dry: bool) -> dict[str, Any]:
    config = configuration(config_path)
    run_check(config)
    if dry:
        return {"status": "PASS_OFFLINE", "allocations": 0}
    with register.reservation_lock(Path(config["reservation_lock"])):
        admission = check_admission(config_path)
        started = datetime.fromisoformat(admission["capture"]["started_utc"])
        remaining = 600 - (utc_now() - started).total_seconds()
        if remaining < 60:
            raise ValueError("insufficient live window; rerun admit (no reviewer restart)")
        # The qualified wrapper independently refreshes membership under alloc.lock.
        env = dict(
            os.environ,
            OPS_ALLOC_LOCK_TIMEOUT_SECONDS=str(min(remaining, 600)),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=config["dispatch_pythonpath"],
        )
        subprocess.run(config["dispatch_command"], cwd=config["dispatch_cwd"], env=env, check=True)
    return {"status": "DISPATCHED", "reservation_id": config["reservation_id"]}


def settlement_evidence(event: dict[str, Any]) -> dict[str, Any]:
    """Prove local close while retaining unpaid debit independently of release."""
    receipt = json.loads(bound(event["terminal_receipt"]).read_text())
    bound(event["verified_exports"])
    release = json.loads(bound(event["release_disappearance"]).read_text())
    session_text = bound(release["sessions"]).read_text()
    endpoint = receipt.get("endpoint")
    if receipt.get("stop_verified") is not True:
        # The pinned broker publishes this create-only receipt before any CLI
        # request for an oversized upload. Unknown allocation is never cancelled.
        no_allocation = (
            endpoint is None
            and receipt.get("generation") is None
            and receipt.get("failure_class") == "UPLOAD_CAP"
            and receipt.get("elapsed_seconds") == 0
            and receipt.get("cli_calls") == []
            and event.get("allocation_never_requested") is True
            and bool(event.get("no_allocation_attestation"))
        )
        if not no_allocation:
            raise ValueError("settlement needs verified release or proven pre-allocation failure")
        bound(event["no_allocation_proof"])
    if (endpoint and endpoint in session_text) or (
        "No active sessions found on server." not in session_text
        and not re.search(r"^\[[^\]]+\]\s+\S+\s+\| Hardware:", session_text, re.M)
    ):
        raise ValueError("settlement needs actual endpoint disappearance")
    finished = (receipt.get("terminal") or {}).get("finished_at")
    if (
        finished is not None
        and datetime.fromisoformat(release["started_utc"]).timestamp() < finished
    ):
        raise ValueError("release capture predates terminal receipt")
    default_hold = event["charged"]["units"] + event["remaining_hold"]["units"]
    if event.get("remaining_account_units", default_hold) < default_hold:
        bound(event["provider_settlement"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("admit", "dispatch", "check", "audit-record", "settle", "delta", "reconcile"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.dry_run and args.operation != "dispatch":
        parser.error("--dry-run is supported only for dispatch; no live action performed")
    if args.operation == "admit":
        result = admit(args.config)
    elif args.operation == "dispatch":
        result = dispatch(args.config, args.dry_run)
    elif args.operation == "check":
        result = check_admission(args.config)
    else:
        config = configuration(args.config)
        if args.record is None:
            parser.error("independent audit/settlement --record required")
        event = verified(args.record)
        expected = {
            "audit-record": "audit",
            "settle": "settlement",
            "delta": "delta",
            "reconcile": "reconcile",
        }[args.operation]
        if event.get("type") != expected:
            raise ValueError("wrong record type")
        for source in event.get("sources", []):
            bound(source)
        for source in event.get("settlement_evidence", []):
            bound(source)
        if expected == "audit":
            bound(event["independent_source"])
        elif expected == "settlement":
            receipt = settlement_evidence(event)
        with register.reservation_lock(Path(config["reservation_lock"])):
            if expected == "settlement":
                current = register.state(register.read(Path(config["register"])))
                identity = current["reservations"][event["reservation_id"]]["identity"]
                if receipt.get("job_id") != identity[0]:
                    raise ValueError("settlement receipt has wrong job identity")
            result = register.append(Path(config["register"]), event)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
