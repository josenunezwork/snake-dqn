"""Outcome-blind, one-look checker for the C1 RPC qualification profile.

The checker deliberately knows no label or rescue fields.  It first publishes
an inventory of immutable unit ledgers, then (and only then) reads operational
receipts and integrity summaries for the single platform verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_labels.exports import _verify, publish_json
from research.bl_labels.l4_faults_v2 import classify_session, load_injection_manifest

SCHEMA = "rpc-qualification-profile/v2"
PUBLICATION_SCHEMA = "rpc-qualification-ledger-publication/v1"
VERDICT_SCHEMA = "rpc-qualification-verdict/v2"
FROZEN_BANK = "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"
F1_PIN = "2af48cd0b98a2063f46871cf4d320a5541b789af"
BRANCH_REF = "refs/heads/ops-l4-faults-v2-20261024"
PLATFORMS = ("cpu", "l4")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"qualification plan lacks {name}")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _profile(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _read(plan_path)
    if plan.get("schema") != SCHEMA or plan.get("bank_sha256") != FROZEN_BANK:
        raise ValueError("foreign qualification plan/bank")
    if plan.get("worker_pin") != F1_PIN:
        raise ValueError("qualification plan has foreign frozen worker pin")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 7:
        raise ValueError("qualification plan requires six loops and one injection")
    seen: set[str] = set()
    loops = {platform: 0 for platform in PLATFORMS}
    injection = 0
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("qualification plan has malformed job")
        identity = job.get("job_id")
        platform, mode, namespace = job.get("platform"), job.get("mode"), job.get("namespace")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("qualification plan has duplicate job ID")
        seen.add(identity)
        if platform not in PLATFORMS or mode not in {"loop", "injection"}:
            raise ValueError("qualification plan has foreign job class")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("qualification job lacks namespace")
        if mode == "loop":
            loops[platform] += 1
        else:
            injection += 1
            if platform != "cpu":
                raise ValueError("live injection requires CPU")
    if loops != {"cpu": 3, "l4": 3} or injection != 1:
        raise ValueError("qualification plan does not have the required 3+3+1 shape")
    namespaces = [str(job["namespace"]) for job in jobs]
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("qualification plan reuses a namespace")
    return plan, jobs


def _job_paths(plan_path: Path, job: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    base = plan_path.parent
    manifest = _path(base, job.get("manifest"), "manifest")
    output = _path(base, job.get("output_dir"), "output_dir")
    receipt = _path(base, job.get("receipt", str(output / "receipt.json")), "receipt")
    runtime = _path(base, job.get("runtime", str(output / "runtime-result.json")), "runtime")
    return manifest, output, receipt, runtime


def _manifest(job: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = _read(manifest_path)
    rq = manifest.get("rq")
    if not isinstance(rq, dict) or rq.get("schema") != SCHEMA:
        raise ValueError("manifest lacks qualification binding")
    for key in ("mode", "platform", "namespace"):
        if rq.get(key) != job.get(key):
            raise ValueError(f"manifest differs from plan: {key}")
    if rq.get("bank_sha256") != FROZEN_BANK or rq.get("worker_pin") != F1_PIN:
        raise ValueError("manifest differs from frozen F1 identity")
    envelope = rq.get("envelope")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("effective"), dict):
        raise ValueError("manifest lacks C1 envelope")
    effective = envelope["effective"]
    cap = _finite(effective.get("unit_cap"), "effective unit cap")
    timeout = _finite(effective.get("timeout_minutes"), "effective timeout")
    if job["mode"] == "loop":
        expected = 5.0 if job["platform"] == "l4" else 0.30
        if cap != expected or timeout != 160:
            raise ValueError("manifest breaches C1 session envelope")
    elif cap != 0.12 or timeout != 30:
        raise ValueError("manifest breaches C1 injection envelope")
    if (
        manifest.get("timeout_minutes") != timeout
        or manifest.get("budget", {}).get("unit_cap") != cap
    ):
        raise ValueError("manifest operative caps differ from its frozen envelope")
    return manifest


def _ledger_inventory(job: dict[str, Any], output: Path) -> dict[str, Any]:
    """Read only identity wrappers plus immutable hashes before any verdict read."""
    # ``output`` is the supervisor collection root; immutable exports live below it.
    ledger_path = output / "exports" / "verified-ledger.json"
    if not ledger_path.is_file():
        return {"path": str(ledger_path), "status": "MISSING"}
    _verify(ledger_path)
    ledger = _read(ledger_path)
    if (
        ledger.get("schema") != "rpc-qualification-verified-ledger/v1"
        or ledger.get("namespace") != job["namespace"]
        or ledger.get("allocation") != job.get("allocation")
        or ledger.get("source_commit") != F1_PIN
        or ledger.get("bank_sha256") != FROZEN_BANK
        or not isinstance(ledger.get("units"), list)
    ):
        raise ValueError("foreign qualification ledger")
    seen: set[int] = set()
    for row in ledger["units"]:
        if not isinstance(row, dict) or not isinstance(row.get("index"), int):
            raise ValueError("malformed qualification ledger row")
        index = row["index"]
        if index < 0 or index in seen or row.get("path") != f"units/unit-{index:08d}.json":
            raise ValueError("duplicate or foreign ledger unit")
        seen.add(index)
        payload = ledger_path.parent / row["path"]
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("ledger unit differs from immutable payload")
        unit = _read(payload)
        if (
            unit.get("schema") != "rpc-qualification-loop/v1"
            or unit.get("status") != "COMPLETE"
            or unit.get("namespace") != job["namespace"]
            or unit.get("allocation") != job.get("allocation")
            or unit.get("index") != index
            or unit.get("unit_id") != f"{job['namespace']}/{unit.get('allocation')}/{index:08d}"
        ):
            raise ValueError("foreign qualification unit")
    return {
        "path": str(ledger_path),
        "sha256": _sha(ledger_path),
        "size": ledger_path.stat().st_size,
        "units": len(seen),
    }


def _sealed_evidence(job: dict[str, Any], output: Path) -> dict[str, Any]:
    """Hash opaque evidence before reading any integrity or verdict fields."""
    paths = [output / "receipt.json", output / "manifest.json", output / "injection-proof.json"]
    # The verified ledger binds every unit's path/size/hash. Keep its publication
    # index small enough to commit; do not duplicate thousands of unit sidecars.
    paths += [
        output / "exports" / name
        for name in ("runtime-result.json", "verified-ledger.json", "injection-proof.json")
    ]
    paths += (
        sorted((output / "injection-evidence").rglob("*"))
        if (output / "injection-evidence").is_dir()
        else []
    )
    result = {}
    for path in paths:
        if path.is_file():
            result[str(path.relative_to(output))] = {
                "sha256": _sha(path),
                "size": path.stat().st_size,
            }
    return result


def publish(plan_path: Path, out: Path) -> dict[str, Any]:
    """Publish all seven ledgers and opaque evidence hashes before the one look."""
    plan, jobs = _profile(plan_path)
    ledgers = {}
    for job in jobs:
        manifest, output, _receipt, _runtime = _job_paths(plan_path, job)
        if manifest.is_file():
            _manifest(job, manifest)
        try:
            inventory = _ledger_inventory(job, output)
        except (ValueError, OSError, KeyError) as error:
            inventory = {"status": "INVALID", "reason": str(error)}
        inventory["evidence"] = _sealed_evidence(job, output)
        ledgers[job["job_id"]] = inventory
    value = {
        "schema": PUBLICATION_SCHEMA,
        "plan_sha256": _sha(plan_path),
        "branch_ref": BRANCH_REF,
        "ledgers": ledgers,
        "published_before_verdict": True,
    }
    repo = _path(plan_path.parent, plan.get("repo"), "repository")
    relative = Path(plan["publication_ledger"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("publication ledger must be repository relative")
    publish_json(repo / relative, value)
    if out.resolve() != (repo / relative).resolve():
        publish_json(out, value)
    return value


def _bound_receipt(
    job: dict[str, Any], manifest_path: Path, output: Path, *, require_success: bool = True
) -> tuple[dict, dict]:
    expected = _manifest(job, manifest_path)
    canonical = output / "manifest.json"
    actual = _read(canonical)
    receipt = _read(output / "receipt.json")
    if receipt.get("manifest_sha256") != _sha(canonical):
        raise ValueError("receipt canonical manifest binding differs")
    # Account inventory is freshly reviewed by dispatch under its account lock.
    # Supervisor may materialize documented defaults, but cannot change any
    # approved field except this exact live account map.
    for key, value in expected.items():
        if key == "budget":
            if any(
                actual.get(key, {}).get(k) != v
                for k, v in value.items()
                if k != "other_assignments"
            ):
                raise ValueError("dispatched budget differs from approved envelope")
        elif actual.get(key) != value:
            raise ValueError(f"canonical manifest differs from approved field: {key}")
    if receipt.get("job_id") != actual.get("job_id") or receipt.get("commit") != actual.get(
        "commit"
    ):
        raise ValueError("receipt job/source identity differs")
    terminal = receipt.get("terminal")
    if require_success and (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error")
        or not isinstance(terminal, dict)
        or terminal.get("returncode") != 0
        or terminal.get("generation") != receipt.get("generation")
        or terminal.get("job_id") != receipt.get("job_id")
    ):
        raise ValueError("missing successful terminal or verified release/backend disappearance")
    elapsed = _finite(receipt.get("elapsed_seconds"), "session elapsed")
    budget = actual["budget"]
    if require_success and (
        elapsed < 0
        or elapsed > actual["timeout_minutes"] * 60
        or elapsed * budget["rate_units_per_hour"] / 3600 > budget["unit_cap"]
    ):
        raise ValueError("session envelope breach")
    return actual, receipt


def _bound_export(path: Path, receipt: dict[str, Any], name: str) -> None:
    digest = _verify(path)
    rows = [row for row in receipt.get("exports", []) if row.get("path") == name]
    if (
        len(rows) != 1
        or rows[0].get("sha256") != digest
        or rows[0].get("size") != path.stat().st_size
    ):
        raise ValueError(f"receipt export binding differs: {name}")


def _injections(manifest: dict[str, Any]) -> dict[str, Any]:
    """The session's create-only, pre-dispatch injection manifest (sha-bound)."""
    planned = manifest.get("rq", {}).get("injection_manifest")
    if not isinstance(planned, dict) or not isinstance(planned.get("path"), str):
        raise ValueError("session lacks its pre-dispatch injection manifest")
    path = Path(planned["path"])
    if _sha(path) != planned.get("sha256"):
        raise ValueError("injection manifest differs from its dispatched binding")
    return load_injection_manifest(path, job_id=manifest.get("job_id"))


def _raw_faults(
    receipt: dict[str, Any], injections: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Counted faults under l4-label-faults/v2 (C1-C6) with the raw records kept.

    Recovered requests are recorded (``recovered``), never counted; deliberate
    faults are excluded only by the session's pre-dispatch manifest identity.
    """
    result = classify_session(receipt, injections)
    faults = [
        {"source": "v2:" + row["class"], "index": row.get("index"), "record": row}
        for row in result["counted"]
    ]
    summary = {key: value for key, value in result.items() if key != "counted"}
    return faults, summary


def _check_loop(plan_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    manifest_path, output, receipt_path, runtime_path = _job_paths(plan_path, job)
    manifest, receipt = _bound_receipt(job, manifest_path, output)
    runtime = _read(runtime_path)
    _bound_export(runtime_path, receipt, "runtime-result.json")
    _bound_export(output / "exports/verified-ledger.json", receipt, "verified-ledger.json")
    if runtime.get("schema") != "rpc-qualification-loop/v1" or runtime.get("status") != "COMPLETE":
        raise ValueError("qualification loop is incomplete")
    required = {
        "namespace": job["namespace"],
        "platform": job["platform"],
        "bank_sha256": FROZEN_BANK,
        "source_commit": F1_PIN,
        "workers": 12 if job["platform"] == "l4" else 2,
        "allocation": job["allocation"],
    }
    if (
        runtime.get("source_closure", {}).get("archive_sha256")
        != manifest["rq"]["frozen_source_sha256"]
    ):
        raise ValueError("runtime frozen source attestation differs")
    if any(runtime.get(key) != value for key, value in required.items()):
        raise ValueError("qualification runtime has foreign identity")
    if (
        _finite(runtime.get("loop_seconds"), "loop seconds") < 3600
        or _finite(runtime.get("measured_loop_seconds"), "measured loop seconds") < 3600
    ):
        raise ValueError("qualification loop is shorter than 60 minutes")
    integrity = runtime.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("all_attempted_roots_clean") is not True
        or integrity.get("repeat_digest_consistent") is not True
        or integrity.get("bare_audit_mismatches") != 0
    ):
        raise ValueError("qualification runtime integrity fails")
    repeats = runtime.get("root_repeat_counts")
    if (
        not isinstance(repeats, dict)
        or len(repeats) != 64
        or any(type(count) is not int or count < 2 for count in repeats.values())
    ):
        raise ValueError("qualification loop did not repeat every frozen F1 root")
    ledger = _read(output / "exports" / "verified-ledger.json")
    if {row["path"] for row in ledger["units"]} != {
        row["path"] for row in receipt.get("exports", []) if row["path"].startswith("units/")
    }:
        raise ValueError("ledger does not cover every attempted committed unit")
    if runtime.get("units_completed") != len(ledger["units"]):
        raise ValueError("runtime unit count differs from verified ledger")
    observed: dict[str, tuple[str, str, int]] = {}
    for row in ledger["units"]:
        unit = _read((output / "exports") / row["path"])
        integrity_row = unit.get("integrity")
        if not isinstance(integrity_row, dict):
            raise ValueError("qualification unit lacks integrity")
        if (
            integrity_row.get("digest_checks") != 1
            or integrity_row.get("bare_audit_mismatches") != 0
        ):
            raise ValueError("qualification unit integrity fails")
        root = unit.get("root_id")
        digest = integrity_row.get("digest")
        label = integrity_row.get("label_row_sha256")
        if not all(isinstance(value, str) and value for value in (root, digest, label)):
            raise ValueError("qualification unit integrity is malformed")
        if unit.get("root_id") != runtime.get("root_order", [])[row["index"] % 64]:
            raise ValueError("unit order differs from the frozen bank allocation")
        _bound_export(output / "exports" / row["path"], receipt, row["path"])
        prior = observed.get(root)
        if prior and prior[:2] != (digest, label):
            raise ValueError("qualification repeat digest/hash differs")
        observed[root] = (digest, label, (prior[2] if prior else 0) + 1)
    if len(observed) != 64 or any(value[2] < 2 for value in observed.values()):
        raise ValueError("qualification unit ledger lacks repeated frozen roots")
    faults, v2 = _raw_faults(receipt, _injections(manifest))
    return {
        "session": job["job_id"],
        "platform": job["platform"],
        "status": "FAIL" if faults else "PASS",
        "v2": v2,
        "counted_faults": len(faults),
        "faults": faults,
        "raw_records": receipt,
        "release_verified": True,
        "backend_disappeared": True,
    }


def _check_injection(plan_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    manifest_path, output, receipt_path, _runtime_path = _job_paths(plan_path, job)
    manifest, receipt = _bound_receipt(job, manifest_path, output, require_success=False)
    evidence = _read(output / "injection-proof.json")
    _bound_export(output / "exports/verified-ledger.json", receipt, "verified-ledger.json")
    _bound_export(_runtime_path, receipt, "runtime-result.json")
    if (
        evidence.get("schema") != "rpc-qualification-injection-proof/v1"
        or evidence.get("status") != "COMPLETE"
    ):
        raise ValueError("foreign/incomplete injection proof")
    units = evidence.get("injected_units")
    if not isinstance(units, list) or len(units) != 3:
        raise ValueError("injection needs at least three units")
    ledger = _read(output / "exports/verified-ledger.json")
    ledger_units = {row["unit_id"]: row for row in ledger["units"]}
    if len(ledger_units) != 3:
        raise ValueError("injection ledger does not have exactly three committed units")
    request_ids, unit_ids, injected_indices = set(), set(), set()
    for item in units:
        if not isinstance(item, dict):
            raise ValueError("malformed injection unit")
        request_id, unit_id = item.get("request_id"), item.get("unit_id")
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in request_ids
            or not isinstance(unit_id, str)
            or not unit_id
            or unit_id in unit_ids
        ):
            raise ValueError("injection identities are not unique")
        request_ids.add(request_id)
        unit_ids.add(unit_id)
        if item.get("execution_count") != 1:
            raise ValueError("injection duplicated execution")
        if unit_id not in ledger_units:
            raise ValueError("injection proof has foreign unit identity")
        marker_values = {}
        for marker in ("started", "done", "result", "execution_started", "execution_done"):
            meta = item.get(marker)
            if not isinstance(meta, dict):
                raise ValueError("injection marker missing")
            if (
                not isinstance(meta.get("path"), str)
                or not isinstance(meta.get("sha256"), str)
                or len(meta["sha256"]) != 64
                or not isinstance(meta.get("size"), int)
                or meta["size"] <= 0
            ):
                raise ValueError("injection marker metadata is malformed")
            local = meta.get("local_path")
            if not isinstance(local, str):
                raise ValueError("injection marker was not independently recovered on Mac")
            path = Path(local)
            if (
                not path.resolve().is_relative_to(output.resolve())
                or _sha(path) != meta["sha256"]
                or path.stat().st_size != meta["size"]
            ):
                raise ValueError("injection local marker differs from evidence")
            value = _read(path)
            marker_values[marker] = value
            if value.get("request_id") != request_id:
                raise ValueError("injection local marker has foreign request identity")
            if marker in {"execution_started", "execution_done", "result"} and (
                value.get("unit_id") != unit_id or value.get("execution_count") != 1
            ):
                raise ValueError("injection duplicated execution or foreign unit")
            if marker == "done" and value.get("ok") is not True:
                raise ValueError("injected RPC did not complete successfully")
        if item["result"]["sha256"] != ledger_units[unit_id]["sha256"]:
            raise ValueError("injection committed result differs from verified ledger")
        if marker_values["execution_started"] != marker_values["execution_done"]:
            raise ValueError("injection execution markers disagree")
        code_hash = marker_values["started"].get("sha256")
        if (
            not isinstance(code_hash, str)
            or len(code_hash) != 64
            or marker_values["done"].get("sha256") != code_hash
        ):
            raise ValueError("injection RPC code identity differs")
        index = item.get("drop_call_index")
        if not isinstance(index, int) or index < 0:
            raise ValueError("injection drop call missing")
        injected_indices.add(index)
        calls = receipt.get("cli_calls")
        if not isinstance(calls, list) or index >= len(calls) or not isinstance(calls[index], dict):
            raise ValueError("injection dropped call is absent from raw receipt")
        raw = calls[index]
        if (
            raw.get("injected") is not True
            or raw.get("injection_kind") != "post_send_response_loss"
            or raw.get("unit_id") != unit_id
            or raw.get("request_id") != request_id
            or raw.get("status") != "failed"
            or raw.get("request_sha256") != code_hash
        ):
            raise ValueError("injection dropped call is not the deliberate tagged response loss")
        completed = [
            call
            for call in calls
            if call.get("command") == "exec"
            and call.get("request_id") == request_id
            and call.get("status") == "completed"
            and call.get("request_sha256") == code_hash
        ]
        if len(completed) < 2:
            raise ValueError("injection lacks recovered and completed-ID reissued calls")
    reissue = evidence.get("completed_reissue")
    if (
        not isinstance(reissue, dict)
        or reissue.get("request_id") not in request_ids
        or reissue.get("same_result_sha256") is not True
    ):
        raise ValueError("injection lacks completed-request reissue proof")
    injections = _injections(manifest)
    planned_ids = [entry["request_id"] for entry in injections["entries"]]
    if sorted(planned_ids) != sorted(request_ids):
        raise ValueError("injected units do not use the pre-dispatch request identities")
    faults, v2 = _raw_faults(receipt, injections)
    excluded = {row["index"] for row in v2["excluded_injections"] if "index" in row}
    if not injected_indices <= excluded or v2["unrealized_injections"]:
        raise ValueError("deliberate drops differ from the manifest-identified exclusions")
    elapsed = _finite(receipt.get("elapsed_seconds"), "injection elapsed")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or elapsed > manifest["timeout_minutes"] * 60
        or elapsed * manifest["budget"]["rate_units_per_hour"] / 3600
        > manifest["budget"]["unit_cap"]
    ):
        faults.append(
            {
                "source": "injection_operational_failure",
                "record": {
                    "status": receipt.get("status"),
                    "stop_verified": receipt.get("stop_verified"),
                    "elapsed_seconds": elapsed,
                },
            }
        )
    return {
        "session": job["job_id"],
        "platform": "cpu",
        "status": "FAIL" if faults else "PASS",
        "counted_faults": len(faults),
        "faults": faults,
        "raw_records": receipt,
        "injected_units": len(units),
        "no_duplicate": True,
        "v2": v2,
    }


def _remote_json(plan_path: Path, relative: str) -> dict[str, Any]:
    plan = _read(plan_path)
    repo = _path(plan_path.parent, plan.get("repo"), "repository")
    subprocess.run(["git", "fetch", "origin", BRANCH_REF], cwd=repo, check=True, timeout=60)
    raw = subprocess.check_output(
        ["git", "show", f"FETCH_HEAD:{relative}"], cwd=repo, text=True, timeout=30
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("remote publication is not an object")
    return value


def _verify_remote_publication(plan_path: Path, publication: dict[str, Any]) -> None:
    plan = _read(plan_path)
    if (
        publication.get("branch_ref") != plan.get("publication_ref")
        or publication.get("branch_ref") != BRANCH_REF
    ):
        raise ValueError("ledgers were not published to the branch-only ref")
    remote = plan.get("publication_ledger")
    if not isinstance(remote, str) or not remote:
        raise ValueError("plan lacks remote ledger publication index")
    index = _remote_json(plan_path, remote)
    if index.get("branch_ref") != BRANCH_REF or index.get("plan_sha256") != _sha(plan_path):
        raise ValueError("remote ledger index differs from qualification plan")
    if index != publication:
        raise ValueError("remote published ledgers differ from local publication")


def verdict(plan_path: Path, publication_path: Path, out: Path) -> dict[str, Any]:
    plan, jobs = _profile(plan_path)
    publication = _read(publication_path)
    if publication.get("schema") != PUBLICATION_SCHEMA or publication.get("plan_sha256") != _sha(
        plan_path
    ):
        raise ValueError("missing or foreign ledger publication")
    _verify_remote_publication(plan_path, publication)
    fake_error = None
    try:
        fake = plan.get("fake_proof")
        if not isinstance(fake, dict):
            raise ValueError("qualification lacks deterministic fake-CLI proof")
        fake_path = _path(plan_path.parent, fake.get("path"), "fake CLI proof")
        if _sha(fake_path) != fake.get("sha256"):
            raise ValueError("fake-CLI proof differs from plan binding")
        fake_value = _read(fake_path)
        if (
            fake_value.get("schema") != "rpc-qualification-fake-cli-proof/v1"
            or fake_value.get("status") != "PASS"
        ):
            raise ValueError("fake-CLI proof did not pass")
    except (ValueError, OSError) as error:
        fake_error = str(error)
    expected = {job["job_id"] for job in jobs}
    if set(publication.get("ledgers", {})) != expected:
        raise ValueError("all seven ledgers were not published before verdict")
    # Consume exactly one operational look after remote publication verification.
    with (plan_path.parent / "verdict-look-started.json").open("x") as handle:
        handle.write(_sha(publication_path) + "\n")

    def assess(job: dict[str, Any]) -> dict[str, Any]:
        _manifest_path, output, receipt_path, _runtime = _job_paths(plan_path, job)
        try:
            sealed = publication["ledgers"][job["job_id"]]
            if _sealed_evidence(job, output) != sealed.get("evidence"):
                raise ValueError("evidence changed after ledger publication")
            current = _ledger_inventory(job, output)
            if {k: v for k, v in sealed.items() if k != "evidence"} != current:
                raise ValueError("ledger changed after publication")
            return (_check_loop if job["mode"] == "loop" else _check_injection)(plan_path, job)
        except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
            try:
                receipt = _read(receipt_path) if receipt_path.is_file() else {}
            except (ValueError, OSError) as receipt_error:
                receipt = {"malformed_receipt": str(receipt_error)}
            try:
                try:
                    injections = _injections(_read(_manifest_path))
                except (ValueError, OSError, KeyError, TypeError):
                    injections = None  # stricter: nothing is excluded
                faults, _summary = _raw_faults(receipt, injections)
            except (ValueError, KeyError, TypeError) as census_error:
                faults = [
                    {"source": "missing_or_malformed_call_census", "record": str(census_error)}
                ]
            return {
                "session": job["job_id"],
                "platform": job["platform"],
                "status": "FAIL",
                "reason": str(error),
                "counted_faults": len(faults),
                "faults": faults,
                "raw_records": receipt,
                "no_duplicate": False if job["mode"] == "injection" else None,
            }

    def dispatched(job: dict[str, Any]) -> bool:
        return _job_paths(plan_path, job)[0].is_file()

    # Attempt 2 (judgment §4 step 4): the CPU loop series is optional and
    # all-or-none; L4 loops and the injection proof are required.
    cpu_loops = [job for job in jobs if job["mode"] == "loop" and job["platform"] == "cpu"]
    cpu_run = any(dispatched(job) for job in cpu_loops)
    sessions = [
        assess(job)
        for job in jobs
        if job["mode"] == "loop" and (job["platform"] == "l4" or cpu_run)
    ]
    injection = assess(next(job for job in jobs if job["mode"] == "injection"))
    by_platform: dict[str, list[dict[str, Any]]] = {platform: [] for platform in PLATFORMS}
    for session in sessions:
        by_platform[session["platform"]].append(session)
    result = {
        "schema": VERDICT_SCHEMA,
        "plan_sha256": _sha(plan_path),
        "one_look": True,
        "platforms": {},
        "injection": injection,
        "fake_proof": {"status": "FAIL" if fake_error else "PASS", "reason": fake_error},
    }
    for platform, rows in by_platform.items():
        if platform == "cpu" and not cpu_run:
            result["platforms"]["cpu"] = {
                "status": "NOT_RUN",
                "reason": "optional CPU loop series not dispatched",
                "injection_status": injection.get("status"),
            }
            continue
        faults = [fault for row in rows for fault in row["faults"]]
        if platform == "cpu":
            faults.extend(injection["faults"])
        result["platforms"][platform] = {
            "status": (
                "PASS"
                if not faults
                and fake_error is None
                and all(row.get("status") == "PASS" for row in rows)
                and (
                    injection.get("status") == "PASS"
                    if platform == "cpu"
                    else injection.get("no_duplicate") is True
                )
                else "FAIL"
            ),
            "sessions": rows,
            "counted_faults": len(faults),
            "faults": faults,
        }
    publish_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "verdict"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "verdict":
            command.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "publish":
        publish(args.plan, args.out)
    else:
        verdict(args.plan, args.publication, args.out)


if __name__ == "__main__":
    main()
