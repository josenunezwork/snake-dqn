"""Verified import of L4 label units and the Mac + L4 union-ledger close.

Authority: ``bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md`` §2.2 (C5),
§3.1–§3.5, §4 steps 2 and 8. Hash-only: payloads are hashed and hard-linked,
never parsed, except the terminal session wrapper's death count (the same field
the factory's own resume path reads) to decide whether a parent is committed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .exports import (
    LEDGER_SCHEMA,
    UNIT_DIR,
    _verify,
    _write_once,
    empty_ledger,
    publish_json,
    unit_id,
)
from .l4_delegation import (
    IMPORT_DIR,
    IMPORT_SCHEMA,
    DelegationRefused,
    allocation_sha256,
    committed_from_units,
    file_sha256,
    read_records,
    scan_units,
)

QUARANTINE_MARKER = "QUARANTINE.json"
QUARANTINE_JOB_PREFIX = "bl-l4-identity-quarantine-"
UNION_SCHEMA = "bl-labels-f0-union-close/v1"
TRANSPORT_CLASSES = frozenset({"C1", "C2", "C3", "C6"})
INTEGRITY_CLASSES = frozenset({"C4", "C5"})


class C5Breach(ValueError):
    """Unit-integrity breach (judgment §2.2 C5): the tranche is INVALID."""


def _parent(name: str) -> str:
    return name.split("--life-", 1)[0].removesuffix("--session")


def quarantined(path: Path, job_id: str | None = None) -> bool:
    """True for the step-7 identity namespace, which is never importable."""
    if job_id and job_id.startswith(QUARANTINE_JOB_PREFIX):
        return True
    path = Path(path).resolve()
    return any((folder / QUARANTINE_MARKER).exists() for folder in (path, *path.parents))


def _default_classifier() -> Callable[..., dict[str, Any]]:
    from research.bl_labels.l4_faults_v2 import classify_session

    return classify_session


def _link_triplet(source: Path, target: Path) -> None:
    """Create-only hard link (copy across devices) of a verified export triplet."""
    target.parent.mkdir(parents=True, exist_ok=True)
    for old, new in (
        (source, target),
        (source.with_name(source.name + ".sha256"), target.with_name(target.name + ".sha256")),
        (source.with_name(source.name + ".complete"), target.with_name(target.name + ".complete")),
    ):
        try:
            os.link(old, new)
        except FileExistsError as error:
            raise C5Breach(f"C5: create-only target already exists: {new}") from error
        except OSError:
            _write_once(new, old.read_bytes())
    _verify(target)


def import_session(
    delegation_dir: Path,
    control: dict[str, Any],
    delegation_id: str,
    session_control: Path,
    session_output: Path,
    receipt: Path,
    mac_roots: list[Path],
    union_root: Path,
    *,
    injection_manifest: dict[str, Any] | None = None,
    allow_transport_fault_import: bool = False,
    classifier: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    """Verify one stopped L4 session and import its committed parents create-only.

    Refuses: quarantine, foreign namespace/allocation/collector, unverified stop,
    any C4/C5 (INVALID), any runtime-vs-Mac hash mismatch or duplicate unit (C5).
    A transport-class v2 FAIL (C1/C2/C3/C6) refuses unless the operator passes
    ``allow_transport_fault_import`` (judgment §4 step 8: stop, import committed
    units, reclaim); the record then says so and the fault counts toward closing
    the lane.
    """
    namespace, allocation = control["namespace"], control["allocation"]
    session_output, receipt, session_control = map(Path, (session_output, receipt, session_control))
    if quarantined(session_output) or quarantined(session_control):
        raise DelegationRefused("quarantined identity sessions are never importable")
    receipt_value = json.loads(receipt.read_text())
    manifest = (
        receipt_value.get("manifest") if isinstance(receipt_value.get("manifest"), dict) else {}
    )
    job_id = receipt_value.get("job_id") or manifest.get("job_id")
    if quarantined(session_output, job_id) or quarantined(session_control, job_id):
        raise DelegationRefused("quarantined identity sessions are never importable")
    delegations, reclaims = read_records(delegation_dir, namespace, allocation)
    if delegation_id not in delegations:
        raise DelegationRefused(f"no delegation-{delegation_id}")
    if delegation_id in reclaims:
        raise DelegationRefused(f"delegation-{delegation_id} is already reclaimed")
    delegation = delegations[delegation_id]
    delegated = list(delegation["parents"])
    by_id = {unit_id(row): row for row in allocation}
    order = {unit_id(row): index for index, row in enumerate(allocation)}
    _verify(session_control)
    l4_control = json.loads(session_control.read_text())
    if l4_control.get("namespace") != namespace or l4_control.get("allocation") != [
        by_id[parent] for parent in delegated
    ]:
        raise DelegationRefused("L4 session control is not exactly this delegated block")
    if (
        receipt_value.get("stop_verified") is not True
        or receipt_value.get("cleanup_error") is not None
    ):
        raise DelegationRefused("import requires a verified stop and backend disappearance")

    classify = classifier or _default_classifier()
    faults = classify(receipt_value, injection_manifest=injection_manifest)
    if faults.get("schema") != "l4-label-faults/v2":
        raise DelegationRefused("transport classification is not l4-label-faults/v2")
    classes = {str(row.get("class")) for row in faults.get("counted", [])}
    if classes & INTEGRITY_CLASSES:
        raise C5Breach(f"INVALID: v2 integrity classes {sorted(classes & INTEGRITY_CLASSES)}")
    transport_fault = faults.get("verdict") != "PASS"
    if transport_fault and not allow_transport_fault_import:
        raise DelegationRefused(f"v2 verdict {faults.get('verdict')}: {sorted(classes)}")
    if transport_fault and not classes <= TRANSPORT_CLASSES:
        raise DelegationRefused("v2 FAIL without a recognised transport class")

    for marker in sorted(session_output.glob("INVALID-*.json")):
        raise C5Breach(f"INVALID: session exported {marker.name}")
    runtime_path = session_output / "runtime-result.json"
    _verify(runtime_path)
    runtime = json.loads(runtime_path.read_text())
    diagnostics = runtime.get("operational_diagnostics", {})
    if (
        runtime.get("namespace") != namespace
        or runtime.get("result_class") != "SCIENCE"
        or runtime.get("status") == "INVALID"
    ):
        raise DelegationRefused("L4 runtime result is foreign, a probe, or INVALID")
    if diagnostics.get("code_sha") != delegation["collector_source_commit"]:
        raise DelegationRefused("L4 runtime code sha differs from the delegated collector pin")

    # Every sidecar-complete unit: runtime sha recorded by the runtime's .sha256,
    # re-verified here on the Mac (exports._verify recomputes the bytes).
    units: dict[str, tuple[Path, str]] = {}
    partial: list[str] = []
    folder = session_output / UNIT_DIR
    for payload in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        name = payload.stem
        sidecars = [payload.with_name(payload.name + s) for s in (".sha256", ".complete")]
        if not all(item.is_file() for item in sidecars):
            partial.append(name)
            continue
        if _parent(name) not in delegated:
            raise C5Breach(f"C5: L4 unit {name} is outside delegation-{delegation_id}")
        try:
            units[name] = (payload, _verify(payload))
        except ValueError as error:
            raise C5Breach(f"C5: runtime sha256 != Mac sha256 for {name}") from error
    mac_units = scan_units(mac_roots)
    union_units = scan_units(sorted((Path(union_root) / "l4").glob("*")))
    for name in units:
        if name in mac_units or name in union_units:
            raise C5Breach(f"C5: unit {name} committed twice (never silently ignored)")
    committed = committed_from_units(units)
    session_tag = f"d{delegation_id}-{file_sha256(receipt)[:12]}"
    target_root = Path(union_root) / "l4" / session_tag
    if target_root.exists():
        raise C5Breach(f"C5: import target {target_root} already exists")
    rows = []
    for name, (payload, digest) in sorted(
        units.items(), key=lambda kv: (order[_parent(kv[0])], kv[0])
    ):
        if _parent(name) not in committed:
            continue
        target = target_root / UNIT_DIR / payload.name
        _link_triplet(payload, target)
        mac_digest = _verify(target)
        if mac_digest != digest:
            raise C5Breach(f"C5: imported copy of {name} differs")
        rows.append(
            {
                "unit_id": name,
                "parent": _parent(name),
                "allocation_index": order[_parent(name)],
                "runtime_sha256": digest,
                "mac_sha256": mac_digest,
                "size": target.stat().st_size,
                "path": str(target.relative_to(union_root)),
            }
        )
    record = {
        "schema": IMPORT_SCHEMA,
        "namespace": namespace,
        "allocation_sha256": allocation_sha256(allocation),
        "delegation_id": delegation_id,
        "delegation_sha256": file_sha256(Path(delegation_dir) / f"delegation-{delegation_id}.json"),
        "receipt_sha256": file_sha256(receipt),
        "runtime_result_sha256": file_sha256(runtime_path),
        "session_control_sha256": file_sha256(session_control),
        "session_tag": session_tag,
        "collector_source_commit": delegation["collector_source_commit"],
        "platform": {
            "platform": diagnostics.get("platform"),
            "code_sha": diagnostics.get("code_sha"),
            "cpu_model": diagnostics.get("cpu_model"),
            "nproc": diagnostics.get("nproc"),
            "kernel_attestation": diagnostics.get("kernel_attestation")
            or "NOT_EXPORTED_BY_COLLECTOR_COMMIT",
            "label": "colab-l4-x86_64",
        },
        "faults_v2": faults,
        "status": "IMPORTED_AFTER_TRANSPORT_FAULT" if transport_fault else "IMPORTED",
        "committed_parents": [p for p in delegated if p in committed],
        "uncommitted_parents": [p for p in delegated if p not in committed],
        "uncommitted_units_not_imported": sorted(
            name for name in units if _parent(name) not in committed
        ),
        "partial_sidecar_units_ignored": sorted(partial),
        "units": rows,
    }
    out = Path(delegation_dir) / IMPORT_DIR / f"import-{session_tag}.json"
    publish_json(out, record)
    return out


def contiguous_prefix(allocation: list[dict[str, Any]], committed: set[str]) -> int:
    """Length of the longest fully committed allocation-index prefix (F4 nested doses)."""
    length = 0
    for row in allocation:
        if unit_id(row) not in committed:
            break
        length += 1
    return length


def union_close(
    control: dict[str, Any],
    mac_roots: list[Path],
    delegation_dir: Path,
    union_root: Path,
    out_dir: Path,
    *,
    declared_pending: list[str] | None = None,
    funding_receipt: Path | None = None,
) -> dict[str, Any]:
    """Close the Mac + L4 union: exactly one committed copy per unit, prefix dose ledger."""
    namespace, allocation = control["namespace"], control["allocation"]
    d4096 = control.get("mac_gate", {}).get("stage") == "d4096" or namespace.endswith("-mac-d4096")
    if d4096:
        from . import d4096_resume, d4096_sealed, mac_gates

        d4096_sealed.require_not_discarded(None)
        d4096_sealed.require_funding_receipt(funding_receipt)
        mac_gates.require_admission(control)
        pack = Path(control["mac_gate"]["package_root"])
        attempts, _elapsed = d4096_resume.attempt_chain(pack, control, final=True)
        if [path.resolve() for path in mac_roots] != [path.resolve() for path in attempts]:
            raise DelegationRefused("D4096 union must charge exactly every verified Mac attempt")
    ids = [unit_id(row) for row in allocation]
    breaches: list[str] = []
    mac_units = scan_units(mac_roots)
    l4_units: dict[str, tuple[Path, str]] = {}
    imports = sorted((Path(delegation_dir) / IMPORT_DIR).glob("import-*.json"))
    for path in imports:
        _verify(path)
        record = json.loads(path.read_text())
        if record.get("schema") != IMPORT_SCHEMA or record.get("namespace") != namespace:
            raise DelegationRefused(f"foreign import record {path.name}")
        for row in record["units"]:
            payload = Path(union_root) / row["path"]
            digest = _verify(payload)
            if digest != row["mac_sha256"] or digest != row["runtime_sha256"]:
                breaches.append(f"C5 hash: {row['unit_id']}")
            if row["unit_id"] in l4_units:
                breaches.append(f"C5 duplicate across L4 sessions: {row['unit_id']}")
            l4_units[row["unit_id"]] = (payload, digest)
    for name in sorted(set(mac_units) & set(l4_units)):
        breaches.append(f"C5 duplicate Mac+L4: {name}")
    for name in sorted(set(mac_units) | set(l4_units)):
        if _parent(name) not in ids:
            breaches.append(f"C5 foreign unit: {name}")
    mac_parents = {_parent(n) for n in mac_units}
    l4_parents = {_parent(n) for n in l4_units}
    for parent in sorted(mac_parents & l4_parents):
        breaches.append(f"C5 parent split across platforms: {parent}")
    union = {**mac_units, **l4_units}
    committed = committed_from_units(union)
    pending = [p for p in ids if p not in committed]
    if d4096 and pending:
        raise DelegationRefused("D4096 final union cannot publish while any parent is pending")
    if declared_pending is not None and sorted(declared_pending) != sorted(pending):
        breaches.append("C5 ledger close: allocated != committed ∪ declared-pending")
    prefix = contiguous_prefix(allocation, committed)
    prefix_parents = set(ids[:prefix])
    result: dict[str, Any] = {
        "schema": UNION_SCHEMA,
        "namespace": namespace,
        "allocation_sha256": allocation_sha256(allocation),
        "verdict": "INVALID" if breaches else "CLEAN",
        "c5_breaches": breaches,
        "allocated": len(ids),
        "committed_parents": len(committed),
        "declared_pending": pending,
        "prefix_length": prefix,
        "unused_beyond_prefix_l4_parents": sorted(
            (p for p in committed & l4_parents if p not in prefix_parents), key=ids.index
        ),
        "beyond_prefix_mac_parents": sorted(
            (p for p in committed & mac_parents if p not in prefix_parents), key=ids.index
        ),
        "platform_by_parent_counts": {
            "darwin-arm64": len(committed & mac_parents & prefix_parents),
            "colab-l4-x86_64": len(committed & l4_parents & prefix_parents),
        },
        "import_records": {p.name: file_sha256(p) for p in imports},
    }
    if d4096:
        result.update(
            mac_attempt_exports=[str(path.resolve()) for path in attempts],
            cumulative_mac_elapsed_seconds=_elapsed,
            funding_receipt_sha256=_verify(funding_receipt),
        )
    out_dir = Path(out_dir)
    if not breaches:
        rows = []
        for name in sorted(union, key=lambda n: (ids.index(_parent(n)), n)):
            if _parent(name) not in prefix_parents:
                continue
            payload, digest = union[name]
            _link_triplet(payload, out_dir / UNIT_DIR / payload.name)
            rows.append(
                {
                    "unit_id": name,
                    "path": f"{UNIT_DIR}/{payload.name}",
                    "sha256": digest,
                    "size": payload.stat().st_size,
                }
            )
        ledger = empty_ledger(namespace, allocation)
        ledger["units"] = rows
        assert ledger["schema"] == LEDGER_SCHEMA
        publish_json(out_dir / "verified-ledger.json", ledger)
        result["prefix_ledger_sha256"] = file_sha256(out_dir / "verified-ledger.json")
    publish_json(out_dir / "union-close.json", result)
    return result
