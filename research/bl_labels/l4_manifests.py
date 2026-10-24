"""L4 label-session manifests over a delegated block, and the step-7 identity packet.

Authority: ``bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md`` §3.6–§3.7,
§4 steps 2, 7, 8. Prepares files only; dispatch goes through
``tools/ops/dispatch.sh`` with a broker qualified under ``l4-label-faults/v2``.

Session control = the Mac D4096 namespace plus exactly the delegated allocation
items (byte-identical dicts, allocation order). ``collect_unit`` reads only the
item, so L4 units are the same bytes the Mac would publish for those parents.
The Colab entrypoint is the existing F0 one (``research/bl_labels/colab.sh`` →
``runtime.py``) at the collector commit; nothing in the runtime is changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .exports import UNIT_DIR, _verify, empty_ledger, publish_json, unit_id
from .l4_delegation import (
    F1_RULER_PIN,
    DelegationRefused,
    allocation_sha256,
    collector_commit,
    committed_from_units,
    read_records,
    scan_units,
)
from .l4_import import QUARANTINE_JOB_PREFIX, QUARANTINE_MARKER

L4_RATE_UNITS_PER_HOUR = 1.54  # research/bl_labels/f1_manifests.py
L4_WORKERS = 12
IDENTITY_PARENTS = 36
IDENTITY_SALT = "l4-payload-identity-v1"
IDENTITY_SCHEMA = "bl-labels-l4-identity-packet/v1"
COMPARE_SCHEMA = "bl-labels-l4-identity-compare/v1"


def _require_commit(commit: str, control: dict[str, Any]) -> None:
    if commit == F1_RULER_PIN:
        raise DelegationRefused("the F1 ruler pin 2af48cd0 is never a label collector pin")
    if commit != collector_commit(control):
        raise DelegationRefused("L4 must run the identical collector commit as the Mac tranche")


def envelope(base_minutes: float) -> dict[str, float]:
    """Owner 2x applied once to the whole session; units follow the L4 rate."""
    base_units = round(base_minutes / 60 * L4_RATE_UNITS_PER_HOUR * 1.2, 3)
    return {
        "base_minutes": base_minutes,
        "effective_minutes": 2 * base_minutes,
        "base_units": base_units,
        "effective_units": 2 * base_units,
        "effective_reserve_units": round(0.2 * 2 * base_units, 3),
        "effective_finalize_seconds": 2 * 270,
    }


def build_manifest(
    *,
    commit: str,
    job_id: str,
    output_dir: Path,
    control_path: Path,
    ledger_path: Path,
    b8: Path,
    frp3: Path,
    base_minutes: float,
    claim: str,
    online: bool = True,
) -> dict[str, Any]:
    """Reuse the F0 CPU-canary manifest shape (same entrypoint), retargeted to L4 W12."""
    from . import manifests as f0

    output_dir = Path(output_dir).resolve()
    sizes = envelope(base_minutes)
    if online:
        f0.require_source_pin(commit)
        qualified = json.loads(f0.QUALIFIED_RECORD.read_text())
        qualified_transport = {
            "record": str(f0.QUALIFIED_RECORD),
            "sha256": f0._sha(f0.QUALIFIED_RECORD),
            "commit": qualified.get("qualified_commit"),
            "tree_hash": qualified.get("tree_hash"),
        }
        pool = f0._pool_inputs(b8, frp3)
        source = f0.source_paths()
        source_digest = f0.source_sha256()
    else:  # offline fixtures only; never dispatchable
        qualified_transport = {"record": "OFFLINE_FIXTURE"}
        pool, source, source_digest = [], [], "OFFLINE_FIXTURE"
    value: dict[str, Any] = {
        "job_id": job_id,
        "commit": commit,
        "script": f0.SCRIPT,
        "claim": claim,
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{f0.PACKAGE}/requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "F0_NAMESPACE": json.loads(Path(control_path).read_text())["namespace"],
            "F0_COMMIT": commit,
            "F0_SOURCE_SHA256": source_digest,
            "F0_WORKERS": str(L4_WORKERS),
            "F0_DEVICE": "cpu",
            "F0_SESSION_SECONDS": str(int(sizes["effective_minutes"] * 60)),
            "F0_FINALIZE_SECONDS": str(sizes["effective_finalize_seconds"]),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": L4_WORKERS,
            "est_minutes": base_minutes,
        },
        "timeout_minutes": sizes["effective_minutes"],
        "output_dir": str(output_dir),
        "inputs": [
            f0.input_row(Path(control_path), "inputs/control.json"),
            *f0.ledger_inputs(Path(ledger_path)),
            *pool,
        ],
        "budget": {
            "unit_cap": sizes["effective_units"],
            "rate_units_per_hour": L4_RATE_UNITS_PER_HOUR,
            "reserve_units": sizes["effective_reserve_units"],
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 1024**3,
        "source_paths": source,
        "required_exports": ["runtime-result.json"],
        "qualified_transport": qualified_transport,
        "resume": {
            "unit_kind": "life",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 0,
        },
        "l4_lane": {
            "fault_definition": "l4-label-faults/v2",
            "authority": "JUDGMENT_L4_FAULT_DEFINITION_20261024.md",
            "envelope": sizes,
            "collector_source_commit": commit,
        },
    }
    if online:
        from research.bl_upload import assert_upload_budget, convert_public_checkpoints

        value = convert_public_checkpoints(value, repo=f0.ROOT)
        assert_upload_budget(value, repo=f0.ROOT)
    return value


def prepare_session(
    delegation_dir: Path,
    control: dict[str, Any],
    delegation_id: str,
    session_dir: Path,
    *,
    commit: str,
    b8: Path,
    frp3: Path,
    base_minutes: float = 120.0,
    broker: Path | None = None,
    online: bool = True,
) -> Path:
    """Fresh namespace/state/output for one L4 production session over one delegation."""
    _require_commit(commit, control)
    namespace, allocation = control["namespace"], control["allocation"]
    delegations, reclaims = read_records(delegation_dir, namespace, allocation)
    if delegation_id not in delegations or delegation_id in reclaims:
        raise DelegationRefused(f"delegation-{delegation_id} is absent or reclaimed")
    record = delegations[delegation_id]
    if record["collector_source_commit"] != commit:
        raise DelegationRefused("delegation binds a different collector commit")
    session_dir = Path(session_dir)
    if session_dir.exists() and any(session_dir.iterdir()):
        raise DelegationRefused("L4 session directory must be fresh")
    by_id = {unit_id(row): row for row in allocation}
    sub = [by_id[parent] for parent in record["parents"]]
    control_path = session_dir / "control.json"
    publish_json(
        control_path,
        {
            "schema": "bl-labels-f0-control/v1",
            "namespace": namespace,
            "allocation": sub,
            "workers": L4_WORKERS,
        },
    )
    ledger_path = session_dir / "verified-ledger.json"
    publish_json(ledger_path, empty_ledger(namespace, sub))
    job_id = f"bl-l4-d4096-d{delegation_id}-{commit[:8]}-{session_dir.name}"
    manifest = build_manifest(
        commit=commit,
        job_id=job_id,
        output_dir=session_dir / "output",
        control_path=control_path,
        ledger_path=ledger_path,
        b8=b8,
        frp3=frp3,
        base_minutes=base_minutes,
        claim=(
            f"L4 D4096 production over delegation-{delegation_id}; "
            "v2 faults; hash-verified import"
        ),
        online=online,
    )
    manifest["l4_lane"]["delegation"] = {
        "id": delegation_id,
        "sha256": hashlib.sha256(
            (Path(delegation_dir) / f"delegation-{delegation_id}.json").read_bytes()
        ).hexdigest(),
        "allocation_sha256": allocation_sha256(allocation),
    }
    manifest_path = session_dir / "manifest.json"
    publish_json(manifest_path, manifest)
    _write_dispatch(session_dir, job_id, manifest_path, broker)
    return manifest_path


def _write_dispatch(session_dir: Path, name: str, manifest: Path, broker: Path | None) -> None:
    broker_text = str(broker) if broker else "<v2-qualified broker worktree>"
    text = (
        "# Prepared only. Dispatch exactly once after its approval/review; no automatic retry.\n"
        f'/Users/josenunez/Projects/ml/snake-dqn/tools/ops/dispatch.sh "{name}" "{manifest}" \\\n'
        f'  --broker "{broker_text}" --cwd "{broker_text}" '
        f'--state-dir "{session_dir}/state" --dispatch-txt "{session_dir}/DISPATCH.txt"\n'
    )
    (session_dir / "DISPATCH.txt").write_text(text)


def identity_rank(namespace: str, parent: str) -> str:
    """Documented outcome-blind selection key: sha256('l4-payload-identity-v1/<ns>/<id>')."""
    return hashlib.sha256(f"{IDENTITY_SALT}/{namespace}/{parent}".encode()).hexdigest()


def select_identity_parents(
    control: dict[str, Any], mac_roots: list[Path], count: int = IDENTITY_PARENTS
) -> list[str]:
    """The ``count`` Mac-committed parents with the smallest rank, in allocation order."""
    namespace, allocation = control["namespace"], control["allocation"]
    committed = committed_from_units(scan_units(mac_roots))
    if len(committed) < count:
        raise DelegationRefused(f"only {len(committed)} Mac-committed parents; need {count}")
    chosen = set(sorted(committed, key=lambda parent: identity_rank(namespace, parent))[:count])
    return [unit_id(row) for row in allocation if unit_id(row) in chosen]


def prepare_identity_packet(
    control: dict[str, Any],
    mac_roots: list[Path],
    packet_dir: Path,
    *,
    commit: str,
    b8: Path,
    frp3: Path,
    base_minutes: float = 120.0,
    broker: Path | None = None,
    online: bool = True,
) -> Path:
    """Quarantined L4 re-run of 36 Mac-committed parents; hash-only expectations."""
    _require_commit(commit, control)
    namespace, allocation = control["namespace"], control["allocation"]
    packet_dir = Path(packet_dir)
    if packet_dir.exists() and any(packet_dir.iterdir()):
        raise DelegationRefused("identity packet directory must be fresh")
    parents = select_identity_parents(control, mac_roots)
    units = scan_units(mac_roots)
    expected = {
        name: digest
        for name, (_path, digest) in sorted(units.items())
        if name.split("--life-", 1)[0].removesuffix("--session") in set(parents)
    }
    publish_json(
        packet_dir / QUARANTINE_MARKER,
        {
            "schema": "bl-labels-l4-quarantine/v1",
            "reason": "step-7 payload-identity canary; never imported into any dose",
        },
    )
    by_id = {unit_id(row): row for row in allocation}
    control_path = packet_dir / "control.json"
    publish_json(
        control_path,
        {
            "schema": "bl-labels-f0-control/v1",
            # Same F0 namespace: unit bytes embed it, so identity requires it.
            # Quarantine is enforced by the packet marker and job prefix instead.
            "namespace": namespace,
            "allocation": [by_id[parent] for parent in parents],
            "workers": L4_WORKERS,
        },
    )
    ledger_path = packet_dir / "verified-ledger.json"
    publish_json(ledger_path, empty_ledger(namespace, [by_id[p] for p in parents]))
    publish_json(
        packet_dir / "mac-expected.json",
        {
            "schema": IDENTITY_SCHEMA,
            "namespace": namespace,
            "allocation_sha256": allocation_sha256(allocation),
            "collector_source_commit": commit,
            "selection": f"36 smallest sha256('{IDENTITY_SALT}/<namespace>/<allocation_id>') "
            "among Mac-committed parents",
            "parents": parents,
            "mac_unit_sha256": expected,
            "mac_roots": [str(Path(root).resolve()) for root in mac_roots],
        },
    )
    job_id = f"{QUARANTINE_JOB_PREFIX}{commit[:8]}-{packet_dir.name}"
    manifest = build_manifest(
        commit=commit,
        job_id=job_id,
        output_dir=packet_dir / "output",
        control_path=control_path,
        ledger_path=ledger_path,
        b8=b8,
        frp3=frp3,
        base_minutes=base_minutes,
        claim="Step-7 L4 payload-identity canary (quarantined; hash-only; never imported)",
        online=online,
    )
    manifest["l4_lane"]["quarantine"] = True
    manifest_path = packet_dir / "manifest.json"
    publish_json(manifest_path, manifest)
    _write_dispatch(packet_dir, job_id, manifest_path, broker)
    return manifest_path


def compare_identity(packet_dir: Path, l4_output: Path, b8: Path, frp3: Path) -> dict[str, Any]:
    """Per-life sha256 L4 vs Mac; never opens payload contents."""
    packet_dir = Path(packet_dir)
    expected_path = packet_dir / "mac-expected.json"
    _verify(expected_path)
    expected = json.loads(expected_path.read_text())
    observed = {name: digest for name, (_p, digest) in scan_units([Path(l4_output)]).items()}
    mac = expected["mac_unit_sha256"]
    identical = sorted(n for n in mac if observed.get(n) == mac[n])
    mismatch = sorted(n for n in mac if n in observed and observed[n] != mac[n])
    missing = sorted(n for n in mac if n not in observed)
    extra = sorted(n for n in observed if n not in mac)
    verdict = "MISMATCH" if mismatch or extra else "INCOMPLETE" if missing else "IDENTICAL_ALL"
    audit_command = None
    if mismatch:
        units = " ".join(
            f'--unit "{Path(l4_output).resolve() / UNIT_DIR / (n + ".json")}"' for n in mismatch
        )
        audit_command = (
            "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m "
            f"research.bl_labels.replay_audit {units} "
            f'--out "{packet_dir / "replay-audit.json"}" --b8 "{b8}" --frp3 "{frp3}"'
        )
    result = {
        "schema": COMPARE_SCHEMA,
        "verdict": verdict,
        "identical": len(identical),
        "mismatch": mismatch,
        "missing_l4": missing,
        "extra_l4": extra,
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        "replay_audit_command": audit_command,
        "rule": "any MISMATCH -> Mac replay_audit; every divergence a classified float tie, "
        "else X86_LABELS_NOT_MIXABLE and the L4 lane closes",
    }
    publish_json(packet_dir / "identity-compare.json", result)
    return result
