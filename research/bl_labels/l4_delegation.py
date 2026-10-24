"""Create-only Mac/L4 parent delegation for F0/F4 label production.

Authority: ``bl-labels-f0/JUDGMENT_L4_FAULT_DEFINITION_20261024.md`` §3.4, §4 step 2.
The Mac and L4 never race for a parent:

* ``delegation-NN.json`` hands a contiguous block of *tail* pending parents to L4;
* the Mac runner, only when started with a delegation directory, claims every
  parent at admission (create-only ``mac-admitted/<parent>.json``) under the same
  ``flock`` as delegation writes, and skips delegated parents;
* ``reclaim-NN.json`` returns only the uncommitted parents of a delegation, after
  the L4 session's verified stop and its verified import record exist.

Everything here reads sidecar-verified wrapper files and hashes only. No label
row, outcome or rescue count is opened.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exports import UNIT_DIR, _parent_life_ids, _session_life_ids, _verify, publish_json, unit_id

DELEGATION_SCHEMA = "bl-labels-l4-delegation/v1"
RECLAIM_SCHEMA = "bl-labels-l4-reclaim/v1"
ADMIT_SCHEMA = "bl-labels-mac-admit/v1"
IMPORT_SCHEMA = "bl-labels-l4-import/v1"
F1_RULER_PIN = "2af48cd0b98a2063f46871cf4d320a5541b789af"
MAX_L4_SHARE = 0.5
ADMIT_DIR = "mac-admitted"
IMPORT_DIR = "imports"
_RECORD = re.compile(r"^(delegation|reclaim)-(\d{2,})\.json$")


class DelegationRefused(ValueError):
    """A delegation, reclaim or claim would break the disjoint split."""


def allocation_sha256(allocation: list[dict[str, Any]]) -> str:
    payload = json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collector_commit(control: dict[str, Any]) -> str:
    """Return the collector source commit the Mac tranche runs; refuse the ruler pin."""
    gate = control.get("mac_gate") if isinstance(control.get("mac_gate"), dict) else {}
    commit = gate.get("collector_source_commit") or control.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise DelegationRefused("control does not bind a full collector source commit")
    if commit == F1_RULER_PIN:
        raise DelegationRefused("the F1 ruler pin 2af48cd0 is never a label collector pin")
    return commit


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    """Serialize delegation writes against Mac admission claims."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict[str, Any]:
    _verify(path)
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise DelegationRefused(f"malformed record {path}")
    return value


def read_records(
    directory: Path, namespace: str, allocation: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return verified ``(delegations, reclaims)`` keyed by two-digit id."""
    directory = Path(directory)
    digest = allocation_sha256(allocation)
    ids = {unit_id(row) for row in allocation}
    delegations: dict[str, dict[str, Any]] = {}
    reclaims: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return delegations, reclaims
    for path in sorted(directory.glob("*.json")):
        match = _RECORD.match(path.name)
        if not match:
            continue
        if not path.with_name(path.name + ".complete").is_file():
            continue  # create-only write still in flight under another process
        value = _read(path)
        if value.get("namespace") != namespace or value.get("allocation_sha256") != digest:
            raise DelegationRefused(f"foreign delegation record {path.name}")
        kind, number = match.groups()
        if kind == "delegation":
            if value.get("schema") != DELEGATION_SCHEMA or value.get("delegation_id") != number:
                raise DelegationRefused(f"malformed delegation {path.name}")
            if not set(value.get("parents", [])) <= ids:
                raise DelegationRefused(f"delegation {path.name} names foreign parents")
            delegations[number] = value
        else:
            if value.get("schema") != RECLAIM_SCHEMA or value.get("delegation_id") != number:
                raise DelegationRefused(f"malformed reclaim {path.name}")
            reclaims[number] = value
    for number, value in reclaims.items():
        if number not in delegations:
            raise DelegationRefused(f"reclaim-{number} has no delegation")
        if value.get("delegation_sha256") != file_sha256(directory / f"delegation-{number}.json"):
            raise DelegationRefused(f"reclaim-{number} binds different delegation bytes")
        if not set(value.get("reclaimed_parents", [])) <= set(delegations[number]["parents"]):
            raise DelegationRefused(f"reclaim-{number} returns parents it never delegated")
    return delegations, reclaims


def mac_excluded(directory: Path, namespace: str, allocation: list[dict[str, Any]]) -> set[str]:
    """Parents the Mac must not run: live delegations plus L4-committed parents."""
    delegations, reclaims = read_records(directory, namespace, allocation)
    excluded: set[str] = set()
    for number, value in delegations.items():
        parents = set(value["parents"])
        if number in reclaims:
            parents -= set(reclaims[number]["reclaimed_parents"])
        excluded |= parents
    return excluded


def live_delegated(directory: Path, namespace: str, allocation: list[dict[str, Any]]) -> set[str]:
    """Parents in delegations that have not been reclaimed yet."""
    delegations, reclaims = read_records(directory, namespace, allocation)
    return {
        parent
        for number, value in delegations.items()
        if number not in reclaims
        for parent in value["parents"]
    }


def admitted_parents(directory: Path) -> set[str]:
    folder = Path(directory) / ADMIT_DIR
    if not folder.is_dir():
        return set()
    return {
        path.stem
        for path in folder.glob("*.json")
        if path.with_name(path.name + ".complete").is_file()
    }


def scan_units(roots: list[Path]) -> dict[str, tuple[Path, str]]:
    """Return ``unit name -> (payload, sha256)`` for sidecar-complete exports only.

    The same unit in two Mac roots (resume ledger copy + attempt output) must have
    identical bytes; differing bytes are a C5 breach and refuse.
    """
    found: dict[str, tuple[Path, str]] = {}
    for root in roots:
        folder = Path(root) / UNIT_DIR
        if not folder.is_dir():
            continue
        for payload in sorted(folder.glob("*.json")):
            if not payload.with_name(payload.name + ".complete").is_file():
                continue
            digest = _verify(payload)
            previous = found.get(payload.stem)
            if previous is not None and previous[1] != digest:
                raise DelegationRefused(f"C5: unit {payload.stem} has two different payloads")
            found.setdefault(payload.stem, (payload, digest))
    return found


def _parent(name: str) -> str:
    return name.split("--life-", 1)[0].removesuffix("--session")


def committed_from_units(units: dict[str, tuple[Path, str]]) -> set[str]:
    """Parents whose terminal session and every death-life it names are present."""
    names = set(units)
    committed: set[str] = set()
    for name, (payload, _digest) in units.items():
        if name.endswith("--session"):
            parent = name.removesuffix("--session")
            if _session_life_ids(payload, parent) == _parent_life_ids(names, parent):
                committed.add(parent)
    return committed


def _next_id(directory: Path) -> str:
    numbers = [
        int(match.group(2))
        for path in Path(directory).glob("delegation-*.json")
        if (match := _RECORD.match(path.name))
    ]
    return f"{(max(numbers) + 1) if numbers else 1:02d}"


def plan_block(
    allocation: list[dict[str, Any]],
    *,
    unavailable: set[str],
    remaining_uncommitted: set[str],
    outstanding: set[str],
    count: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> list[int]:
    """Choose or validate a contiguous tail block of available allocation indices."""
    ids = [unit_id(row) for row in allocation]
    available = [index for index, parent in enumerate(ids) if parent not in unavailable]
    if (count is None) == (start is None or end is None):
        raise DelegationRefused("give exactly one of count or start/end")
    if count is not None:
        if count <= 0 or count > len(available):
            raise DelegationRefused("block size exceeds available pending parents")
        block = available[-count:]
    else:
        assert start is not None and end is not None
        block = list(range(start, end + 1))
        if not block or start < 0 or end >= len(ids):
            raise DelegationRefused("block is outside the allocation")
        if any(index not in set(available) for index in block):
            raise DelegationRefused(
                "block overlaps committed, started, admitted or delegated parents"
            )
    if block != list(range(block[0], block[-1] + 1)):
        raise DelegationRefused("block is not a contiguous allocation-index range")
    if any(index > block[-1] for index in available):
        raise DelegationRefused("block is not taken from the tail of pending parents")
    share = len(outstanding) + len(block)
    if share > MAX_L4_SHARE * len(remaining_uncommitted):
        raise DelegationRefused(
            f"L4 share {share}/{len(remaining_uncommitted)} exceeds 50% of remaining parents"
        )
    return block


def write_delegation(
    directory: Path,
    control: dict[str, Any],
    mac_roots: list[Path],
    *,
    count: int | None = None,
    start: int | None = None,
    end: int | None = None,
    sizing_note: str = "",
) -> Path:
    """Create the next ``delegation-NN.json`` under the admission lock."""
    namespace, allocation = control["namespace"], control["allocation"]
    commit = collector_commit(control)
    directory = Path(directory)
    with locked(directory):
        delegations, reclaims = read_records(directory, namespace, allocation)
        units = scan_units(mac_roots)
        committed = committed_from_units(units)
        started = {_parent(name) for name in units} | admitted_parents(directory)
        excluded = mac_excluded(directory, namespace, allocation)
        outstanding = live_delegated(directory, namespace, allocation)
        l4_committed = excluded - outstanding
        ids = [unit_id(row) for row in allocation]
        remaining = {p for p in ids if p not in committed and p not in l4_committed}
        block = plan_block(
            allocation,
            unavailable=committed | started | excluded,
            remaining_uncommitted=remaining,
            outstanding=outstanding,
            count=count,
            start=start,
            end=end,
        )
        number = _next_id(directory)
        record = {
            "schema": DELEGATION_SCHEMA,
            "delegation_id": number,
            "namespace": namespace,
            "allocation_sha256": allocation_sha256(allocation),
            "collector_source_commit": commit,
            "indices": [block[0], block[-1]],
            "parents": [ids[index] for index in block],
            "remaining_uncommitted_at_write": len(remaining),
            "outstanding_l4_at_write": len(outstanding),
            "prior_delegations": sorted(delegations),
            "prior_reclaims": sorted(reclaims),
            "sizing_note": sizing_note,
            "authority": "JUDGMENT_L4_FAULT_DEFINITION_20261024.md §3.4, §4 step 8",
        }
        path = directory / f"delegation-{number}.json"
        publish_json(path, record)
    return path


def write_reclaim(
    directory: Path,
    control: dict[str, Any],
    delegation_id: str,
    receipt: Path,
    import_record: Path,
) -> Path:
    """Return a delegation's uncommitted parents after verified stop + import."""
    namespace, allocation = control["namespace"], control["allocation"]
    directory = Path(directory)
    with locked(directory):
        delegations, reclaims = read_records(directory, namespace, allocation)
        if delegation_id not in delegations:
            raise DelegationRefused(f"no delegation-{delegation_id}")
        if delegation_id in reclaims:
            raise DelegationRefused(f"delegation-{delegation_id} is already reclaimed")
        stop = json.loads(Path(receipt).read_text())
        if stop.get("stop_verified") is not True or stop.get("cleanup_error") is not None:
            raise DelegationRefused("reclaim requires a verified stop and backend disappearance")
        imported = _read(Path(import_record))
        if (
            imported.get("schema") != IMPORT_SCHEMA
            or imported.get("delegation_id") != delegation_id
            or imported.get("receipt_sha256") != file_sha256(Path(receipt))
            or imported.get("namespace") != namespace
        ):
            raise DelegationRefused("import record does not bind this delegation and receipt")
        parents = delegations[delegation_id]["parents"]
        committed = set(imported.get("committed_parents", []))
        if not committed <= set(parents):
            raise DelegationRefused("import record commits parents outside the delegation")
        record = {
            "schema": RECLAIM_SCHEMA,
            "delegation_id": delegation_id,
            "namespace": namespace,
            "allocation_sha256": allocation_sha256(allocation),
            "delegation_sha256": file_sha256(directory / f"delegation-{delegation_id}.json"),
            "receipt": {"path": str(Path(receipt).resolve()), "sha256": file_sha256(receipt)},
            "import_record": {
                "path": str(Path(import_record).resolve()),
                "sha256": file_sha256(import_record),
            },
            "committed_parents": [p for p in parents if p in committed],
            "reclaimed_parents": [p for p in parents if p not in committed],
        }
        path = directory / f"reclaim-{delegation_id}.json"
        publish_json(path, record)
    return path


class MacDelegationGate:
    """Mac-runner hook; constructed only when a delegation directory is configured."""

    def __init__(self, directory: Path, namespace: str, allocation: list[dict[str, Any]]):
        self.directory = Path(directory)
        self.namespace = namespace
        self.allocation = allocation
        self.directory.mkdir(parents=True, exist_ok=True)
        read_records(self.directory, namespace, allocation)  # refuse foreign records early

    def claim(self, parent: str) -> bool:
        """At a parent boundary: False if delegated, else record a create-only claim."""
        with locked(self.directory):
            if parent in mac_excluded(self.directory, self.namespace, self.allocation):
                return False
            marker = self.directory / ADMIT_DIR / f"{parent}.json"
            if marker.exists():
                _verify(marker)  # resume of a parent the Mac already owns
                return True
            publish_json(
                marker,
                {"schema": ADMIT_SCHEMA, "namespace": self.namespace, "allocation_id": parent},
            )
            return True

    def admissible(self, parents: list[str]) -> set[str]:
        """Previously skipped parents the Mac may run now (reclaimed)."""
        with locked(self.directory):
            excluded = mac_excluded(self.directory, self.namespace, self.allocation)
        return {parent for parent in parents if parent not in excluded}
