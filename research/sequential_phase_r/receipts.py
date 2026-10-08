"""Create-only plan and look receipts of a sequential Phase R (stdlib + the decision module).

Layout under a study's sequential root (e.g. ``<artifacts>/<study>/phaseR/seq-v1``)::

    plan.json            the frozen plan dict, banks and study metadata (written once, before
                         any Phase R shard)
    looks/look-<k>.json  one receipt per analysed look: plan sha256, previous receipt sha256,
                         the exact record set (path + sha256), the deltas digest, the decision,
                         the action (CONTINUE / STOP / HALT)

Every file is written create-only (a fsynced temporary file hard-linked to its final name, so a
second writer fails and nothing is ever overwritten). :func:`look_gate` is what a shard (Mac or
pod) calls before running look ``k``: it returns the binding to record in the shard's start
marker and refuses unless look ``k - 1``'s receipt exists and says CONTINUE.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PLAN_SCHEMA = "sequential-phase-r-plan/v1"
RECEIPT_SCHEMA = "sequential-phase-r-look/v1"
PLAN_NAME = "plan.json"
LOOKS_DIR = "looks"


class ReceiptError(RuntimeError):
    """A receipt is missing, malformed, out of order or would be overwritten."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: Mapping[str, Any]) -> str:
    """Write ``payload`` as JSON create-only; returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise ReceiptError(f"{path} exists (create-only)") from exc
    finally:
        os.unlink(tmp)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError(f"cannot read {path}: {exc}") from exc


def receipt_path(root: Path, look: int) -> Path:
    return Path(root) / LOOKS_DIR / f"look-{int(look)}.json"


def write_plan(
    root: Path,
    plan_dict: Mapping[str, Any],
    banks: Mapping[str, Sequence[int]],
    study: Mapping[str, Any],
) -> str:
    """Freeze the plan before any Phase R shard. ``banks`` maps ``str(seed)`` to the ordered
    world seeds of that seed's bank (the same order in every mix); ``study`` carries the study
    id, commit, ratification sha256 and the pre-registration path + sha256."""
    if Path(root, LOOKS_DIR).exists() and any(Path(root, LOOKS_DIR).iterdir()):
        raise ReceiptError("look receipts already exist: the plan cannot be (re)written")
    n_worlds = int(plan_dict["n_worlds"])
    seeds = [str(s) for s in plan_dict["rule"]["seeds"]]
    if sorted(banks) != sorted(seeds):
        raise ReceiptError(f"banks must cover exactly the seeds {seeds}")
    for seed, bank in banks.items():
        if len(bank) != n_worlds or len(set(int(w) for w in bank)) != n_worlds:
            raise ReceiptError(f"bank of seed {seed} must hold {n_worlds} distinct worlds")
    payload = {
        "schema": PLAN_SCHEMA,
        "plan": dict(plan_dict),
        "plan_sha256": canonical_sha(dict(plan_dict)),
        "banks": {str(k): [int(w) for w in v] for k, v in sorted(banks.items())},
        "study": dict(study),
    }
    return write_once(Path(root) / PLAN_NAME, payload)


def read_plan(root: Path) -> tuple[dict[str, Any], str]:
    path = Path(root) / PLAN_NAME
    if not path.exists():
        raise ReceiptError(f"no plan at {path}")
    payload = load(path)
    if payload.get("schema") != PLAN_SCHEMA:
        raise ReceiptError("plan schema mismatch")
    if canonical_sha(payload["plan"]) != payload["plan_sha256"]:
        raise ReceiptError("plan sha256 does not match its content")
    return payload, sha256_file(path)


def look_gate(root: Path, look: int) -> dict[str, Any]:
    """Binding a look-``look`` shard must record before its first episode (refuses otherwise)."""
    payload, plan_file_sha = read_plan(root)
    n_looks = len(payload["plan"]["look_sizes"])
    if not 0 <= int(look) < n_looks:
        raise ReceiptError(f"look {look} is not a planned look (0..{n_looks - 1})")
    previous_sha: Optional[str] = None
    if int(look) > 0:
        prev = receipt_path(root, int(look) - 1)
        if not prev.exists():
            raise ReceiptError(f"look {look - 1} has no receipt: look {look} may not start")
        receipt = load(prev)
        if receipt.get("action") != "CONTINUE" or not receipt.get("decision", {}).get("valid"):
            raise ReceiptError(
                f"look {look - 1} ended the study ({receipt.get('decision', {}).get('status')}, "
                f"{receipt.get('action')}): look {look} may not start"
            )
        previous_sha = sha256_file(prev)
    if receipt_path(root, int(look)).exists():
        raise ReceiptError(f"look {look} is already analysed")
    lo = 0 if int(look) == 0 else int(payload["plan"]["look_sizes"][int(look) - 1])
    hi = int(payload["plan"]["look_sizes"][int(look)])
    return {
        "look": int(look),
        "plan_file_sha256": plan_file_sha,
        "plan_sha256": payload["plan_sha256"],
        "previous_receipt_sha256": previous_sha,
        "world_index_range": [lo, hi],
    }


def write_look_receipt(
    root: Path,
    look: int,
    decision: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    deltas_sha256: str,
    flags: Mapping[str, bool],
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Create-only ``looks/look-<k>.json``. ``records`` are ``{"key": [...], "path", "sha256"}``
    rows of exactly the look's record set."""
    payload_plan, plan_file_sha = read_plan(root)
    if decision.get("plan_sha256") != payload_plan["plan_sha256"]:
        raise ReceiptError("the decision was computed under a different plan")
    if int(decision.get("look", -1)) != int(look):
        raise ReceiptError("decision look index mismatch")
    if not decision.get("valid") and (
        decision.get("status") != "INVALID_ANALYSIS" or decision.get("action") != "HALT"
    ):
        raise ReceiptError("an invalid decision must be INVALID_ANALYSIS / HALT")
    previous_sha = None
    if int(look) > 0:
        prev = receipt_path(root, int(look) - 1)
        if not prev.exists():
            raise ReceiptError(f"look {look - 1} has no receipt")
        if load(prev).get("action") != "CONTINUE":
            raise ReceiptError(f"look {look - 1} ended the study: no look {look}")
        previous_sha = sha256_file(prev)
    rows = sorted(
        (
            {"key": list(r["key"]), "path": str(r["path"]), "sha256": str(r["sha256"])}
            for r in records
        ),
        key=lambda r: canonical(r["key"]),
    )
    payload = {
        "schema": RECEIPT_SCHEMA,
        "look": int(look),
        "plan_sha256": payload_plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha,
        "previous_receipt_sha256": previous_sha,
        "n_per_seed_mix": int(payload_plan["plan"]["look_sizes"][int(look)]),
        "records": rows,
        "records_sha256": canonical_sha(rows),
        "deltas_sha256": deltas_sha256,
        "flags": {str(k): bool(v) for k, v in sorted(flags.items())},
        "decision": json.loads(canonical(dict(decision))),
        "action": decision["action"],
        "extra": dict(extra or {}),
    }
    return write_once(receipt_path(root, look), payload)


def last_receipt(root: Path) -> Optional[dict[str, Any]]:
    """The latest look receipt (``None`` before look 0 is analysed)."""
    found = sorted(
        Path(root, LOOKS_DIR).glob("look-*.json"), key=lambda p: int(p.stem.split("-")[1])
    )
    return load(found[-1]) if found else None
