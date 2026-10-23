"""Frozen-pin F0 speed equivalence and profile harness.

This module is deliberately outside the factory's science payloads.  It runs
each pin in a separate worktree, compares only sealed exports, and writes a
create-only receipt.  It never prints label rows or outcome summaries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .exports import _verify, publish_json

SCHEMA = "bl-labels-f0-speed-equiv/v1"
OLD_PIN = "61900af28932e47e22e332840fc4406559adb062"
NAMESPACE = "f0-speed-equiv-v1"
F1_BANK_SHA256 = "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"
F1_BANK = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-10a/"
    "f1-root-bank.json.xz"
)
RULING_SHA256 = "f3d4e6254771830625ec25938f9eb6dd5a9750e383cb12b34a7a04b687b8518c"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Apply only root timing, root provenance elapsed, and actual row code SHA."""
    if isinstance(value, list):
        return [_normalise(item, path=(*path, "[]")) for item in value]
    if not isinstance(value, dict):
        return value
    is_row = (
        value.get("schema") == "bl-labels-t5-row/v1"
        and len(path) >= 2
        and path[-2:] in (("rows", "[]"), ("unknown_suffix", "[]"))
    )
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not path and key == "timing_seconds":
            continue
        if path == ("provenance",) and key == "elapsed_seconds":
            continue
        if is_row and key == "code_sha":
            continue
        result[key] = _normalise(item, path=(*path, key))
    return result


def canonical_payload(value: Mapping[str, Any]) -> bytes:
    """Return the byte representation used for life/session identity."""
    return canonical(_normalise(dict(value)))


def _row_identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row.get("world")), int(row.get("life")), int(row.get("decision"))


def _rows(payloads: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for payload in payloads:
        # Life exports are the complete, non-overlapping F4 source.  The
        # terminal unit/session is compared byte-for-byte separately.
        if payload.get("schema") != "bl-labels-f0-life/v1":
            continue
        candidate = payload.get("rows", [])
        if not isinstance(candidate, list):
            raise ValueError("F0 payload rows must be a list")
        rows.extend(item for item in candidate if isinstance(item, Mapping))
    return sorted(rows, key=_row_identity)


def _selectable(row: Mapping[str, Any]) -> list[bool]:
    # F4 uses physical legal AND an advisory action mask.  The latter is an
    # external T-EQ diagnostic collected alongside a session, never a science
    # row field.  Its all-false normal-action fallback is the F3 helper rule.
    advisory = row.get("_speed_equiv_advisory")
    if (
        not isinstance(advisory, list)
        or len(advisory) != 6
        or not all(type(x) is bool for x in advisory)
    ):
        raise ValueError("T-EQ row lacks external advisory mask")
    return advisory if any(advisory) else [True, True, True, False, False, False]


def f4_pairs(payloads: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Extract the exact F4 preference and ordinary-growth pair membership.

    This mirrors ``fit._preference_loss`` and ``fit._ordinary_growth_loss``:
    exact, legal/resolved actions only; all survivor-over-dier pairs, then
    ordinary survivor mass-improving pairs.  The records carry the mass gap so
    equality proves the actual F4 input, rather than only matching ``a_star``.
    """
    survival: list[dict[str, Any]] = []
    growth: list[dict[str, Any]] = []
    for row in _rows(payloads):
        status, died, mass = row.get("status"), row.get("died"), row.get("mass")
        if not all(isinstance(x, list) and len(x) == 6 for x in (status, died, mass)):
            raise ValueError("F0 row lacks six-action F4 fields")
        physical = row.get("legal")
        if (
            not isinstance(physical, list)
            or len(physical) != 6
            or not all(type(x) is bool for x in physical)
        ):
            raise ValueError("F0 row has malformed physical legal mask")
        advisory = _selectable(row)
        valid = [
            status[action] == "EXACT"
            and physical[action]
            and advisory[action]
            and died[action] is not None
            for action in range(6)
        ]
        ident = {
            "world": row.get("world"),
            "life": row.get("life"),
            "decision": row.get("decision"),
        }
        survivors = [action for action in range(6) if valid[action] and not bool(died[action])]
        dying = [action for action in range(6) if valid[action] and bool(died[action])]
        for good in survivors:
            for poor in dying:
                survival.append({**ident, "survivor": good, "dying": poor})
        ordinary = row.get("sampling_kind") == "ordinary"
        if ordinary:
            for good in survivors:
                for poor in survivors:
                    if mass[good] is None or mass[poor] is None:
                        raise ValueError("ordinary exact survivor lacks mass")
                    gap = np.float32(mass[good]) - np.float32(mass[poor])
                    if gap > np.float32(0):
                        growth.append(
                            {
                                **ident,
                                "good": good,
                                "poor": poor,
                                "mass_gap": float(min(np.float32(4.0), gap)),
                            }
                        )
    return {"survival": sorted(survival, key=canonical), "growth": sorted(growth, key=canonical)}


def pair_digests(payloads: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {name: digest(rows) for name, rows in f4_pairs(payloads).items()}


def _payload_identity(value: Mapping[str, Any]) -> str:
    allocation = value.get("allocation_id")
    if not isinstance(allocation, str):
        candidate = value.get("allocation")
        allocation = candidate.get("allocation_id") if isinstance(candidate, Mapping) else None
    if not isinstance(allocation, str):
        raise ValueError("payload has no allocation identity")
    life = value.get("life")
    return f"{value.get('schema')}:{allocation}:{'' if life is None else life}"


def compare_payloads(
    old: Iterable[Mapping[str, Any]],
    new: Iterable[Mapping[str, Any]],
    *,
    compare_pairs: bool = True,
) -> dict[str, Any]:
    old_values, new_values = list(old), list(new)
    if not old_values or not new_values:
        return {"status": "FAIL", "reason": "empty science payloads"}
    old_map = {_payload_identity(value): value for value in old_values}
    new_map = {_payload_identity(value): value for value in new_values}
    if len(old_map) != len(old_values) or len(new_map) != len(new_values):
        return {"status": "FAIL", "reason": "duplicate payload identity"}
    if old_map.keys() != new_map.keys():
        return {"status": "FAIL", "reason": "payload identities differ"}
    hashes: dict[str, dict[str, str]] = {}
    for key in sorted(old_map):
        before, after = (
            hashlib.sha256(canonical_payload(old_map[key])).hexdigest(),
            hashlib.sha256(canonical_payload(new_map[key])).hexdigest(),
        )
        hashes[key] = {"old": before, "new": after}
        if before != after:
            return {
                "status": "FAIL",
                "reason": f"unallowlisted payload difference: {key}",
                "payload_hashes": hashes,
            }
    if not compare_pairs:
        return {"status": "PASS", "payload_hashes": hashes}
    old_pairs, new_pairs = pair_digests(old_values), pair_digests(new_values)
    if old_pairs != new_pairs:
        return {
            "status": "FAIL",
            "reason": "F4 pair or mass-gap difference",
            "payload_hashes": hashes,
            "pairs": {"old": old_pairs, "new": new_pairs},
        }
    return {"status": "PASS", "payload_hashes": hashes, "pairs": old_pairs}


def beq_allocation(namespace: str = NAMESPACE) -> list[dict[str, Any]]:
    """Freeze the excluded 12-parent B-EQ composition before an execution."""
    spec = (
        ("watch6", 2, 3000),
        ("gate-frozen", 4, 5000),
        ("gate-mixed", 3, 5000),
        ("gate-scripted", 3, 5000),
    )
    rows: list[dict[str, Any]] = []
    for stream, count, horizon in spec:
        for index in range(count):
            token = f"{namespace}/{stream}/{index}".encode()
            rows.append(
                {
                    "allocation_id": f"beq-{stream}-{index:02d}",
                    "seed": int.from_bytes(hashlib.sha256(token).digest()[:4], "big"),
                    "stream": stream,
                    "horizon": horizon,
                    "allocation_index": len(rows),
                }
            )
    if len({row["seed"] for row in rows}) != 12:
        raise AssertionError("B-EQ seed collision")
    return rows


def create_manifest(path: Path) -> dict[str, Any]:
    value = {
        "schema": "bl-labels-f0-speed-equiv-bank/v1",
        "namespace": NAMESPACE,
        "excluded_from_production": True,
        "allocation": beq_allocation(),
        "ruling_sha256": RULING_SHA256,
    }
    publish_json(path, value)
    return value


def f1_identity(
    old: Path,
    new: Path,
    *,
    bank: Path = F1_BANK,
    expected_old_sha: str = OLD_PIN,
    expected_new_sha: str | None = None,
) -> dict[str, Any]:
    """Compare every schedule of exactly the frozen 64 roots, with source binding."""
    from .f1_ruler import bank_digest, load_bank

    if bank_digest(bank) != F1_BANK_SHA256:
        raise ValueError("frozen F1 bank digest differs")
    expected = {str(row["root_id"]) for row in load_bank(bank)["roots"]}
    if len(expected) != 64:
        raise ValueError("F1 bank allocation differs")

    def records(path: Path, pin: str | None) -> dict[str, tuple[str, str, str]]:
        _verify(path)
        value = json.loads(path.read_text())
        if (
            value.get("status") != "COMPLETE"
            or value.get("bank_sha256") != F1_BANK_SHA256
            or (pin is not None and value.get("source_commit") != pin)
        ):
            raise ValueError("F1 result source/bank/status differs")
        configs = value.get("configs", [])
        if len(configs) != 3 or {c.get("schedule_label") for c in configs} != {
            "one",
            "half",
            "all",
        }:
            raise ValueError("F1 requires all three frozen schedules")
        reference = None
        for config in configs:
            if (
                config.get("source_commit") != value.get("source_commit")
                or config.get("bank_sha256") != F1_BANK_SHA256
            ):
                raise ValueError("F1 schedule source/bank differs")
            rows = config.get("records", [])
            if len(rows) != 64 or len({r["root_id"] for r in rows}) != 64:
                raise ValueError("F1 duplicate/missing root")
            result = {
                str(r["root_id"]): tuple(
                    r.get(k) for k in ("label_sha256", "digest", "state_digest")
                )
                for r in rows
            }
            if set(result) != expected or any(
                any(
                    not isinstance(x, str)
                    or len(x) != 64
                    or any(c not in "0123456789abcdef" for c in x)
                    for x in triple
                )
                for triple in result.values()
            ):
                raise ValueError("F1 identity lacks frozen roots or hashes")
            if any(
                r.get("digest_checks") != 1 or r.get("bare_audit_mismatches") != 0 for r in rows
            ):
                raise ValueError("F1 integrity gate failed")
            if reference is not None and result != reference:
                raise ValueError("F1 worker-width identity failed")
            reference = result
        assert reference is not None
        return reference

    before, after = records(old, expected_old_sha), records(new, expected_new_sha)
    return {
        "status": "PASS" if before == after else "FAIL",
        "old": digest(before),
        "new": digest(after),
        "roots": 64,
        "bank_sha256": F1_BANK_SHA256,
    }


def verify_receipt(
    path: Path,
    expected_old_sha: str,
    expected_new_sha: str,
    expected_bank_sha: str | None = None,
    expected_beq_manifest_sha: str | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("T-EQ receipt is not PASS")
    if (
        value.get("old", {}).get("commit") != expected_old_sha
        or value.get("new", {}).get("commit") != expected_new_sha
    ):
        raise ValueError("T-EQ receipt pin differs")
    # The existing mac_gates caller calls this ``expected_bank_sha`` for the
    # frozen F1 root bank.  B-EQ is a separate excluded parent manifest.
    if expected_bank_sha is not None and value.get("f1_bank_sha256") != expected_bank_sha:
        raise ValueError("T-EQ receipt F1 bank differs")
    if (
        expected_beq_manifest_sha is not None
        and value.get("bank", {}).get("sha256") != expected_beq_manifest_sha
    ):
        raise ValueError("T-EQ receipt B-EQ manifest differs")
    _verify(path)
    from . import speed_run

    if value.get("ruling_sha256") != RULING_SHA256 or value.get("call_counter_gate") != "PASS":
        raise ValueError("T-EQ ruling/call-counter evidence missing")
    comparisons = value.get("comparisons", {})
    required = {"old_vs_new_w1", "old_vs_new_w6", "new_w1_vs_w6", "f1"}
    if set(comparisons) != required or any(c.get("status") != "PASS" for c in comparisons.values()):
        raise ValueError("T-EQ full comparison gate missing")
    for name in required - {"f1"}:
        comparison = comparisons[name]
        hashes = comparison.get("payload_hashes", {})
        pairs = comparison.get("pairs", {})
        if (
            not hashes
            or any(h.get("old") != h.get("new") for h in hashes.values())
            or set(pairs.get("old", {})) != {"survival", "growth"}
            or pairs.get("old") != pairs.get("new")
        ):
            raise ValueError("T-EQ payload/pair identity missing")
    if (
        comparisons["f1"].get("roots") != 64
        or comparisons["f1"].get("bank_sha256") != F1_BANK_SHA256
    ):
        raise ValueError("T-EQ F1 identity missing")
    bank = Path(value["bank"]["path"])
    speed_run.verify_bank(bank, expected_new_sha)
    if _verify(bank) != value["bank"]["sha256"]:
        raise ValueError("T-EQ B-EQ binding changed")
    ledger = speed_run.verify_publication(path.parent, speed_run.ROOT, value["publication_path"])
    if _verify(path.parent / "phase-ledger.json") != value.get("phase_ledger_sha256"):
        raise ValueError("T-EQ phase ledger changed")
    for phase in ("old-w1", "new-w1", "old-w6", "new-w6", "old-f1", "new-f1"):
        speed_run.load_phase(path.parent, phase, ledger, metadata_only=True)
    return value


def main() -> None:
    """Public WP-1 entrypoint; production orchestration lives in speed_run."""
    from .speed_run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
