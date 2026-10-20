"""Mac A2 replay audit for the registered deterministic Round 0 sample."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.t34_exit_b8.exports import load_world
from research.t34_round0.runner import (
    Round0Error,
    _load,
    _metrics_from_trace,
    _validate_audit,
    _validate_bank,
    sha256,
)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    digest = sha256(path)
    with path.with_name(f"{path.name}.sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {path.name}\n")
    path.with_name(f"{path.name}.complete").touch(exist_ok=False)


def _registration_identity(registration: Path, bank: Path, audit: Path) -> Mapping[str, Any]:
    from research.t34_round0.registration import require_registration

    value = require_registration(
        _load(registration), bank, audit, repo=Path(__file__).resolve().parents[2]
    )
    if not isinstance(value, Mapping):
        raise Round0Error("registration validator did not return a bank")
    return value


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise Round0Error("A2 replay audit is Mac-only")
    bank, audit_path, ratification = Path(args.bank), Path(args.audit), Path(args.ratification)
    rows = _validate_bank(_load(bank), preflight=False)
    _validate_audit(audit_path, bank)
    _registration_identity(ratification, bank, audit_path)
    roots = [Path(path) for path in args.chunk_dirs]
    if not roots or len({root.resolve() for root in roots}) != len(roots):
        raise Round0Error("replay requires unique chunk directories")
    from research.t34_round0.history import require_active_phase, require_chunks, selected_rows

    require_active_phase(_load(ratification), "replay")
    require_chunks(roots, _load(ratification))
    expected = {
        "bank_sha256": sha256(bank),
        "audit_sha256": sha256(audit_path),
        "ratification_sha256": sha256(ratification),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
    }
    records: dict[str, tuple[Path, Path, Mapping[str, Any]]] = {}
    identities: list[Mapping[str, Any]] = []
    chunks, drop_reports = [], []
    for root in roots:
        if (root / "fault.json").is_file():
            return {
                "schema": "t34-round0-replay/v1",
                "status": "STOP_CONTROLLER_FAULT",
                "reason": "explained unconditional fault; proceed only to sole STOP decision",
                "fault_sha256": sha256(root / "fault.json"),
            }
        receipt = _load(root / "chunk.json")
        if receipt.get("schema") != "t34-round0-chunk/v1" or receipt.get("status") != "complete":
            raise Round0Error("chunk receipt is not complete")
        identity = receipt.get("run_identity", {})
        if any(identity.get(key) != value for key, value in expected.items()):
            raise Round0Error("chunk identity differs from replay inputs")
        identities.append(identity)
        chunks.append(receipt)
        allocated = receipt.get("slice", {}).get("count", 0)
        completed = len(receipt.get("pairs", []))
        if type(allocated) is not int or allocated <= 0 or completed > allocated:
            raise Round0Error("invalid chunk completion counts")
        dropped_fraction = (allocated - completed) / allocated
        drop_reports.append(
            {
                "chunk": str(root),
                "allocated": allocated,
                "completed": completed,
                "dropped_fraction": dropped_fraction,
            }
        )
        if dropped_fraction > 0.02:
            report = {
                "schema": "t34-round0-replay/v1",
                "status": "STOP_DROPPED_WORLDS",
                "chunks": drop_reports,
                "reason": ">2% dropped worlds in a chunk",
            }
            return report
        pair_index = {Path(item["path"]).name: item for item in receipt.get("pairs", [])}
        for path in (root / "pairs").glob("*.json"):
            pair = _load(path)
            indexed = pair_index.get(path.name)
            if not indexed or indexed.get("sha256") != sha256(path):
                raise Round0Error("chunk pair receipt hash differs")
            world_id = pair.get("world", {}).get("world_id")
            if not isinstance(world_id, str) or world_id in records:
                raise Round0Error("missing or duplicate pair world identity")
            records[world_id] = (root, path, pair)
    selected = selected_rows(rows, chunks)
    divergences, checked = [], []
    for row in selected:
        record = records.get(row["world_id"])
        root, pair_path, pair = record if record else (None, None, None)
        if pair_path is None:
            divergences.append({"world_id": row["world_id"], "reason": "missing pair"})
            continue
        trace_hashes: dict[str, str] = {}
        for arm in ("bare", "teacher"):
            meta = pair.get("arms", {}).get(arm, {})
            trace = meta.get("trace", {})
            path = root / str(trace.get("path", ""))
            try:
                if sha256(path) != trace.get("sha256"):
                    raise Round0Error("trace bytes differ")
                exported = load_world(path)
                if exported.get("labels"):
                    raise Round0Error("Round 0 trace contains labels")
                measured = _metrics_from_trace(exported, Path(args.checkpoint))
                for key in (
                    "mass_integral",
                    "survival_fraction",
                    "no_death",
                    "deaths",
                    "decisions",
                    "boost_decisions",
                    "action_counts",
                ):
                    if measured[key] != meta.get(key):
                        raise Round0Error(f"outcome mismatch: {key}")
                if (
                    trace.get("final_state_hash")
                    != exported["hashes"][str(len(exported["actions"]))]
                ):
                    raise Round0Error("final state hash differs")
                trace_hashes[arm] = sha256(path)
            except Exception as exc:
                divergences.append({"world_id": row["world_id"], "arm": arm, "reason": str(exc)})
        if all(arm in trace_hashes for arm in ("bare", "teacher")):
            checked.append(
                {
                    "world_id": row["world_id"],
                    "pair_sha256": sha256(pair_path),
                    "traces": trace_hashes,
                }
            )
    report = {
        "schema": "t34-round0-replay/v1",
        "status": "PASS" if not divergences and len(checked) == len(selected) else "FAIL",
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "run_identity": dict(expected, commit=identities[0].get("commit")),
        "selected_ids": [row["world_id"] for row in selected],
        "sampling": "first ceil(10% of allocated pairs) within each bank-order chunk",
        "chunks": drop_reports,
        "checked_pairs": checked,
        "divergences": divergences,
    }
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deterministic Round 0 traces on a Mac")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--ratification", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunk-dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        from research.t34_round0.history import policy, run_phase

        registration = _load(Path(args.ratification))
        root, _, _ = policy(registration)
        if Path(args.out) != root / "replay/replay.json":
            raise Round0Error("replay must use the fixed study replay/replay.json")
        report = run_phase(registration, "replay", lambda _out: audit(args))
    except Round0Error as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
