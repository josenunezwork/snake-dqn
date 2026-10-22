"""Mac-only, pre-data identity harness for the L2 exact-stop production path.

This is deliberately not an L2 bank runner.  It derives five development
worlds in a namespace that is audited as a whole against every visible bank,
then runs those worlds through :func:`ceiling._worker` in both kernel modes.
The only conclusion it can produce is whether the modes have identical
scientific receipts on those development worlds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from research.bl_served_search import analyze, bank, ceiling
from research.bl_served_search.exports import digest, export_json
from research.t1_commitment_20261015.replay import configure_cpu

SCHEMA = "bl-served-search-mac-identity/v1"
NAMESPACE = "bl-served-search-mac-identity-20261022/v1"
HORIZON = 5000
MODES = ("all", "exact_stop")
MIXES = ("frozen", "frozen", "mixed", "mixed", "scripted")
LEASE_SECONDS = 2 * 60 * 60

# These fields establish how the search was executed, or measure it.  They are
# intentionally excluded from the scientific identity comparison below.
_DIAGNOSTIC_FIELDS = frozenset(
    {
        "native_v8",
        "search_mode",
        "native_mode",
        "worker_threads",
        "batch_diagnostics",
        "branches_evaluated",
        "branches_skipped",
        "rollout_env_frames",
        "rollout_wall_seconds",
        "wall_seconds",
        "batches",
    }
)


class ModeRunError(RuntimeError):
    """Keep completed receipts when a mode faults or exceeds the P3 hard cap."""

    def __init__(self, message: str, rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rows = rows


def canonical(value: Any) -> str:
    """Return a stable JSON representation for receipt identity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed(round_index: int, index: int, mix: str) -> int:
    name = f"{NAMESPACE}/round-{round_index}/{mix}/{index}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def derive(round_index: int) -> list[dict[str, Any]]:
    """Derive all five development worlds; rejected rounds are never repaired."""
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round must be a nonnegative integer")
    occurrence: dict[str, int] = {}
    rows = []
    for ordinal, mix in enumerate(MIXES):
        index = occurrence.get(mix, 0)
        occurrence[mix] = index + 1
        world_id = f"{NAMESPACE}/round-{round_index}/{mix}/{index}"
        rows.append(
            {
                "world_id": world_id,
                "world_seed": _seed(round_index, index, mix),
                "mix": mix,
                "horizon": HORIZON,
                # Reuse bank.materialize's planning path only.  These rows are
                # never a registered selection bank despite its required tag.
                "pool": "selection",
                "development_only": True,
                "identity_ordinal": ordinal,
            }
        )
    return rows


def allocate(out: Path, *, max_rounds: int = 32) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Publish every whole-round audit, materializing only the first clear one."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("identity output directory is create-only")
    out.mkdir(parents=True)
    for round_index in range(max_rounds):
        rows = derive(round_index)
        # bank.audit is the project no-peek audit: its scanner prunes the sealed
        # Round 0 subtree before traversal and explicitly checks strict banks.
        audit = bank.audit(rows)
        audit.update(
            namespace=NAMESPACE, round_index=round_index, purpose="P3 non-bank development"
        )
        export_json(out / f"seed-audit-round-{round_index:02d}.json", audit)
        if audit["status"] != "CLEAR":
            continue
        planned = bank.materialize(rows)
        allocation = {
            "schema": "bl-served-search-mac-identity-allocation/v1",
            "namespace": NAMESPACE,
            "round_index": round_index,
            "development_only": True,
            "horizon": HORIZON,
            "mix_counts": {mix: MIXES.count(mix) for mix in sorted(set(MIXES))},
            "rows": planned,
            "seed_audit_sha256": digest(out / f"seed-audit-round-{round_index:02d}.json"),
        }
        export_json(out / "allocation.json", allocation)
        return planned, audit
    raise RuntimeError("no clear whole P3 allocation; no seed swaps are permitted")


def scientific_projection(row: Any) -> Any:
    """Remove timing and branch diagnostics before comparing scientific receipts."""
    if isinstance(row, dict):
        return {
            key: scientific_projection(value)
            for key, value in row.items()
            if key not in _DIAGNOSTIC_FIELDS
        }
    if isinstance(row, list):
        return [scientific_projection(value) for value in row]
    return row


def validate_row(row: dict[str, Any], expected: dict[str, Any], *, mode: str) -> None:
    """Apply the production analyzer's execution and scientific-row checks."""
    error = analyze._row_error(row, expected, search_mode=mode, native_mode="kernel")
    if error:
        raise ValueError(f"invalid {mode} production row {row.get('world_id')}: {error}")


def compare_rows(all_row: dict[str, Any], stopped_row: dict[str, Any]) -> None:
    """Fail on any action, trajectory, ledger, or scientific receipt difference."""
    if all_row.get("world_id") != stopped_row.get("world_id"):
        raise ValueError("mode rows have different world identities")
    all_batches = all_row["dense"]["batch_diagnostics"]
    stopped_batches = stopped_row["dense"]["batch_diagnostics"]
    all_actions = [(entry["decision"], entry["chosen"]) for entry in all_batches]
    stopped_actions = [(entry["decision"], entry["chosen"]) for entry in stopped_batches]
    if all_actions != stopped_actions:
        raise ValueError(f"chosen action mismatch: {all_row['world_id']}")
    if scientific_projection(all_row) != scientific_projection(stopped_row):
        raise ValueError(f"scientific row mismatch: {all_row['world_id']}")


def _stop_pool(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    """Terminate only the two worker processes owned by this P3 pool."""
    processes = list((pool._processes or {}).values())
    for process in processes:
        process.terminate()
    until = time.monotonic() + 5.0
    for process in processes:
        process.join(max(0.0, until - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(1.0)
    pool.shutdown(wait=False, cancel_futures=True)


def _run_mode(
    rows: list[dict[str, Any]],
    *,
    b8_path: Path,
    mode: str,
    workers: int,
    deadline: float,
    guard: Callable[[], Any],
) -> list[dict[str, Any]]:
    """Run a mode through production workers, preserving allocation order."""
    configure_cpu()
    context = multiprocessing.get_context("spawn")
    positions = {str(row["world_id"]): index for index, row in enumerate(rows)}
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context)
    result: list[dict[str, Any]] = []
    mode_started = time.monotonic()
    try:
        pending = {
            pool.submit(ceiling._worker, (row, str(b8_path), mode, "kernel")) for row in rows
        }
        while pending:
            guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("P3 two-hour wall cap expired")
            done, pending = concurrent.futures.wait(
                pending,
                timeout=min(30.0, remaining),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                result.append(future.result())
            # Operational heartbeat only: no rows, trajectories, outcomes, or
            # aggregate scientific metrics are exposed before the receipt.
            print(
                canonical(
                    {
                        "schema": "bl-served-search-mac-identity-progress/v1",
                        "mode": mode,
                        "completed": len(result),
                        "total": len(rows),
                        "elapsed_seconds": time.monotonic() - mode_started,
                        "remaining_seconds": max(0.0, deadline - time.monotonic()),
                    }
                ),
                flush=True,
            )
    except BaseException as error:
        _stop_pool(pool)
        raise ModeRunError(str(error), result) from error
    else:
        pool.shutdown(wait=True)
    result.sort(key=lambda row: positions[str(row["world_id"])])
    try:
        for row in result:
            validate_row(
                row, next(item for item in rows if item["world_id"] == row["world_id"]), mode=mode
            )
    except BaseException as error:
        raise ModeRunError(str(error), result) from error
    return result


def git_commit(root: Path) -> str:
    """Read the source commit without mutating the checkout."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def run(
    out: Path,
    *,
    b8_path: Path,
    workers: int = 2,
    max_rounds: int = 32,
    deadline: float | None = None,
    guard: Callable[[], Any] | None = None,
    execution_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate five audited worlds, execute both modes, and write one receipt."""
    if workers != 2:
        raise ValueError("P3 requires exactly two worker processes")
    if not b8_path.is_file() or ceiling.digest_file(b8_path) != ceiling.B8_SHA256:
        raise ValueError("P3 requires the byte-pinned B8 checkpoint")
    started = time.monotonic()
    deadline = started + LEASE_SECONDS if deadline is None else deadline
    if deadline <= started:
        raise ValueError("P3 deadline is already expired")
    guard = (lambda: None) if guard is None else guard
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] | None = None
    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    status, error = "PASS", None
    try:
        rows, audit = allocate(out, max_rounds=max_rounds)
        for mode in MODES:
            results[mode] = _run_mode(
                rows, b8_path=b8_path, mode=mode, workers=workers, deadline=deadline, guard=guard
            )
        for all_row, stopped_row in zip(results["all"], results["exact_stop"]):
            compare_rows(all_row, stopped_row)
        fired = {
            mix: sum(row["dense"]["batches"] for row in results["exact_stop"] if row["mix"] == mix)
            for mix in set(MIXES)
        }
        if any(count <= 0 for count in fired.values()):
            raise ValueError(f"NOT_READY vacuous P3 search coverage: {fired}")
    except ModeRunError as exc:
        status, error = "INCONCLUSIVE", str(exc)
        # Retain the rows returned before the owned pool was terminated.
        for mode in MODES:
            if not results[mode] and exc.rows:
                results[mode] = exc.rows
                break
    except ValueError as exc:
        status, error = ("INCONCLUSIVE" if str(exc).startswith("NOT_READY") else "INVALID"), str(
            exc
        )
    except BaseException as exc:
        status, error = "INCONCLUSIVE", f"{type(exc).__name__}: {exc}"
    counters = {
        mode: {
            "native_v8": {
                key: sum(int(row["native_v8"][key]) for row in results[mode])
                for key in (
                    "kernel_rows",
                    "stock_rows",
                    "fallback_rows",
                    "noncandidate_rows",
                    "calls",
                )
            },
            "branches_evaluated": sum(
                int(row["dense"]["branches_evaluated"]) for row in results[mode]
            ),
            "branches_skipped": sum(int(row["dense"]["branches_skipped"]) for row in results[mode]),
        }
        for mode in MODES
    }
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "identity": {"status": status, "schema": SCHEMA},
        "authority": "JUDGMENT_MAC_AMENDMENT.md P3; development identity only, not bank evidence",
        "units": 0,
        "platform": "mac",
        "workers": workers,
        "threads": 1,
        "horizon": HORIZON,
        "modes": list(MODES),
        "execution_counters": counters,
        "allocation": {
            "path": str(Path(out) / "allocation.json"),
            "sha256": (
                digest(Path(out) / "allocation.json")
                if (Path(out) / "allocation.json").is_file()
                else None
            ),
            "seed_audit_status": None if audit is None else audit["status"],
            "seed_audit_sha256": (
                None
                if audit is None
                else digest(Path(out) / f"seed-audit-round-{audit['round_index']:02d}.json")
            ),
            "never_opened": None if audit is None else audit["history"]["never_opened"],
            "coverage": None if audit is None else audit["coverage"],
        },
        "b8_sha256": ceiling.B8_SHA256,
        "source_manifest": ceiling.source_manifest(),
        "commit": git_commit(Path(__file__).resolve().parents[2]),
        "execution_identity": execution_identity,
        "rows": results,
        "comparison": {
            "status": status,
            "worlds": len(rows),
            "mix_counts": {mix: MIXES.count(mix) for mix in sorted(set(MIXES))},
            "requirements": [
                "chosen action at every fired batch",
                "full mass and alive trajectories",
                "mass and world ledgers plus paired deltas",
                "analyzer row validation in both kernel modes",
            ],
            "excluded_fields": sorted(_DIAGNOSTIC_FIELDS),
            "fired_batches_by_mix": {
                mix: sum(
                    row["dense"]["batches"] for row in results["exact_stop"] if row["mix"] == mix
                )
                for mix in set(MIXES)
            },
        },
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
    }
    export_json(Path(out) / "receipt.json", receipt)
    if status != "PASS":
        raise RuntimeError(f"P3 {status}: {error}")
    return receipt


def _until() -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS))
        .isoformat()
        .replace("+00:00", "Z")
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b8-path", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="derive and audit only; do not run worlds"
    )
    parser.add_argument("--run", action="store_true", help="execute the five P3 development worlds")
    parser.add_argument(
        "--lease",
        type=Path,
        default=Path(__file__).resolve().parents[2].parent
        / "snake-dqn-artifacts/big-lever/exp/mac-lease.json",
    )
    parser.add_argument("--lease-owner", default="claude-p3-mac-identity")
    args = parser.parse_args(argv)
    if args.check == args.run:
        parser.error("choose exactly one of --check or --run")
    if args.check:
        rows = derive(0)
        report = bank.audit(rows)
        print(
            canonical({"status": report["status"], "candidate_count": len(rows), "audit": report})
        )
        return 0
    try:
        priority = os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError) as error:
        parser.error(f"cannot attest Mac nice level: {error}")
    if priority != 10:
        parser.error("P3 must be launched under /usr/bin/nice -n 10")
    from research.bl_served_search.mac_lease import acquire, release, require

    def interrupted(signum: int, frame: Any) -> None:
        raise SystemExit("SIGTERM received by P3 coordinator")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    lease = acquire(args.lease, args.lease_owner, "P3 production-path identity", _until())
    try:
        until = datetime.fromisoformat(lease["until"].replace("Z", "+00:00"))
        remaining = (until - datetime.now(timezone.utc)).total_seconds()
        receipt = run(
            args.out,
            b8_path=args.b8_path,
            deadline=time.monotonic() + remaining,
            guard=lambda: require(args.lease, args.lease_owner, lease["id"]),
            execution_identity={
                "lease": lease,
                "nice": priority,
                "pid": os.getpid(),
                "host": {
                    "node": platform.node(),
                    "machine": platform.machine(),
                    "system": platform.system(),
                },
                "launcher": "nice -n 10 caffeinate -is",
            },
        )
    finally:
        try:
            release(args.lease, args.lease_owner, lease["id"])
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    print(canonical({"status": receipt["status"], "receipt": str(args.out / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
