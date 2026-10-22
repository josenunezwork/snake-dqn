"""Mac-only, pre-data identity harness for the L2 exact-stop production path.

This is deliberately not an L2 bank runner.  It derives five development
worlds in a namespace that is audited as a whole against every visible bank,
then runs those worlds through :func:`ceiling._worker` in both stock modes.
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
from research.bl_served_search.exports import digest, export_json, verify_export
from research.t1_commitment_20261015.replay import configure_cpu

SCHEMA = "bl-served-search-mac-identity/v1"
NAMESPACE = "bl-served-search-mac-identity-20261022/v1"
HORIZON = 5000
MODES = ("all", "exact_stop")
MIXES = ("frozen", "frozen", "mixed", "mixed", "scripted")
LEASE_SECONDS = 2 * 60 * 60
P3_ALLOCATION = (
    Path(__file__).resolve().parents[3]
    / "snake-dqn-artifacts/big-lever/exp/bl-l2/mac-prerequisites-01/p3-identity-02/allocation.json"
)
P3_ALLOCATION_SHA256 = "348187b77dd8f407722a4654efdef5ebf512e4239d3b0be37cb9bac9fab6b5bf"
EXECUTION_MODES = {
    "all": {"search_mode": "all", "native_mode": "stock"},
    "exact_stop": {"search_mode": "exact_stop", "native_mode": "stock"},
}
_VETO_COUNTER_KEYS = (
    "calls",
    "rows",
    "replayed_rows",
    "fallback_rows",
    "tail_release_rows",
    "python_calls",
    "audited_rows",
    "overrides",
    "v5_changes",
    "v7_changes",
    "head_changes",
    "no_spacious",
    "tie_rows",
    "tie_overrides",
    "boost_base_rows",
    "long_rows",
    "kernel_rows",
    "stock_rows",
    "noncandidate_rows",
    "root_stock_rows",
)

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


def materialize(
    rows: list[dict[str, Any]],
    *,
    context_factory: Any | None = None,
    roster_builder: Any | None = None,
) -> list[dict[str, Any]]:
    """Materialize the one non-bank P3 group without an empty-pool call.

    ``bank.materialize`` deliberately visits both its registered selection and
    preflight pools.  P3 has a single five-world development group, so calling
    that helper would incorrectly submit an empty preflight seed vector.
    """
    if not rows or any(row.get("pool") != "selection" for row in rows):
        raise ValueError("P3 materialization requires one nonempty development selection group")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if context_factory is None or roster_builder is None:
        from research.redesign_scope_20261007 import grid_h5000_identity as gid
        from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

        context_factory = gid._context if context_factory is None else context_factory
        roster_builder = planned_rosters if roster_builder is None else roster_builder
    context = context_factory(1)
    plans = roster_builder(
        context, [row["world_seed"] for row in rows], [row["mix"] for row in rows]
    )
    if not isinstance(plans, list) or len(plans) != len(rows):
        raise ValueError("P3 roster materializer returned an incomplete development group")
    return [{**row, "plan": plan} for row, plan in zip(rows, plans)]


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
        planned = materialize(rows)
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


def load_allocation(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the immutable P3-02 allocation without deriving replacement worlds."""
    path = Path(path)
    verify_export(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P3 allocation is not an object")
    round_index = value.get("round_index")
    if type(round_index) is not int:
        raise ValueError("P3 allocation round index is invalid")
    audit_path = path.parent / f"seed-audit-round-{round_index:02d}.json"
    verify_export(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("P3 allocation audit is not an object")
    if (
        value.get("schema") != "bl-served-search-mac-identity-allocation/v1"
        or digest(path) != P3_ALLOCATION_SHA256
        or value.get("namespace") != NAMESPACE
        or round_index != 1
        or value.get("development_only") is not True
        or value.get("horizon") != HORIZON
        or value.get("mix_counts") != {"frozen": 2, "mixed": 2, "scripted": 1}
        or value.get("seed_audit_sha256") != digest(audit_path)
        or audit.get("schema") != "bl-served-search-seed-audit/v1"
        or audit.get("status") != "CLEAR"
        or audit.get("namespace") != NAMESPACE
        or audit.get("round_index") != round_index
        or audit.get("candidate_count") != 5
        or audit.get("unique_count") != 5
        or audit.get("history", {}).get("never_opened") != bank.SEALED
    ):
        raise ValueError("P3 allocation/audit identity does not bind the audited round-1 worlds")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(MIXES):
        raise ValueError("P3 allocation does not contain exactly five worlds")
    expected_rows = derive(1)
    for row, expected in zip(rows, expected_rows):
        if not isinstance(row, dict) or any(row.get(key) != expected[key] for key in expected):
            raise ValueError("P3 allocation world identity differs from audited round-1 allocation")
        plan = row.get("plan")
        if (
            not isinstance(plan, dict)
            or plan.get("seed") != row["world_seed"]
            or plan.get("mix") != row["mix"]
            or not isinstance(plan.get("specs"), list)
            or not isinstance(plan.get("member_sha256s"), list)
            or len(plan["specs"]) != 5
            or len(plan["member_sha256s"]) != 5
        ):
            raise ValueError("P3 allocation has an incomplete historical roster plan")
    return rows, audit


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
    error = analyze._row_error(row, expected, search_mode=mode, native_mode="stock")
    if error:
        raise ValueError(f"invalid {mode} production row {row.get('world_id')}: {error}")
    counters = row.get("native_v8")
    if not isinstance(counters, dict) or any(
        type(value) is not int or value != 0 for key, value in counters.items() if key != "mode"
    ):
        raise ValueError(
            f"invalid {mode} production row {row.get('world_id')}: nonzero veto counter"
        )


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
        pending = {pool.submit(ceiling._worker, (row, str(b8_path), mode, "stock")) for row in rows}
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
    expected_ids = set(positions)
    received_ids = [str(row.get("world_id")) for row in result]
    if (
        len(result) != len(rows)
        or set(received_ids) != expected_ids
        or len(set(received_ids)) != len(rows)
    ):
        raise ModeRunError("P3 mode returned incomplete or duplicate world rows", result)
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


def _execution_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize partial worker rows without treating malformed evidence as valid."""
    native = dict.fromkeys(_VETO_COUNTER_KEYS, 0)
    branches_evaluated = branches_skipped = 0
    rejected_rows = []
    for index, row in enumerate(rows):
        world_id = row.get("world_id") if isinstance(row, dict) else None
        reasons = []
        stats = row.get("native_v8") if isinstance(row, dict) else None
        if not isinstance(stats, dict):
            reasons.append("missing/invalid native_v8")
        else:
            invalid = [
                key
                for key in _VETO_COUNTER_KEYS
                if type(stats.get(key)) is not int or stats[key] < 0
            ]
            if invalid:
                reasons.append("missing/invalid native counters: " + ", ".join(invalid))
            else:
                for key in _VETO_COUNTER_KEYS:
                    native[key] += stats[key]
        dense = row.get("dense") if isinstance(row, dict) else None
        if not isinstance(dense, dict) or any(
            type(dense.get(key)) is not int or dense[key] < 0
            for key in ("branches_evaluated", "branches_skipped")
        ):
            reasons.append("missing/invalid dense branch counters")
        else:
            branches_evaluated += dense["branches_evaluated"]
            branches_skipped += dense["branches_skipped"]
        if reasons:
            rejected_rows.append({"index": index, "world_id": world_id, "reasons": reasons})
    return {
        "native_v8": native,
        "branches_evaluated": branches_evaluated,
        "branches_skipped": branches_skipped,
        "valid_rows": len(rows) - len(rejected_rows),
        "rejected_rows": rejected_rows,
    }


def _fired_batches_by_mix(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    """Report only well-formed diagnostic batch counts from retained partial rows."""
    totals = {mix: dict.fromkeys(MODES, 0) for mix in set(MIXES)}
    for mode in MODES:
        for row in results[mode]:
            mix = row.get("mix") if isinstance(row, dict) else None
            if not isinstance(mix, str) or mix not in totals:
                continue
            dense = row.get("dense")
            batches = dense.get("batches") if isinstance(dense, dict) else None
            if type(batches) is int and batches >= 0:
                totals[mix][mode] += batches
    return totals


def run(
    out: Path,
    *,
    b8_path: Path,
    allocation: Path = P3_ALLOCATION,
    workers: int = 2,
    max_rounds: int = 32,
    deadline: float | None = None,
    guard: Callable[[], Any] | None = None,
    execution_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute both stock modes on the immutable five-world P3-02 allocation."""
    if workers != 2:
        raise ValueError("P3 requires exactly two worker processes")
    if not b8_path.is_file() or ceiling.digest_file(b8_path) != ceiling.B8_SHA256:
        raise ValueError("P3 requires the byte-pinned B8 checkpoint")
    out = Path(out)
    if out.exists() or any(Path(str(out) + suffix).exists() for suffix in (".sha256", ".complete")):
        raise FileExistsError("P3 output directory is create-only")
    started = time.monotonic()
    deadline = started + LEASE_SECONDS if deadline is None else deadline
    if deadline <= started:
        raise ValueError("P3 deadline is already expired")
    guard = (lambda: None) if guard is None else guard
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] | None = None
    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    status, error = "NOT_READY", None
    readiness = {
        "mode_a": {**EXECUTION_MODES["exact_stop"], "status": "NOT_READY"},
        "mode_b": {**EXECUTION_MODES["all"], "status": "NOT_READY"},
    }
    try:
        rows, audit = load_allocation(allocation)
        results["all"] = _run_mode(
            rows, b8_path=b8_path, mode="all", workers=workers, deadline=deadline, guard=guard
        )
        fired_b = {
            mix: sum(row["dense"]["batches"] for row in results["all"] if row["mix"] == mix)
            for mix in set(MIXES)
        }
        if any(count <= 0 for count in fired_b.values()):
            raise ValueError(f"NOT_READY vacuous mode-B search coverage: {fired_b}")
        readiness["mode_b"] = {**EXECUTION_MODES["all"], "status": "READY"}
        try:
            results["exact_stop"] = _run_mode(
                rows,
                b8_path=b8_path,
                mode="exact_stop",
                workers=workers,
                deadline=deadline,
                guard=guard,
            )
            for all_row, stopped_row in zip(results["all"], results["exact_stop"]):
                compare_rows(all_row, stopped_row)
            fired_a = {
                mix: sum(
                    row["dense"]["batches"] for row in results["exact_stop"] if row["mix"] == mix
                )
                for mix in set(MIXES)
            }
            if any(count <= 0 for count in fired_a.values()):
                raise ValueError(f"NOT_READY vacuous mode-A search coverage: {fired_a}")
            readiness["mode_a"] = {**EXECUTION_MODES["exact_stop"], "status": "READY"}
            status = "READY_MODE_A"
        except ModeRunError as exc:
            results["exact_stop"] = exc.rows
            readiness["mode_a"] = {
                **EXECUTION_MODES["exact_stop"],
                "status": "NOT_READY",
                "reason": str(exc),
            }
            status, error = "READY_MODE_B", str(exc)
        except ValueError as exc:
            readiness["mode_a"] = {
                **EXECUTION_MODES["exact_stop"],
                "status": "NOT_READY",
                "reason": str(exc),
            }
            status, error = "READY_MODE_B", str(exc)
    except ModeRunError as exc:
        results["all"] = exc.rows
        status, error = "NOT_READY", str(exc)
    except ValueError as exc:
        status, error = "NOT_READY", str(exc)
    except BaseException as exc:
        status, error = "NOT_READY", f"{type(exc).__name__}: {exc}"
    counters = {mode: _execution_counters(results[mode]) for mode in MODES}
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "identity": {"status": status, "schema": SCHEMA},
        "authority": (
            "JUDGMENT_OPPONENT_PREMISE.md P3'; development identity only, not bank evidence"
        ),
        "units": 0,
        "platform": "mac",
        "workers": workers,
        "threads": 1,
        "horizon": HORIZON,
        "modes": list(MODES),
        "execution_modes": EXECUTION_MODES,
        "readiness": readiness,
        "permitted_search_modes": (
            ["all", "exact_stop"]
            if status == "READY_MODE_A"
            else ["all"] if status == "READY_MODE_B" else []
        ),
        "execution_counters": counters,
        "allocation": {
            "path": str(allocation),
            "sha256": digest(allocation) if Path(allocation).is_file() else None,
            "seed_audit_status": None if audit is None else audit["status"],
            "seed_audit_sha256": (
                None
                if audit is None
                else digest(
                    Path(allocation).parent / f"seed-audit-round-{audit['round_index']:02d}.json"
                )
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
                "analyzer row validation in both stock modes",
                "all-zero opponent-veto counters",
            ],
            "excluded_fields": sorted(_DIAGNOSTIC_FIELDS),
            "fired_batches_by_mix": _fired_batches_by_mix(results),
        },
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
    }
    export_json(Path(out) / "receipt.json", receipt)
    if status == "NOT_READY":
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
    parser.add_argument("--b8-path", type=Path)
    parser.add_argument("--allocation", type=Path, default=P3_ALLOCATION)
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
        rows, report = load_allocation(args.allocation)
        print(
            canonical(
                {
                    "status": report["status"],
                    "candidate_count": len(rows),
                    "allocation": str(args.allocation),
                    "audit": report,
                }
            )
        )
        return 0
    if args.b8_path is None:
        parser.error("--run requires --b8-path")
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
            allocation=args.allocation,
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
