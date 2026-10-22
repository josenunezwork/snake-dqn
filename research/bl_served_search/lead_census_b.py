"""PLAN_v3 M0 Part B: diagnostic bare replay, run only by Claude under the Mac lease.

Searches never control the live hero. Each completed world is an immutable unit;
the separate aggregator is the sole look over verified units. No dispatch/retry is automatic.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.lead_census_a import (
    B8_SHA,
    EXPECTED,
    OUTPUT,
    ROOT,
    SCHEMA,
    SOURCE,
    read_sources,
    rescue_record,
    value_sha,
)

NAMESPACE = "bl-c15-m0-d300-lad/v1"
CAP_SECONDS = 3600  # PLAN_v3 already includes the owner's 2x envelopes; do not double again.
LEADS = (48, 72, 96, 144, 192, 240)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class CapExpired(RuntimeError):
    """A mechanical deadline leaves the current unexported world incomplete."""


def source_manifest() -> dict[str, str]:
    """Bind the unchanged replay path and the M0 implementation, never a directory scan."""
    from research.bl_served_search.ceiling import source_manifest as base

    root = Path(__file__).resolve().parents[2]
    paths = ("lead_census_a.py", "lead_census_b.py", "run_lead_census_b.sh")
    return {
        **base(),
        **{f"research/bl_served_search/{p}": digest(Path(__file__).with_name(p)) for p in paths},
        "research/bl_ppo/exports.py": digest(root / "research/bl_ppo/exports.py"),
        "research/mac_lease.py": digest(root / "research/mac_lease.py"),
    }


def load_part_a(path: Path) -> tuple[dict, list[dict]]:
    """Revalidate the published A table and the source shards before any gameplay."""
    verify_export(path)
    part_a = json.loads(path.read_text())
    if part_a.get("schema") != SCHEMA or part_a.get("status") != "PART_A_COMPLETE_PART_B_NOT_RUN":
        raise ValueError("complete Part A required")
    rows, hashes = read_sources(SOURCE)
    selected = [r for r in rows if r["world_ledger"]["productive_rescue"] is True]
    if (
        part_a["rows"] != [rescue_record(r) for r in selected]
        or part_a["rescue_counts"] != EXPECTED
        or any(
            part_a["input_sha256"].get(p) != h
            for p, h in hashes.items()
            if p.endswith("shard.json")
        )
    ):
        raise ValueError("Part A/shards disagree")
    if part_a["authority_sha256"] != {
        str(p): digest(p) for p in (ROOT / "PLAN_v3.md", ROOT / "round-15/JUDGMENT.md")
    }:
        raise ValueError("M0 authority changed")
    return part_a, selected


def binding(part_a_path: Path, records: list[dict], sources: dict[str, str]) -> dict:
    """Bind a deterministic 28-world allocation and code to every ledger/unit."""
    return {
        "namespace": NAMESPACE,
        "part_a_sha256": verify_export(part_a_path),
        "allocation_sha256": value_sha([r["world_id"] for r in records]),
        "source_manifest": sources,
    }


def unit_path(attempt: Path, index: int) -> Path:
    return attempt / "units" / f"world-{index:02d}.json"


def verify_ledger(ledger: dict, identity: dict, records: list[dict]) -> dict[int, dict]:
    """Skip exactly path/size/hash-verified units of this allocation, never foreign work."""
    if ledger.get("schema") != "bl-c15-m0-ledger/v1" or ledger.get("binding") != identity:
        raise ValueError("foreign ledger namespace/allocation/source")
    entries = ledger.get("units")
    if not isinstance(entries, list):
        raise ValueError("ledger units must be a list")
    result = {}
    for entry in entries:
        index = entry["allocation_index"]
        if type(index) is not int or not 0 <= index < len(records) or index in result:
            raise ValueError("duplicate/foreign allocation index")
        path = Path(entry["path"])
        if (
            entry["world_id"] != records[index]["world_id"]
            or path.name != f"world-{index:02d}.json"
            or verify_export(path) != entry["sha256"]
            or path.stat().st_size != entry["size"]
        ):
            raise ValueError("ledger unit identity/hash/size mismatch")
        # Do not read outcome contents here: only the aggregator opens result payloads.
        result[index] = entry
    charged = ledger.get("charged_seconds")
    attempts = ledger.get("completed_attempts")
    if (
        not isinstance(charged, (float, int))
        or isinstance(charged, bool)
        or not math.isfinite(charged)
        or charged < 0
        or charged >= CAP_SECONDS
        or type(attempts) is not int
        or attempts not in (0, 1)
        or (attempts == 0 and (charged != 0 or entries))
    ):
        raise ValueError("invalid/expired attempt envelope")
    return result


def make_ledger(part_a_path: Path, output: Path, previous: Path | None = None) -> dict:
    """Create empty first ledger or one outcome-blind mechanical-completion ledger."""
    part_a, _ = load_part_a(part_a_path)
    records = part_a["rows"]
    identity = binding(part_a_path, records, source_manifest())
    units, charged, attempts = [], 0.0, 0
    if previous is not None:
        terminal_path = previous / "attempt.json"
        verify_export(terminal_path)
        terminal = json.loads(terminal_path.read_text())
        if (
            terminal.get("binding") != identity
            or terminal.get("status") != "INCOMPLETE_MECHANICAL"
            or terminal.get("attempt_number") != 1
            or terminal.get("verified_stop") is not True
            or terminal.get("failure_kind") not in ("cap", "host_guard", "worker_exit")
        ):
            raise ValueError("retry requires first mechanically failed, verified-stopped attempt")
        old_ledger_path = Path(terminal["ledger_path"])
        if verify_export(old_ledger_path) != terminal["ledger_sha256"]:
            raise ValueError("prior ledger changed")
        old = json.loads(old_ledger_path.read_text())
        prior = verify_ledger(old, identity, records)
        if prior:
            raise ValueError("only one completion retry is permitted by PLAN_v3")
        files = sorted((previous / "units").glob("*.json"))
        allowed = {unit_path(previous, i).name: i for i in range(len(records))}
        if any(p.name not in allowed for p in files):
            raise ValueError("foreign unit filename")
        for marker in (previous / "units").glob("*.json.complete"):
            payload = Path(str(marker).removesuffix(".complete"))
            if payload.name not in allowed or not payload.is_file():
                raise ValueError("foreign/orphan completion marker")
        for path in files:
            # A payload published before a mechanical interruption is not a committed unit
            # until its marker is present. Preserve it; recompute only in the fresh attempt.
            if not Path(str(path) + ".complete").exists():
                continue
            index = allowed[path.name]
            units.append(
                {
                    "allocation_index": index,
                    "world_id": records[index]["world_id"],
                    "path": str(path.resolve()),
                    "sha256": verify_export(path),
                    "size": path.stat().st_size,
                }
            )
        charged, attempts = float(terminal["series_charged_seconds"]), 1
        if len(units) == len(records):
            raise ValueError("no missing units to complete")
    ledger = {
        "schema": "bl-c15-m0-ledger/v1",
        "binding": identity,
        "units": units,
        "charged_seconds": charged,
        "completed_attempts": attempts,
    }
    verify_ledger(ledger, identity, records)
    export_json(output, ledger)
    return ledger


def probe_step(world: Any, decision: int, search: bool, check: Callable[[], None]) -> dict:
    """Search prepared independent clones, then dispatch and execute only bare B8."""
    import numpy as np

    from research.bl_served_search.ceiling import _choose, _legal
    from research.bl_served_search.exact_stop import run
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q, rollout_all
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected = {}

    def choose(prepared: Any) -> Any:
        check()
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        advisory = prepared.get_advisory_action_mask()[0, 0]
        obs = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
        legal, bare = _legal(q, advisory)
        selected.update(bare=bare, proposed=bare, length=int(prepared.length[0, 0]), batch=None)
        if search and np.asarray(advisory, dtype=bool).any():
            root = world.clone()

            def evaluate(action: int) -> dict:
                check()
                clock = 0 if world.mix == "scripted" else decision
                outcome = rollout_all(root.clone(), [action], 300, clock, record_outcomes=True)[
                    action
                ]
                if clock != decision and outcome.get("death_decision") is not None:
                    outcome["death_decision"] += decision
                return outcome

            result = run([int(a) for a in np.flatnonzero(legal)], bare, q, evaluate, _choose)
            selected.update(
                proposed=int(result["chosen"]), batch={"decision": decision, "bare": bare, **result}
            )
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        if int(actions[0, 0]) != bare:
            raise RuntimeError("B8 dispatch/Q mismatch")
        # Never write proposed into actions: a probe is diagnostic only, including delay ranking.
        selected["executed"] = int(actions[0, 0])
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return selected


def replay_world(
    row: dict, record: dict, b8_path: Path, checkpoint_dir: Path, check: Callable[[], None]
) -> dict:
    """Reconstruct the entire bare prefix and attest it against every saved baseline frame."""
    from research.bl_k1.native import RowStats, configure
    from research.bl_served_search.ceiling import (
        opponent_attestation,
        portable_plan,
        worker_threads,
    )
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    configure_cpu()
    plan = portable_plan(row["plan"], checkpoint_dir, allow_absolute=True)
    stats = RowStats()
    f, death, a = record["f"], row["baseline"]["death_frame"], None
    batches, controls, frames_checked, changes_proposed = [], 0, 0, 0
    empty_advisory_decisions = []
    with configure("stock", stats):
        world = build_world("b8_s3", plan, hero_checkpoint=b8_path)
        attestation = opponent_attestation(world)
        if attestation != {
            key: row[key] for key in ("opponent_member_sha256s", "opponent_veto_slots")
        }:
            raise RuntimeError("T3 opponent identity differs")
        if any(attestation["opponent_veto_slots"]):
            raise RuntimeError("M0 T3 roster must have no opponent veto")
        for decision in range(int(death) + 1):
            check()
            search = decision >= f and (decision - f) % 5 == 0
            selected = probe_step(world, decision, search, check)
            alive = bool(world.sim.alive[0, 0])
            mass = int(world.sim.length[0, 0]) if alive else 0
            if (
                selected["executed"] != selected["bare"]
                or alive != row["trajectories"]["baseline_alive"][decision]
                or mass != row["trajectories"]["baseline_mass"][decision]
            ):
                raise RuntimeError(f"bare trajectory/control mismatch at {decision}")
            frames_checked += 1
            batch = selected["batch"]
            if search and batch is None:
                empty_advisory_decisions.append(decision)
            if batch is not None:
                outcomes = batch["outcomes"]
                bare_outcome = outcomes[batch["bare"]]
                expected_survival = death >= decision + 300
                if bool(bare_outcome["survives"]) != expected_survival:
                    raise RuntimeError("bare branch survival mismatch")
                if not expected_survival and bare_outcome["death_decision"] != death:
                    raise RuntimeError("bare branch death mismatch")
                controls += 1
                changes_proposed += selected["proposed"] != selected["bare"]
                if any(
                    o.get("survives") is True
                    for action, o in outcomes.items()
                    if action != batch["bare"]
                ):
                    a = decision
                if decision == f:
                    original = next(
                        b for b in row["dense"]["batch_diagnostics"] if b["decision"] == f
                    )
                    actual = {k: batch[k] for k in original}
                    # In-memory outcome keys are integers; saved JSON keys are strings.
                    if (
                        json.loads(json.dumps(actual)) != original
                        or selected["length"] != record["hero_length_at_f"]
                    ):
                        raise RuntimeError("f reconstruction differs from dense detection batch")
                batches.append(batch)
            if decision == f and batch is None:
                raise RuntimeError("missing search at detection")
            if not alive:
                if (
                    decision != death
                    or int(world.sim.get_death_cause()[0, 0]) != record["baseline_death_cause"]
                ):
                    raise RuntimeError("baseline death identity mismatch")
                if selected["length"] != record["baseline_death_length"]:
                    raise RuntimeError("baseline death length differs")
                break
        if frames_checked != death + 1 or bool(world.sim.alive[0, 0]):
            raise RuntimeError("bare replay did not terminate at saved death")
    return {
        "world_id": record["world_id"],
        "mix": record["mix"],
        "f": f,
        "D": record["D"],
        "a": a,
        "a_minus_f": a - f if a is not None else None,
        "batches": batches,
        "baseline_frames_verified": frames_checked,
        "bare_branch_controls_verified": controls,
        "empty_advisory_decisions": empty_advisory_decisions,
        "proposed_changes": changes_proposed,
        "executed_changes": 0,
        "search_mode": "exact_stop",
        "native_mode": "stock",
        "worker_threads": worker_threads(),
        "native_v8": stats.as_dict(),
        "cadence": "f + 5*k through bare death; empty advisory skipped as in T3",
        "opponents": attestation,
    }


def _worker(
    row: dict,
    record: dict,
    b8: str,
    checkpoints: str,
    deadline: float,
    path: str,
    identity: dict,
    index: int,
) -> None:
    def check() -> None:
        if time.monotonic() >= deadline:
            raise CapExpired("M0 series 1 Mac-hour exhausted")

    try:
        result = replay_world(row, record, Path(b8), Path(checkpoints), check)
        export_json(
            Path(path),
            {
                "schema": "bl-c15-m0-unit/v1",
                "binding": identity,
                "allocation_index": index,
                "result": result,
            },
        )
    except Exception as error:
        export_json(
            Path(path).parent.parent / "errors" / Path(path).name,
            {
                "failure_kind": "cap" if isinstance(error, CapExpired) else "validation_error",
                "exception_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


def run(args: argparse.Namespace) -> dict:
    """Hold the same process-live Mac-wide fence as M1 and the other timed lanes."""
    from research.mac_lease import held

    with held(
        args.lease_owner, "PLAN_v3 M0 Part B bare LAD replay", seconds=CAP_SECONDS, path=args.lease
    ) as lease:
        return run_leased(args, lease)


def run_leased(args: argparse.Namespace, lease: Any) -> dict:
    """Coordinate one serial worker under an owned canonical Mac lease."""
    from research.bl_mac_latency.host import GuardError, check_guard, snapshot
    from research.mac_lease import LeaseError

    def require() -> None:
        current = json.loads(lease.path.read_text())
        if (
            current.get("id") != lease.lease_id
            or current.get("owner") != lease.owner
            or current.get("until_epoch", 0) <= time.time()
        ):
            raise LeaseError("Mac-wide lease lost/expired")

    if platform.system() != "Darwin" or os.nice(0) < 10:
        raise ValueError("Mac only, nice >= 10")
    if any(os.environ.get(k) != "1" for k in THREAD_VARS):
        raise ValueError("all numerical thread caps must be one")
    started = time.monotonic()
    part_a, rows = load_part_a(args.part_a)
    records = part_a["rows"]
    identity = binding(args.part_a, records, source_manifest())
    verify_export(args.ledger)
    ledger = json.loads(args.ledger.read_text())
    verified = verify_ledger(ledger, identity, records)
    if digest(args.b8_path) != B8_SHA:
        raise ValueError("wrong B8 bytes")
    require()
    host = snapshot()
    if not host["quiet"]:
        raise ValueError("Mac idle receipt required; foreign load present")
    args.out.mkdir(parents=True, exist_ok=False)
    # Three host-guard subprocesses each have a 15-second timeout. Reserve that
    # worst case plus owned-child stop/join/export time inside the one-hour cap.
    deadline = started + CAP_SECONDS - ledger["charged_seconds"] - 60
    export_json(
        args.out / "start.json",
        {
            "binding": identity,
            "host": host,
            "ledger_path": str(args.ledger.resolve()),
            "ledger_sha256": digest(args.ledger),
            "attempt_number": ledger["completed_attempts"] + 1,
            "cap_seconds": CAP_SECONDS,
            "base_forecast_seconds": 1800,
            "budget_note": "effective PLAN_v3 <=1 Mac-h already doubled; shared across attempts",
            "workers": 1,
            "math_threads_per_worker": 1,
            "lease": lease.receipt,
        },
    )
    failure, completed, child = None, len(verified), None
    context = multiprocessing.get_context("spawn")
    try:
        for index, (row, record) in enumerate(zip(rows, records)):
            if index in verified:
                continue
            require()
            check_guard()
            if time.monotonic() >= deadline:
                failure = "cap"
                break
            child = context.Process(
                target=_worker,
                args=(
                    row,
                    record,
                    str(args.b8_path),
                    str(args.checkpoint_dir),
                    deadline,
                    str(unit_path(args.out, index)),
                    identity,
                    index,
                ),
            )
            child.start()
            last_report = last_guard = time.monotonic()
            while child.is_alive():
                child.join(timeout=min(5, max(0.01, deadline - time.monotonic())))
                now = time.monotonic()
                if now >= deadline:
                    failure = "cap"
                    break
                if now - last_guard >= 60:
                    require()
                    check_guard()
                    last_guard = now
                if now - last_report >= 120:
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(records),
                                "elapsed_seconds": now - started,
                            }
                        ),
                        flush=True,
                    )
                    last_report = now
            if failure:
                break
            if child.exitcode != 0:
                error_path = args.out / "errors" / unit_path(args.out, index).name
                if error_path.exists():
                    verify_export(error_path)
                    failure = json.loads(error_path.read_text())["failure_kind"]
                else:
                    failure = "worker_exit"
                break
            verify_export(unit_path(args.out, index))
            completed += 1
            print(json.dumps({"completed": completed, "total": len(records)}), flush=True)
    except Exception as error:
        failure = (
            "host_guard" if isinstance(error, (GuardError, LeaseError)) else "validation_error"
        )
        raise
    except BaseException:
        failure = "operator_interruption"
        raise
    finally:
        if child is not None and child.is_alive():
            child.terminate()  # Only this coordinator's own child.
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join()
        elapsed = time.monotonic() - started
        terminal = {
            "schema": "bl-c15-m0-attempt/v1",
            "binding": identity,
            "status": (
                "COMPLETE_UNAGGREGATED"
                if failure is None
                else "INVALID" if failure == "validation_error" else "INCOMPLETE_MECHANICAL"
            ),
            "failure_kind": failure,
            "completed_units": completed,
            "total_units": len(records),
            "verified_stop": child is None or not child.is_alive(),
            "attempt_number": ledger["completed_attempts"] + 1,
            "ledger_path": str(args.ledger.resolve()),
            "ledger_sha256": digest(args.ledger),
            "elapsed_seconds": elapsed,
            "series_charged_seconds": ledger["charged_seconds"] + elapsed,
        }
        export_json(args.out / "attempt.json", terminal)
    return terminal


def curves(results: list[dict]) -> dict:
    """Fixed denominators, equality reachable: a >= f + L + 2."""
    return {
        group: [
            {
                "L": lead,
                "reachable": sum(
                    r["a"] is not None and r["a"] >= r["f"] + lead + 2 for r in selected
                ),
                "denominator": len(selected),
                "fraction": sum(
                    r["a"] is not None and r["a"] >= r["f"] + lead + 2 for r in selected
                )
                / len(selected),
            }
            for lead in sorted(set(LEADS) | {432} | set(range(75, 433, 5)))
        ]
        for group in ("frozen+mixed", "scripted")
        for selected in [[r for r in results if (r["mix"] == "scripted") == (group == "scripted")]]
    }


def validate_result(result: dict, record: dict) -> None:
    """Recompute LAD, cadence and bare controls from the retained diagnostic batches."""
    death, f = record["baseline_death_frame"], record["f"]
    scheduled = list(range(f, death + 1, 5))
    batches = result["batches"]
    decisions = [batch["decision"] for batch in batches]
    empty = result["empty_advisory_decisions"]
    if (
        decisions != sorted(set(decisions))
        or empty != sorted(set(empty))
        or set(decisions) & set(empty)
        or sorted(decisions + empty) != scheduled
        or f not in decisions
        or result["search_mode"] != "exact_stop"
        or result["native_mode"] != "stock"
    ):
        raise ValueError("invalid cadence/search identity")
    last = None
    for batch in batches:
        order = batch["evaluation_order"]
        outcomes = batch["outcomes"]
        bare = batch["bare"]
        if (
            not order
            or order[0] != bare
            or len(set(order)) != len(order)
            or batch["branches_evaluated"] != len(order)
            or len(outcomes) != len(order) + batch["branches_skipped"]
        ):
            raise ValueError("invalid branch counters")
        evaluated = {a: outcomes[str(a)] for a in order}
        survivors = [a for a in order if evaluated[a].get("survives") is True]
        if any(a < 0 or a > 5 for a in order) or any(
            o.get("survives") not in (True, False) for o in evaluated.values()
        ):
            raise ValueError("invalid branch outcomes")
        for outcome in evaluated.values():
            frames = outcome.get("frames_executed")
            if type(frames) is not int or not 1 <= frames <= 300:
                raise ValueError("invalid H300 branch frame count")
            survived = outcome["survives"]
            expected_death = None if survived else batch["decision"] + frames - 1
            if (survived and frames != 300) or outcome.get("death_decision") != expected_death:
                raise ValueError("invalid H300 survivor/death coordinate")
            if "horizon" in outcome and outcome["horizon"] != 300:
                raise ValueError("clipped branch horizon")
            if "frames_survived" in outcome and outcome["frames_survived"] != (
                300 if survived else frames - 1
            ):
                raise ValueError("invalid alive branch frames")
        if (survivors and (survivors != [order[-1]] or batch["chosen"] != survivors[0])) or (
            not survivors and batch["branches_skipped"] != 0
        ):
            raise ValueError("invalid exact-stop termination")
        if not survivors:
            best = max(outcome["frames_executed"] for outcome in evaluated.values())
            expected_chosen = next(a for a in order if evaluated[a]["frames_executed"] == best)
            if batch["chosen"] != expected_chosen:
                raise ValueError("all-die delay ranking differs")
        for action, outcome in outcomes.items():
            if int(action) not in order and outcome != {"status": "NOT_EVALUATED"}:
                raise ValueError("invalid skipped outcome")
        bare_outcome = outcomes[str(bare)]
        if (
            bare_outcome["survives"] is not False
            or bare_outcome["death_decision"] != death
            or bare_outcome["frames_executed"] != death - batch["decision"] + 1
        ):
            raise ValueError("bare control differs from baseline death")
        if any(a != bare for a in survivors):
            last = batch["decision"]
    if (
        result["a"] != last
        or result["a_minus_f"] != (last - f if last is not None else None)
        or result["proposed_changes"] != sum(batch["chosen"] != batch["bare"] for batch in batches)
    ):
        raise ValueError("LAD/proposal summary differs from batches")


def validate_execution_attestation(result: dict, source_row: dict) -> None:
    """Verify worker math threads, stock counters and the exact non-vetoed T3 roster."""
    threads = result["worker_threads"]
    if (
        threads.get("torch") != 1
        or threads.get("numba") != 1
        or any(threads.get("environment", {}).get(k) != "1" for k in THREAD_VARS[:5])
    ):
        raise ValueError("worker numerical thread attestation differs")
    from research.bl_k1.native import RowStats

    if result["native_v8"] != RowStats().as_dict():
        raise ValueError("unexpected native rows in non-vetoed stock T3 replay")
    if result["opponents"] != {
        key: source_row[key] for key in ("opponent_member_sha256s", "opponent_veto_slots")
    }:
        raise ValueError("opponent attestation differs from L2 source")


def aggregate(part_a_path: Path, attempt: Path, output: Path) -> dict:
    """Take one look only after all 28 current/past units and counters verify."""
    part_a, source_rows = load_part_a(part_a_path)
    records = part_a["rows"]
    identity = binding(part_a_path, records, source_manifest())
    verify_export(attempt / "attempt.json")
    terminal = json.loads((attempt / "attempt.json").read_text())
    if (
        terminal["binding"] != identity
        or terminal["status"] != "COMPLETE_UNAGGREGATED"
        or terminal["verified_stop"] is not True
        or terminal["series_charged_seconds"] > CAP_SECONDS
    ):
        raise ValueError("INCOMPLETE: terminal completion required; no partial look")
    ledger_path = Path(terminal["ledger_path"])
    if verify_export(ledger_path) != terminal["ledger_sha256"]:
        raise ValueError("ledger changed")
    prior = verify_ledger(json.loads(ledger_path.read_text()), identity, records)
    current = list((attempt / "units").glob("*.json"))
    allowed = {unit_path(attempt, i).name for i in range(len(records)) if i not in prior}
    if {p.name for p in current} != allowed:
        raise ValueError("INCOMPLETE: missing, duplicate, or foreign units")
    results, hashes = [], {}
    for index, record in enumerate(records):
        path = Path(prior[index]["path"]) if index in prior else unit_path(attempt, index)
        hashes[str(path)] = verify_export(path)
        unit = json.loads(path.read_text())
        result = unit["result"]
        if (
            unit.get("schema") != "bl-c15-m0-unit/v1"
            or unit["binding"] != identity
            or unit["allocation_index"] != index
            or result["world_id"] != record["world_id"]
            or result["f"] != record["f"]
            or result["D"] != record["D"]
            or result["mix"] != record["mix"]
            or result["executed_changes"] != 0
            or result["baseline_frames_verified"] != record["baseline_death_frame"] + 1
            or result["bare_branch_controls_verified"] != len(result["batches"])
        ):
            raise ValueError("invalid unit counters/identity")
        validate_result(result, record)
        validate_execution_attestation(result, source_rows[index])
        results.append(result)
    receipt = {
        "schema": "bl-c15-m0-part-b/v1",
        "binding": identity,
        "status": "COMPLETE_D1_PENDING_M1",
        "development_only": True,
        "serving_or_release": "NOT_GRANTED",
        "rows": results,
        "curves": curves(results),
        "part_a_tables": part_a["tables"],
        "unit_sha256": hashes,
        "decision": "NOT_READY_M1_DOOM_W4_P99",
        "lead_rule": "L*=max(72,min(432,5*ceil((24*(p99_doom_W4+0.25)+2)/5)))",
        "close_rule": "F(L*)<0.50 => CLOSED; otherwise M0 admits C15 build subject to M1 PASS",
        "all_die_budget_required": part_a["tables"]["frozen+mixed"]["all_die_at_f"] > 18 / 3,
    }
    export_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-a", type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    ledger = commands.add_parser("ledger")
    ledger.add_argument("--out", required=True, type=Path)
    ledger.add_argument("--previous", type=Path)
    replay = commands.add_parser("run")
    replay.add_argument("--ledger", required=True, type=Path)
    replay.add_argument("--out", required=True, type=Path)
    replay.add_argument(
        "--lease",
        type=Path,
        default=None,
        help="default: canonical primary-checkout exp/mac-lease.json",
    )
    replay.add_argument("--lease-owner", default="claude-m0-part-b")
    replay.add_argument(
        "--b8-path",
        type=Path,
        default=Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"),
    )
    replay.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes"),
    )
    combine = commands.add_parser("aggregate")
    combine.add_argument("--attempt", required=True, type=Path)
    combine.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "check":
        part_a, _ = load_part_a(args.part_a)
        print(
            json.dumps(
                {
                    "status": "READY_FOR_CLAUDE_LEASED_REPLAY",
                    "worlds": len(part_a["rows"]),
                    "part_a_sha256": digest(args.part_a),
                    "source_manifest": source_manifest(),
                }
            )
        )
    elif args.command == "ledger":
        result = make_ledger(args.part_a, args.out, args.previous)
        print(
            json.dumps({"ledger_sha256": digest(args.out), "verified_units": len(result["units"])})
        )
    elif args.command == "run":
        print(json.dumps(run(args)))
    else:
        result = aggregate(args.part_a, args.attempt, args.out)
        print(json.dumps({"status": result["status"], "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
