"""PLAN_v3 C1: Claude-operated, leased Mac development bank; one sealed look."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from research.bl_k1 import m1
from research.bl_mac_latency import host
from research.bl_served_search import c1_bank, ceiling
from research.bl_served_search.exports import digest, export_json, verify_export

SCHEMA = "bl-plan-v3-c1/v1"
BASE_SECONDS = 8 * 3600
CAP_SECONDS = 16 * 3600  # Already doubled by PLAN_v3. Never double again.
WORKERS = 2
_STOP: Any = None
_DEADLINE = 0.0


def read(path: Path) -> dict:
    """Open only complete, hash-verified JSON exports."""
    verify_export(path)
    return json.loads(path.read_text())


def sources() -> dict[str, str]:
    """Bind the runner, Watch mirror, math, kernels, controls, and identities."""
    root = c1_bank.ROOT
    paths = list(root.glob("research/bl_served_search/c1*.py"))
    paths += [
        root / "research/bl_served_search/run_c1.sh",
        root / "research/published_ledger.py",
        root / "research/compute/thermal_guard.py",
    ]
    for directory in ("research/t34_pathb", "research/t34_shadow", "src/training", "src/utils"):
        paths += list((root / directory).rglob("*.py"))
    return {**m1.sources(), **{str(p.relative_to(root)): digest(p) for p in paths}}


def binding() -> dict:
    """Freeze sources/allocation/caps before any unit; no adaptive design fields."""
    if digest(c1_bank.PLAN) != c1_bank.PLAN_SHA:
        raise ValueError("PLAN_v3 authority bytes changed")
    value = {
        "schema": SCHEMA,
        "namespace": c1_bank.NAMESPACE,
        "sources": sources(),
        "allocation": c1_bank.allocation(),
        "plan_sha256": c1_bank.PLAN_SHA,
        "base_seconds": BASE_SECONDS,
        "cap_seconds": CAP_SECONDS,
        "workers": WORKERS,
        "threads": 1,
        "search_mode": "exact_stop",
        "native_mode": "kernel",
        "max_mechanical_completion_attempts": 1,
        "development_only": True,
    }
    return {**value, "identity": ceiling.digest_value(value)}


def inventory(path: Path, unit_id: str) -> dict:
    """Outcome-blind path/size/hash inventory, never a scientific look."""
    return {
        "unit_id": unit_id,
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": verify_export(path),
    }


def verify_completion(descriptor: dict, data: dict) -> None:
    """Bind sealed payload bytes to outcome-free completion metadata."""
    path = Path(descriptor["path"])
    expected = next((r for r in data["allocation"] if r["unit_id"] == descriptor["unit_id"]), None)
    if expected is None:
        raise ValueError("foreign completed C1 allocation")
    metadata = read(path.with_name(path.stem + ".ops.json"))
    if metadata != {
        "schema": SCHEMA + "/completed-pair",
        "namespace": c1_bank.NAMESPACE,
        "identity": data["identity"],
        "allocation": expected,
        "payload": descriptor,
    }:
        raise ValueError("foreign namespace/allocation or mismatched completed pair")


def verify_ledger(path: Path, data: dict) -> dict[str, dict]:
    """Skip exactly verified allocation units; refuse foreign namespaces/files."""
    value = read(path)
    if value.get("schema") != SCHEMA + "/ledger" or value.get("identity") != data["identity"]:
        raise ValueError("foreign C1 ledger identity")
    result = {}
    allowed = {row["unit_id"] for row in data["allocation"]}
    for row in value["units"]:
        unit = row["unit_id"]
        if unit not in allowed or unit in result:
            raise ValueError("foreign or duplicate C1 allocation")
        target = Path(row["path"])
        if not target.is_absolute() or not target.is_relative_to(c1_bank.STUDY_ROOT):
            raise ValueError("foreign C1 artifact root")
        if target.name != f"{unit}.json" or inventory(target, unit) != row:
            raise ValueError("C1 unit path/size/hash mismatch")
        verify_completion(row, data)
        result[unit] = row
    return result


def recovery(previous_path: Path | None, ledger: dict, data: dict) -> tuple[int, float]:
    """Only one verified mechanical completion; no scientific retries."""
    if previous_path is None:
        if ledger:
            raise ValueError("nonempty ledger requires previous stopped attempt")
        return 1, 0.0
    previous = read(previous_path)
    if (
        previous.get("schema") != SCHEMA + "/attempt"
        or previous.get("identity") != data["identity"]
        or previous.get("attempt_number") != 1
        or previous.get("status") != "INCOMPLETE"
        or previous.get("verified_stop") is not True
        or previous.get("mechanical_failure") is not True
        or previous.get("completed_unit_exports") != ledger
        or len(ledger) >= c1_bank.COUNT
    ):
        raise ValueError("recovery requires one verified mechanical stop and exact ledger")
    charged = previous.get("charged_seconds")
    if (
        type(charged) not in (float, int)
        or not math.isfinite(charged)
        or not 0 <= charged < CAP_SECONDS
    ):
        raise ValueError("invalid or exhausted cumulative C1 budget")
    return 2, charged


def make_ledger(path: Path, data: dict, previous_path: Path | None) -> None:
    """First attempt has an empty ledger; recovery copies verified units only."""
    frozen_path = c1_bank.STUDY_ROOT / "study.json"
    if frozen_path.exists():
        if read(frozen_path) != data:
            raise ValueError("C1 source/allocation drift; no new bank allowed")
    else:
        if previous_path is not None:
            raise ValueError("missing frozen C1 study")
        proof = c1_bank.freshness()
        export_json(c1_bank.STUDY_ROOT / "freshness.json", proof)
        export_json(frozen_path, data)
    units = {} if previous_path is None else read(previous_path)["completed_unit_exports"]
    recovery(previous_path, units, data)
    candidate = {
        "schema": SCHEMA + "/ledger",
        "identity": data["identity"],
        "units": list(units.values()),
    }
    # Validate via the same checks after create-only publication; never parse raw outcomes.
    export_json(path, candidate)
    verify_ledger(path, data)


def worker_init(stop: Any, deadline: float) -> None:
    """Each W2 process has exactly one numerical thread and inherits nice 10."""
    global _STOP, _DEADLINE
    import numba

    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    numba.set_num_threads(1)
    _STOP, _DEADLINE = stop, deadline


def worker_check() -> None:
    if _STOP.is_set() or time.monotonic() >= _DEADLINE:
        raise TimeoutError("mechanical C1 stop/cumulative cap")


def run_unit(row: dict, data: dict, out: Path) -> dict:
    """Publish a whole paired session only; return no efficacy to coordinator."""
    import torch

    from research.bl_served_search.c1_watch import run_arm

    worker_check()
    threads = ceiling.worker_threads()
    if (
        threads != {"environment": dict.fromkeys(ceiling.THREAD_ENV, "1"), "torch": 1, "numba": 1}
        or torch.get_num_interop_threads() != 1
        or os.getpriority(os.PRIO_PROCESS, 0) < 10
    ):
        raise ValueError("C1 worker threads/nice mismatch")
    arms = {name: run_arm(row["seed"], name == "dense", worker_check) for name in ("bare", "dense")}
    worker_check()
    unit = {
        **row,
        "schema": SCHEMA + "/unit",
        "namespace": c1_bank.NAMESPACE,
        "identity": data["identity"],
        "worker_threads": threads,
        "worker_pid": os.getpid(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "arms": arms,
    }
    path = out / f"{row['unit_id']}.json"
    export_json(path, unit)
    descriptor = inventory(path, row["unit_id"])
    export_json(
        path.with_name(path.stem + ".ops.json"),
        {
            "schema": SCHEMA + "/completed-pair",
            "namespace": c1_bank.NAMESPACE,
            "identity": data["identity"],
            "allocation": row,
            "payload": descriptor,
        },
    )
    return descriptor


def verify_unit(unit: dict, expected: dict, data: dict) -> None:
    """Validate execution attestation before the registered scientific look."""
    if (
        any(unit.get(k) != v for k, v in expected.items())
        or unit.get("schema") != SCHEMA + "/unit"
        or unit.get("identity") != data["identity"]
        or unit.get("namespace") != c1_bank.NAMESPACE
        or unit.get("worker_threads")
        != {"environment": dict.fromkeys(ceiling.THREAD_ENV, "1"), "torch": 1, "numba": 1}
        or unit.get("nice", 0) < 10
    ):
        raise ValueError("foreign C1 unit or execution attestation")
    import numpy as np

    from research.bl_served_search.exact_stop import run as exact_replay
    from research.t3_rollout_labels_20261015.trigger import DenseTrigger
    from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

    if set(unit["arms"]) != {"bare", "dense"}:
        raise ValueError("C1 paired arms missing")
    for name, arm in unit["arms"].items():
        if arm["mirror_identity_frames"] != c1_bank.HORIZON:
            raise ValueError("missing Watch mirror identity")
        roster = arm["roster"]
        if (
            roster["hero_checkpoint_sha256"] != B8_S3_SHA256
            or roster["opponent_checkpoint"]["sha256"] != FRP3_S12_SHA256
            or len(roster["snakes"]) != 6
        ):
            raise ValueError("C1 roster/checkpoint identity mismatch")
        if [row["frame"] for row in arm["frames"]] != list(range(1, c1_bank.HORIZON + 1)):
            raise ValueError("missing full-horizon raw frames")
        actual_frames = [row["frame"] for row in arm["frames"] if row["alive"] or row["death"]]
        if [row["frame"] for row in arm["decisions"]] != actual_frames:
            raise ValueError("actual hero decision denominator mismatch")
        batch_rows = {row["frame"]: row for row in arm["batches"]}
        if len(batch_rows) != len(arm["batches"]) or (name == "bare" and batch_rows):
            raise ValueError("duplicate batches or bare arm searched")
        trigger, life = None, None
        expected_frames = set()
        for index, decision in enumerate(arm["decisions"]):
            if decision["life"] != life:
                trigger, life = DenseTrigger(min_length=600, interval=5), decision["life"]
            q = np.asarray(decision["q"], dtype=np.float32)
            advisory = np.asarray(decision["advisory"], dtype=bool)
            legal, bare = ceiling._legal(q, advisory)
            if bare != decision["bare"]:
                raise ValueError("raw B8 Q/mask/bare mismatch")
            fired = name == "dense" and trigger.fires(index, decision["length"], True)
            if fired and advisory.any():
                expected_frames.add(decision["frame"])
                batch = batch_rows[decision["frame"]]
                outcomes = batch["outcomes"]
                replay = exact_replay(
                    np.flatnonzero(legal).tolist(),
                    bare,
                    q,
                    lambda a: outcomes[str(a)],
                    ceiling._choose,
                )
                for key in (
                    "chosen",
                    "evaluation_order",
                    "branches_evaluated",
                    "branches_skipped",
                    "outcomes",
                ):
                    expected_value = replay[key]
                    if key == "outcomes":
                        expected_value = {str(k): v for k, v in expected_value.items()}
                    if expected_value != batch[key]:
                        raise ValueError("frozen exact-stop rule/all-legal coverage mismatch")
                counters = batch["native_v8"]
                if counters["kernel_rows"] <= 0 or any(
                    counters[k] != 0 for k in ("stock_rows", "fallback_rows", "noncandidate_rows")
                ):
                    raise ValueError("missing per-decision native attestation")
                for action in batch["evaluation_order"]:
                    outcome = outcomes[str(action)]
                    if not 1 <= outcome["frames_executed"] <= 300 or (
                        outcome["survives"] and outcome["frames_executed"] != 300
                    ):
                        raise ValueError("short H300 or malformed branch")
                chosen = replay["chosen"]
            else:
                chosen = bare
            if decision["chosen"] != chosen:
                raise ValueError("delivered action differs from frozen rule")
        if expected_frames != set(batch_rows):
            raise ValueError("dense trigger eligibility mismatch")


def single_look(data: dict, ledger: dict, deadline: float | None = None, check: Any = None) -> dict:
    """All hashes/allocations verify before opening any outcome; look is create-only."""
    from research.bl_served_search.c1_reduce import reduce_sessions

    def check_cap() -> None:
        if check is not None:
            check()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("C1 cumulative cap exhausted at look boundary")

    check_cap()
    if set(ledger) != {row["unit_id"] for row in data["allocation"]}:
        raise ValueError("no partial C1 scientific look")
    for unit_id, descriptor in ledger.items():
        if inventory(Path(descriptor["path"]), unit_id) != descriptor:
            raise ValueError("C1 unit changed before look")
        verify_completion(descriptor, data)
    check_cap()
    export_json(
        c1_bank.STUDY_ROOT / "look-started.json", {"identity": data["identity"], "units": ledger}
    )
    units = []
    for expected in data["allocation"]:
        unit = read(Path(ledger[expected["unit_id"]]["path"]))
        verify_unit(unit, expected, data)
        units.append(unit)
    check_cap()
    result = reduce_sessions(units)
    check_cap()
    result.update(identity=data["identity"], schema=SCHEMA + "/decision", development_only=True)
    export_json(c1_bank.STUDY_ROOT / "decision.json", result)
    return result


def run(data: dict, ledger: dict, number: int, charged: float, ledger_path: Path) -> None:
    """One exclusive lease, idle receipt, W2, host guard, and cumulative wall cap."""
    from research.mac_lease import held
    from research.published_ledger import require_published_ledger

    started = time.monotonic()
    publication = require_published_ledger(c1_bank.ROOT, "research/bl_served_search/c1.py")
    if publication["authoring_head"] != publication["origin_main"]:
        raise ValueError("merge current origin/main before C1 dispatch")
    if platform.system() != "Darwin" or os.environ.get("SNAKE_DQN_DEVICE") != "cpu":
        raise ValueError("C1 is Mac CPU only")
    if any(os.environ.get(key) != "1" for key in ceiling.THREAD_ENV):
        raise ValueError("C1 requires one thread per worker")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("use the nice 10 C1 launcher")
    if read(c1_bank.STUDY_ROOT / "study.json") != data:
        raise ValueError("C1 requires frozen study and ledger")
    out = c1_bank.STUDY_ROOT / f"attempt-{number:02d}"
    if out.exists():
        raise ValueError("prior attempt exists; reconcile its verified stop")
    remaining_seconds = CAP_SECONDS - charged - (time.monotonic() - started)
    if remaining_seconds <= 0:
        raise TimeoutError("C1 budget exhausted during admission")
    with held(
        "claude-c1", "PLAN_v3 C1 served-roster development bank", seconds=remaining_seconds
    ) as lease:
        deadline = started + CAP_SECONDS - charged
        out.mkdir(parents=True, exist_ok=False)
        completed = dict(ledger)
        error, invalid, verified_stop = None, False, True
        pool = monitor = caffeine = stop = None
        monitor_started = False
        result = None
        pending = {}

        def admit() -> None:
            if time.monotonic() >= deadline:
                raise TimeoutError("C1 cumulative cap exhausted")
            if monitor is not None and monitor.error:
                raise TimeoutError(monitor.error)
            if caffeine is not None and caffeine.poll() is not None:
                raise TimeoutError("caffeinate exited")

        def observe() -> dict:
            if json.loads(lease.path.read_text()).get("id") != lease.lease_id:
                raise ValueError("C1 lease ownership changed")
            if time.time() >= lease.until_epoch:
                raise TimeoutError("C1 lease expired")
            return {**host.check_guard(), "process_screen": host.process_screen()}

        try:
            # All setup, including the full freshness rescan, is charged under
            # the same lease and cumulative deadline as the paired sessions.
            export_json(out / "freshness.json", c1_bank.freshness())
            admit()
            idle = host.snapshot()
            export_json(out / "idle.json", idle)
            if not idle["quiet"]:
                raise TimeoutError("initial host not idle; no workers admitted")
            export_json(
                out / "start.json",
                {
                    "identity": data["identity"],
                    "mac_lease": lease.receipt,
                    "ledger_sha256": digest(ledger_path),
                    "charged_before": charged,
                    "publication": publication,
                },
            )
            caffeine = subprocess.Popen(["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())])
            context = multiprocessing.get_context("spawn")
            stop = context.Event()
            monitor = m1.HostMonitor(observe)
            monitor.thread.start()
            monitor_started = True
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=WORKERS,
                mp_context=context,
                initializer=worker_init,
                initargs=(stop, deadline),
            )
            remaining = iter(row for row in data["allocation"] if row["unit_id"] not in ledger)
            exhausted = False
            while pending or not exhausted:
                admit()
                while not exhausted and len(pending) < WORKERS:
                    row = next(remaining, None)
                    if row is None:
                        exhausted = True
                    else:
                        pending[pool.submit(run_unit, row, data, out)] = row
                done, _ = concurrent.futures.wait(
                    pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    row = pending.pop(future)
                    descriptor = future.result()
                    if descriptor != inventory(Path(descriptor["path"]), row["unit_id"]):
                        raise ValueError("worker export inventory mismatch")
                    verify_completion(descriptor, data)
                    completed[row["unit_id"]] = descriptor
                    print(json.dumps({"completed_pairs": len(completed), "total": 24}), flush=True)
            # A fault appearing with the final future still blocks the look.
            admit()
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            invalid = isinstance(exc, ValueError)
            if isinstance(exc, host.GuardError):
                export_json(out / "failed-host.json", exc.observation)
        finally:
            if stop is not None:
                stop.set()
            if pool is not None:
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                except BaseException as exc:
                    verified_stop = False
                    error = f"unverified worker shutdown: {exc}"
            try:
                # A pair is complete only after the outcome-free .ops marker,
                # published last, binds both arms' immutable payload bytes.
                for row in data["allocation"]:
                    path = out / f"{row['unit_id']}.json"
                    metadata = path.with_name(path.stem + ".ops.json")
                    if Path(str(metadata) + ".complete").exists():
                        descriptor = inventory(path, row["unit_id"])
                        verify_completion(descriptor, data)
                        completed[row["unit_id"]] = descriptor
            except Exception as exc:
                invalid, error = True, f"invalid completed export: {exc}"
        if monitor is not None and monitor.error:
            error = monitor.error
        try:
            if sources() != data["sources"]:
                invalid, error = True, "C1 source drift"
        except Exception as exc:
            invalid, error = True, f"C1 source inventory unavailable: {exc}"
        complete = len(completed) == 24 and error is None and verified_stop
        if complete:
            try:
                admit()
                result = single_look(data, completed, deadline, admit)
            except TimeoutError as exc:
                error, complete = str(exc), False
            except (Exception, KeyboardInterrupt) as exc:
                invalid, error, complete = True, f"C1 look invalid: {exc}", False
        if monitor_started:
            monitor.close()
        if caffeine is not None:
            try:
                caffeine.terminate()
                caffeine.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                verified_stop, complete = False, False
                error = f"unverified caffeinate shutdown: {exc}"
        spent = charged + time.monotonic() - started
        if spent >= CAP_SECONDS:
            complete, error = False, "C1 cumulative wall cap exhausted"
        status = "INVALID" if invalid else ("COMPLETE" if complete else "INCOMPLETE")
        export_json(
            out / "receipt.json",
            {
                "schema": SCHEMA + "/attempt",
                "identity": data["identity"],
                "attempt_number": number,
                "status": status,
                "reason": error,
                "mechanical_failure": status == "INCOMPLETE",
                "verified_stop": verified_stop,
                "completed_unit_exports": completed,
                "charged_seconds": spent,
                "mac_lease": lease.receipt,
                "host": [] if monitor is None else monitor.rows,
                "base_seconds": BASE_SECONDS,
                "cap_seconds": CAP_SECONDS,
            },
        )
        if result is not None:
            print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--make-ledger", type=Path)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    data = binding()
    if args.make_ledger:
        make_ledger(args.make_ledger, data, args.previous_attempt)
        return
    ledger = verify_ledger(args.ledger, data) if args.ledger else {}
    number, charged = recovery(args.previous_attempt, ledger, data)
    if args.check:
        print(
            json.dumps(
                {
                    "status": "CHECK_ONLY",
                    "identity": data["identity"],
                    "freshness": c1_bank.freshness(),
                    "allocation": data["allocation"],
                    "base_seconds": BASE_SECONDS,
                    "cap_seconds": CAP_SECONDS,
                    "remaining_seconds": CAP_SECONDS - charged,
                    "workers": 2,
                    "threads": 1,
                    "skipped": list(ledger),
                    "production_run": False,
                }
            )
        )
        return
    if args.ledger is None:
        parser.error("--run requires an explicit verified ledger, empty on attempt 1")
    run(data, ledger, number, charged, args.ledger)


if __name__ == "__main__":
    main()
