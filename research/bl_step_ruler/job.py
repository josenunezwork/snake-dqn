"""Step -1 world-step ruler runtime: dev-namespace F0 parents, counters only.

Each allocation is one unchanged F0 ``collect_unit`` parent run under the
observational tick wrappers of :mod:`ticks`. Label rows, outcomes, rescue and
decisive counts are never written: ``publish_life`` is a no-op and only work and
timing counters leave the worker. One unit = one parent, exported atomically as
``units/<allocation_id>.json`` with ``.sha256`` then ``.complete`` last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path
from typing import Any

UNIT_SCHEMA = "bl-step-ruler-unit/v1"
LEDGER_SCHEMA = "bl-step-ruler-ledger/v1"
CONTROL_SCHEMA = "bl-step-ruler-control/v1"
UNIT_DIR = "units"
# Work counters only. Outcome-bearing collector counters are dropped on purpose.
WORK_COUNTERS = ("deaths", "roots", "digest_checks", "bare_rerolls")
SEALED_COUNTERS = (
    "decisive_rows",
    "rescue_bearing_lives",
    "h1440",
    "reversals",
    "bare_audit_mismatches",
)
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def source_sha256(root: Path | None = None) -> str:
    """Digest this package plus the collector closure files it measures."""
    root = root or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "research/bl_step_ruler").glob("*.py")) + [
        root / "research/bl_labels/factory.py",
        root / "research/bl_served_search/b2_engine.py",
    ]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def empty_ledger(namespace: str, allocation: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": namespace,
        "allocation_sha256": allocation_sha256(allocation),
        "units": [],
    }


def allocation_sha256(allocation: list[dict[str, Any]]) -> str:
    payload = json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_ledger(path: Path, namespace: str, allocation: list[dict[str, Any]]) -> set[str]:
    """Return verified unit IDs; refuse foreign namespaces/allocations/duplicates."""
    value = json.loads(Path(path).read_text())
    if value.get("schema") != LEDGER_SCHEMA or value.get("namespace") != namespace:
        raise ValueError("foreign step-ruler ledger namespace")
    if value.get("allocation_sha256") != allocation_sha256(allocation):
        raise ValueError("foreign step-ruler ledger allocation")
    known = {item["allocation_id"] for item in allocation}
    ids = [row.get("id") for row in value.get("units", [])]
    if len(ids) != len(set(ids)) or not set(ids) <= known:
        raise ValueError("duplicate or foreign step-ruler ledger unit")
    for row in value["units"]:
        if row.get("path") != f"{UNIT_DIR}/{row['id']}.json":
            raise ValueError("step-ruler ledger path differs from unit id")
        if type(row.get("size")) is not int or len(str(row.get("sha256", ""))) != 64:
            raise ValueError("step-ruler ledger row lacks size/hash")
    return set(ids)


def verify_unit(path: Path) -> str:
    """Check payload/sidecar/marker triplet; return the payload SHA256."""
    path = Path(path)
    marker = path.with_name(path.name + ".complete")
    sidecar = path.with_name(path.name + ".sha256")
    if not (path.is_file() and sidecar.is_file() and marker.is_file()):
        raise ValueError(f"incomplete step-ruler unit triplet: {path.name}")
    if marker.read_bytes() != b"complete\n":
        raise ValueError(f"bad completion marker: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sidecar.read_text() != f"{digest}  {path.name}\n":
        raise ValueError(f"step-ruler unit digest differs: {path.name}")
    return digest


def build_ledger(
    control: dict[str, Any], exports: list[Path], out: Path, prior: Path | None = None
) -> dict[str, Any]:
    """Mac-side, path/size/hash-only union of verified units (create-only)."""
    from research.bl_labels.exports import publish_json

    namespace, allocation = control["namespace"], control["allocation"]
    known = {item["allocation_id"] for item in allocation}
    rows: dict[str, dict[str, Any]] = {}
    if prior is not None:
        verify_unit(prior)
        load_ledger(prior, namespace, allocation)
        for row in json.loads(prior.read_text())["units"]:
            rows[row["id"]] = row
    for root in exports:
        directory = Path(root) / UNIT_DIR
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            unit = path.stem
            if unit not in known:
                raise ValueError(f"foreign step-ruler unit {unit}")
            digest = verify_unit(path)
            row = {
                "id": unit,
                "path": f"{UNIT_DIR}/{unit}.json",
                "sha256": digest,
                "size": path.stat().st_size,
                "source": str(path.resolve()),
            }
            if unit in rows:
                if rows[unit]["sha256"] != digest:
                    raise ValueError(f"duplicate step-ruler unit with different bytes: {unit}")
                continue
            rows[unit] = row
    value = {
        **empty_ledger(namespace, allocation),
        "units": [rows[key] for key in sorted(rows)],
    }
    publish_json(out, value)
    return value


def _unit_payload(
    item: dict[str, Any], namespace: str, result: dict[str, Any], ticks: dict[str, Any]
) -> dict[str, Any]:
    counters = result.get("counters", {})
    work = {key: int(counters.get(key, 0)) for key in WORK_COUNTERS}
    return {
        "schema": UNIT_SCHEMA,
        "namespace": namespace,
        "allocation_id": item["allocation_id"],
        "stream": item["stream"],
        "seed": item["seed"],
        "horizon": item["horizon"],
        "status": result.get("status"),
        "world_steps": ticks["world_steps"],
        "tick_counters": ticks,
        "work_counters": work,
        "timing_seconds": dict(result.get("timing_seconds", {})),
        "elapsed_seconds": float(result.get("provenance", {}).get("elapsed_seconds", 0.0)),
        "provenance": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "nproc": os.cpu_count(),
            "code_sha": os.environ.get("F0_COMMIT"),
            "source_sha256": os.environ.get("STEP_RULER_SOURCE_SHA256"),
            "thread_env": {key: os.environ.get(key) for key in THREAD_ENV},
        },
        "sealed_counters_dropped": list(SEALED_COUNTERS),
    }


def _collect(job: tuple[dict, str, str, str, str, float | None]) -> dict[str, Any]:
    """Spawned worker: one numerical thread; returns a counters-only status."""
    item, b8, frp3, namespace, output, deadline = job
    os.environ.update({key: "1" for key in THREAD_ENV})
    from research.bl_labels import factory
    from research.bl_labels.exports import publish_json

    from .ticks import TickCounter, instrument

    def progress(value: dict) -> None:
        print(
            f"STEP_RULER_PROGRESS allocation={item['allocation_id']} "
            f"done={value.get('done_decisions')} total={value.get('total_decisions')}",
            flush=True,
        )

    supplied = dict(item)
    supplied["publish_life"] = lambda _value: None  # rows are never persisted
    supplied["progress"] = progress
    if deadline is not None:
        supplied["deadline_monotonic"] = deadline
    counter = TickCounter()
    started = time.monotonic()
    try:
        with instrument(counter):
            result = factory.collect_unit(supplied, Path(b8), Path(frp3), namespace)
    except factory.FactoryInvalid as exc:
        return {"allocation_id": item["allocation_id"], "status": "INVALID", "reason": str(exc)}
    except RuntimeError as exc:
        if "INCOMPLETE F0 unit wall cap" not in str(exc):
            raise
        return {
            "allocation_id": item["allocation_id"],
            "status": "INCOMPLETE",
            "reason": str(exc),
            "partial_world_steps": counter.world_steps(),
            "seconds": time.monotonic() - started,
        }
    if counter.stack != ["play"]:
        raise RuntimeError("step-ruler phase stack not balanced")
    payload = _unit_payload(item, namespace, result, counter.as_dict())
    del result  # label rows are dropped with the result object
    publish_json(Path(output) / UNIT_DIR / f"{item['allocation_id']}.json", payload)
    return {
        "allocation_id": item["allocation_id"],
        "status": "COMPLETE",
        "world_steps": payload["world_steps"],
        "seconds": time.monotonic() - started,
    }


def self_check(b8: Path, frp3: Path, seed: int) -> dict[str, Any]:
    """Exercise the real replay and branch seams once and check the counters.

    Two natural decisions are recorded from a dev Watch6 world, then
    ``factory._restore_root`` replays them (2 replay ticks, 1 replay restore)
    and ``factory._branch`` runs one H300 branch (<=300 branch ticks, 1 branch
    restore). Only counts are returned; the branch outcome is discarded.
    """
    from research.bl_labels import factory

    from .ticks import TickCounter, instrument

    counter = TickCounter()
    with instrument(counter):
        driver = factory.make_driver({"seed": seed, "stream": "watch6"}, b8, frp3)
        checkpoint = factory.digest(b8)
        restore = type(driver).restore
        driver.prepare()
        snapshots = {driver.decision: driver.snapshot()}
        tape: dict[int, Any] = {}
        for _ in range(2):
            packet = driver.prepared_decision()
            tape[driver.decision] = factory._TapeEntry(
                driver.decision,
                driver.life,
                int(driver.sim.frame[0]),
                int(driver.sim.length[0, 0]),
                packet.bare,
                tuple(float(x) for x in packet.q_values()),
                packet.legal,
                factory._digest_driver(driver, checkpoint),
                {},
            )
            driver.step(packet=packet)
            driver.prepare()
        packet = driver.prepared_decision()
        root = factory._TapeEntry(
            driver.decision,
            driver.life,
            int(driver.sim.frame[0]),
            int(driver.sim.length[0, 0]),
            packet.bare,
            tuple(float(x) for x in packet.q_values()),
            packet.legal,
            factory._digest_driver(driver, checkpoint),
            {},
        )
        before = counter.as_dict()
        restored = factory._restore_root(snapshots, tape, root, checkpoint, restore=restore)
        payload = restored.snapshot()
        branch = factory._branch(payload, int(root.action), factory.HORIZON, restore=restore)
        branch_frames = int(branch["frames_executed"])
        del branch
    after = counter.as_dict()
    expected = {
        "play_ticks": 2,
        "play_restores": 0,
        "replay_ticks": 2,
        "replay_restores": 1,
        "branch_restores": 1,
        "branch_ticks": branch_frames,
    }
    observed = {
        "play_ticks": before["ticks"]["play"],
        "play_restores": after["restores"]["play"],
        "replay_ticks": after["ticks"]["replay"],
        "replay_restores": after["restores"]["replay"],
        "branch_restores": after["restores"]["branch"],
        "branch_ticks": after["ticks"]["branch"],
    }
    status = (
        "PASS"
        if observed == expected
        and 1 <= branch_frames <= factory.HORIZON
        and counter.stack == ["play"]
        else "FAIL"
    )
    return {
        "schema": "bl-step-ruler-self-check/v1",
        "status": status,
        "expected": expected,
        "observed": observed,
        "counters": after,
    }


def run(
    control: dict[str, Any],
    ledger: Path,
    b8: Path,
    frp3: Path,
    output: Path,
    *,
    deadline_monotonic: float | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if control.get("schema") != CONTROL_SCHEMA:
        raise ValueError("foreign step-ruler control")
    namespace, allocation = control["namespace"], control["allocation"]
    done = load_ledger(ledger, namespace, allocation)
    pending = [item for item in allocation if item["allocation_id"] not in done]
    nproc = os.cpu_count() or 1
    width = workers if workers is not None else nproc
    width = max(1, min(width, nproc, len(pending) or 1))
    jobs = [
        (item, str(b8), str(frp3), namespace, str(output), deadline_monotonic) for item in pending
    ]
    statuses: list[dict[str, Any]] = []
    if jobs:
        # imap preserves frozen allocation order; Watch6 parents are listed first.
        with mp.get_context("spawn").Pool(processes=width) as pool:
            for value in pool.imap(_collect, jobs):
                statuses.append(value)
                print(f"STEP_RULER_UNIT {json.dumps(value, sort_keys=True)}", flush=True)
                if value["status"] == "INVALID":
                    pool.terminate()
                    break
    invalid = [row for row in statuses if row["status"] == "INVALID"]
    complete = [row["allocation_id"] for row in statuses if row["status"] == "COMPLETE"]
    missing = sorted({item["allocation_id"] for item in pending} - set(complete))
    status = "INVALID" if invalid else ("INCOMPLETE" if missing else "COMPLETE")
    return {
        "schema": "bl-step-ruler-runtime/v1",
        "namespace": namespace,
        "status": status,
        "skipped_verified": sorted(done),
        "completed": complete,
        "missing": missing,
        "unit_statuses": statuses,
        "invalid": invalid,
        "workers": width,
        "operational_diagnostics": {
            "nproc": nproc,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "elapsed_seconds": time.monotonic() - started,
            "code_sha": os.environ.get("F0_COMMIT"),
        },
        "claim": "dev-namespace work/timing counters only; no labels, outcomes or verdict",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--finalize-seconds", type=float)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--self-check-seed", type=int)
    args = parser.parse_args()
    from research.bl_labels.exports import publish_json

    expected = os.environ.get("STEP_RULER_SOURCE_SHA256")
    if expected and expected != source_sha256():
        raise SystemExit("step-ruler source digest differs from manifest")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    control = json.loads(args.control.read_text())
    deadline = None
    if args.session_seconds is not None:
        from research.bl_labels.runtime import session_budget

        budget = session_budget(
            Path(os.environ["COLAB_CLI_ROOT"]), args.session_seconds, args.finalize_seconds
        )
        publish_json(args.output / "session-budget.json", budget)
        deadline = budget["collection_deadline_monotonic"]
    elif args.deadline_seconds is not None:
        deadline = time.monotonic() + args.deadline_seconds
    if args.self_check_seed is not None:
        check = self_check(args.b8, args.frp3, args.self_check_seed)
        publish_json(args.output / "self-check.json", check)
        print(json.dumps({"self_check": check["status"]}), flush=True)
        if check["status"] != "PASS":
            raise SystemExit(3)
    result = run(
        control,
        args.ledger,
        args.b8,
        args.frp3,
        args.output,
        deadline_monotonic=deadline,
        workers=args.workers,
    )
    publish_json(args.output / "runtime-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "completed", "missing")}), flush=True)
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
