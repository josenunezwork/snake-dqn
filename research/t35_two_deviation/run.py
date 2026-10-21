"""T35 two-deviation rescue probe entrypoint.

Subcommands::

    event      run one registered event in this process (one thread) -> create-only receipt
    launch     run a fixed event set as P concurrent single-thread ``event`` processes
    summarize  aggregate receipts and apply the registered verdict

Every numerical thread limit must be 1 (``launch`` sets them for its children).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.t35_two_deviation import events as ev  # noqa: E402
from research.t35_two_deviation.search import SearchParams  # noqa: E402

SCHEMA = "t35-two-deviation-event/v1"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CLOSURE = (
    "research/t35_two_deviation",
    "research/t34_shadow",
    "research/t3_rollout_labels_20261015",
    "research/t1_commitment_20261015",
    "research/b8_death_census_20261014",
    "research/redesign_scope_20261007",
    "research/redesign_m2_20261008",
    "research/apex_safety_20260926",
    "src",
)
REGISTERED_CAP_SECONDS = 3 * 3600


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def provenance() -> dict[str, Any]:
    """Pinned source closure, thread limits and host state."""
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise RuntimeError("all numerical thread limits must explicitly equal one")
    if os.environ.get("SNAKE_DQN_DEVICE") != "cpu":
        raise RuntimeError("T35 requires SNAKE_DQN_DEVICE=cpu")
    host: dict[str, Any] = {"platform": platform.platform(), "load_average": os.getloadavg()}
    if platform.system() == "Darwin":
        power = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        if "AC Power" not in power:
            raise RuntimeError("T35 on the Mac requires AC power")
        host["power"] = power
        host["thermal"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    paths = sorted({p for d in CLOSURE for p in (ROOT / d).rglob("*.py")})
    return {
        "argv": sys.argv,
        "python": sys.version,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain", "--", *CLOSURE], cwd=ROOT, text=True
        ),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in paths},
        "threads": {name: os.environ[name] for name in THREAD_VARS},
        "nice": os.nice(0),
        "host": host,
    }


def _event(event_id: str) -> dict[str, Any]:
    for event in ev.all_die_events():
        if event["id"] == event_id:
            return event
    raise SystemExit(f"unknown all-die event: {event_id}")


def identity_mismatches(result: dict[str, Any], truth: dict[str, Any]) -> list[Any]:
    """Compare fresh layer-1 outcomes with the M2 stride-one truth scan."""
    bad = []
    for decision, fresh in result["layer1_outcomes"].items():
        saved = truth.get(decision)
        if saved is None or saved["greedy"] != fresh["greedy"]:
            bad.append((decision, "greedy_or_missing"))
            continue
        if set(saved["outcomes"]) != set(fresh["outcomes"]):
            bad.append((decision, "action_set"))
            continue
        for action, outcome in fresh["outcomes"].items():
            if outcome != saved["outcomes"][action]:
                bad.append((decision, action))
    return bad


def run_event(args: argparse.Namespace) -> int:
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t35_two_deviation.backend import BankBackend, Deadline, prepare_event
    from research.t35_two_deviation.search import two_deviation_search

    receipt_path = args.out / "events" / f"{args.event}.json"
    if receipt_path.exists():
        raise FileExistsError(f"create-only receipt exists: {receipt_path}")
    seconds = args.deadline_epoch - time.time() if args.deadline_epoch else args.seconds
    params = SearchParams(
        beam=args.beam, window2=args.window2, horizon=args.horizon, confirm_extra=args.confirm
    )
    event = _event(args.event)
    progress_path = args.out / "progress" / f"{args.event}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last = [0.0]

    def progress(info: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - last[0] >= args.progress_seconds:
            last[0] = now
            with progress_path.open("a") as stream:
                stream.write(json.dumps({"t": round(now - started, 1), **info}) + "\n")

    meta = None
    base = {"schema": SCHEMA, "event": event, "params": asdict(params)}
    try:
        configure_cpu()
        meta = provenance()
        deadline = Deadline(seconds)
        states, prep = prepare_event(event["world"], event["start"], event["doom"], deadline)
        window = list(range(event["start"], event["doom"] + 1))
        if args.window1_last:
            window = window[-args.window1_last :]
        search_started = time.perf_counter()
        result = two_deviation_search(
            BankBackend(deadline), states, window, event["doom"], params, progress
        )
        search_seconds = time.perf_counter() - search_started
        mismatches = identity_mismatches(result, ev.m2_truth_outcomes(event))
        status = result["status"] if not mismatches else "INCOMPLETE_IDENTITY"
        receipt = {
            **base,
            "status": status,
            "registered_shape": args.window1_last is None
            and params == SearchParams()
            and args.seconds_registered,
            "identity_mismatches": mismatches[:50],
            "preparation": prep,
            "search": result,
            "search_seconds": search_seconds,
            "row_steps_per_second": (
                result["row_steps"] / search_seconds if search_seconds else None
            ),
            "wall_seconds": time.monotonic() - started,
            "provenance": meta,
        }
        _write_once(receipt_path, receipt)
        print(json.dumps({"event": args.event, "status": status, "receipt": str(receipt_path)}))
        return 0
    except Exception as error:  # TimeoutError, ControlMismatch, reconstruction asserts
        _write_once(
            receipt_path,
            {
                **base,
                "status": "INCOMPLETE",
                "error_type": type(error).__name__,
                "error": repr(error)[:2000],
                "wall_seconds": time.monotonic() - started,
                "provenance": meta,
            },
        )
        print(json.dumps({"event": args.event, "status": "INCOMPLETE", "error": repr(error)[:300]}))
        return 1


def event_set(name: str) -> list[str]:
    """Registered event sets: ``primary`` (10), ``extend`` (remaining 23), or explicit ids."""
    order = ev.round_robin_order(ev.all_die_events())
    if name == "primary":
        if tuple(order[: ev.PRIMARY_COUNT]) != ev.PRIMARY_EVENTS:
            raise RuntimeError("primary selection drifted from the registered list")
        return list(ev.PRIMARY_EVENTS)
    if name == "extend":
        return order[ev.PRIMARY_COUNT :]
    return [item for item in name.split(",") if item]


def launch(args: argparse.Namespace) -> int:
    """Run events as concurrent single-thread processes; one shared absolute deadline."""
    ids = event_set(args.events)
    # Longest-first (LPT) packing by the outcome-blind M2 layer-1 cost proxy.
    by_id = {event["id"]: event for event in ev.all_die_events()}
    ids.sort(key=lambda item: (-ev.m2_layer1_row_steps(by_id[item]), item))
    if args.cap_seconds > REGISTERED_CAP_SECONDS and args.events == "primary":
        raise SystemExit("primary run cap exceeds the registered 3 h")
    args.out.mkdir(parents=True, exist_ok=True)
    plan = (
        args.out / f"plan-{args.events if args.events in ('primary', 'extend') else 'custom'}.json"
    )
    deadline = time.time() + args.cap_seconds
    _write_once(
        plan,
        {"events": ids, "procs": args.procs, "cap_seconds": args.cap_seconds, "deadline": deadline},
    )
    env = {**os.environ, **{name: "1" for name in THREAD_VARS}, "SNAKE_DQN_DEVICE": "cpu"}
    pending, running, codes = list(ids), [], {}
    while pending or running:
        while pending and len(running) < args.procs:
            event_id = pending.pop(0)
            log = args.out / "logs" / f"{event_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "nice", "-n", str(args.nice), sys.executable, str(Path(__file__)), "event",
                "--event", event_id, "--out", str(args.out),
                "--deadline-epoch", str(deadline), *args.extra,
            ]  # fmt: skip
            stream = log.open("x")
            running.append(
                (
                    event_id,
                    subprocess.Popen(
                        command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                    ),
                    stream,
                )
            )
        time.sleep(5)
        for item in list(running):
            event_id, process, stream = item
            if process.poll() is not None:
                codes[event_id] = process.returncode
                stream.close()
                running.remove(item)
                print(json.dumps({"event": event_id, "rc": process.returncode}), flush=True)
    return 0 if all(code == 0 for code in codes.values()) else 1


def verdict(rescued: int, unknown: int, total: int) -> str:
    """Registered rule; INCONCLUSIVE when unknown events could change the verdict."""

    def rule(r: int) -> str:
        if r >= 3:
            return "REAL"
        if r <= 1:
            return "KILL"
        return "EXTEND"

    if total != ev.PRIMARY_COUNT:
        raise ValueError("the registered verdict is defined on the 10 primary events")
    low, high = rule(rescued), rule(rescued + unknown)
    return low if low == high else "INCONCLUSIVE"


def summarize(args: argparse.Namespace) -> int:
    ids = event_set(args.events)
    rows = []
    for event_id in ids:
        path = args.out / "events" / f"{event_id}.json"
        receipt = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        search = receipt.get("search", {})
        rows.append(
            {
                "event": event_id,
                "status": receipt["status"],
                "registered_shape": receipt.get("registered_shape"),
                "rescue": search.get("rescue"),
                "rescued_ranks": search.get("rescued_ranks"),
                "unconfirmed_survivors": len(search.get("unconfirmed_survivors", []) or []),
                "row_steps": search.get("row_steps"),
                "search_seconds": receipt.get("search_seconds"),
            }
        )
    rescued = sum(r["status"] == "RESCUED_DEPTH2" and r["registered_shape"] for r in rows)
    negative = sum(r["status"] == "NOT_RESCUED" and r["registered_shape"] for r in rows)
    unknown = len(rows) - rescued - negative
    total_steps = sum(r["row_steps"] or 0 for r in rows)
    summary = {
        "schema": "t35-two-deviation-summary/v1",
        "events": rows,
        "rescued": rescued,
        "not_rescued": negative,
        "unknown": unknown,
        "rescued_at_rank_above_4": sum(
            bool(r["rescued_ranks"]) and max(r["rescued_ranks"]) > 4 for r in rows
        ),
        "rescued_only_at_rank_above_4": sum(
            bool(r["rescued_ranks"]) and min(r["rescued_ranks"]) > 4 for r in rows
        ),
        "row_steps": total_steps,
        "row_steps_per_rescue": total_steps / rescued if rescued else None,
        "verdict": verdict(rescued, unknown, len(rows)) if args.events == "primary" else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.write:
        _write_once(args.out / f"summary-{args.events}.json", summary)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("event")
    one.add_argument("--event", required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--seconds", type=float, default=REGISTERED_CAP_SECONDS)
    one.add_argument("--deadline-epoch", type=float, default=None)
    one.add_argument("--beam", type=int, default=8)
    one.add_argument("--window2", type=int, default=300)
    one.add_argument("--horizon", type=int, default=300)
    one.add_argument("--confirm", type=int, default=300)
    one.add_argument("--window1-last", type=int, default=None, help="preflight only")
    one.add_argument("--progress-seconds", type=float, default=60.0)
    many = sub.add_parser("launch")
    many.add_argument("--events", default="primary")
    many.add_argument("--out", type=Path, required=True)
    many.add_argument("--procs", type=int, default=3)
    many.add_argument("--nice", type=int, default=10)
    many.add_argument("--cap-seconds", type=float, default=REGISTERED_CAP_SECONDS)
    many.add_argument("extra", nargs=argparse.REMAINDER, help="passed to every event")
    summ = sub.add_parser("summarize")
    summ.add_argument("--events", default="primary")
    summ.add_argument("--out", type=Path, required=True)
    summ.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "event":
        args.seconds_registered = args.seconds <= REGISTERED_CAP_SECONDS
    if args.command == "launch" and args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {"event": run_event, "launch": launch, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
