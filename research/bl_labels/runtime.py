"""CPU-runtime entrypoint: fixed-order collection and completion-last exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

from .exports import (
    completed_session_parents,
    load_ledger,
    publish_json,
    publish_life,
    publish_session,
)


def session_budget(root: Path, session_seconds: float, finalize_seconds: float) -> dict:
    """Subtract measured setup from the qualified watchdog's remaining window.

    Its timeout already deducts Mac-side allocation/setup and the supervisor's
    cleanup guards from the whole-session cap. Its state clock starts before
    uploads, public inputs, uv/Python and dependency installation. Never start a
    fresh independent runtime timer with the full manifest session allowance.
    """
    spec = json.loads((root / "spec.json").read_text())
    state = json.loads((root / "state.json").read_text())
    if (
        spec.get("job_id") != state.get("job_id")
        or spec.get("generation") != state.get("generation")
        or Path(spec["root"]).resolve() != root.resolve()
        or float(spec["env"]["F0_SESSION_SECONDS"]) != session_seconds
        or float(spec["env"]["F0_FINALIZE_SECONDS"]) != finalize_seconds
    ):
        raise ValueError("F0 session budget identity differs from qualified runtime")
    timeout = float(spec["timeout_seconds"])
    elapsed = time.time() - float(state["started_at"])
    if not all(math.isfinite(v) for v in (session_seconds, finalize_seconds, timeout, elapsed)):
        raise ValueError("F0 session budget must be finite")
    if not (0 < timeout <= session_seconds and 0 < finalize_seconds < timeout and elapsed >= 0):
        raise ValueError("F0 session budget has invalid bounds or clock")
    remaining = timeout - elapsed - finalize_seconds
    if remaining <= 0:
        raise ValueError("F0 setup exhausted whole-session budget before collection")
    return {
        "schema": "bl-labels-f0-session-budget/v1",
        "session_seconds": session_seconds,
        "qualified_window_seconds": timeout,
        "pre_watchdog_setup_and_guard_seconds": session_seconds - timeout,
        "measured_setup_seconds": elapsed,
        "finalize_seconds": finalize_seconds,
        "runtime_seconds": remaining,
        "collection_deadline_monotonic": time.monotonic() + remaining,
        "job_id": spec["job_id"],
        "generation": spec["generation"],
    }


def _source_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    # The manifest adds the launcher bytes; this runtime verifies the Python
    # closure it actually imports before collecting labels.
    return digest.hexdigest()


def _collect(
    job: tuple[dict, Path, Path, str, Path, list[dict], set[int], bool, float | None],
) -> tuple[dict, str | None]:
    """Top-level process target: each worker has exactly one CPU thread."""
    item, b8, frp3, namespace, output, allocation, skip_lives, suppress_session, deadline = job
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1"
    )
    from .factory import FactoryInvalid, collect_unit

    def write_life(value: dict) -> None:
        if value.get("schema") == "bl-labels-f0-session/v1":
            if suppress_session:
                return
            publish_session(output, namespace, allocation, value)
        else:
            publish_life(output, namespace, allocation, value)

    def progress(value: dict) -> None:
        print(
            f"F0_PROGRESS allocation={item['allocation_id']} done={value.get('done_decisions')} "
            f"total={value.get('total_decisions')}",
            flush=True,
        )

    try:
        supplied = dict(item)
        supplied["publish_life"] = write_life
        supplied["skip_lives"] = skip_lives
        supplied["progress"] = progress
        if deadline is not None:
            supplied["deadline_monotonic"] = deadline
        return collect_unit(supplied, b8, frp3, namespace), None
    except (FactoryInvalid, ValueError) as exc:
        publish_json(
            output / f"INVALID-{item['allocation_id']}.json",
            {
                "schema": "bl-labels-f0-invalid/v1",
                "namespace": namespace,
                "allocation_id": item["allocation_id"],
                "reason": str(exc),
            },
        )
        return {
            "schema": "bl-labels-f0-unit/v1",
            "status": "INVALID",
            "namespace": namespace,
            "allocation": item,
            "invalid_reason": str(exc),
            "rows": [],
            "snapshots": [],
            "counters": {},
        }, str(exc)
    except RuntimeError as exc:
        if "INCOMPLETE F0 unit wall cap" not in str(exc):
            raise
        return {
            "schema": "bl-labels-f0-unit/v1",
            "status": "INCOMPLETE",
            "namespace": namespace,
            "allocation": item,
            "reason": str(exc),
            "rows": [],
            "snapshots": [],
            "counters": {},
        }, None


def run(
    control: dict,
    ledger_path: Path,
    b8: Path,
    frp3: Path,
    output: Path,
    deadline_seconds: float | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> dict:
    started = time.monotonic()
    namespace, allocation = control["namespace"], control["allocation"]
    done = load_ledger(ledger_path, namespace, allocation)
    transport_probe = os.environ.get("F0_TRANSPORT_PROBE") == "1"
    requested = control.get("workers", "nproc")
    workers = (
        (os.cpu_count() or 1)
        if requested == "nproc"
        else min(max(1, int(requested)), os.cpu_count() or 1)
    )

    completed, invalid, incomplete = [], None, False
    done_by_parent: dict[str, set[int]] = {}
    session_parents = {
        name.removesuffix("--session") for name in done if name.endswith("--session")
    }
    completed_sessions = completed_session_parents(ledger_path, done)
    for name in done:
        parent, marker, life = name.partition("--life-")
        if marker:
            done_by_parent.setdefault(parent, set()).add(int(life))
    pending = [item for item in allocation if item["allocation_id"] not in completed_sessions]
    deadline = deadline_monotonic
    if deadline is None and deadline_seconds is not None:
        deadline = started + deadline_seconds
    # imap preserves the frozen allocation order even if a later life finishes first.
    with mp.get_context("spawn").Pool(processes=min(workers, len(pending) or 1)) as pool:
        jobs = [
            (
                item,
                b8,
                frp3,
                namespace,
                output,
                allocation,
                (
                    set(range(int(item["horizon"]) + 1))
                    if transport_probe
                    else done_by_parent.get(item["allocation_id"], set())
                ),
                item["allocation_id"] in session_parents,
                deadline,
            )
            for item in pending
        ]
        for value, error in pool.imap(_collect, jobs):
            completed.append(value["allocation"]["allocation_id"])
            incomplete = incomplete or value.get("status") == "INCOMPLETE"
            if error or value.get("status") == "INVALID":
                invalid = error or str(value.get("invalid_reason", "factory returned INVALID"))
                pool.terminate()
                break
    status = "INVALID" if invalid else ("INCOMPLETE" if incomplete else "COMPLETE")
    result = {
        "schema": "bl-labels-f0-runtime/v1",
        "namespace": namespace,
        "completed": completed,
        "skipped_verified": sorted(done),
        "status": status,
        "transport_probe": transport_probe,
        "result_class": "EXCLUDED_TRANSPORT_PROBE" if transport_probe else "SCIENCE",
        "invalid_reason": invalid,
        "workers": workers,
        "operational_diagnostics": {
            "cpu_model": platform.processor() or platform.machine(),
            "nproc": os.cpu_count(),
            "worker_processes": workers,
            "elapsed_seconds": time.monotonic() - started,
            "stall_census": "NOT_MEASURED",
            "code_sha": os.environ.get("F0_COMMIT"),
            "platform": platform.platform(),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--finalize-seconds", type=float)
    args = parser.parse_args()
    if (args.session_seconds is None) != (args.finalize_seconds is None):
        parser.error("session and finalize seconds must be supplied together")
    if args.session_seconds is not None and args.deadline_seconds is not None:
        parser.error("session-derived and standalone runtime deadlines are mutually exclusive")
    expected = os.environ.get("F0_SOURCE_SHA256")
    if expected and expected != _source_sha256():
        raise SystemExit("F0 runtime Python source digest differs from manifest")
    # Preserve the frozen portable allocation bytes for callback and ledger
    # identity. GateDriver resolves input: checkpoint references internally
    # under SNAKE_CKPT_DIR after the qualified bootstrap binds the catalog.
    control = json.loads(args.control.read_text())
    deadline_monotonic = None
    if args.session_seconds is not None:
        budget = session_budget(
            Path(os.environ["COLAB_CLI_ROOT"]), args.session_seconds, args.finalize_seconds
        )
        publish_json(args.output / "session-budget.json", budget)
        deadline_monotonic = budget["collection_deadline_monotonic"]
        print(json.dumps(budget, sort_keys=True), flush=True)
    result = run(
        control,
        args.ledger,
        args.b8,
        args.frp3,
        args.output,
        args.deadline_seconds,
        deadline_monotonic=deadline_monotonic,
    )
    publish_json(args.output / "runtime-result.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
