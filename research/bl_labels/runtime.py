"""CPU-runtime entrypoint: fixed-order collection and completion-last exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

from .exports import load_ledger, publish_json, publish_life, publish_session


def _source_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    # The manifest adds the launcher bytes; this runtime verifies the Python
    # closure it actually imports before collecting labels.
    return digest.hexdigest()


def _collect(
    job: tuple[dict, Path, Path, str, Path, list[dict], set[int], float | None],
) -> tuple[dict, str | None]:
    """Top-level process target: each worker has exactly one CPU thread."""
    item, b8, frp3, namespace, output, allocation, skip_lives, deadline = job
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1"
    )
    from .factory import FactoryInvalid, collect_unit

    def write_life(value: dict) -> None:
        if value.get("schema") == "bl-labels-f0-session/v1":
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
) -> dict:
    started = time.monotonic()
    namespace, allocation = control["namespace"], control["allocation"]
    done = load_ledger(ledger_path, namespace, allocation)
    requested = control.get("workers", "nproc")
    workers = (
        (os.cpu_count() or 1)
        if requested == "nproc"
        else min(max(1, int(requested)), os.cpu_count() or 1)
    )

    completed, invalid, incomplete = [], None, False
    done_by_parent: dict[str, set[int]] = {}
    completed_sessions: set[str] = set()
    for name in done:
        parent, marker, life = name.partition("--life-")
        if marker:
            done_by_parent.setdefault(parent, set()).add(int(life))
        elif name.endswith("--session"):
            completed_sessions.add(name.removesuffix("--session"))
    pending = [item for item in allocation if item["allocation_id"] not in completed_sessions]
    deadline = time.monotonic() + deadline_seconds if deadline_seconds else None
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
                done_by_parent.get(item["allocation_id"], set()),
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
    args = parser.parse_args()
    expected = os.environ.get("F0_SOURCE_SHA256")
    if expected and expected != _source_sha256():
        raise SystemExit("F0 runtime Python source digest differs from manifest")
    # Preserve the frozen portable allocation bytes for callback and ledger
    # identity. GateDriver resolves input: checkpoint references internally
    # under SNAKE_CKPT_DIR after the qualified bootstrap binds the catalog.
    control = json.loads(args.control.read_text())
    result = run(
        control,
        args.ledger,
        args.b8,
        args.frp3,
        args.output,
        args.deadline_seconds,
    )
    publish_json(args.output / "runtime-result.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
