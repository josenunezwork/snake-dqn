"""Generate immutable session units using the adopted Watch6 exact-stop engine."""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools.ops.package_spec import publish_immutable, publish_json, read_ledger, unit_id
from web.backend.watch_replay import B8_SHA, FRP3_SHA, SCHEMA, canonical, verify_payload

ROOT = Path(__file__).resolve().parents[2]
_STOP = None


def initialize_worker(stop_event: object) -> None:
    """Share the stop fence across queued and currently running session units."""
    global _STOP
    _STOP = stop_event


def allocation(namespace: str, count: int, controller: str, horizon: int = 6000) -> dict:
    """Freeze identities before execution; no outcome-dependent session choice."""
    unit_id(namespace, "sessions", 0)
    if not 1 <= count <= 128 or controller not in {"exact-stop-l2-prime", "bare-b8"}:
        raise ValueError("invalid session allocation")
    if not 1 <= horizon <= 6000:
        raise ValueError("invalid horizon")
    return {
        "namespace": namespace,
        "count": count,
        "controller": controller,
        "horizon": horizon,
        "rows": [
            {
                "index": i,
                "seed": int.from_bytes(
                    hashlib.sha256(f"{namespace}/{i}".encode()).digest()[:4], "big"
                ),
            }
            for i in range(count)
        ],
    }


def identity(contract: dict) -> dict:
    """Bind every allocation dimension and the executable commit in the ledger."""
    return {"contract_sha256": hashlib.sha256(canonical(contract)).hexdigest()}


def completed(ledger: Path, contract: dict) -> list[dict]:
    rows = read_ledger(ledger, contract["namespace"], "sessions", identity(contract))
    if any(not 0 <= r["index"] < contract["count"] for r in rows):
        raise ValueError("foreign allocation index")
    return rows


def generate(task: tuple[dict, dict, str]) -> dict:
    """Publish one all-or-nothing session; interrupted partial units never complete."""
    from research.watch_replay.engine import run_arm

    contract, row, directory = task
    out = Path(directory)
    path = out / f"{row['index']:06d}.watch.gz"
    if path.exists() or Path(str(path) + ".complete").exists():
        raise FileExistsError("never overwrite or recompute a completed unit")
    started, last = time.monotonic(), time.monotonic()
    faulthandler.dump_traceback_later(900, exit=True)
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def check() -> None:
        nonlocal last
        if stop or (_STOP is not None and _STOP.is_set()):
            raise InterruptedError("mechanical stop; partial session discarded")
        if time.monotonic() - last >= 60:
            print(
                f"watch-replay index={row['index']} working "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
            last = time.monotonic()

    with tempfile.TemporaryFile() as frames:

        def record(frame: dict) -> None:
            # Keep gameplay/inspector/veto counters byte-stable across workers.
            # Diagnostic v8 wall times describe generation hardware, not gameplay,
            # and are not fields consumed by the browser's protocol contract.
            veto = frame["session"].get("safety_veto")
            if isinstance(veto, dict):
                veto.pop("diagnostics", None)
            frames.write(canonical(frame))
            # Completed real-frame progress, never mere heartbeat, renews the watchdog.
            faulthandler.dump_traceback_later(900, exit=True)

        try:
            raw = run_arm(
                row["seed"],
                contract["controller"] == "exact-stop-l2-prime",
                check,
                horizon=contract["horizon"],
                record=record,
            )
            stats = {
                "hero_mass_integral": sum(f["mass"] for f in raw["frames"]) / contract["horizon"],
                "hero_deaths": sum(f["death"] for f in raw["frames"]),
                "decisions": len(raw["decisions"]),
                "search_roots": len(raw["batches"]),
                "interventions": sum(b["chosen"] != b["bare"] for b in raw["batches"]),
                "mirror_identity_frames": raw["mirror_identity_frames"],
            }
            header = {
                "schema": SCHEMA,
                "protocol_version": 2,
                "world": "watch6-mechanics-v2-continuous-respawn",
                "unit_id": unit_id(contract["namespace"], "sessions", row["index"]),
                "index": row["index"],
                "seed": row["seed"],
                "horizon": contract["horizon"],
                "controller": contract["controller"],
                "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
                "source_commit": contract["commit"],
                "allocation": identity(contract),
                "fixture": contract["horizon"] != 6000,
                "stats": stats,
            }
            buffer = io.BytesIO()
            with gzip.GzipFile(
                fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=6
            ) as zipped:
                zipped.write(canonical(header))
                frames.seek(0)
                while chunk := frames.read(1024**2):
                    zipped.write(chunk)
            payload = buffer.getvalue()
            verify_payload(payload, product=not header["fixture"])
            result = publish_immutable(path, payload)
        finally:
            faulthandler.cancel_dump_traceback_later()
    seconds = time.monotonic() - started
    print(f"watch-replay index={row['index']} complete seconds={seconds:.3f}", flush=True)
    return {**result, "index": row["index"], "seconds": seconds}


def run(contract: dict, ledger: Path, out: Path, workers: int) -> list[dict]:
    """Skip only uploaded Mac-verified units; merge in frozen allocation order."""
    if not 1 <= workers <= min(os.cpu_count() or 1, 10):
        raise ValueError("invalid worker count")
    verified = completed(ledger, contract)
    skip = {r["index"] for r in verified}
    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "contract.json", contract)
    tasks = [(contract, row, str(out)) for row in contract["rows"] if row["index"] not in skip]
    if workers == 1:
        rows = [generate(t) for t in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        stop_event = context.Event()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(stop_event,),
        ) as pool:

            def stop_pool(signum: int, frame: object) -> None:
                stop_event.set()

            signal.signal(signal.SIGUSR1, stop_pool)
            signal.signal(signal.SIGTERM, stop_pool)
            rows = list(pool.map(generate, tasks))
    publish_json(
        out / "operational.json",
        {
            "allocation": identity(contract),
            "completed": len(rows),
            "skipped": sorted(skip),
            "total": contract["count"],
            "units": [{k: r[k] for k in ("index", "sha256", "size")} for r in rows],
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    expected = allocation(
        contract["namespace"], contract["count"], contract["controller"], contract["horizon"]
    )
    if contract != {**expected, "commit": contract["commit"]}:
        raise ValueError("malformed frozen contract")
    print(json.dumps(run(contract, args.ledger, args.out, args.workers), sort_keys=True))


if __name__ == "__main__":
    main()
