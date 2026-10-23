"""Outcome-blind, leased Mac W6/W12 calibration of the frozen F1 labeller.

Only complete paired cells count: 192 identical roots per width, alternating
order, startup and one full warm-up bank excluded. No sleep/setup time can
satisfy the 1800-second measured-work minimum. This is not an F0 outcome look.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path
from queue import Empty
from typing import Any

from research.mac_lease import held

from . import f1_ruler, mac_driver, mac_gates
from .exports import publish_json
from .mac_package import F1_BANK_PATH, F1_BANK_SHA256

SCHEMA = "bl-labels-f0-w12-calibration/v1"
OWNER_FILE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "OWNER_STANDING_APPROVAL_20261023.txt"
)
OWNER_SHA256 = "32255309812d4df71aa4f6e0635cf0307e3356708262f3cbc99948ebf49569e3"
PACK = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/mac-pack-02")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owner_approval(path: Path) -> str:
    """Require the exact owner-approved W12 text, rather than a truthy filename."""
    digest = sha(path)
    if digest != OWNER_SHA256:
        raise ValueError("owner W12 approval differs from the reviewed standing approval")
    return digest


class CalibrationHold(RuntimeError):
    """Mechanical, preemption or host failure: no production fallback decision."""


class Monitor:
    """Five-second, fail-closed host/RSS/preemption sampling, operations only."""

    def __init__(self, output: Path, timing: Path, deadline: float) -> None:
        self.output, self.timing, self.deadline = output, timing, deadline
        self.next_sample = 0.0
        self.next_report = 0.0
        self.thermal_not_ok = 0

    def poll(self, processes: list[Any], *, force: bool = False) -> None:
        if time.monotonic() >= self.deadline:
            raise CalibrationHold("calibration exhausted its whole-run wall cap")
        if mac_driver._timing_armed(self.timing):
            raise CalibrationHold("calibration preempted by Mac timing arm")
        if not force and time.monotonic() < self.next_sample:
            return
        try:
            sample = mac_driver._guard_snapshot(self.output)
            pids = [p.pid for p in processes if p.pid and p.is_alive()]
            rss = mac_driver._rss_many(pids)
            if set(rss) != set(pids):
                raise CalibrationHold("calibration worker RSS observation incomplete")
            if (
                any(v > mac_driver.MAX_WORKER_RSS_BYTES for v in rss.values())
                or sum(rss.values()) > mac_driver.MAX_TOTAL_RSS_BYTES
            ):
                raise CalibrationHold("calibration RSS cap exceeded")
        except Exception as error:
            self.thermal_not_ok += 1
            raise CalibrationHold(f"host observation failed: {error}") from error
        with (self.output / "host-samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"monotonic": time.monotonic(), **sample, "rss": rss}) + "\n")
        self.next_sample = time.monotonic() + 5
        if time.monotonic() >= self.next_report:
            print(
                json.dumps({"event": "calibration_heartbeat", "workers_alive": len(pids)}),
                flush=True,
            )
            self.next_report = time.monotonic() + 55


class _Ready:
    def wait(self, **_kwargs: Any) -> None:
        pass


def _initialize(ready: Any) -> None:
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError("calibration worker must run at nice 10")
    f1_ruler._warmup(_Ready())
    ready.put(os.getpid())


def _collect(pool: Any, function: Any, jobs: list[Any], monitor: Monitor) -> list[Any]:
    pending = pool.imap(function, jobs, chunksize=1)
    results = []
    last_result = time.monotonic()
    while len(results) < len(jobs):
        monitor.poll(pool._pool)
        if any(p.exitcode is not None for p in pool._pool):
            raise CalibrationHold("calibration worker exited")
        try:
            results.append(pending.next(timeout=1))
            last_result = time.monotonic()
        except mp.TimeoutError:
            if time.monotonic() - last_result >= 300:
                raise CalibrationHold("calibration no-completed-unit watchdog")
    monitor.poll(pool._pool, force=True)
    return results


def integrity(records: list[dict], root_ids: list[str]) -> list[tuple[str, ...]]:
    """Inspect only frozen root IDs and integrity hashes, never outcome content."""
    if [r.get("root_id") for r in records] != root_ids:
        raise CalibrationHold("calibration has foreign/incomplete root allocation")
    values = []
    for row in records:
        hashes = tuple(row.get(k) for k in ("digest", "state_digest", "label_sha256"))
        if (
            row.get("digest_checks") != 1
            or row.get("bare_audit_mismatches") != 0
            or any(
                not isinstance(v, str)
                or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v)
                for v in hashes
            )
        ):
            raise CalibrationHold("calibration integrity failure")
        values.append(hashes)
    return values


def wall_ratio(cells: list[dict]) -> float:
    """Worker-wall per identical unit, rather than unnormalised batch wall."""
    totals = {w: sum(c["seconds"] for c in cells if c["workers"] == w) for w in (6, 12)}
    counts = {w: sum(c["units"] for c in cells if c["workers"] == w) for w in (6, 12)}
    if not all(totals[w] > 0 and counts[w] > 0 for w in totals) or counts[6] != counts[12]:
        raise CalibrationHold("calibration requires equal complete measured work at both widths")
    return (12 * totals[12] / counts[12]) / (6 * totals[6] / counts[6])


def measure(
    bank: dict, b8: Path, frp3: Path, output: Path, monitor: Monitor
) -> tuple[list[dict], int]:
    cache = output / "root-cache"
    cache.mkdir()
    context = mp.get_context("spawn")
    ready = context.Queue()
    paths = {}
    # Preparation is serial and guarded; no timing in it contributes to calibration.
    with context.Pool(1, initializer=_initialize, initargs=(ready,)) as pool:
        jobs = []
        for source in bank["source_sessions"]:
            roots = [r for r in bank["roots"] if r["source_session"] == source["name"]]
            if roots:
                jobs.append((source, roots, b8, frp3, cache, monitor.deadline))
        for result in _collect(pool, f1_ruler._prepare_source_job, jobs, monitor):
            paths.update(result)
    ready.close()
    ids = [r["root_id"] for r in bank["roots"]]
    cells, baseline, reference = [], {}, None
    pair = 0
    # Three copies of 64 roots divide equally by both widths. Each pair has
    # exactly the same ordered work; only complete pairs count toward duration.
    while sum(c["seconds"] for c in cells) < 1800 or pair < 3:
        for width in ((6, 12) if pair % 2 == 0 else (12, 6)):
            monitor.poll([], force=True)
            ready = context.Queue()
            with context.Pool(width, initializer=_initialize, initargs=(ready,)) as pool:
                seen = set()
                startup_deadline = time.monotonic() + 180
                while len(seen) < width:
                    monitor.poll(pool._pool)
                    if time.monotonic() >= startup_deadline:
                        raise CalibrationHold("calibration workers did not become ready")
                    try:
                        seen.add(ready.get(timeout=1))
                    except Empty:
                        pass
                jobs = [(str(paths[root]), monitor.deadline) for root in ids]
                warm = _collect(pool, f1_ruler._label_root, jobs, monitor)
                hashes = integrity(warm, ids)
                if reference is not None and hashes != reference:
                    raise CalibrationHold("calibration worker-count/repeat integrity mismatch")
                reference = hashes
                started = time.monotonic()
                records = _collect(pool, f1_ruler._label_root, jobs * 3, monitor)
                elapsed = time.monotonic() - started
                if integrity(records, ids * 3) != reference * 3:
                    raise CalibrationHold("calibration worker-count/repeat integrity mismatch")
            ready.close()
            baseline.setdefault(width, elapsed)
            cells.append(
                {"pair": pair, "workers": width, "units": len(jobs) * 3, "seconds": elapsed}
            )
            print(json.dumps({"event": "calibration_cell", **cells[-1]}), flush=True)
            if elapsed > baseline[width] * 1.30:
                return cells, 1
        pair += 1
    return cells, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--owner-approval", type=Path, default=OWNER_FILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path(F1_BANK_PATH))
    parser.add_argument("--timing-armed", type=Path, default=mac_driver.DEFAULT_TIMING_ARMED)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or not 1800 < args.timeout_seconds <= 7200:
        parser.error("whole-run timeout must be >1800 and <=7200 seconds")
    os.environ.update(mac_driver.THREAD_ENV)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    approval_sha = owner_approval(args.owner_approval)
    pin = mac_gates._source_pin(args.pack)
    if os.environ.get("F0_COMMIT", pin) != pin:
        raise ValueError("F0_COMMIT differs from frozen package source")
    os.environ.update(F0_COMMIT=pin, F1_COMMIT=pin)
    if f1_ruler.bank_digest(args.bank) != F1_BANK_SHA256:
        raise ValueError("calibration requires the frozen 64-root F1 bank")
    bank = f1_ruler.load_bank(args.bank)
    for path in (args.b8, args.frp3):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError("calibration output must be fresh")
    if args.check:
        print(json.dumps({"status": "CHECKED", "code_sha": pin, "bank_sha256": F1_BANK_SHA256}))
        return
    # No output/work is created until this process acquires the shared lease.
    with held(
        "bl-f0-w12-calibration", "outcome-blind W6/W12 calibration", seconds=args.timeout_seconds
    ) as lease:
        args.output.mkdir(parents=True, exist_ok=False)
        monitor = Monitor(args.output, args.timing_armed, time.monotonic() + args.timeout_seconds)
        base = {
            "schema": SCHEMA,
            "platform": "darwin-arm64",
            "code_sha": pin,
            "bank_sha256": F1_BANK_SHA256,
            "runner_sha256": sha(Path(__file__)),
            "owner_approval_sha256": approval_sha,
            "b8_sha256": sha(args.b8),
            "frp3_sha256": sha(args.frp3),
            "mac_lease": lease.receipt,
            "outcome_exports": False,
        }
        try:
            monitor.poll([], force=True)
            cells, slowdown = measure(bank, args.b8, args.frp3, args.output, monitor)
            duration = sum(c["seconds"] for c in cells)
            ratio = wall_ratio(cells) if not slowdown else None
            passed = not slowdown and duration >= 1800 and ratio is not None and 0 < ratio <= 1.30
            receipt = {
                **base,
                "status": "PASS" if passed else "FAIL",
                "duration_seconds": duration,
                "w12_w6_wall_ratio": ratio,
                "thermal_not_ok": monitor.thermal_not_ok,
                "slowdown_trips": slowdown,
                "cells": cells,
                "slowdown_baseline": (
                    "first measured 192-root cell per width; later cell >1.30x trips"
                ),
                "duration_definition": (
                    "sum of complete measured labeller windows; excludes warmup/setup"
                ),
            }
            monitor.poll([], force=True)
        except Exception as error:
            receipt = {
                **base,
                "status": "HOLD",
                "reason": str(error),
                "duration_seconds": 0,
                "w12_w6_wall_ratio": None,
                "thermal_not_ok": monitor.thermal_not_ok,
                "slowdown_trips": 0,
            }
        samples = args.output / "host-samples.jsonl"
        if samples.exists():
            receipt["host_samples_sha256"] = sha(samples)
            receipt["host_samples_bytes"] = samples.stat().st_size
        publish_json(args.output / "receipt.json", receipt)
        if receipt["status"] == "PASS":
            mac_driver._verify_w12_calibration(args.output / "receipt.json")
        print(
            json.dumps(
                {k: receipt[k] for k in ("status", "duration_seconds", "w12_w6_wall_ratio")}
            ),
            flush=True,
        )
        if receipt["status"] != "PASS":
            raise SystemExit(2)


def _terminated(_signal: int, _frame: Any) -> None:
    raise CalibrationHold("calibration terminated")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminated)
    main()
