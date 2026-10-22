"""Exclusive, no-peek Mac coordinator for the eight L2' amendment chunks.

This module deliberately consumes only producer operational attestations after a
chunk.  Scientific shard rows stay sealed until the registered analyzer look.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from research.bl_mac_latency import host
from research.bl_served_search import ceiling, mac_lease
from research.bl_served_search.exports import export_json, verify_export
from research.published_ledger import require_published_ledger

OWNER = "claude-l2prime-mac"
CHUNK_COUNT = 8
WORKERS = 2
CHUNK_TIMEOUT_SECONDS = 142 * 60
LEASE_HOURS = 28.3
DEFAULT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2")
DEFAULT_CHECKPOINT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
BANK_SHA256 = "0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3"


class AdmissionError(RuntimeError):
    """A prospective prerequisite is absent or belongs to different source bytes."""


def _load_export(path: Path) -> tuple[dict[str, Any], str]:
    """Read a complete receipt, never accepting a partial producer export."""
    sha256 = verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AdmissionError(f"receipt is not an object: {path}")
    return value, sha256


def _source_bound_receipt(
    path: Path, name: str, source: dict[str, str]
) -> tuple[dict[str, Any], str]:
    value, sha256 = _load_export(path)
    if value.get("status") != "PASS":
        raise AdmissionError(f"{name} receipt is not PASS")
    if value.get("source_manifest") != source:
        raise AdmissionError(f"{name} receipt binds another source manifest")
    if name == "review":
        if (
            value.get("schema") != "bl-l2-mac-independent-review/v1"
            or value.get("independent") is not True
            or value.get("review_scope") != "P1_MAC_L2"
        ):
            raise AdmissionError("review receipt lacks independent P1 review identity")
    if name == "P3":
        comparison = value.get("comparison", {})
        if (
            value.get("schema") != "bl-served-search-mac-identity/v1"
            or comparison.get("status") != "PASS"
            or comparison.get("worlds") != 5
            or comparison.get("mix_counts") != {"frozen": 2, "mixed": 2, "scripted": 1}
            or value.get("modes") != ["all", "exact_stop"]
            or value.get("horizon") != 5000
            or value.get("b8_sha256") != ceiling.B8_SHA256
            or any(
                comparison.get("fired_batches_by_mix", {}).get(mix, 0) <= 0
                for mix in ("frozen", "mixed", "scripted")
            )
        ):
            raise AdmissionError("P3 receipt lacks five-world exercised production identity")
    if name in {"review", "P3"}:
        commit = value.get("commit")
        if not isinstance(commit, str) or not commit:
            raise AdmissionError(f"{name} receipt lacks bound commit")
        root = Path(__file__).resolve().parents[2]
        reachable = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False
        )
        if reachable.returncode:
            raise AdmissionError(f"{name} receipt commit is not reachable from launch HEAD")
    return value, sha256


def _check_clean_checkout() -> dict[str, str]:
    """Fetch and bind the published amendment before any bank process starts."""
    root = Path(__file__).resolve().parents[2]
    proof = require_published_ledger(root, "research/bl_served_search/AMENDMENT_MAC_L2.md")
    if proof["authoring_head"] != proof["origin_main"]:
        raise AdmissionError("launch checkout is not origin/main")
    return {"head": proof["authoring_head"], "origin_main": proof["origin_main"], "ledger": proof}


def _write_sample(samples: Path, phase: str, snapshot: dict[str, Any]) -> None:
    """Append only host observations; their content never controls a running chunk."""
    samples.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"phase": phase, "at": datetime.now(timezone.utc).isoformat(), **snapshot}, sort_keys=True
    )
    with samples.open("x" if not samples.exists() else "a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _operational_ok(
    path: Path, *, source: dict[str, str], lease_id: str, shard_index: int
) -> dict[str, Any]:
    """Validate a producer's non-scientific attestation without opening shard rows."""
    value, sha256 = _load_export(path)
    expected = {
        "schema": "bl-l2-mac-operational/v1",
        "status": "completed",
        "workers": WORKERS,
        "lease_id": lease_id,
        "source_manifest": source,
        "chunk_index": shard_index,
        "rows": 20,
        "clean_exit": True,
        "search_mode": "exact_stop",
        "native_mode": "kernel",
    }
    wrong = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if (
        wrong
        or value.get("counters_validation") != "PASS"
        or not isinstance(value.get("shard_sha256"), str)
    ):
        raise AdmissionError("chunk operational attestation is invalid: " + ", ".join(wrong))
    shard_sha256 = verify_export(path.parent / "shard.json")
    if shard_sha256 != value["shard_sha256"]:
        raise AdmissionError("operational shard digest does not bind complete shard export")
    return {"operational_sha256": sha256, "shard_sha256": shard_sha256}


def _run_chunk(
    command: list[str],
    *,
    timeout: float,
    samples: Path,
    sample: Callable[[], dict[str, Any]],
    lease_guard: Callable[[], Any],
) -> int:
    """Run one owned child, recording five-minute host samples until it exits."""
    process = subprocess.Popen(command, start_new_session=True)

    def stop_owned_group() -> None:
        """Join exactly the session we created, never searching by process name."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # The parent can exit before a spawned worker; sweep the session we
        # created even when ``poll`` says the direct child is already gone.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
        if process.poll() is None:
            process.wait()

    started = time.monotonic()
    next_sample = started
    try:
        while process.poll() is None:
            now = time.monotonic()
            try:
                lease_guard()
            except Exception as exc:
                # Lease loss is terminal.  Stop only this owned process group.
                _write_sample(samples, "running_guard_fault", {"error": str(exc)})
                raise RuntimeError(f"lease/host guard failed while chunk ran: {exc}") from exc
            if now >= next_sample:
                try:
                    _write_sample(samples, "running", sample())
                except Exception as exc:
                    # Host samples are diagnostic while a chunk is active.
                    _write_sample(samples, "running_sample_fault", {"error": str(exc)})
                next_sample = now + 300
            if now - started > timeout:
                raise TimeoutError(f"chunk exceeded {timeout} seconds")
            time.sleep(min(1.0, max(0.0, next_sample - now)))
        return int(process.returncode or 0)
    finally:
        # Covers timeout, lease loss, SIGINT/SIGTERM and any logging fault.
        stop_owned_group()


def check() -> dict[str, Any]:
    """Read-only local readiness command for the shell launcher."""
    return {
        "status": "CHECK_ONLY",
        "workers": WORKERS,
        "thread_limits": 1,
        "chunks": CHUNK_COUNT,
        "chunk_timeout_seconds": CHUNK_TIMEOUT_SECONDS,
        "lease_hours": LEASE_HOURS,
        "command": "/usr/bin/nice -n 10 /usr/bin/caffeinate -is",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create one exclusive run, then execute chunks 0..7 once each in order."""
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise AdmissionError("Mac amendment coordinator must run at nice 10")
    if ceiling.digest_file(args.bank) != BANK_SHA256:
        raise AdmissionError("Mac amendment requires the registered BANK_L2 bytes")
    source = ceiling.source_manifest()
    review, review_sha = _source_bound_receipt(args.review_receipt, "review", source)
    p3, p3_sha = _source_bound_receipt(args.p3_receipt, "P3", source)
    checkout = _check_clean_checkout()
    if args.out.exists():
        raise AdmissionError("run output is create-only")
    initial = host.snapshot()
    if not initial.get("quiet") or not initial.get("phase0_session_absent"):
        raise AdmissionError("start gate is not QUIET")
    until = (datetime.now(timezone.utc) + timedelta(hours=LEASE_HOURS)).isoformat()
    lease = mac_lease.acquire(args.lease, OWNER, "L2' Mac amendment chunks 0-7", until)

    def interrupted(_signum: int, _frame: Any) -> None:
        raise SystemExit("SIGTERM received by L2' coordinator")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    terminal: dict[str, Any] = {
        "schema": "bl-served-search-l2-mac-run/v1",
        "status": "INCONCLUSIVE",
        "lease_id": lease["id"],
        "source_manifest": source,
        "review_sha256": review_sha,
        "p3_sha256": p3_sha,
        "checkout": checkout,
        "chunks": [],
    }
    failure: BaseException | None = None
    try:
        args.out.mkdir(parents=True, exist_ok=False)
        export_json(
            args.out / "start-gate.json",
            {"status": "QUIET", "host": initial, "lease": lease, **checkout},
        )
        samples = args.out / "host-samples.jsonl"
        for index in range(CHUNK_COUNT):
            # Thermal/AC loss blocks only a new admission; it never kills an admitted chunk.
            while True:
                mac_lease.require(args.lease, OWNER, lease["id"])
                try:
                    preflight = host.check_guard()
                    break
                except Exception as exc:
                    _write_sample(samples, "admission_guard_wait", {"error": str(exc)})
                    time.sleep(60)
            _write_sample(samples, "before_chunk", preflight)
            mac_lease.require(args.lease, OWNER, lease["id"])
            chunk = args.out / f"chunk-{index:02d}"
            command = [
                sys.executable,
                "-m",
                "research.bl_served_search.ceiling",
                "--allow-production",
                "--bank",
                str(args.bank),
                "--out",
                str(chunk / "shard.json"),
                "--shard-index",
                str(index),
                "--shard-count",
                "8",
                "--workers",
                "2",
                "--b8-path",
                str(args.b8_path),
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--runtime-parity-receipt",
                str(args.runtime_parity_receipt),
                "--search-mode",
                "exact_stop",
                "--native-mode",
                "kernel",
                "--lease",
                str(args.lease),
                "--lease-owner",
                OWNER,
                "--lease-id",
                lease["id"],
            ]
            code = _run_chunk(
                command,
                timeout=CHUNK_TIMEOUT_SECONDS,
                samples=samples,
                sample=host.snapshot,
                lease_guard=lambda: mac_lease.require(args.lease, OWNER, lease["id"]),
            )
            if code:
                raise RuntimeError(f"chunk {index} exited {code}")
            mac_lease.require(args.lease, OWNER, lease["id"])
            operational = _operational_ok(
                chunk / "operational.json", source=source, lease_id=lease["id"], shard_index=index
            )
            terminal["chunks"].append({"index": index, **operational})
        terminal["status"] = "TERMINAL_VALID"
    except BaseException as exc:
        terminal["error"] = str(exc)
        terminal["interrupted"] = isinstance(exc, (KeyboardInterrupt, SystemExit))
        failure = exc
    finally:
        # A foreign/replaced lease must never be removed by this launcher.
        try:
            mac_lease.release(args.lease, OWNER, lease["id"])
            terminal["lease_released"] = True
        except Exception as exc:
            terminal["lease_released"] = False
            terminal["lease_release_error"] = str(exc)
        export_json(args.out / "terminal.json", terminal)
        signal.signal(signal.SIGTERM, previous_sigterm)
    if failure is not None:
        raise failure
    if terminal["status"] != "TERMINAL_VALID":
        raise SystemExit(1)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "mac-amendment-01")
    parser.add_argument("--lease", type=Path, default=DEFAULT_ROOT.parent / "mac-lease.json")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--b8-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--runtime-parity-receipt", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--p3-receipt", type=Path)
    args = parser.parse_args()
    if args.check == args.run:
        parser.error("choose exactly one of --check or --run")
    if args.check:
        print(json.dumps(check(), sort_keys=True))
        return
    required = (
        args.bank,
        args.b8_path,
        args.runtime_parity_receipt,
        args.review_receipt,
        args.p3_receipt,
    )
    if not all(required):
        parser.error("--run requires bank, B8, parity, review and P3 receipts")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
