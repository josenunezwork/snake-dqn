"""R30-only short check during a verified fresh W6 ordinary-production window.

This does not change the running factory, its lease or the global launcher.
Only a current attempt that froze its own first-hour baseline is eligible.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

from research.bl_ppo.exports import digest, verify_export
from research.compute import factory_interim_launcher as launcher
from research.mac_lease import default_lease_path

APPROVED_OPERATIONAL_PIN = "587ff73ecdb9ebb25a104c3e45450ceb851c0358"
INTERIM_SHA = "ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd"


class OrdinaryWindow:
    """Read only operational identities; never open labels or outcome telemetry."""

    def __init__(self, start_path: Path, pid: int) -> None:
        verify_export(start_path)
        self.start_path, self.start_sha = start_path, digest(start_path)
        self.start = json.loads(start_path.read_text())
        self.process = psutil.Process(pid)
        self.birth = self.process.create_time()
        self.argv = self.process.cmdline()
        self.lease_path = default_lease_path()
        self.lease = self.lease_path.read_bytes()
        baseline_paths = list(start_path.parent.glob("slowdown-baseline-*-w6.json"))
        if len(baseline_paths) != 1:
            raise ValueError("HOLD: one fresh W6 baseline required")
        baseline_path = baseline_paths[0]
        verify_export(baseline_path)
        baseline = json.loads(baseline_path.read_text())
        self.baseline_path, self.baseline_sha = baseline_path, digest(baseline_path)
        started = self.start["started_epoch"]
        if (
            self.start.get("schema") != "bl-labels-f0-mac-attempt-start/v1"
            or self.start.get("workers") != 6
            or self.start.get("owner_w12_ok") is not False
            or self.start.get("operational_source_commit") != APPROVED_OPERATIONAL_PIN
            or not self.start["namespace"].endswith("-mac-d1024")
            or abs(self.birth - started) > 60
            or len(list(start_path.parent.glob("attempt-*.start.json"))) != 1
            or baseline.get("schema") != "bl-labels-f0-slowdown-baseline/v1"
            or baseline.get("namespace") != self.start["namespace"]
            or baseline.get("source_commit") != self.start["source_commit"]
            or baseline.get("stream") != "watch6"
            or baseline.get("width") != 6
            or baseline.get("baseline", {}).get("frozen") is not True
            or baseline["baseline"].get("sample_count", 0) < 60
            or baseline.get("created_epoch", 0) < started + 3600
            or baseline["created_epoch"] > time.time()
        ):
            raise ValueError("HOLD: fresh first-hour/rearm/W12 window not cleared")
        if self.argv[1:3] != ["-m", "research.bl_labels.mac_driver"]:
            raise ValueError("HOLD: exact factory driver required")
        for flag, value in (
            ("--state-dir", str(start_path.parent)),
            ("--workers", "6"),
            ("--output", self.start["output"]),
        ):
            if self.argv.count(flag) != 1 or self.argv[self.argv.index(flag) + 1] != value:
                raise ValueError("HOLD: factory argv does not match start")
        control_path = Path(self.argv[self.argv.index("--control") + 1])
        if digest(control_path) != self.start["control_sha256"]:
            raise ValueError("HOLD: factory control identity differs")
        self.journal = start_path.with_name(
            start_path.name.removesuffix(".start.json") + ".factory-process.jsonl"
        )
        self.check()

    def check(self) -> None:
        if (
            launcher.TIMING_ARMED_PATH.exists()
            or self.lease_path.read_bytes() != self.lease
            or self.process.create_time() != self.birth
            or self.process.cmdline() != self.argv
            or digest(self.start_path) != self.start_sha
            or digest(self.baseline_path) != self.baseline_sha
            or any(self.start_path.parent.glob("attempt-*.terminal.json"))
            or (Path(self.start["output"]) / "runtime-result.json").exists()
        ):
            raise ValueError("HOLD: ordinary factory identity/window changed")
        lease = json.loads(self.lease)
        if (
            lease.get("owner") != "bl-labels-f0-mac-factory"
            or lease.get("purpose") != "F0 Mac hindsight label factory"
            or lease.get("until_epoch", 0) <= time.time()
        ):
            raise ValueError("HOLD: ordinary factory lease required")
        # Read only the last complete operational observation, with bounded I/O.
        with self.journal.open("rb") as stream:
            stream.seek(max(0, self.journal.stat().st_size - 65536))
            lines = stream.read(65536).splitlines()
        observation = json.loads(lines[-1])
        if (
            observation.get("attempt_id") != self.journal.name.split(".")[0][8:]
            or observation.get("decision") != "observation_complete"
            or observation.get("interim_ruling_sha256") != INTERIM_SHA
            or not 0 <= time.monotonic() - observation["sample_monotonic"] <= 20
        ):
            raise ValueError("HOLD: fresh successful factory observation required")
        workers = [
            child
            for child in self.process.children()
            if "--multiprocessing-fork" in child.cmdline()
        ]
        if not 1 <= len(workers) <= 6 or any(worker.nice() != 10 for worker in workers):
            raise ValueError("HOLD: ordinary W6 nice10 workers required")

    def evidence(self) -> dict:
        return {
            "ruling_sha256": INTERIM_SHA,
            "start_path": str(self.start_path),
            "start_sha256": self.start_sha,
            "baseline_path": str(self.baseline_path),
            "baseline_sha256": self.baseline_sha,
            "factory_pid": self.process.pid,
            "factory_creation_epoch": self.birth,
            "operational_source_commit": self.start["operational_source_commit"],
            "lease_preserved": True,
            "fresh_w6_baseline_and_rearm_cleared": True,
        }


def run(command: list[str], manifest: Path, sha: str, start: Path, pid: int) -> dict:
    """Reuse reviewed serial command, host lock, 120s deadline and owned cleanup."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        raise ValueError("HOLD: preflight launcher must already be nice>=15")
    deadline = time.monotonic() + 120
    work_deadline = deadline - 6
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(_signal: int, _frame: object) -> None:
        raise ValueError("HOLD: R30 serial check interrupted")

    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        with launcher._elapsed_alarm(deadline):
            with launcher.serial_test_slot():
                with launcher._elapsed_alarm(work_deadline):
                    launcher._reject_test_command(
                        command,
                        review_manifest=str(manifest),
                        review_sha256=sha,
                        worktree=Path.cwd(),
                    )
                    window = OrdinaryWindow(start, pid)
                child = None
                try:
                    # Block asynchronous interruption only across handle registration.
                    blocked = {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
                    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
                    try:
                        child = subprocess.Popen(
                            ["/usr/bin/nice", "-n", "0", *command],
                            env=launcher._env(),
                            start_new_session=True,
                            preexec_fn=lambda: signal.pthread_sigmask(
                                signal.SIG_SETMASK, prior_mask
                            ),
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                    if os.getpriority(os.PRIO_PROCESS, child.pid) < 15:
                        raise ValueError("HOLD: numerical child must inherit nice>=15")
                    with launcher._elapsed_alarm(work_deadline):
                        while child.poll() is None:
                            window.check()
                            time.sleep(0.1)
                        window.check()
                        result = {"returncode": child.returncode, **window.evidence()}
                finally:
                    for sig in old_handlers:
                        signal.signal(sig, signal.SIG_IGN)
                    if child is not None:
                        launcher._cleanup(child, deadline)
        return result
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    import argparse

    from research.bl_ppo.exports import export_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run(command, args.manifest, args.sha, args.start, args.pid)
    export_json(args.evidence, result)
    raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
