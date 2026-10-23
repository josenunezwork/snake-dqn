"""Read-only Mac host evidence and fail-closed power/lid/thermal guard."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from research.compute.thermal_guard import parse_pmset_therm, thermal_reasons


def _read(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=True)
    return result.stdout


class GuardError(RuntimeError):
    """Carry the failed host observation into a create-only stop receipt."""

    def __init__(self, observation: dict[str, Any]) -> None:
        self.observation = observation
        super().__init__(f"host guard: {observation.get('reasons')}")


def process_screen(*, include_identity: bool = False) -> dict[str, Any]:
    """Record a point-in-time CPU screen, excluding only this measurement PID."""
    raw = _read(["ps", "-ww", "-Ao", "pid,ppid,%cpu,comm", "-r"])
    heavy = []
    identities = {}
    for line in raw.splitlines()[1:]:
        fields = line.strip().split(None, 3)
        if (
            include_identity
            and len(fields) == 4
            and Path(fields[3]).name.lower().startswith("python")
        ):
            import psutil

            try:
                process = psutil.Process(int(fields[0]))
                identities[fields[0]] = {"argv": process.cmdline(), "cwd": process.cwd()}
            except (psutil.Error, OSError):
                pass  # Unreadable/disappeared identities receive no exemption.
        if len(fields) == 4 and int(fields[0]) != os.getpid() and float(fields[2]) >= 50:
            heavy.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "cpu_percent": float(fields[2]),
                    "command": fields[3],
                }
            )
    return {
        "processes": raw,
        "other_heavy_processes": heavy,
        "measurement_pid_excluded": os.getpid(),
        "threshold_percent_one_core": 50,
        **({"process_identity": identities} if include_identity else {}),
    }


def check_guard() -> dict[str, Any]:
    """Stop admitting work on battery, closed lid, unknown state or throttling."""
    if platform.system() != "Darwin":
        raise RuntimeError("Mac-only measurement")
    power = _read(["pmset", "-g", "batt"])
    lid = _read(["ioreg", "-r", "-k", "AppleClamshellState"])
    thermal = _read(["pmset", "-g", "therm"])
    reasons = thermal_reasons(parse_pmset_therm(thermal))
    if "AC Power" not in power:
        reasons.append("not_on_ac")
    if '"AppleClamshellState" = No' not in lid:
        reasons.append("lid_not_known_open")
    value = {
        "time": datetime.datetime.now().astimezone().isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "power": power,
        "lid_open": '"AppleClamshellState" = No' in lid,
        "thermal": thermal,
        "reasons": reasons,
    }
    if reasons:
        raise GuardError(value)
    return value


def snapshot(*, process_policy: Callable[[dict], list[dict]] | None = None) -> dict[str, Any]:
    """Keep measured occupancy distinct from the owner's Phase-0 completion."""
    value = check_guard()
    screen = process_screen(include_identity=process_policy is not None)
    processes = screen["processes"]
    heavy = screen["other_heavy_processes"] if process_policy is None else process_policy(screen)
    tmux = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=10)
    value.update(
        schema="bl-mac-host/v1",
        pid=os.getpid(),
        platform=platform.platform(),
        load_average=list(os.getloadavg()),
        cpu_count=os.cpu_count(),
        top=_read(["top", "-l", "1", "-n", "0"]),
        processes=processes,
        process_screen=screen,
        heavy_processes=heavy,
        quiet=not heavy,
        tmux=tmux.stdout,
        tmux_exit=tmux.returncode,
        phase0_session_absent="phase0-selection:" not in tmux.stdout,
        phase0_finished_authority="owner instruction 2026-10-21",
        status="QUIET" if not heavy else "PHASE0_IDLE_HOST_NOT_QUIET",
        status_document_sha256=__import__("hashlib")
        .sha256(Path("docs/STATUS.md").read_bytes())
        .hexdigest(),
    )
    return value
