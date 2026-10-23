"""Owner-authorized Mac background load: executable locations and sampled disclosure."""

from __future__ import annotations

import re
from pathlib import Path

ESTIMAND = "served-Mac timing under normal macOS background load (owner-authorized)"
OWNER_DIRECTIVE_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "OWNER_DIRECTIVE_M1_DAEMONS_20261022.txt"
)

# Owner clarification 2026-10-23 generalizes the 2026-10-22 directive to
# executable locations, independent of daemon name. Keep directory boundaries;
# user applications and command-line arguments never establish system identity.
SYSTEM_EXECUTABLE_ROOTS = ("/System/", "/usr/libexec/", "/usr/sbin/", "/sbin/")
CHECKOUT_PARENT = Path("/Users/josenunez/Projects/ml")


def checkout(path: str) -> Path | None:
    """Accept only direct snake-dqn checkouts, excluding the artifact directory."""
    root = Path(path)
    if (
        root.is_absolute()
        and root.parent == CHECKOUT_PARENT
        and (root.name == "snake-dqn" or root.name.startswith("snake-dqn-"))
        and root.name != "snake-dqn-artifacts"
        and ".." not in root.parts
    ):
        return root
    return None


def background_kind(process: dict, identity: dict) -> str | None:
    """Trust system executable locations or our supervisor entrypoint, never args text."""
    command = process.get("command", "")
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(command).parts:
        return "macos_system_daemon"
    if not re.fullmatch(r"[Pp]ython(?:\d+(?:\.\d+)*)?", Path(process.get("command", "")).name):
        return None
    argv = identity.get("argv", [])
    if not argv or not re.fullmatch(r"[Pp]ython(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
        return None
    # Only the actual Python entrypoint is considered; -c/arguments mentioning
    # a supervisor do not authorize an unrelated Python workload.
    if len(argv) >= 3 and argv[1] == "-m":
        module = argv[2]
        if (
            module == "research.colab_cli" or module.startswith("research.colab_cli.")
        ) and checkout(identity.get("cwd", "")):
            return "own_colab_supervisor"
    if len(argv) >= 2:
        script = Path(argv[1])
        if not script.is_absolute():
            script = Path(identity.get("cwd", "")) / script
        if script.parts[-3:] == ("tools", "ops", "dispatch.py") and checkout(
            str(script.parents[2])
        ):
            return "own_colab_supervisor"
    return None


def allowed_background(screen: dict) -> list[dict]:
    """Keep CPU for every allowed process, including observations below 50 percent."""
    result = []
    for line in screen.get("processes", "").splitlines()[1:]:
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, ppid, cpu, command = fields
        row = {"pid": int(pid), "ppid": int(ppid), "cpu_percent": float(cpu), "command": command}
        identity = screen.get("process_identity", {}).get(pid, {})
        kind = background_kind(row, identity)
        if kind:
            result.append({**row, "kind": kind, **({"identity": identity} if identity else {})})
    return result


def background_summary(segments: list[tuple[list[dict], int]]) -> dict:
    """Forward-hold each sample to the next sample/end, never across segment gaps.

    CPU is summed across PIDs of the same exact command. Absent commands count
    as zero. Means are time-weighted; coverage and the point-sample limitation
    are explicit. These disclosures never enter a bar or restart predicate.
    """
    intervals = []
    for rows, finished in segments:
        for index, row in enumerate(rows):
            started = row["monotonic_ns"]
            end = rows[index + 1]["monotonic_ns"] if index + 1 < len(rows) else finished
            values: dict[str, float] = {}
            for process in row.get("allowed_background_processes", []):
                command = process["command"]
                values[command] = values.get(command, 0.0) + process["cpu_percent"]
            intervals.append((max(0, end - started) / 1e9, values))
    duration = sum(seconds for seconds, _ in intervals)

    def stats(command: str | None) -> dict:
        points = [
            (seconds, sum(values.values()) if command is None else values.get(command, 0.0))
            for seconds, values in intervals
        ]
        return {
            "max_cpu_percent": max((cpu for _, cpu in points), default=0.0),
            "mean_cpu_percent": (
                sum(seconds * cpu for seconds, cpu in points) / duration if duration else None
            ),
            "time_at_or_above_50_seconds": sum(seconds for seconds, cpu in points if cpu >= 50.0),
        }

    return {
        "method": (
            "time-weighted forward hold of point samples; absent processes zero; "
            "no gap extrapolation"
        ),
        "sample_count": len(intervals),
        "observed_seconds": duration,
        "total": stats(None),
        "by_command": {
            command: stats(command)
            for command in sorted({command for _, values in intervals for command in values})
        },
    }
