"""Owner-authorized M1 background load: exact identities and sampled disclosure."""

from __future__ import annotations

import re
from pathlib import Path

ESTIMAND = "served-Mac timing under normal macOS background load (owner-authorized)"
OWNER_DIRECTIVE_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "OWNER_DIRECTIVE_M1_DAEMONS_20261022.txt"
)

_METADATA = (
    "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/"
    "Metadata.framework/Versions/A/Support/"
)
# No basename, substring, directory-prefix, or wildcard daemon exemptions.
SYSTEM_DAEMONS = frozenset(
    {
        *(
            _METADATA + name
            for name in (
                "corespotlightd",
                "managedcorespotlightd",
                "mds",
                "mds_stores",
                "mdworker",
                "mdworker_shared",
            )
        ),
        "/System/Library/PrivateFrameworks/FileProvider.framework/Support/fileproviderd",
        "/System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd",
        "/System/Library/PrivateFrameworks/PhotoAnalysis.framework/Versions/A/Support/"
        "photoanalysisd",
        "/System/Library/PrivateFrameworks/CloudKitDaemon.framework/Support/cloudd",
        "/System/Library/PrivateFrameworks/iCloudDriveCore.framework/Versions/A/Support/bird",
        "/usr/libexec/nsurlsessiond",
        "/usr/libexec/spotlightknowledged",
        "/usr/libexec/spotlightknowledged.importer",
        "/usr/libexec/spotlightknowledged.updater",
        "/System/Library/CoreServices/Spotlight.app/Contents/MacOS/Spotlight",
        "/System/Library/ExtensionKit/Extensions/SpotlightSettingsIntentsExtension.appex/"
        "Contents/MacOS/SpotlightSettingsIntentsExtension",
        "/System/Library/ExtensionKit/Extensions/SpotlightPreferenceExtension.appex/"
        "Contents/MacOS/SpotlightPreferenceExtension",
        "/System/Applications/Mail.app/Contents/PlugIns/"
        "com.apple.mail.SpotlightIndexExtension.appex/"
        "Contents/MacOS/com.apple.mail.SpotlightIndexExtension",
    }
)
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
    """Trust exact system paths or a Python entrypoint in our checkout, never args text."""
    if process.get("command") in SYSTEM_DAEMONS:
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
