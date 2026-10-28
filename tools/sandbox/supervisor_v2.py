"""Future-only pinned sandbox adapter with immutable direct-child accounting.

Set SNAKE_SANDBOX_COLLECTOR_DIR to a fresh absolute output directory. This module
must be selected explicitly for a future dispatch; supervisor.py is unchanged.
It never attaches to existing roots and does not grant timing-panel admission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tools.sandbox.collector_v2 import Collector, collector_route, read_collector


def main() -> None:
    """Install collection before qualified main; keep all safety paths intact."""
    if "--export-collector" in sys.argv[1:]:
        from research.bl_paced_identity.record import export_json

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export-collector", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        export_json(args.output, read_collector(args.export_collector))
        return
    if "--collector-smoke" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--collector-smoke", type=Path, required=True)
        directory = parser.parse_args().collector_smoke
        collector = Collector(directory)
        result = collector.run(
            [sys.executable, "-c", "print('collector-smoke')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = collector.seal()
        print(
            json.dumps(
                {"returncode": result.returncode, "stdout": result.stdout.strip(), "status": status}
            )
        )
        return
    args = sys.argv[1:]
    if (
        len(args) != 6
        or args[0] != "--state-dir"
        or args[2] != "--colab"
        or args[4] not in {"validate", "run"}
    ):
        raise ValueError("future collector accepts only exact sandbox run argv")
    from tools.sandbox.supervisor import main as qualified_sandbox_main

    if args[4] == "validate":
        # Validate is allocation-free and must not consume the future journal.
        qualified_sandbox_main()
        return
    from tools.ops.dispatch import sandbox_collector_directory

    packet = json.loads(Path(args[-1]).read_text())
    expected = sandbox_collector_directory(packet)
    directory = Path(os.environ.get("SNAKE_SANDBOX_COLLECTOR_DIR", ""))
    if expected is None or directory != expected or not directory.is_absolute():
        raise ValueError("collector directory must match the opted-in packet output sibling")
    collector = Collector(directory)

    try:
        with collector_route(collector):
            qualified_sandbox_main()
    finally:
        # Accounting failure must not replace an existing control/cleanup error.
        try:
            collector.seal()
        except Exception as exc:
            print("collector evidence INVALID: " + type(exc).__name__, file=sys.stderr)


if __name__ == "__main__":
    main()
