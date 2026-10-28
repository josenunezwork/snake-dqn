"""Operator front door for ruling section 6(5), exactly 600 host-only seconds.

With today's collector this refuses before observation: direct-child accounting
does not supply an admissible live stream. A declaration/filename/flag cannot
turn that missing implementation into a qualification receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_served_search.supervision_dispatch import capture_dispatch_registry
from research.bl_served_search.supervision_io import BLOCKER


def main(argv: list[str] | None = None) -> int:
    """Require a prospectively observed live v2 dispatch, never a PID-only claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-manifest", type=Path, required=True)
    parser.add_argument("--desktop-app-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-only-seconds", type=int, choices=[600], default=600)
    parser.add_argument(
        "--check",
        action="store_true",
        help="inspect live dispatch declaration and report missing qualification capability",
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("qualification output must not exist")
    from research.bl_served_search.desktop_apps import DesktopRegistry
    from research.bl_served_search.remote_supervision import validate_registry

    DesktopRegistry(args.desktop_app_registry).read_export()
    checked = validate_registry(capture_dispatch_registry(args.dispatch_manifest))
    if not checked["valid"]:
        raise ValueError(f"invalid supervision registry: {checked['errors']}")
    if args.check:
        print(
            json.dumps(
                {
                    "policy": "classified-desktop-supervised-v2",
                    "requested_seconds": args.host_only_seconds,
                    "status": "NOT_ADMISSIBLE",
                    "reason": BLOCKER,
                    "qualification_published": False,
                }
            )
        )
        return 2
    # No metadata flag, invented receipt or filename can open the missing
    # observer gate. This command deliberately publishes no qualification.
    raise RuntimeError(BLOCKER)


if __name__ == "__main__":
    raise SystemExit(main())
