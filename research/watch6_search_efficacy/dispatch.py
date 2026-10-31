"""Source-bound admission precedes the qualified operator's existing dispatcher."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from research.watch6_search_efficacy.admission import admit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--operator-root",
        type=Path,
        help="Operator checkout with multi-tree qualification support; broker/source remain here",
    )
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text())
    admit(value)
    root = Path(__file__).resolve().parents[2]
    operator = args.operator_root.resolve() if args.operator_root else root
    command = [
        str(operator / "tools/ops/dispatch.sh"),
        value["job_id"],
        str(args.manifest.resolve()),
        "--broker",
        str(root),
        "--cwd",
        str(root),
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
