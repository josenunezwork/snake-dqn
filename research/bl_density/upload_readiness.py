"""Package-local Z-9 upload-readiness check, without provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, qualify

ROOT = Path(__file__).resolve().parents[2]


def check(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Check exact local inputs against the qualified staging budget."""
    return assert_upload_budget(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = (
        qualify(manifest, repo=ROOT, receipt_path=args.receipt) if args.receipt else check(manifest)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
