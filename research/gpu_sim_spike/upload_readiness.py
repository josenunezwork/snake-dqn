"""Package-local upload budget gate for E1 GPU-spike manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, qualify

ROOT = Path(__file__).resolve().parents[2]


def check(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Refuse a manifest whose initial upload exceeds the qualified budget."""
    return assert_upload_budget(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full and args.receipt is None:
        parser.error("--full requires a create-only --receipt")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = (
        qualify(manifest, repo=ROOT, receipt_path=args.receipt)
        if args.full or args.receipt
        else check(manifest)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
