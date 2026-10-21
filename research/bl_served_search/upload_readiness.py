"""Package-local upload budget gate for served-search manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, qualify

ROOT = Path(__file__).resolve().parents[2]


def check(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Refuse a manifest whose supervisor upload forecast exceeds the budget."""
    return assert_upload_budget(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.full and args.receipt is None:
        parser.error("--full requires a create-only --receipt")
    result = (
        qualify(manifest, repo=ROOT, receipt_path=args.receipt)
        if args.full or args.receipt is not None
        else check(manifest, repo=ROOT)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
