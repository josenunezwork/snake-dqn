"""E4/E2 upload admission: a generic supervisor validation is insufficient for Z-9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_upload.qualify import assert_upload_budget, qualify, verify_receipt

__all__ = ["assert_upload_budget", "qualify", "verify_receipt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="require an existing bound receipt")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = (
        verify_receipt(manifest, args.receipt)
        if args.verify
        else qualify(manifest, receipt_path=args.receipt)
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
