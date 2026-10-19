"""Owner-run, local-only settlement for a reviewed S1 qualification failure.

This program updates only the named ledger. It never invokes Colab, the
supervisor, tmux, a dispatcher, or a publishing command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.s1_mature_20261018.ledger import Ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    result = Ledger(args.ledger).settle_pretraining_qualification_failure(
        args.job_id, args.receipt, args.review
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
