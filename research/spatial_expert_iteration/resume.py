"""Mac-only round ledger adapter; byte/hash verification is owned by the pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.spatial_student_pilot.resume import build_ledger

from .config import RoundConfig
from .prepare import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = RoundConfig.from_dict(json.loads(args.config.read_text()))
    write_json(args.output, build_ledger(cfg, args.roots, args.receipts))


if __name__ == "__main__":
    main()
