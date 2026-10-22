"""Admit V3 transport while executing all O2 V2 science in its pinned snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.t34_round0.analyze import InvalidEvidence


def analyze(registration_path: Path, run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Run all scientific imports in the frozen snapshot with the original look marker."""
    from research.t34_o2_harvest.frozen_sources_v3 import ROOT, invoke

    canonical = ROOT / "research/t34_o2_harvest/REGISTRATION_O2_V2.json"
    if registration_path.resolve() != canonical.resolve():
        raise InvalidEvidence("analysis requires the canonical frozen package registration")

    return invoke(
        registration_path.resolve(),
        analysis={
            "run_dirs": [str(path.resolve()) for path in run_dirs],
            "output_dir": str(output_dir.resolve()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.registration, args.run_dir, args.out_dir)["status"])


if __name__ == "__main__":
    main()
