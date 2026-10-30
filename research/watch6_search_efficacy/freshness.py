"""Pre-data seed audit over published banks and visible historical inventories."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch

from research.bl_served_search import c1_bank
from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import publish_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    own_root = p.ROOT.parent / "snake-dqn-artifacts/t34-sol/watch6-efficacy-fix"
    original_run = subprocess.run

    def audit_run(command: list, **kwargs):
        # The shared auditor hard-codes C1's own-root exclusion. Replace it,
        # never exclude the older C1 allocation from this new study's history.
        command = [
            "!**/t34-sol/watch6-efficacy-fix/**" if arg == "!**/big-lever/exp/bl-c1/**" else arg
            for arg in command
        ]
        return original_run(command, **kwargs)

    # Reuse the existing exhaustive auditor, positive controls, mandatory FRP
    # training-exclusion inventory, and sealed-outcome exclusion. No own-bank
    # seed list is persisted until this audit has passed.
    with (
        patch.object(c1_bank, "allocation", p.allocation),
        patch.object(c1_bank, "NAMESPACE", p.NAMESPACE),
        patch.object(c1_bank, "COUNT", p.COUNT),
        patch.object(c1_bank, "STUDY_ROOT", own_root),
        patch.object(c1_bank.subprocess, "run", audit_run),
    ):
        result = c1_bank.freshness(repo=p.ROOT)
    publish_json(args.out, {**result, "allocation": p.allocation(), "fixtures": p.FIXTURE_SEEDS})


if __name__ == "__main__":
    main()
