"""R6 E-scaling operator companion: L4 compile, A100 canary, then A100 main.

The same governed lanes as ``r2_packet.py`` (dry-run first, create-only bound
manifests, one fresh canary attempt after a pre-work kernel-creation failure,
main built without regenerating the executed canary), selected for the R6
family: artifact ``gpu-r6-escaling-20261024``, broker
``snake-dqn-gpur6-qualified.noindex``, packer ``research/gpu_sim_spike/r6_pack.py``.
Nothing here dispatches unless the operator omits ``--dry-run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ops import r2_packet  # noqa: E402

R6 = r2_packet.Family(
    "r6",
    r2_packet.BASE / "snake-dqn-artifacts/big-lever/exp/gpu-r6-escaling-20261024",
    r2_packet.BASE / "snake-dqn-gpur6-qualified.noindex",
    "R6_BENCH_SECONDS",
    "r6_packet.py",
)


def main() -> None:
    r2_packet.FAMILY = R6
    r2_packet.main()


if __name__ == "__main__":
    main()
