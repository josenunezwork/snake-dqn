"""Bounded headless serving check for a declared checkpoint contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct script execution sets sys.path to src/scripts, while the web package
# lives at the repository root. Keep the documented entrypoint usable without
# asking callers to export PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.backend.session import GameSession  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint to serve")
    parser.add_argument("--frames", type=int, default=10, help="Bounded Watch frames (1..100)")
    parser.add_argument("--output", required=True, help="Disposable JSON receipt path")
    args = parser.parse_args()
    if not 1 <= args.frames <= 100:
        parser.error("--frames must be in 1..100")

    session = GameSession(checkpoint=args.checkpoint)
    for _ in range(args.frames):
        session.step()
    frame = session.snapshot()
    receipt = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames_requested": args.frames,
        "frame": frame["frame"],
        "obs_spec": frame["obs_spec"],
        "serving_contract": frame.get("serving_contract"),
        "error": session.last_error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    contract = receipt["serving_contract"]
    required_digests = {
        "obs_contract_digest",
        "model_head_digest",
        "effective_world_digest",
        "runtime_contract_digest",
        "action_mask_contract_digest",
        "run_provenance_digest",
    }
    if (
        receipt["error"]
        or receipt["obs_spec"] != "raster31v3"
        or not isinstance(contract, dict)
        or not required_digests.issubset(contract)
        or contract.get("deployment_profile") != "promotion-v2-watch-rect"
        or not isinstance(contract.get("deployment_target_manifest_digest"), str)
        or not isinstance(contract.get("checkpoint_sha256"), str)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
