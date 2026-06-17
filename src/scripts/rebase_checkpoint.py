#!/usr/bin/env python3
"""Rebase a checkpoint's contract metadata so a new curriculum/reward can resume it.

Curriculum warm-start technique: copy a checkpoint and rewrite ONLY the contract
metadata that the resume validator checks, keeping weights/gamma/n_step/network
byte-identical. Used to change actor geometry (board_scale, snakes) or reward
shaping (reward_contract.*) while reusing learned weights.

Usage:
  ./venv/bin/python src/scripts/rebase_checkpoint.py SRC DST \
      actor_board_scale=0.8 actor_env_num_snakes=8 reward_contract.boost_segment=3.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def set_in_apex_config(ck: dict, key: str, value: float) -> None:
    """Set key in every contract location the validator inspects."""
    ac = ck.setdefault("apex_config", {})
    if key.startswith("reward_contract."):
        child = key.split(".", 1)[1]
        for loc in (ck, ac):
            rc = loc.get("reward_contract")
            if isinstance(rc, dict):
                rc[child] = value
    else:
        ac[key] = value
        if key in ck:
            ck[key] = value


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, dst = argv[0], argv[1]
    overrides = {}
    for kv in argv[2:]:
        k, v = kv.split("=", 1)
        overrides[k] = float(v)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for k, v in overrides.items():
        set_in_apex_config(ck, k, v)
    torch.save(ck, dst)
    # verify
    d = torch.load(dst, map_location="cpu", weights_only=False)
    ac = d.get("apex_config", {})
    print(f"Rebased {src} -> {dst}")
    for k, v in overrides.items():
        if k.startswith("reward_contract."):
            child = k.split(".", 1)[1]
            print(f"  {k} = {ac.get('reward_contract', {}).get(child)}")
        else:
            print(f"  {k} = {ac.get(k)}")
    print(f"  step_count={d.get('step_count')} (weights unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
