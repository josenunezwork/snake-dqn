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


def parse_value(v: str):
    """Coerce an override string to int, then float, else keep the raw string.

    Contract overrides are usually numeric (board_scale, num_snakes, reward floats)
    but some are non-numeric (e.g. arena_type). Try int -> float -> str so an
    integer like num_snakes=8 stays an int and a label stays a string instead of
    crashing float().
    """
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def set_in_apex_config(ck: dict, key: str, value) -> None:
    """Set key in every contract location the validator inspects.

    The resume validator reads contract values from three locations -- the
    checkpoint top level, ``apex_config``, and the learner's separate ``config``
    blob (see ``checkpoint_contract.checkpoint_contract_values``). We must keep all
    three consistent or a rebased key (gamma/hidden/board_scale/...) desyncs: the
    validator would compare the stale ``config`` copy and reject the resume. We
    update ``apex_config`` unconditionally and the top-level / ``config`` copies
    only where the key already exists, so we never invent keys in a blob that did
    not declare them.
    """
    ac = ck.setdefault("apex_config", {})
    cfg = ck.get("config")
    if key.startswith("reward_contract."):
        child = key.split(".", 1)[1]
        locs = [ck, ac]
        if isinstance(cfg, dict):
            locs.append(cfg)
        for loc in locs:
            rc = loc.get("reward_contract")
            if isinstance(rc, dict):
                rc[child] = value
    else:
        ac[key] = value
        if key in ck:
            ck[key] = value
        if isinstance(cfg, dict) and key in cfg:
            cfg[key] = value


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, dst = argv[0], argv[1]
    overrides = {}
    for kv in argv[2:]:
        if "=" not in kv:
            print(f"ERROR: override {kv!r} is not of the form key=value")
            return 1
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
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
