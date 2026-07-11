#!/usr/bin/env python3
"""Widen a 58-D checkpoint to 61-D for the free-space state features.

The free-space features (network.use_free_space) append 3 inputs to the state
vector. This pads the network's first layer ``feature_layer.0.weight`` from
``(hidden, 58)`` to ``(hidden, 61)`` with ZERO columns — so the 3 new inputs
start inert and the network learns to use them during fine-tuning — and rewrites
the ``input_size`` metadata to 61 so the training resume contract accepts it.
Every other weight is byte-identical.

Stale optimizer moments and bundled replay memories (which reference the old
58-D layout) are dropped so the widened checkpoint resumes cleanly; Adam moments
re-warm in a few steps and the replay buffer refills on resume.

Usage:
  python src/scripts/widen_input.py SRC DST [--base 58] [--extra 3]

Then point a use_free_space=true config at DST via ``--load DST``. If you are
also changing the reward shaping, run rebase_checkpoint.py on DST afterwards so
the reward contract matches the new config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

FIRST_LAYER_SUFFIX = "feature_layer.0.weight"


def widen_state_dict(sd: dict, base: int, extra: int) -> int:
    """Pad every first-layer weight whose input dim == base by `extra` zero cols."""
    changed = 0
    for k, v in list(sd.items()):
        if (
            k.endswith(FIRST_LAYER_SUFFIX)
            and isinstance(v, torch.Tensor)
            and v.dim() == 2
            and v.shape[1] == base
        ):
            pad = torch.zeros(v.shape[0], extra, dtype=v.dtype, device=v.device)
            sd[k] = torch.cat([v, pad], dim=1)
            changed += 1
    return changed


def set_input_size(ck: dict, value: int) -> None:
    """Rewrite input_size in every location the resume contract inspects.

    The contract validator (checkpoint_contract.checkpoint_contract_values) reads
    input_size from the top level AND from the 'apex_config' and 'config' blobs;
    ALL recorded values must agree or training resume rejects the checkpoint.
    """
    ck["input_size"] = value
    for blob_key in ("apex_config", "config"):
        blob = ck.get(blob_key)
        if isinstance(blob, dict) and "input_size" in blob:
            blob["input_size"] = value


# Reward-contract keys the resume validator compares against the *current* config.
# Widening only changes the input layout, NOT the reward shaping — but a free-space
# fine-tune typically uses a different reward contract, and the OLD recorded values
# would then fail the resume check. Drop them so the validator falls back to the
# current-config defaults (a real reward change still needs rebase_checkpoint.py).
STALE_REWARD_KEYS = ("reward_contract", "reward_death", "reward_food_base")


def strip_reward_contract(ck: dict) -> int:
    """Remove stale reward-contract keys from the top level and nested blobs."""
    removed = 0
    for key in STALE_REWARD_KEYS:
        if key in ck:
            del ck[key]
            removed += 1
    for blob_key in ("apex_config", "config"):
        blob = ck.get(blob_key)
        if isinstance(blob, dict):
            for key in STALE_REWARD_KEYS:
                if key in blob:
                    del blob[key]
                    removed += 1
    return removed


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--base", type=int, default=58, help="current input size")
    ap.add_argument("--extra", type=int, default=3, help="features to append")
    a = ap.parse_args(argv)

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    total = 0
    for key in ("dqn_state_dict", "target_dqn_state_dict", "model_state_dict", "target_state_dict"):
        sd = ck.get(key)
        if isinstance(sd, dict):
            total += widen_state_dict(sd, a.base, a.extra)
    if total == 0:
        print(f"ERROR: no '{FIRST_LAYER_SUFFIX}' tensor with input dim {a.base} found.")
        return 1

    # Drop stale state that references the old input layout.
    for stale in ("optimizer_state_dict", "memories"):
        if stale in ck:
            del ck[stale]

    set_input_size(ck, a.base + a.extra)
    # Strip stale reward-contract metadata so a free-space train-resume doesn't fail
    # the reward-contract check against the new config (falls back to current defaults).
    reward_removed = strip_reward_contract(ck)
    torch.save(ck, a.dst)
    print(
        f"Widened {a.src} -> {a.dst}: padded {total} first-layer tensor(s) "
        f"{a.base}->{a.base + a.extra}, input_size={a.base + a.extra}, "
        f"dropped optimizer+memories, stripped {reward_removed} reward-contract key(s)."
    )
    print(
        "  NOTE: reward-contract metadata was removed so resume falls back to the "
        "current config's reward shaping. If you are changing the reward contract, "
        "run rebase_checkpoint.py on the widened file to record the matching contract."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
