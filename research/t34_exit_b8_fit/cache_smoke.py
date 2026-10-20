"""Mac-only bounded cache/timing smoke. Synthetic labels carry no research claim."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

from .interface import FakeReplaySource
from .train import _cpu_guard, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--checkpoint", type=Path, help="optional read-only B8 initializer")
    parser.add_argument("--tower", action="store_true", help="include optional S1a planes/tower")
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("this smoke is Mac only")
    _cpu_guard()
    torch.set_num_interop_threads(1)
    torch.manual_seed(34)
    from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

    model = (
        load_ego2s_checkpoint(args.checkpoint, torch.device("cpu"))
        if args.checkpoint
        else Ego2sNet(7, 60)
    )
    if args.tower:
        from .network import ExitFitTowerNet

        model = ExitFitTowerNet(model, aux_head=False)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = (
        hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    )
    result = fit(
        source=FakeReplaySource(worlds=2, rows_per_world=8, seed=34),
        model=model,
        previous_model=copy.deepcopy(model),
        device="cpu",
        batch_size=4,
        max_updates=8,
        allow_fake=True,
        seed=34,
        cache_dir=args.out / "cache",
        cache_identity={
            "fixture_seed": 34,
            "checkpoint_sha256": checkpoint_sha,
            "tower": args.tower,
        },
        include_tower=args.tower,
    )
    receipt = {
        "schema": "t34-fitcache-smoke/v1",
        "status": "MAC_SYNTHETIC_SMOKE_ONLY",
        "checkpoint_sha256": checkpoint_sha,
        "tower": args.tower,
        "updates": result["updates"],
        "history": result["history"],
        "fit_timing": result["fit_timing"],
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = args.out / "timing.json"
    path.write_bytes(encoded)
    path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  timing.json\n"
    )
    path.with_suffix(".json.complete").write_text("complete\n")
    print(json.dumps({"receipt": str(path), "updates": result["updates"]}))


if __name__ == "__main__":
    main()
