"""Run a capped, real-actor RETURN calibration; never a campaign-length run.

Uses the registered RETURN recipe, the actual bank and B8 initializations. The
only training change is an early stop after 300 measured learner updates (also
bounded at 200k transitions). Diagnostics are read without flushing the fast
learner pipeline. The normal trainer owns and cleans up its own actors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ART = ROOT.parent / "snake-dqn-artifacts"
UPDATE_LIMIT = 300


def recipe(seed: int, weight: float, out: Path, device: str) -> list[str]:
    """Return the production recipe with an unconditional short-run ceiling."""
    return [
        "--seed",
        str(seed),
        "--init",
        str(ART / f"redesign-m4b-mac-20261013/B8_s{seed - 10}/final.pth"),
        "--out",
        str(out),
        "--demo-data",
        str(ART / "redesign-m2b-20261008/data"),
        "--transitions",
        "200000",
        "--device",
        device,
        "--actors",
        "2",
        "--worlds",
        "128",
        "--segment-frames",
        "5000",
        "--reward",
        "mass",
        "--mass-kappa",
        "1.0",
        "--kill-scale",
        "0",
        "--gamma-start",
        "0.998",
        "--gamma-end",
        "0.998",
        "--gamma-hold",
        "0",
        "--gamma-anneal-end",
        "1",
        "--eps-end-at",
        "1",
        "--mixed-frac",
        "0.25",
        "--anchor-mode",
        "v8margin",
        "--anchor-hold",
        "0",
        "--anchor-floor-at",
        "1",
        "--anchor-floor",
        "0.1",
        "--onpolicy-labels",
        "--label-frac",
        "1",
        "--obs-version",
        "bv8",
        "--v61-backend",
        "numba",
        "--learner-fast",
        "--batch-scale",
        "2",
        "--obs-pack",
        "--checkpoint-every",
        "2000000",
        "--tight-telemetry",
        "--td-huber-delta",
        "1.0",
        "--return-bank",
        str(ART / "t4e-return-bank-20261016/banks/train"),
        "--return-frac",
        "0.5",
        "--return-weight",
        str(weight),
        "--return-huber-delta",
        "100",
        "--return-gradient-telemetry",
    ]


def main() -> int:
    """Instrument only this process and stop via the trainer's normal exit path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(10, 12), required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    if not 0 < options.weight <= 1:
        parser.error("weight must be in (0, 1]")
    if options.out.exists():
        parser.error("output already exists")

    import torch

    from research.redesign_m3c_20261008 import learner_m3c, train_m3c

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    argv = recipe(options.seed, options.weight, options.out, device)
    original = learner_m3c.Learner.update
    records = []

    def measured_update(learner, lam: float) -> None:
        prior_pending = learner._pending
        original(learner, lam)
        # Fast mode finalizes the previous update and stages the current one.
        # Read the current pending measurement so the final normal flush applies
        # update 300, rather than accidentally staging update 301 to observe 300.
        pending = learner._pending
        current = pending.get("return_gradient_telemetry") if pending is not None else None
        if current is not None and pending is not prior_pending:
            row = {"update": len(records) + 1, "transitions": learner.st["transitions"], **current}
            records.append(row)
            with (options.out / "calibration_updates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if len(records) % 50 == 0:
                print(json.dumps({"calibration_update": len(records), **current}), flush=True)
            if len(records) >= UPDATE_LIMIT:
                learner.args.transitions = learner.st["transitions"]
                learner.st["since_update"] = 0

    learner_m3c.Learner.update = measured_update
    print(json.dumps({"device": device, "argv": argv, "update_limit": UPDATE_LIMIT}), flush=True)
    try:
        code = train_m3c.main(argv)
    finally:
        learner_m3c.Learner.update = original
    if options.out.exists():
        hashes = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "research/redesign_m3c_20261008/learner_m3c.py",
                ROOT / "research/redesign_m3c_20261008/train_m3c.py",
            )
        }
        (options.out / "calibration_receipt.json").write_text(
            json.dumps(
                {
                    "schema": "t4e-calibration-smoke/v1",
                    "device": device,
                    "argv": argv,
                    "update_limit": UPDATE_LIMIT,
                    "measured_updates": len(records),
                    "trainer_exit": code,
                    "source_sha256": hashes,
                },
                indent=2,
            )
            + "\n"
        )
    return code if code else (0 if len(records) == UPDATE_LIMIT else 2)


if __name__ == "__main__":
    raise SystemExit(main())
