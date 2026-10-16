"""Run an explicitly-MPS, capped RETURN_C contrast calibration smoke.

This entry point is intentionally unable to start an 8M campaign: it uses the
real two-actor recipe but stops after 300 measured learner updates and has a
200k-transition ceiling.  The caller must supply an existing macro-bank path.
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
MACRO_BANK = ART / "t4f-return-bank-20261016/banks/train"


def recipe(seed: int, weight: float, out: Path, *, bank: Path = MACRO_BANK) -> list[str]:
    """Return RETURN_C's production recipe with only a short-run ceiling."""
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
        "mps",
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
        "500000",
        "--tight-telemetry",
        "--td-huber-delta",
        "1.0",
        "--return-bank",
        str(bank),
        "--return-frac",
        "0.5",
        "--return-weight",
        str(weight),
        "--return-huber-delta",
        "100",
        "--return-mode",
        "contrast",
        "--return-contrast-epsilon",
        "2.0",
        "--boost-halt-rate",
        "0.02",
        "--boost-halt-low-length-rate",
        "0.01",
        "--return-gradient-telemetry",
    ]


def main() -> int:
    """Collect exactly 300 measurement rows through normal trainer cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(10, 12), required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=MACRO_BANK)
    options = parser.parse_args()
    if not 0 < options.weight <= 1:
        parser.error("weight must be in (0, 1]")
    if options.out.exists() or not options.bank.is_dir():
        parser.error("output already exists or macro bank is absent")
    import torch

    if not torch.backends.mps.is_available():
        parser.error("RETURN_C calibration requires explicit available MPS")
    from research.redesign_m3c_20261008 import learner_m3c, train_m3c

    argv = recipe(options.seed, options.weight, options.out, bank=options.bank)
    original, records = learner_m3c.Learner.update, []

    def measured_update(learner, lam: float) -> None:
        prior_pending = learner._pending
        original(learner, lam)
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
    print(json.dumps({"device": "mps", "argv": argv, "update_limit": UPDATE_LIMIT}), flush=True)
    try:
        code = train_m3c.main(argv)
    finally:
        learner_m3c.Learner.update = original
    if options.out.exists():
        source = (
            Path(__file__),
            ROOT / "research/redesign_m3c_20261008/learner_m3c.py",
            ROOT / "research/redesign_m3c_20261008/train_m3c.py",
        )
        (options.out / "calibration_receipt.json").write_text(
            json.dumps(
                {
                    "schema": "t4f-contrast-calibration-smoke/v1",
                    "device": "mps",
                    "argv": argv,
                    "update_limit": UPDATE_LIMIT,
                    "measured_updates": len(records),
                    "trainer_exit": code,
                    "source_sha256": {
                        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in source
                    },
                },
                indent=2,
            )
            + "\n"
        )
    return code if code else (0 if len(records) == UPDATE_LIMIT else 2)


if __name__ == "__main__":
    raise SystemExit(main())
