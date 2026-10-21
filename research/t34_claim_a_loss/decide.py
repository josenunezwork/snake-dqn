"""The frozen D0 rule: pick the corrected Claim A objective and update dose, or KILL.

Input: one diagnostics report from :mod:`research.t34_claim_a_loss.harness` run on
the **excluded A5 v2 exports** with the registered cell set (``D0_CELLS``).  The rule
is spec ``docs/research/specs/2026-10-21_t34_claim_a_amendment.md`` §4, frozen before
the report exists; every constant below is quoted there.

Steps (mechanical, no judgment):

1. **Free-Q gate.** A corrected loss (``split``/``split_tau2``) is *eligible* only if
   its free-Q probe flips >= 99 % of decisive rows to a*.  If no corrected loss flips
   even 70 %, the projection is closed (``KILL``).
2. **Permissive gate.** For each eligible corrected cell, the permissive agreement is
   the best pooled *training-fold* decisive agreement at any checkpoint.  If no
   eligible cell reaches 70 %, the projection is closed (``KILL``).
3. **Guards** (FINAL_PLAN §8 tripwires on cross-fold held-out anchor rows, every fold,
   every checkpoint up to the dose): max length-decile disagreement <= 5 %, boost share
   <= 1.5x the previous student's, one-action share <= 90 %.
4. **Dose.** Default 2,048 updates.  If the guards fail at or before 2,048, the dose is
   the largest guarded checkpoint in {512, 1024}; none -> the cell is ineligible.  If
   2,048 is guarded, 4,096 is guarded, and pooled held-out decisive agreement at 4,096
   exceeds that at 2,048 by >= 0.05, the dose is 4,096.
5. **Per trunk** (trainable / residual) pick the eligible cell with the highest pooled
   held-out decisive agreement at its dose; ties -> smaller dose, smaller lr, ``split``
   before ``split_tau2``, then name.
6. ``PASS`` if at least one trunk has a selection, else ``KILL`` (guards).

The frozen-loss cells are controls: reported, never selected.  Held-out decisive
agreement here is a cross-fold *trend* on ~225 rows; it chooses among corrected
cells and is reported, but the 70 % held-out bar applies to the production fit on
the R1 held-out bank, unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .harness import D0_CELLS, D0_FREE_Q

SCHEMA = "t34-claim-a-d0-decision/v1"
CORRECTED_LOSSES = ("split", "split_tau2")
FREE_Q_ELIGIBLE = 0.99
PERMISSIVE_BAR = 0.70
DEFAULT_DOSE = 2048
FALLBACK_DOSES = (1024, 512)
RAISED_DOSE = 4096
RAISE_MARGIN = 0.05
MAX_DECILE_DISAGREE = 0.05
MAX_BOOST_RATIO = 1.5
MAX_ONE_ACTION_SHARE = 0.90
BLOCKS = 4
BATCH = {"size": 1024, "decisive": 128, "anchor": 896}


class DecisionRefused(ValueError):
    """The report is not a complete registered D0 report."""


def _curve(cell: Mapping[str, Any], updates: int) -> list[dict[str, Any]]:
    return [next(c for c in fold["curve"] if c["updates"] == updates) for fold in cell["folds"]]


def train_agreement(cell: Mapping[str, Any], updates: int) -> float:
    points = [c["train_decisive"] for c in _curve(cell, updates)]
    n = sum(p["n"] for p in points)
    return sum(p["flip"] * p["n"] for p in points) / n if n else 0.0


def heldout_agreement(cell: Mapping[str, Any], updates: int) -> float:
    point = next(p for p in cell["pooled"] if p["updates"] == updates)
    return float(point["heldout_decisive"].get("flip", 0.0))


def guards_hold(cell: Mapping[str, Any], updates: int) -> bool:
    for point in _curve(cell, updates):
        anchor = point["heldout_anchor"]
        if anchor.get("n", 0) == 0:
            return False
        ratio = anchor.get("boost_ratio")
        if anchor["max_decile_disagree"] > MAX_DECILE_DISAGREE:
            return False
        if ratio is not None and ratio > MAX_BOOST_RATIO:
            return False
        if anchor["max_action_share"] > MAX_ONE_ACTION_SHARE:
            return False
    return True


def guarded_through(cell: Mapping[str, Any], updates: int) -> bool:
    checkpoints = [u for u in cell["cell"]["checkpoints"] if 0 < u <= updates]
    return updates in cell["cell"]["checkpoints"] and all(guards_hold(cell, u) for u in checkpoints)


def dose_for(cell: Mapping[str, Any]) -> int | None:
    if guarded_through(cell, DEFAULT_DOSE):
        if (
            guarded_through(cell, RAISED_DOSE)
            and heldout_agreement(cell, RAISED_DOSE)
            >= heldout_agreement(cell, DEFAULT_DOSE) + RAISE_MARGIN
        ):
            return RAISED_DOSE
        return DEFAULT_DOSE
    for dose in FALLBACK_DOSES:
        if guarded_through(cell, dose):
            return dose
    return None


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    cells = {c["cell"]["name"]: c for c in report.get("cells", [])}
    missing = [name for name in D0_CELLS if name not in cells]
    free_q = {item["loss"]: item for item in report.get("free_q", [])}
    missing += [f"free_q:{loss}" for loss in D0_FREE_Q if loss not in free_q]
    if missing:
        raise DecisionRefused(f"report lacks registered D0 items: {missing}")
    controls = {
        name: {
            "train_agreement_max": (
                max(train_agreement(cell, u) for u in cell["cell"]["checkpoints"] if u)
                if cell["cell"]["rule"] == "streams"
                else None
            ),
            "pooled": cell["pooled"],
        }
        for name, cell in cells.items()
        if name in D0_CELLS and cell["cell"]["loss"] not in CORRECTED_LOSSES
    }
    free = {loss: float(free_q[loss]["flip"]) for loss in D0_FREE_Q}
    eligible_losses = [loss for loss in CORRECTED_LOSSES if free[loss] >= FREE_Q_ELIGIBLE]
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "constants": {
            "free_q_eligible": FREE_Q_ELIGIBLE,
            "permissive_bar": PERMISSIVE_BAR,
            "default_dose": DEFAULT_DOSE,
            "fallback_doses": list(FALLBACK_DOSES),
            "raised_dose": RAISED_DOSE,
            "raise_margin": RAISE_MARGIN,
            "max_decile_disagree": MAX_DECILE_DISAGREE,
            "max_boost_ratio": MAX_BOOST_RATIO,
            "max_one_action_share": MAX_ONE_ACTION_SHARE,
            "blocks": BLOCKS,
            "batch": BATCH,
        },
        "free_q_flip": free,
        "eligible_losses": eligible_losses,
        "controls": controls,
        "candidates": {},
        "selection": {},
    }
    if max(free[loss] for loss in CORRECTED_LOSSES) < PERMISSIVE_BAR:
        return out | {
            "decision": "KILL",
            "reason": "no corrected objective flips 70% of decisive rows even with free Q",
        }
    for name in D0_CELLS:
        cell = cells[name]
        spec = cell["cell"]
        if spec["loss"] not in CORRECTED_LOSSES:
            continue
        checkpoints = [u for u in spec["checkpoints"] if u]
        permissive = max(train_agreement(cell, u) for u in checkpoints)
        dose = dose_for(cell)
        out["candidates"][name] = {
            "loss": spec["loss"],
            "trunk": spec["trunk"],
            "lr": spec["lr"],
            "loss_eligible": spec["loss"] in eligible_losses,
            "permissive_train_agreement": permissive,
            "dose": dose,
            "heldout_agreement_at_dose": heldout_agreement(cell, dose) if dose else None,
            "train_agreement_at_dose": train_agreement(cell, dose) if dose else None,
            "decisive_passes_at_dose": (
                next(p for p in cell["pooled"] if p["updates"] == dose)["decisive_passes_mean"]
                if dose
                else None
            ),
            "heldout_curve": {u: heldout_agreement(cell, u) for u in checkpoints},
            "train_curve": {u: train_agreement(cell, u) for u in checkpoints},
            "guards": {u: guards_hold(cell, u) for u in checkpoints},
        }
    permissive_ok = {
        name: c
        for name, c in out["candidates"].items()
        if c["loss_eligible"] and c["permissive_train_agreement"] >= PERMISSIVE_BAR
    }
    if not permissive_ok:
        return out | {
            "decision": "KILL",
            "reason": "no eligible corrected objective reaches 70% decisive agreement even "
            "in-sample (permissive diagnostic)",
        }
    order = {loss: i for i, loss in enumerate(CORRECTED_LOSSES)}
    for trunk in ("trainable", "residual"):
        pool = [
            (name, c)
            for name, c in permissive_ok.items()
            if c["trunk"] == trunk and c["dose"] is not None
        ]
        if not pool:
            out["selection"][trunk] = {"status": "INELIGIBLE"}
            continue
        name, best = min(
            pool,
            key=lambda item: (
                -item[1]["heldout_agreement_at_dose"],
                item[1]["dose"],
                item[1]["lr"],
                order[item[1]["loss"]],
                item[0],
            ),
        )
        out["selection"][trunk] = {
            "status": "SELECTED",
            "cell": name,
            "loss": best["loss"],
            "lr": best["lr"],
            "updates": best["dose"],
            "blocks": BLOCKS,
            "updates_per_block": best["dose"] // BLOCKS,
            "batch": BATCH,
            "heldout_agreement_trend": best["heldout_agreement_at_dose"],
        }
    if all(s["status"] == "INELIGIBLE" for s in out["selection"].values()):
        return out | {
            "decision": "KILL",
            "reason": "no corrected objective keeps the anchor guards for >= 512 updates",
        }
    return out | {"decision": "PASS", "reason": "corrected objective and dose frozen"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = decide(json.loads(raw))
    result["report_sha256"] = hashlib.sha256(raw).hexdigest()
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with args.out.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"decision": result["decision"], "selection": result["selection"]}))


if __name__ == "__main__":
    main()
