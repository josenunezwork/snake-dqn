"""Calibrate the sparse raw-Q trigger on existing T1/M7 development rows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.parity import write_json
from research.t1_commitment_20261015.replay import ARTIFACTS as T1_ROOT

ROOT = census.ARTIFACTS / "t3-rollout-labels-20261015"
M7_ROOT = census.ARTIFACTS / "m7-diagnostics-20261014" / "feature_auc" / "runs"


class DenseTrigger:
    """Episode-local decision clock anchored at the first length-eligible decision.

    The clock advances with episode decisions even if length later falls below the
    cutoff. Frame eligibility gates collection without resetting the clock.
    """

    def __init__(self, min_length: int, min_frame: int = 0, interval: int = 5) -> None:
        if interval < 1:
            raise ValueError("interval must be positive")
        self.min_length = min_length
        self.min_frame = min_frame
        self.interval = interval
        self.first_decision: int | None = None

    def fires(self, decision: int, length: int, alive: bool = True) -> bool:
        """Fire at clock indices 0, 5, ... when all eligibility conditions hold."""
        if not alive or length < self.min_length:
            return False
        if self.first_decision is None:
            self.first_decision = decision
        return decision >= self.min_frame and (decision - self.first_decision) % self.interval == 0


def capped_triggers(frames: np.ndarray, eligible: np.ndarray, interval: int = 5) -> np.ndarray:
    """Accept eligible chronological decisions with at least interval separation."""
    if interval < 1:
        raise ValueError("interval must be positive")
    accepted = np.zeros(len(frames), bool)
    last = -interval
    for index in np.flatnonzero(eligible):
        frame = int(frames[index])
        if frame - last >= interval:
            accepted[index] = True
            last = frame
    return accepted


def calibrate(out: Path = ROOT) -> dict[str, Any]:
    """Choose the most selective death-maximum cutoff satisfying capped coverage."""
    import json

    captures, deaths, inputs = {}, [], []
    for arm in census.B8_ARMS:
        path = M7_ROOT / f"B8_s{arm[-1]}.npz"
        rows_path = T1_ROOT / "rows" / f"{arm}.jsonl"
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        census_path = census.OUT_ROOT / "runs" / f"{arm}.json"
        original = json.loads(census_path.read_text())
        if (
            metadata["checkpoint"] != original["checkpoint"]
            or metadata["seeds"] != original["seeds"]
            or metadata["mix"] != "scripted"
            or metadata["round_index"] != 52
            or metadata["frames"] != 5000
        ):
            raise ValueError(f"M7/census source identity mismatch: {arm}")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("q", "frame", "env", "length", "adv")}
            np.testing.assert_array_equal(archive["seeds"], original["seeds"])
        # Convert before comparison so a float64 nextafter threshold is not rounded
        # back to the float32 score by NumPy's scalar promotion rules.
        data["risk"] = (-data.pop("q").max(axis=1)).astype(np.float64)
        if not np.isfinite(data["risk"]).all():
            raise ValueError(f"nonfinite Q in {path}")
        captures[arm] = data
        inputs.extend(
            {"path": str(p), "sha256": census.sha256_file(p)}
            for p in (path, rows_path, metadata_path, census_path)
        )
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            if int(row["world_seed"]) != original["seeds"][int(row["env"])]:
                raise ValueError(f"T1/M7 world identity mismatch: {arm}")
            if int(row["length"]) < 800:
                continue
            fatal = int(row["fatal_t"])
            window = (
                (data["env"] == int(row["env"]))
                & (data["frame"] >= fatal - 149)
                & (data["frame"] <= fatal)
                & (data["length"] >= 600)
                & data["adv"].any(axis=1)
            )
            if not window.any():
                raise ValueError(f"late death has no eligible trigger decisions: {arm}, {fatal}")
            deaths.append(
                {
                    "arm": arm,
                    "env": int(row["env"]),
                    "fatal_t": fatal,
                    "length": int(row["length"]),
                    "max_risk_last150": float(data["risk"][window].max()),
                }
            )
    required = math.ceil(0.90 * len(deaths))
    candidates = sorted({row["max_risk_last150"] for row in deaths}, reverse=True)
    selected, coverage, arm_stats = None, [], {}
    for cutoff in candidates:
        threshold = float(np.nextafter(cutoff, -np.inf))
        accepted_by_arm = {}
        arm_stats = {}
        for arm, data in captures.items():
            late = data["length"] >= 600
            eligible = late & data["adv"].any(axis=1) & (data["risk"] > threshold)
            accepted = np.zeros(len(eligible), bool)
            for env in np.unique(data["env"]):
                indices = np.flatnonzero(data["env"] == env)
                frames = data["frame"][indices]
                if np.any(np.diff(frames) <= 0):
                    raise ValueError("M7 decisions must be chronological within each world")
                accepted[indices] = capped_triggers(frames, eligible[indices])
            accepted_by_arm[arm] = accepted
            arm_stats[arm] = {
                "decisions": len(eligible),
                "length_eligible_decisions": int(late.sum()),
                "raw_trigger_rows": int(eligible.sum()),
                "rollout_batches_after_cap": int(accepted.sum()),
                "raw_trigger_rate": float(eligible.mean()),
                "capped_trigger_rate": float(accepted.mean()),
                "capped_rate_among_length_eligible": float(accepted.sum() / max(1, late.sum())),
            }
        coverage = []
        for death in deaths:
            data = captures[death["arm"]]
            window = (
                (data["env"] == death["env"])
                & (data["frame"] >= death["fatal_t"] - 149)
                & (data["frame"] <= death["fatal_t"])
            )
            covered = bool((window & accepted_by_arm[death["arm"]]).any())
            coverage.append({**death, "covered_after_cap": covered})
        if sum(row["covered_after_cap"] for row in coverage) >= required:
            selected = threshold
            break
    if selected is None:
        raise ValueError("no threshold reaches the required 90 percent late-death coverage")
    total = sum(row["decisions"] for row in arm_stats.values())
    raw = sum(row["raw_trigger_rows"] for row in arm_stats.values())
    capped = sum(row["rollout_batches_after_cap"] for row in arm_stats.values())
    receipt = {
        "status": "PASS",
        "authority": "development diagnostic, not gate evidence",
        "score": "-max of all six raw Q outputs; strict greater-than threshold",
        "selection": "largest observed per-death maximum cutoff passing coverage after cooldown",
        "threshold": selected,
        "min_length": 600,
        "interval": 5,
        "last_decisions": 150,
        "late_deaths": len(deaths),
        "covered_late_deaths": sum(row["covered_after_cap"] for row in coverage),
        "late_death_coverage": sum(row["covered_after_cap"] for row in coverage) / len(deaths),
        "raw_trigger_rate": raw / total,
        "capped_trigger_rate": capped / total,
        "arm_stats": arm_stats,
        "death_coverage": coverage,
        "inputs": inputs,
    }
    write_json(out / "trigger.json", receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "threshold",
                    "covered_late_deaths",
                    "late_deaths",
                    "raw_trigger_rate",
                    "capped_trigger_rate",
                )
            }
        ),
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    calibrate(parser.parse_args().out)


if __name__ == "__main__":
    main()
