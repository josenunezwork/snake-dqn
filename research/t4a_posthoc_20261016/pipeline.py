"""Readiness gate and aggregation for completed, hash-bound post-hoc captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t3_diagnostics_20261016 import replay as t3  # noqa: E402
from research.t4a_posthoc_20261016 import report, spillover  # noqa: E402
from research.t4a_screen_20261016.watch_t4a import training_status  # noqa: E402

ARMS = ("TWO_SIDED", "LABELS", "LABELS_KEEPV8")
TRAIN = t3.ARTIFACTS / "redesign-t4a-mac-20261016"
OLD = t3.ARTIFACTS / "t3-diagnostics-20261016"
BANK = t3.ARTIFACTS / "t4-prep-20261015/bank-v2"


def _checkpoint(arm: str, seed: int) -> Path:
    root = TRAIN if arm == "TWO_SIDED" else t3.TRAIN
    return root / f"{arm}_s{seed}" / "final.pth"


def readiness(train: Path = TRAIN) -> dict[str, Any]:
    """Accept only all five successful terminal finals, never intermediate weights."""
    statuses = training_status(train)
    names = [f"TWO_SIDED_s{seed}" for seed in t3.SEEDS]
    ready = all(
        statuses.get(name) == 0
        and (train / name / "final.pth").is_file()
        and (train / name / "summary.json").is_file()
        for name in names
    )
    return {"ready": ready, "statuses": statuses, "required": names}


def _bound(path: Path, expected: str) -> None:
    if not path.is_file() or spillover.sha256(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")


def _receipt(path: Path, schema: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != schema:
        raise ValueError(f"unexpected receipt schema: {path}")
    inputs = receipt.get("inputs_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"missing receipt input bindings: {path}")
    for source, digest in inputs.items():
        _bound(Path(source), digest)
    return receipt


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _pool_growth(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for cohort in ("frames", "fixed_h5000_shared_survivor_frames"):
        result[cohort] = {}
        for frame in spillover.GROWTH_FRAMES:
            items = [row[cohort][str(frame)] for row in rows]
            n = sum(item["shared_survivors"] for item in items)
            values = {"shared_survivors": n}
            for metric in (
                "candidate_mean_alive_length",
                "ctrl_mean_alive_length",
                "candidate_minus_ctrl_mean_alive_length",
            ):
                values[metric] = (
                    None
                    if not n
                    else sum(
                        item[metric] * item["shared_survivors"]
                        for item in items
                        if item["shared_survivors"]
                    )
                    / n
                )
            result[cohort][str(frame)] = values
    return result


def summarize(out: Path) -> dict[str, Any]:
    """Verify captures and pool the five seeds without treating worlds as independent."""
    gate = readiness()
    if not gate["ready"]:
        raise ValueError("all five successful TWO_SIDED finals are required")
    evidence = {}
    for seed in t3.SEEDS:
        receipt_path = out / "receipts" / f"CTRL_s{seed}.json"
        receipt = _receipt(receipt_path, "t4a-posthoc-ctrl-capture/v1")
        expected = dict(
            t3._input_identities(
                f"CTRL_s{seed}", json.loads(t3.row_path(f"CTRL_s{seed}").read_text())
            )
        )
        expected.update(
            {str(_checkpoint(arm, seed)): spillover.sha256(_checkpoint(arm, seed)) for arm in ARMS}
        )
        if receipt["inputs_sha256"] != expected:
            raise ValueError("CTRL capture inputs do not match the declared three finals")
        series = out / "series" / f"CTRL_s{seed}.npz"
        if receipt["ctrl_series"] != str(series):
            raise ValueError("CTRL series receipt path mismatch")
        _bound(series, receipt["ctrl_series_sha256"])
        for arm in ARMS:
            trajectory = out / "trajectory" / f"{arm}_s{seed}.npz"
            binding = receipt["trajectory"][arm]
            if binding["path"] != str(trajectory):
                raise ValueError("trajectory receipt path mismatch")
            _bound(trajectory, binding["sha256"])
        evidence[str(receipt_path)] = receipt
    result = {"schema": "t4a-posthoc-complete/v1", "gate": gate, "arms": {}, "receipts": evidence}
    for arm in ARMS:
        banks, trajectories, growth = [], [], []
        seed_results = {}
        for seed in t3.SEEDS:
            name = f"{arm}_s{seed}"
            bank_path = out / "bank" / f"{name}.npz"
            receipt_path = bank_path.with_suffix(".receipt.json")
            bank_receipt = _receipt(receipt_path, "t4a-posthoc-bank-capture/v1")
            bank_inputs = [
                BANK / "manifest.json",
                t3.checkpoint(f"CTRL_s{seed}"),
                _checkpoint(arm, seed),
                *sorted(BANK.glob("*.npz")),
            ]
            if bank_receipt["inputs_sha256"] != {
                str(path): spillover.sha256(path) for path in bank_inputs
            }:
                raise ValueError("bank capture inputs do not match its named candidate/CTRL")
            if bank_receipt["out"] != str(bank_path):
                raise ValueError("bank receipt path mismatch")
            _bound(bank_path, bank_receipt["out_sha256"])
            evidence[str(receipt_path)] = bank_receipt
            bank = _arrays(bank_path)
            trajectory = _arrays(out / "trajectory" / f"{name}.npz")
            banks.append(bank)
            trajectories.append(trajectory)
            series_root = out if arm == "TWO_SIDED" else OLD
            series = series_root / "series" / f"{name}.npz"
            source_receipt_path = series_root / "receipts" / f"{name}.json"
            schema = (
                "t4a-posthoc-candidate-series/v1"
                if arm == "TWO_SIDED"
                else "t3-diagnostics-replay/v1"
            )
            source_receipt = _receipt(source_receipt_path, schema)
            if arm == "TWO_SIDED":
                expected_inputs = {
                    str(path): spillover.sha256(path)
                    for path in (_checkpoint(arm, seed), t3.row_path(f"CTRL_s{seed}"))
                }
            else:
                expected_inputs = t3._input_identities(
                    name, json.loads(t3.row_path(name).read_text())
                )
            if source_receipt["inputs_sha256"] != expected_inputs:
                raise ValueError("growth source inputs do not match named candidate")
            if source_receipt["series"] != str(series):
                raise ValueError("growth series receipt path mismatch")
            _bound(series, source_receipt["series_sha256"])
            evidence[str(source_receipt_path)] = source_receipt
            curve = spillover.growth_summary(series, out / "series" / f"CTRL_s{seed}.npz")
            growth.append(curve)
            seed_results[str(seed)] = {
                "growth": curve,
                "bank_agreement": bank_receipt["candidate_bank_agree"],
            }
            if arm == "TWO_SIDED":
                log = TRAIN / name / "log.jsonl"
                digest = spillover.sha256(log)
                records = [json.loads(line) for line in log.read_text().splitlines() if line]
                last = records[-1]
                if last["transitions"] < 8_000_000:
                    raise ValueError("TWO_SIDED final telemetry has not reached 8M")
                _bound(log, digest)
                seed_results[str(seed)]["final_training_window"] = {
                    key: last[key]
                    for key in (
                        "transitions",
                        "bank_agree_decisive",
                        "bank_agree_counter",
                        "greedy_boost_rate",
                    )
                }
                evidence[str(log)] = {"sha256": digest}
        pooled_bank = {key: np.concatenate([item[key] for item in banks]) for key in banks[0]}
        pooled_trajectory = {
            key: np.concatenate([item[key] for item in trajectories]) for key in trajectories[0]
        }
        stats = spillover.spillover_summary(pooled_bank, pooled_trajectory)
        result["arms"][arm] = {
            "spillover": stats,
            "standardized": report.standardize(stats),
            "growth": _pool_growth(growth),
            "per_seed": seed_results,
            "bank_agreement": {
                key: float(np.mean([row["bank_agreement"][key] for row in seed_results.values()]))
                for key in ("overall", "decisive", "counter")
            },
        }
    points = [item["spillover"]["global_decile_cutpoints"] for item in result["arms"].values()]
    if not all(np.array_equal(points[0], item) for item in points[1:]):
        raise ValueError("candidate captures do not share identical CTRL decile cutpoints")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.ready:
        print(json.dumps(readiness(), sort_keys=True))
        return
    if args.out is None:
        parser.error("--out is required for aggregation")
    result = summarize(args.out)
    spillover._atomic_json(args.out / "posthoc-summary.json", result)
    if args.report:
        prereg = Path("docs/research/specs/t4a_prereg_2026-10-16.md").read_text()
        interpretation = prereg[prereg.index("Interpretation, written before the data:") :]
        args.report.write_text(report.render(result, interpretation))
    print(json.dumps({"summary": str(args.out / "posthoc-summary.json")}))


if __name__ == "__main__":
    main()
