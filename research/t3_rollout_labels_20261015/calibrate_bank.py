"""Calibrate the T3 dense-bank raw-Q prefilter from preserved legacy worlds.

This is deliberately a development admission receipt, never a gate result.  It
first recovers rowwise B8 Q values directly from the saved raw observations,
then replays the same completed worlds without rollout branches to measure the
denominator that the prefilter will actually see.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.replay import configure_cpu
from research.t3_rollout_labels_20261015 import bank_engine
from research.t3_rollout_labels_20261015.trigger import DenseTrigger

SCHEMA = "t3-prefilter/v1"
LEGACY_SCHEMA = "t3-legacy-resume/v1"
DEFAULT_BANK = census.ARTIFACTS / "t3-rollout-labels-20261015" / "bank"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_threshold(scores: np.ndarray, coverage: float = 0.95) -> dict[str, Any]:
    """Return a strict cutoff retaining at least ``coverage`` decisive states.

    The boundary is an order statistic.  Moving one representable float below
    it matters because collection uses ``score > threshold``; equal scores at
    the boundary are retained rather than being lost to a strict comparison.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if not len(values):
        raise ValueError("cannot calibrate an empty decisive-score population")
    if not np.isfinite(values).all():
        raise ValueError("decisive scores must be finite")
    required = int(math.ceil(coverage * len(values)))
    ordered = np.sort(values)
    boundary = float(ordered[len(values) - required])
    threshold = float(np.nextafter(boundary, -np.inf))
    retained = int((values > threshold).sum())
    return {
        "threshold": threshold,
        "boundary_score": boundary,
        "selection": "order-statistic-nextafter-below-boundary/strict-greater-than/v1",
        "required_decisive_rows": required,
        "decisive_rows": int(len(values)),
        "retained_decisive_rows": retained,
        "coverage": retained / len(values),
    }


def _legacy_entries(bank: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    path = bank / "legacy-resume.json"
    raw = json.loads(path.read_text())
    if raw.get("schema") != LEGACY_SCHEMA or not isinstance(raw.get("entries"), dict):
        raise RuntimeError("legacy-resume.json must be a t3-legacy-resume/v1 entry mapping")
    entries = raw["entries"]
    if not entries:
        raise RuntimeError("legacy-resume.json contains no worlds")
    return path, entries


def _checked_world(bank: Path, key: str, entry: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    receipt_path = bank / "worlds" / f"{key}.json"
    rows_path = bank / "worlds" / f"{key}.npz"
    receipt = json.loads(receipt_path.read_text())
    if bank_engine.sha256(receipt_path) != entry.get("receipt_sha256"):
        raise RuntimeError(f"legacy receipt digest mismatch: {key}")
    if bank_engine.sha256(rows_path) != entry.get("rows_sha256") or receipt.get(
        "rows_sha256"
    ) != bank_engine.sha256(rows_path):
        raise RuntimeError(f"legacy rows digest mismatch: {key}")
    if receipt.get("run_identity") != entry.get("run_identity"):
        raise RuntimeError(f"legacy run identity mismatch: {key}")
    if receipt.get("key") != key or receipt.get("status") != "PASS":
        raise RuntimeError(f"legacy receipt metadata mismatch: {key}")
    return rows_path, receipt


def _row_keys(source: Mapping[str, np.ndarray], rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Read persisted row keys, accepting the intentional zero-row legacy schema."""
    if "decision" in source and "frame" in source:
        return source["decision"].astype(np.int64), source["frame"].astype(np.int64)
    if rows:
        raise RuntimeError("nonempty legacy raw shard is missing decision/frame")
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


def _death_endpoint(alive: bool, last_decision: int) -> int | None:
    """Return the terminal decision without exposing later saved-row loop variables."""
    return None if alive else int(last_decision)


def _require_ac_for_twelve_workers(workers: int) -> None:
    """Refuse the requested Mac-wide 12-worker replay while running on battery."""
    if workers != 12:
        return
    state = subprocess.run(
        ["pmset", "-g", "batt"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if state.returncode or "AC Power" not in state.stdout:
        raise RuntimeError("12-worker calibration requires pmset -g batt to report AC Power")


def _check_b8_checkpoint(receipt: Mapping[str, Any], key: str) -> Path:
    """Bind both enrichment and replay to the immutable B8 checkpoint bytes."""
    arm = str(receipt["arm"])
    checkpoint = Path(census.b8_path(arm))
    expected = receipt["run_identity"]["b8_final_sha256"].get(arm)
    if not expected or bank_engine.sha256(checkpoint) != expected:
        raise RuntimeError(f"legacy B8 checkpoint digest mismatch: {key}")
    return checkpoint


def enrich_world(bank: Path, key: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Recover rowwise Q/length metadata from one immutable legacy raw shard."""
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    rows_path, receipt = _checked_world(bank, key, entry)
    checkpoint = _check_b8_checkpoint(receipt, key)
    net = load_ego2s_checkpoint(str(checkpoint), "cpu")
    with np.load(rows_path, allow_pickle=False) as source:
        required = {"local", "global", "scalars", "mask", "action", "decisive"}
        if not required.issubset(source.files):
            raise RuntimeError(f"legacy raw shard schema is incomplete: {key}")
        obs = {name: source[name].copy() for name in ("local", "global", "scalars")}
        action = source["action"].astype(np.int64)
        q = (
            forward_q(net, obs, "rowwise").numpy().astype(np.float32, copy=False)
            if len(action)
            else np.zeros((0, 6), dtype=np.float32)
        )
        mask, action = source["mask"].astype(bool), source["action"].astype(np.int64)
        decision, frame = _row_keys(source, len(action))
        length = np.rint(source["scalars"][:, 1] * 1000.0).astype(np.int64)
        decisive = source["decisive"].astype(bool)
    if q.shape != (len(action), 6) or not np.isfinite(q).all():
        raise RuntimeError(f"malformed recovered Q values: {key}")
    legal_q = np.where(mask, q, -np.inf)
    if len(action) and (
        np.any(~mask.any(axis=1)) or not np.array_equal(legal_q.argmax(axis=1), action)
    ):
        raise RuntimeError(f"recovered Q/action mismatch in saved decision rows: {key}")
    target = bank / "calibration" / f"{key}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, q=q, length=length, decision=decision, frame=frame)
    temporary.replace(target)
    return {
        "key": key,
        "sha256": bank_engine.sha256(target),
        "rows": int(len(action)),
        "decisive_rows": int(decisive.sum()),
        "scores": (-q.max(axis=1)).astype(np.float64),
        "decisive": decisive,
        "inputs": {
            "rows": bank_engine.sha256(rows_path),
            "receipt": entry["receipt_sha256"],
            "checkpoint": bank_engine.sha256(checkpoint),
        },
    }


def _replay_one(bank: str, key: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one world without branches and verify stored captured decisions."""
    configure_cpu()
    root = Path(bank)
    rows_path, receipt = _checked_world(root, key, entry)
    _check_b8_checkpoint(receipt, key)
    with np.load(rows_path, allow_pickle=False) as saved:
        saved_action = saved["action"].astype(np.int64)
        saved_decision, saved_frame = _row_keys(saved, len(saved_action))
        saved_length = np.rint(saved["scalars"][:, 1] * 1000.0).astype(np.int64)
    with np.load(root / "calibration" / f"{key}.npz", allow_pickle=False) as calibration:
        saved_q = calibration["q"].copy()
        calibration_decision = calibration["decision"].copy()
        calibration_frame = calibration["frame"].copy()
        calibration_length = calibration["length"].copy()
    if (
        not np.array_equal(saved_decision, calibration_decision)
        or not np.array_equal(saved_frame, calibration_frame)
        or not np.array_equal(saved_length, calibration_length)
    ):
        raise RuntimeError(f"calibration row keys do not bind source world: {key}")
    world = bank_engine.build_world(str(receipt["arm"]), dict(receipt["roster"]))
    trigger = DenseTrigger(min_length=700)
    scores: list[float] = []
    dense_decisions: list[int] = []
    observed: dict[int, tuple[int, int, int, np.ndarray]] = {}
    started = time.perf_counter()
    last_play_decision = -1
    for decision in range(bank_engine.HORIZON):
        last_play_decision = decision
        capture: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepared(sim: Any) -> None:
            original_prepare(sim)
            frame, alive = int(sim.frame[0]), bool(sim.alive[0, 0])
            length = int(sim.length[0, 0]) if alive else 0
            if bank_engine.dense_eligible(trigger, decision, frame, length, alive):
                from src.simd_env import ego_raster

                obs = ego_raster.build_ego_raster(
                    sim,
                    ego_raster.EgoRasterConfig(version="bv8"),
                    backend="numba",
                    rows=np.array([[0, 0]]),
                )
                capture.update(frame=frame, length=length, q=bank_engine.prepared_q(world, obs))

        world.runtime.prepare = prepared
        try:
            actions, advisory, _ = bank_engine.choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        if capture:
            q = np.asarray(capture["q"], dtype=np.float32)
            if bool(advisory.any()):
                scores.append(float(-q.max()))
                dense_decisions.append(decision)
                observed[decision] = (int(actions[0, 0]), capture["frame"], capture["length"], q)
        if not bool(world.sim.alive[0, 0]):
            break
    if len(scores) != int(receipt["batches"]):
        raise RuntimeError(f"replayed dense-batch count differs from legacy receipt: {key}")
    endpoint = _death_endpoint(bool(world.sim.alive[0, 0]), last_play_decision)
    for index, saved_at_decision in enumerate(saved_decision):
        actual = observed.get(int(saved_at_decision))
        if actual is None:
            raise RuntimeError(
                f"saved trigger decision absent from replay: {key}/{saved_at_decision}"
            )
        action, frame, length, q = actual
        if (
            action != int(saved_action[index])
            or frame != int(saved_frame[index])
            or length != int(saved_length[index])
        ):
            raise RuntimeError(f"saved decision/action/length mismatch: {key}/{saved_at_decision}")
        if not np.array_equal(q, saved_q[index]):
            raise RuntimeError(f"saved Q mismatch: {key}/{saved_at_decision}")
    if endpoint != receipt.get("death_decision"):
        raise RuntimeError(f"saved death endpoint mismatch: {key}")
    return {
        "key": key,
        "dense_batches": len(scores),
        "scores": scores,
        "dense_decisions": dense_decisions,
        "checked_saved_rows": len(saved_decision),
        "death_decision": endpoint,
        "wall_seconds": time.perf_counter() - started,
    }


def calibrate(bank: Path = DEFAULT_BANK, workers: int = 12) -> dict[str, Any]:
    """Write ``bank/prefilter.json`` and calibration sidecars for all legacy worlds."""
    if workers < 1 or workers > 12:
        raise ValueError("workers must be in [1, 12]")
    _require_ac_for_twelve_workers(workers)
    bank = Path(bank)
    started = time.perf_counter()
    legacy_path, entries = _legacy_entries(bank)
    enriched = [enrich_world(bank, key, entry) for key, entry in sorted(entries.items())]
    decisive_scores = np.concatenate(
        [item["scores"][item["decisive"]] for item in enriched if item["decisive"].any()]
    )
    selected = select_threshold(decisive_scores)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        replayed = list(
            pool.map(_replay_one, [str(bank)] * len(entries), entries.keys(), entries.values())
        )
    all_scores = np.asarray(
        [score for row in replayed for score in row["scores"]], dtype=np.float64
    )
    if not len(all_scores):
        raise RuntimeError("replay produced no dense batches")
    skipped = int((all_scores <= selected["threshold"]).sum())
    replay_scores_path = bank / "calibration" / "replay-scores.npz"
    replay_keys = np.asarray([row["key"] for row in replayed for _ in row["scores"]], dtype=np.str_)
    replay_decisions = np.asarray(
        [decision for row in replayed for decision in row["dense_decisions"]], dtype=np.int64
    )
    if not (len(replay_keys) == len(replay_decisions) == len(all_scores)):
        raise RuntimeError("replay score audit arrays have inconsistent lengths")
    np.savez_compressed(
        replay_scores_path, score=all_scores, key=replay_keys, decision=replay_decisions
    )
    replay_scores_sha256 = bank_engine.sha256(replay_scores_path)
    by_mix: dict[str, dict[str, int]] = {}
    decisive_by_arm: dict[str, dict[str, int]] = {}
    decisive_by_mix: dict[str, dict[str, int]] = {}
    for item in enriched:
        receipt = json.loads((bank / "worlds" / f"{item['key']}.json").read_text())
        arm, mix = str(receipt["arm"]), str(receipt["mix"])
        decisive_values = item["scores"][item["decisive"]]
        for buckets, name in ((decisive_by_arm, arm), (decisive_by_mix, mix)):
            bucket = buckets.setdefault(name, {"decisive_rows": 0, "retained_decisive_rows": 0})
            bucket["decisive_rows"] += len(decisive_values)
            bucket["retained_decisive_rows"] += int((decisive_values > selected["threshold"]).sum())
    for replay in replayed:
        mix = str(json.loads((bank / "worlds" / f"{replay['key']}.json").read_text())["mix"])
        bucket = by_mix.setdefault(mix, {"worlds": 0, "dense_batches": 0, "skipped": 0})
        values = np.asarray(replay["scores"], dtype=np.float64)
        bucket["worlds"] += 1
        bucket["dense_batches"] += len(values)
        bucket["skipped"] += int((values <= selected["threshold"]).sum())
    for bucket in by_mix.values():
        bucket["skip_fraction_all_dense_batches"] = bucket["skipped"] / max(
            1, bucket["dense_batches"]
        )
    for buckets in (decisive_by_arm, decisive_by_mix):
        for bucket in buckets.values():
            bucket["coverage"] = bucket["retained_decisive_rows"] / max(1, bucket["decisive_rows"])
    output = {
        "schema": SCHEMA,
        "status": "PASS",
        "authority": "development admission calibration, not gate evidence",
        **selected,
        "score": "-max of all six rowwise B8 Q outputs; strict greater-than threshold",
        "legacy_resume_sha256": bank_engine.sha256(legacy_path),
        "enrichment_sha256": {item["key"]: item["sha256"] for item in enriched},
        "replay_scores_sha256": replay_scores_sha256,
        "source_hashes": {item["key"]: item["inputs"] for item in enriched},
        "code_sha256": {
            "calibrate_bank.py": bank_engine.sha256(Path(__file__)),
            "bank_engine.py": bank_engine.sha256(Path(bank_engine.__file__)),
        },
        "replay": {
            "worlds": len(replayed),
            "dense_batches": int(len(all_scores)),
            "skipped_dense_batches": skipped,
            "skip_fraction_all_dense_batches": skipped / len(all_scores),
            "checked_saved_rows": sum(row["checked_saved_rows"] for row in replayed),
            "worker_seconds": sum(row["wall_seconds"] for row in replayed),
            "wall_seconds": time.perf_counter() - started,
            "by_mix": by_mix,
            "decisive_coverage_by_arm": decisive_by_arm,
            "decisive_coverage_by_mix": decisive_by_mix,
            "caveat": (
                "The legacy sample is concentrated in b8_s0 with limited b8_s1 coverage; it is "
                "an admission calibration only and does not estimate per-arm or per-mix "
                "generalization."
            ),
        },
    }
    _atomic_json(bank / "prefilter.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    receipt = calibrate(args.bank, args.workers)
    print(json.dumps({key: receipt[key] for key in ("threshold", "coverage")}, sort_keys=True))


if __name__ == "__main__":
    main()
