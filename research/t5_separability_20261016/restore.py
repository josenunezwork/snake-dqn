#!/usr/bin/env python3
"""Restore T4e bank states and write the T5 feature/label table.

Each partition is a CPU-only, single-thread reconstruction process.  It replays
each source world once, validates every captured bank observation byte-for-byte,
and writes one row per unique ``(world_id, source_decision)`` state.  Branch
labels remain separate columns: a surviving D300 branch is a truncation, never a
negative death observation inferred from a missing terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

DEFAULT_BANK = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016")
SCHEMA = "t5-restored-return-bank/v1"
WORLD_RE = re.compile(r"world-(\d+)-")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_cpu() -> None:
    """Pin all relevant numerical libraries to CPU and one thread."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from src.core.device_manager import DeviceManager

    DeviceManager.override_device(torch.device("cpu"))


def _load_plans(root: Path) -> dict[int, dict[str, Any]]:
    plans: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "plans").glob("*.json")):
        for job in json.loads(path.read_text())["jobs"]:
            index = int(job["plan_index"])
            if index in plans:
                raise ValueError(f"duplicate T4e plan index: {index}")
            plans[index] = dict(job)
    if not plans:
        raise ValueError("T4e plans are unavailable")
    return plans


def _world_id(path: Path) -> int:
    match = WORLD_RE.match(path.name)
    if match is None:
        raise ValueError(f"invalid T4e world shard name: {path.name}")
    return int(match.group(1))


def _load_split(root: Path, split: str) -> dict[int, dict[str, Any]]:
    """Load branch rows grouped by persisted source world without copying all banks."""
    directory = root / "banks" / split
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = sorted(directory.glob("*.npz"))
    if set(manifest["shards"]) != {path.name for path in paths}:
        raise ValueError(f"{split} shard set differs from immutable manifest")
    worlds: dict[int, dict[str, Any]] = {}
    for path in paths:
        if sha256(path) != manifest["shards"][path.name]:
            raise ValueError(f"{split} shard hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            rows = {key: archive[key].copy() for key in archive.files if key != "manifest_json"}
        world_id = _world_id(path)
        # One played source world can contribute both decisive and matched-outside
        # states.  They are distinct immutable publication shards but must replay
        # through the same world once.
        if world_id in worlds:
            old = worlds[world_id]
            if set(old) != set(rows):
                raise ValueError(f"incompatible published shards for world: {world_id}")
            worlds[world_id] = {name: np.concatenate((old[name], rows[name])) for name in rows}
        else:
            worlds[world_id] = rows
    return worlds


def state_labels(rows: dict[str, np.ndarray]) -> dict[int, dict[str, Any]]:
    """Collapse legal action branch rows into state labels with explicit truncation."""
    required = {"source_decision", "action", "died", "T", "state_scalars", "state_packed"}
    if not required <= set(rows):
        raise ValueError(f"bank rows lack fields: {sorted(required - set(rows))}")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, decision in enumerate(rows["source_decision"]):
        grouped[int(decision)].append(index)
    result: dict[int, dict[str, Any]] = {}
    for decision, indices in grouped.items():
        first = indices[0]
        packed, scalars = rows["state_packed"][first], rows["state_scalars"][first]
        actions = np.asarray([rows["action"][i] for i in indices], dtype=np.int64)
        died = np.asarray([rows["died"][i] for i in indices], dtype=bool)
        horizon = np.asarray([rows["T"][i] for i in indices], dtype=np.int64)
        if np.any(horizon < 1) or np.any(horizon > 300):
            raise ValueError(f"invalid D300 branch length at decision {decision}")
        if len(set(actions.tolist())) != len(actions):
            raise ValueError(f"duplicate action branch at decision {decision}")
        for index in indices[1:]:
            if not (
                np.array_equal(packed, rows["state_packed"][index])
                and np.array_equal(scalars, rows["state_scalars"][index])
            ):
                raise ValueError(f"non-identical state rows at decision {decision}")
        # A nonfatal D300 endpoint is a right-censored/truncated branch.  It is
        # intentionally not relabelled as a death after the recorded horizon.
        died_by_action = np.zeros(6, dtype=bool)
        truncated_by_action = np.zeros(6, dtype=bool)
        horizon_by_action = np.zeros(6, dtype=np.int16)
        legal = np.zeros(6, dtype=bool)
        for action, branch_died, steps in zip(actions, died, horizon):
            legal[action] = True
            horizon_by_action[action] = steps
            died_by_action[action] = bool(branch_died and steps <= 300)
            truncated_by_action[action] = bool(not branch_died and steps == 300)
        normal = legal[:3]
        boost = legal[3:]
        boost_only = bool(
            boost.any()
            and (~died_by_action[3:][boost]).any()
            and normal.any()
            and died_by_action[:3][normal].all()
        )
        result[decision] = {
            "packed": packed,
            "scalars": scalars,
            "legal": legal,
            "died_by_action": died_by_action,
            "truncated_by_action": truncated_by_action,
            "horizon_by_action": horizon_by_action,
            "boost_only_escape": boost_only,
        }
    return result


def _capture_state(
    world: Any, decision: int, *, wanted: bool
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, float]:
    """Advance one exact evaluator world to its prepared decision boundary."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    captured: dict[str, Any] = {}
    original = world.runtime.prepare

    def prepare(sim: Any) -> None:
        original(sim)
        if wanted:
            captured["obs"] = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            # `choose_and_step` advances this mutable simulator after the
            # selector returns.  The hypothetical-action features must be read
            # at this exact prepared boundary, alongside the observation that
            # is byte-compared to the immutable bank.
            from src.simd_env.t5_features import compute_slot_features

            started = time.perf_counter()
            captured["features"] = compute_slot_features(sim, 0, 0).copy()
            captured["feature_seconds"] = time.perf_counter() - started

    world.runtime.prepare = prepare
    try:
        choose_and_step(world, capture_obs=False)
    finally:
        world.runtime.prepare = original
    if wanted and not captured:
        raise RuntimeError(f"failed to capture prepared state at decision {decision}")
    return (
        captured.get("obs"),
        captured.get("features"),
        float(captured.get("feature_seconds", 0.0)),
    )


def replay_world(
    job: dict[str, Any], states: dict[int, dict[str, Any]]
) -> Iterable[tuple[int, np.ndarray, float]]:
    """Yield validated boundary-time feature vectors in ascending decision order."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    world = build_world(str(job["arm"]), job, hero_checkpoint=Path(job["student"]))
    targets = set(states)
    for decision in range(max(targets) + 1):
        expected = states.get(decision)
        obs, features, feature_seconds = _capture_state(
            world, decision, wanted=expected is not None
        )
        if expected is None:
            continue
        assert features is not None and obs is not None
        packed = pack_ref(obs["local"], obs["global"], train_layout())[0]
        # The claim is byte identity, including signed-zero representation.
        scalars = obs["scalars"][0]
        if not (
            packed.dtype == expected["packed"].dtype
            and packed.tobytes() == expected["packed"].tobytes()
            and scalars.dtype == expected["scalars"].dtype
            and scalars.tobytes() == expected["scalars"].tobytes()
        ):
            raise AssertionError(
                f"T4e observation mismatch world={job['plan_index']} decision={decision}"
            )
        yield decision, features, feature_seconds


def restore_bank(
    bank_root: Path,
    out: Path,
    *,
    limit_worlds: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Restore selected T4e worlds and write a compact, provenance-rich NPZ table."""
    configure_cpu()
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must lie in [0, shard count)")
    bank_root = Path(bank_root)
    feature_source = Path("src/simd_env/t5_features.py")
    restore_source = Path(__file__)
    source_hashes = {
        str(feature_source): sha256(feature_source),
        str(restore_source): sha256(restore_source),
    }
    input_paths = [bank_root / "banks" / split / "manifest.json" for split in ("train", "holdout")]
    input_paths += sorted((bank_root / "plans").glob("*.json"))
    input_hashes = {str(path): sha256(path) for path in input_paths}
    plans = _load_plans(bank_root)
    all_rows: list[dict[str, Any]] = []
    proof: dict[str, list[dict[str, Any]]] = defaultdict(list)
    performance: list[dict[str, Any]] = []
    for split in ("train", "holdout"):
        worlds = _load_split(bank_root, split)
        selected = [
            world_id for world_id in sorted(worlds) if world_id % shard_count == shard_index
        ]
        if limit_worlds is not None:
            selected = selected[:limit_worlds]
        for world_id in selected:
            started = time.perf_counter()
            feature_seconds = 0.0
            job = plans.get(world_id)
            if job is None:
                raise ValueError(f"published world {world_id} has no immutable plan")
            states = state_labels(worlds[world_id])
            restored = 0
            for decision, features, elapsed in replay_world(job, states):
                record = states[decision]
                feature_seconds += elapsed
                if features.shape != (24,) or features.dtype != np.float32:
                    raise ValueError("T5 feature API must return float32 (E,S,24)")
                all_rows.append(
                    {
                        "world_id": world_id,
                        "decision": decision,
                        "split": 0 if split == "train" else 1,
                        "family": int(
                            worlds[world_id]["family"][
                                np.flatnonzero(worlds[world_id]["source_decision"] == decision)[0]
                            ]
                        ),
                        "scalars": record["scalars"],
                        "features": features,
                        "legal": record["legal"],
                        "died": record["died_by_action"],
                        "truncated": record["truncated_by_action"],
                        "horizon": record["horizon_by_action"],
                        "boost_only": record["boost_only_escape"],
                    }
                )
                restored += 1
                family = int(
                    worlds[world_id]["family"][
                        np.flatnonzero(worlds[world_id]["source_decision"] == decision)[0]
                    ]
                )
                bucket = f"{split}/{'decisive' if family == 0 else 'outside'}"
                if len(proof[bucket]) < 3:
                    proof[bucket].append(
                        {
                            "world_id": world_id,
                            "decision": decision,
                            "observation_bytes_match": True,
                        }
                    )
            if restored != len(states):
                raise AssertionError(f"replay missed states in world {world_id}")
            performance.append(
                {
                    "world_id": world_id,
                    "split": split,
                    "states": restored,
                    "wall_seconds": time.perf_counter() - started,
                    "feature_seconds": feature_seconds,
                }
            )
            print(
                json.dumps(
                    {
                        "completed_world": world_id,
                        "split": split,
                        "states": restored,
                        "partition_states": len(all_rows),
                        "wall_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    if not all_rows:
        raise ValueError("no states restored")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    arrays = {
        "world": np.asarray([f"world-{row['world_id']:05d}" for row in all_rows], dtype="U16"),
        "split": np.asarray(
            ["train" if row["split"] == 0 else "holdout" for row in all_rows], dtype="U8"
        ),
        "family": np.asarray([row["family"] for row in all_rows], dtype=np.uint8),
        "state_id": np.asarray(
            [f"world-{row['world_id']:05d}:d{row['decision']:05d}" for row in all_rows], dtype="U32"
        ),
        "scalars": np.stack([row["scalars"] for row in all_rows]).astype(np.float32),
        "features": np.stack([row["features"] for row in all_rows]).astype(np.float32),
        "legal": np.stack([row["legal"] for row in all_rows]),
        "died": np.stack([row["died"] for row in all_rows]),
        "horizon": np.stack([row["horizon"] for row in all_rows]),
        "truncated_at_300": np.stack([row["truncated"] for row in all_rows]),
        "boost_only_escape": np.asarray([row["boost_only"] for row in all_rows], dtype=bool),
    }
    np.savez_compressed(out, **arrays)
    after_hashes = {
        str(feature_source): sha256(feature_source),
        str(restore_source): sha256(restore_source),
    }
    if source_hashes != after_hashes:
        raise RuntimeError("T5 feature or restoration source changed during extraction")
    if input_hashes != {str(path): sha256(path) for path in input_paths}:
        raise RuntimeError("T4e bank manifests or plans changed during extraction")
    summary = {
        "schema": SCHEMA,
        "device": "cpu",
        "processes": 1,
        "bank_root": str(bank_root),
        "output": str(out),
        "rows": len(all_rows),
        "by_split": {
            "train": int((arrays["split"] == "train").sum()),
            "holdout": int((arrays["split"] == "holdout").sum()),
        },
        "shard": {"index": shard_index, "count": shard_count},
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "output_sha256": sha256(out),
        "observation_proof": {
            "all_emitted_states_byte_verified": len(all_rows),
            "samples": dict(proof),
        },
        "performance": performance,
        "labels": {
            "died_within_300": "actual branch death at or before D300",
            "truncated_at_300": "survived all 300 branch transitions; not a death label",
        },
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-worlds", type=int, default=None, help="test-only bounded replay")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            restore_bank(
                args.bank,
                args.out,
                limit_worlds=args.limit_worlds,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
