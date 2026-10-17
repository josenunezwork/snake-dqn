"""Restore immutable T5 boundaries and extract T8 arrival-race features.

This command follows T5's restoration path and validates every requested
observation before saving a compact, state-keyed feature table.  It does not
train, roll out a candidate policy, or change a label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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

from research.t5_separability_20261016 import restore  # noqa: E402
from src.simd_env.t8_features import FEATURE_NAMES, compute_slot_features  # noqa: E402

DEFAULT_BANK = restore.DEFAULT_BANK
T5_DATA = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/features.npz"
)
EXPECTED_T5_DATA_SHA = "e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a"
EXPECTED_MANIFESTS = {
    "train": "708e84d9463ebb890edeb3aa2ed99111cfbd259c939a3c2c211d6bf41c236a3d",
    "holdout": "c0ae22541d20927d1a7996f1a5a91c2aceb9591de8eb743c514def86e2c95e6c",
}


def sha256(path: Path) -> str:
    """Return the file's streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_world(
    job: dict[str, Any], states: dict[int, dict[str, Any]]
) -> Iterable[tuple[int, np.ndarray, float]]:
    """Yield arrival features after T5-equivalent observation byte validation."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    world = build_world(str(job["arm"]), job, hero_checkpoint=Path(job["student"]))
    targets = set(states)
    # _capture_arrival owns the observation only while it wraps prepare.  Repeat
    # the small T5 observation comparison here by retaining it on the world.
    original = world.runtime.prepare
    captured: dict[str, Any] = {}
    current_decision = -1

    def prepare(sim: Any) -> None:
        original(sim)
        if current_decision in targets:
            from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

            captured["obs"] = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            started = time.perf_counter()
            captured["features"] = compute_slot_features(sim, 0, 0).copy()
            captured["seconds"] = time.perf_counter() - started

    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

    world.runtime.prepare = prepare
    try:
        for decision in range(max(targets) + 1):
            current_decision = decision
            captured.clear()
            choose_and_step(world, capture_obs=False)
            if decision not in targets:
                continue
            expected = states[decision]
            obs, features = captured["obs"], captured["features"]
            packed = pack_ref(obs["local"], obs["global"], train_layout())[0]
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
            yield decision, features, float(captured["seconds"])
    finally:
        world.runtime.prepare = original


def extract_bank(
    bank_root: Path,
    out: Path,
    *,
    limit_worlds: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Write state IDs and 18 read-only T8 values for the complete T5 bank."""
    restore.configure_cpu()
    bank_root, out = Path(bank_root), Path(out)
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    receipt_path = out.with_suffix(".json")
    if out.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite arrival extraction output: {out}")
    if sha256(T5_DATA) != EXPECTED_T5_DATA_SHA:
        raise ValueError("canonical T5 feature table SHA differs from the T5b-pinned dataset")
    with np.load(T5_DATA, allow_pickle=False) as archive:
        canonical_ids = archive["state_id"].copy()
        canonical_worlds = archive["world"].copy()
    if len(canonical_ids) != 19976 or len(set(canonical_ids.tolist())) != len(canonical_ids):
        raise ValueError("canonical T5 state identities are incomplete or non-unique")
    source_paths = (Path(__file__), Path("src/simd_env/t8_features.py"))
    input_paths = list(source_paths)
    input_paths += [bank_root / "banks" / split / "manifest.json" for split in ("train", "holdout")]
    input_paths += sorted((bank_root / "plans").glob("*.json"))
    before = {str(path): sha256(path) for path in input_paths}
    for split in ("train", "holdout"):
        manifest = bank_root / "banks" / split / "manifest.json"
        if before[str(manifest)] != EXPECTED_MANIFESTS[split]:
            raise ValueError(f"{split} manifest differs from the T5b-pinned receipt")
    plans = restore._load_plans(bank_root)
    checkpoint_paths = sorted({Path(job["student"]) for job in plans.values()}, key=str)
    checkpoint_before = {str(path): sha256(path) for path in checkpoint_paths}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split in ("train", "holdout"):
        worlds = restore._load_split(bank_root, split)
        selected = [
            world_id for world_id in sorted(worlds) if world_id % shard_count == shard_index
        ]
        if limit_worlds is not None:
            selected = selected[:limit_worlds]
        for world_id in selected:
            world_started = time.perf_counter()
            job = plans.get(world_id)
            if job is None:
                raise ValueError(f"published world {world_id} has no immutable plan")
            states = restore.state_labels(worlds[world_id])
            count = 0
            for decision, features, elapsed in replay_world(job, states):
                if features.shape != (18,) or features.dtype != np.float32:
                    raise ValueError("T8 feature API must return float32 action-major (18,)")
                rows.append(
                    {
                        "state_id": f"world-{world_id:05d}:d{decision:05d}",
                        "world": f"world-{world_id:05d}",
                        "split": split,
                        "features": features,
                        "feature_seconds": elapsed,
                    }
                )
                count += 1
            if count != len(states):
                raise AssertionError(f"replay missed states in world {world_id}")
            print(
                json.dumps(
                    {
                        "completed_world": world_id,
                        "split": split,
                        "states": count,
                        "extracted_states": len(rows),
                        "wall_seconds": round(time.perf_counter() - world_started, 3),
                    }
                ),
                flush=True,
            )
    if not rows:
        raise ValueError("no restored T5 states selected")
    ids = np.asarray([row["state_id"] for row in rows])
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("duplicate restored T5 state identity")
    if limit_worlds is None:
        source_index = {state_id: index for index, state_id in enumerate(ids)}
        wanted = np.asarray(
            [
                state_id
                for state_id in canonical_ids
                if int(str(state_id)[6:11]) % shard_count == shard_index
            ]
        )
        if set(ids.tolist()) != set(wanted.tolist()):
            raise ValueError(
                "arrival restoration does not cover exactly its canonical T5 state partition"
            )
        order = np.asarray([source_index[state_id] for state_id in wanted], dtype=np.int64)
        ids = ids[order]
        rows = [rows[index] for index in order]
        wanted_worlds = np.asarray(
            [
                world
                for state_id, world in zip(canonical_ids, canonical_worlds)
                if int(str(state_id)[6:11]) % shard_count == shard_index
            ]
        )
        if not np.array_equal(np.asarray([row["world"] for row in rows]), wanted_worlds):
            raise ValueError("arrival restoration world identities differ from canonical T5 rows")
    after = {str(path): sha256(path) for path in input_paths}
    if after != before:
        changed = [path for path in before if before[path] != after[path]]
        raise RuntimeError(f"arrival extraction inputs changed during replay: {changed}")
    checkpoint_after = {str(path): sha256(path) for path in checkpoint_paths}
    if checkpoint_after != checkpoint_before:
        changed = [
            path for path in checkpoint_before if checkpoint_before[path] != checkpoint_after[path]
        ]
        raise RuntimeError(f"arrival extraction checkpoints changed during replay: {changed}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        state_id=ids,
        world=np.asarray([row["world"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
        features=np.stack([row["features"] for row in rows]),
    )
    receipt = {
        "schema": "t8-arrival-features/v1",
        "authority": "diagnostic only; not policy training or gate evidence",
        "output": str(out),
        "output_sha256": sha256(out),
        "states": len(rows),
        "shard": {"index": shard_index, "count": shard_count},
        "feature_names": [f"a{action}_{name}" for action in range(6) for name in FEATURE_NAMES],
        "feature_conventions": {
            "voronoi": "hero arrival strictly before the simultaneous opponent-head flood",
            "contested": "opponent arrival <= hero arrival over the hero tail-aware region",
            "margin": "minimum hero arrival to a contested cell; width*height+1 when none exists",
        },
        "canonical_t5_data": str(T5_DATA),
        "canonical_t5_data_sha256": EXPECTED_T5_DATA_SHA,
        "source_and_input_sha256": before,
        "checkpoint_sha256": checkpoint_before,
        "feature_seconds": float(sum(row["feature_seconds"] for row in rows)),
        "wall_seconds": time.perf_counter() - started,
    }
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-worlds", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            extract_bank(
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
