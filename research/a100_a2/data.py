"""Immutable input bundle and remote table materialisation for registered A2.

This module deliberately separates the Mac-side byte collection from the
remote-side replay.  ``build_bundle`` only copies and hashes the pre-registered
inputs; it never runs a policy or a fit.  ``materialize_bundle`` runs the
audited replay seam for the permitted A5 exports and expands the already
hash-checked T4e packed observations for T5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.redesign_m3c_20261008.obs_pack import train_layout, unpack
from research.t34_exit_b8.exports import load_world
from research.t34_exit_b8.replay import replay_world
from research.t35_outcome_head.data import assign_folds, world_targets

T5_FEATURES = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/features.npz"
)
T5_FOLDS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/fits/folds.npz"
)
T4E_BANK = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016")
A5_ROOTS = (
    Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/"
        "manifests-final-3ebf3991/a5-l4-s0-output"
    ),
    Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/"
        "manifests-final-3ebf3991/a5-l4-s1-output"
    ),
)
OPPONENT_CHECKPOINTS = (
    "champion_a5_freespace_20260621.pth",
    "best_apex_fs.pth",
    "best_apex_stage1_fs.pth",
    "best_apex_pre_fs.pth",
    "frp3_m3_s12_u60000_20261005.pth",
)
CHECKPOINT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
EXPECTED_T5_SHA = "e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a"
EXPECTED_T4E_MANIFESTS = {
    "train": "708e84d9463ebb890edeb3aa2ed99111cfbd259c939a3c2c211d6bf41c236a3d",
    "holdout": "c0ae22541d20927d1a7996f1a5a91c2aceb9591de8eb743c514def86e2c95e6c",
}
SCHEMA = "a2-normalized-table/v1"


@dataclass(frozen=True)
class BundleInputs:
    """Pre-registered source locations, injectable only for hermetic tests."""

    t5_features: Path = T5_FEATURES
    t5_folds: Path = T5_FOLDS
    t4e_bank: Path = T4E_BANK
    a5_roots: tuple[Path, Path] = A5_ROOTS
    checkpoint_root: Path = CHECKPOINT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, files: list[dict[str, Any]], root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256(destination)
    if digest != sha256(source):
        raise RuntimeError(f"source changed while copying: {source}")
    files.append(
        {
            "path": str(destination.relative_to(root)),
            "sha256": digest,
            "bytes": destination.stat().st_size,
        }
    )


def _t5_ids(features: Path) -> set[tuple[int, int]]:
    with np.load(features, allow_pickle=False) as z:
        state_ids = z["state_id"].copy()
    result: set[tuple[int, int]] = set()
    for state_id in state_ids:
        left, right = str(state_id).split(":d")
        result.add((int(left.split("-")[1]), int(right)))
    return result


def _a5_world_paths(root: Path) -> list[Path]:
    """Return only the T35 allowlisted export path, never an artifact-wide walk."""
    directory = root / "exports" / "worlds"
    paths = sorted(directory.glob("*/worlds/world-*.json"))
    if not paths:
        raise ValueError(f"no allowlisted A5 world exports below {directory}")
    for path in paths:
        for suffix in (".complete", ".sha256"):
            if not path.with_name(path.name + suffix).is_file():
                raise ValueError(f"A5 export lacks required sidecar: {path.name}{suffix}")
    return paths


def _build_bundle_dir(output: Path, inputs: BundleInputs) -> dict[str, Any]:
    """Create a self-contained, hash-bound A2 input directory without replaying.

    ``output`` must not exist.  It contains exactly the T5 table/folds, the
    manifests and T4e shards needed by canonical state IDs, plus every complete
    world JSON and sidecar from the two explicitly allowlisted A5 exports.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output}")
    if sha256(inputs.t5_features) != EXPECTED_T5_SHA and inputs.t5_features == T5_FEATURES:
        raise ValueError("published T5 feature table hash differs from the A2 registration")
    output.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    _copy(inputs.t5_features, output / "t5/features.npz", files, output)
    _copy(inputs.t5_folds, output / "t5/folds.npz", files, output)
    wanted = _t5_ids(inputs.t5_features)
    found: set[tuple[int, int]] = set()
    for split in ("train", "holdout"):
        directory = inputs.t4e_bank / "banks" / split
        manifest_path = directory / "manifest.json"
        manifest_digest = sha256(manifest_path)
        if inputs.t4e_bank == T4E_BANK and manifest_digest != EXPECTED_T4E_MANIFESTS[split]:
            raise ValueError(f"T4e {split} manifest hash differs from registered receipt")
        manifest = json.loads(manifest_path.read_text())
        _copy(manifest_path, output / "t4e/banks" / split / "manifest.json", files, output)
        for name, expected in manifest["shards"].items():
            source = directory / name
            if sha256(source) != expected:
                raise ValueError(f"T4e shard hash mismatch: {name}")
            with np.load(source, allow_pickle=False) as z:
                decisions = z["source_decision"].astype(np.int64, copy=False)
            world = int(name.split("-")[1])
            contained = {(world, int(d)) for d in decisions}
            if wanted & contained:
                _copy(source, output / "t4e/banks" / split / name, files, output)
                found |= wanted & contained
    if found != wanted:
        raise ValueError(f"T5 canonical IDs missing from copied T4e shards: {len(wanted - found)}")
    worlds = 0
    for index, root in enumerate(inputs.a5_roots):
        for source in _a5_world_paths(root):
            rel = source.relative_to(root / "exports" / "worlds")
            base = output / "a5" / f"s{index}" / rel
            _copy(source, base, files, output)
            _copy(
                source.with_name(source.name + ".complete"),
                base.with_name(base.name + ".complete"),
                files,
                output,
            )
            _copy(
                source.with_name(source.name + ".sha256"),
                base.with_name(base.name + ".sha256"),
                files,
                output,
            )
            worlds += 1
    if inputs.a5_roots == A5_ROOTS and worlds != 39:
        raise ValueError(f"expected exactly 39 permitted A5 worlds, found {worlds}")
    # ``trajectory.build_world`` resolves the frozen roster through
    # SNAKE_CKPT_DIR.  B8 is deliberately excluded: the caller binds it as the
    # separate registered checkpoint input and passes it to materialize_bundle.
    for name in OPPONENT_CHECKPOINTS:
        source = inputs.checkpoint_root / name
        if not source.is_file():
            raise ValueError(f"missing frozen replay opponent checkpoint: {source}")
        _copy(source, output / "checkpoints" / name, files, output)
    receipt = {
        "schema": "a2-input-bundle/v1",
        "t5_sha256": sha256(inputs.t5_features),
        "a5_worlds": worlds,
        "files": sorted(files, key=lambda x: x["path"]),
    }
    (output / "bundle.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt | {"bundle": str(output), "bytes": sum(item["bytes"] for item in files)}


def build_bundle(output: Path, inputs: BundleInputs = BundleInputs()) -> dict[str, Any]:
    """Build a directory bundle or a portable ``.tar.gz`` bundle archive.

    The archive contains exactly one directory with ``bundle.json`` at its
    root.  Its entries are the already hash-bound files, so the supervisor can
    bind the final archive digest and the remote revalidates every member.
    """
    output = Path(output)
    if not output.name.endswith(".tar.gz"):
        return _build_bundle_dir(output, inputs)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bundle archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a2-build-") as temporary:
        root = Path(temporary) / output.name[: -len(".tar.gz")]
        receipt = _build_bundle_dir(root, inputs)
        with tarfile.open(output, "x:gz", format=tarfile.PAX_FORMAT) as stream:
            stream.add(root, arcname=root.name, recursive=True)
    return receipt | {
        "bundle": str(output),
        "archive_sha256": sha256(output),
        "archive_bytes": output.stat().st_size,
    }


def _verify_bundle(bundle: Path) -> dict[str, Any]:
    receipt = json.loads((bundle / "bundle.json").read_text())
    if receipt.get("schema") != "a2-input-bundle/v1":
        raise ValueError("unrecognized A2 bundle schema")
    for item in receipt["files"]:
        path = bundle / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError(f"bundle input hash mismatch: {item['path']}")
    return receipt


def _t5_raw(bundle: Path) -> dict[str, np.ndarray]:
    with np.load(bundle / "t5/features.npz", allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    if data["scalars"].shape != (19976, 60) or int(data["legal"].sum()) != 103264:
        raise ValueError("unexpected registered T5 dimensions")
    with np.load(bundle / "t5/folds.npz", allow_pickle=False) as z:
        if not np.array_equal(z["world"], data["world"]) or not np.array_equal(
            z["state_id"], data["state_id"]
        ):
            raise ValueError("saved T5 folds do not align with state identities")
        fold = z["fold"].astype(np.int16, copy=True)
    lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "holdout"):
        directory = bundle / "t4e/banks" / split
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected in manifest["shards"].items():
            path = directory / name
            if not path.is_file():
                continue
            if sha256(path) != expected:
                raise ValueError(f"copied T4e shard hash mismatch: {name}")
            with np.load(path, allow_pickle=False) as z:
                world = int(name.split("-")[1])
                # NPZ indexing decompresses the whole column: load once per shard.
                decisions = z["source_decision"]
                shard_packed, shard_scalars = z["state_packed"], z["state_scalars"]
                for i, decision in enumerate(decisions):
                    key = (world, int(decision))
                    value = (shard_packed[i].copy(), shard_scalars[i].copy())
                    old = lookup.get(key)
                    if old is not None and (
                        not np.array_equal(old[0], value[0]) or not np.array_equal(old[1], value[1])
                    ):
                        raise ValueError(f"conflicting T4e observation for {key}")
                    lookup[key] = value
    packed: list[np.ndarray] = []
    for state_id, scalars in zip(data["state_id"], data["scalars"]):
        name, decision = str(state_id).split(":d")
        value = lookup.get((int(name.split("-")[1]), int(decision)))
        if (
            value is None
            or value[1].dtype != scalars.dtype
            or value[1].tobytes() != scalars.tobytes()
        ):
            raise ValueError(f"canonical T5 join failed for {state_id}")
        packed.append(value[0])
    local, global_ = unpack(np.stack(packed), np.arange(len(packed)), train_layout())
    return {
        "local": local,
        "global": global_,
        "scalars": data["scalars"].astype(np.float32),
        "world": data["world"].astype("U64"),
        "fold": fold,
        "legal": data["legal"].astype(bool),
        # T4e's T is the transition count to a death event.  The registered
        # hazard convention is event interval ``alive_steps + 1``; therefore
        # a T=30 death has alive_steps=29, while a T=300 survivor remains
        # right-censored at 300.
        "died": data["died"].astype(bool),
        "alive_steps": _t5_alive_steps(data["died"], data["horizon"], data["legal"]),
        "observed": data["legal"].astype(bool),
        "decisive": data["boost_only_escape"].astype(bool),
        "a_star": np.full(len(fold), -1, dtype=np.int16),
        "q": np.full((len(fold), 6), np.nan, dtype=np.float32),
    }


def _t5_alive_steps(died: np.ndarray, horizon: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Translate T4e transition-count labels to the registered event intervals."""
    died = np.asarray(died, dtype=bool)
    horizon = np.asarray(horizon)
    legal = np.asarray(legal, dtype=bool)
    if (
        died.shape != horizon.shape
        or legal.shape != horizon.shape
        or np.any(horizon[legal] < 1)
        or np.any(horizon[legal] > 300)
    ):
        raise ValueError("invalid T4e D300 event/censor label")
    return np.where(died, horizon - 1, horizon).astype(np.int16)


def _bundle_a5_paths(bundle: Path) -> Iterable[Path]:
    for lane in ("s0", "s1"):
        yield from sorted((bundle / "a5" / lane).glob("*/worlds/world-*.json"))


def _a5_raw(bundle: Path, checkpoint: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    rows: list[dict[str, np.ndarray]] = []
    anchors = searched = decisive = 0
    all_world_ids: list[str] = []
    has_searched: list[bool] = []
    for path in _bundle_a5_paths(bundle):
        for suffix in (".complete", ".sha256"):
            if not path.with_name(path.name + suffix).is_file():
                raise ValueError(f"A5 materialization input lacks sidecar: {path}")
        world = load_world(path)
        target = world_targets(world)
        replay = replay_world(world, str(checkpoint))
        if not replay.accepted:
            raise ValueError(f"A5 replay rejected {target.world_id}: {replay.mismatch}")
        obs = replay.observations
        if len(obs) != target.n or not np.array_equal(
            np.asarray([x["decision"] for x in obs]), target.decisions
        ):
            raise ValueError(f"A5 replay/label decision mismatch: {target.world_id}")
        all_world_ids.append(target.world_id)
        anchors += target.n
        keep = np.flatnonzero(target.searched)
        has_searched.append(bool(len(keep)))
        searched += len(keep)
        decisive += int(target.decisive[keep].sum())
        if not len(keep):
            continue
        rows.append(
            {
                "local": np.stack([obs[i]["local"] for i in keep]).astype(np.uint8),
                "global": np.stack([obs[i]["global"] for i in keep]).astype(np.uint8),
                "scalars": np.stack([obs[i]["scalars"] for i in keep]).astype(np.float32),
                "world": np.full(len(keep), target.world_id, dtype="U128"),
                "legal": target.legal[keep].astype(bool),
                "died": (~target.t_survive[keep].astype(bool)),
                "alive_steps": target.t_alive_steps[keep].astype(np.int16),
                "observed": target.t_tested[keep].astype(bool),
                "decisive": target.decisive[keep].astype(bool),
                "a_star": target.a_star[keep].astype(np.int16),
                "q": target.q_export[keep].astype(np.float32),
            }
        )
    if len(all_world_ids) != 39 or anchors != 190779 or searched != 296 or decisive != 225:
        raise ValueError("A5 registered count check failed; do not shrink the table")
    if len(set(all_world_ids)) != len(all_world_ids):
        raise ValueError("duplicate A5 world identity")
    folds_by_world = assign_folds(all_world_ids, has_searched, 5)
    data = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    data["fold"] = np.asarray([folds_by_world[str(w)] for w in data["world"]], dtype=np.int16)
    return data, {
        "anchors": anchors,
        "searched": searched,
        "decisive": decisive,
        "worlds": len(all_world_ids),
    }


def _save_table(path: Path, data: dict[str, np.ndarray], table: str) -> Path:
    required = {
        "local",
        "global",
        "scalars",
        "world",
        "fold",
        "legal",
        "died",
        "alive_steps",
        "observed",
        "decisive",
        "a_star",
        "q",
    }
    if set(data) != required:
        raise ValueError(f"{table}: normalized field mismatch")
    n = len(data["world"])
    shapes = {
        "local": (n, 7, 31, 31),
        "global": (n, 4, 37, 37),
        "scalars": (n, 60),
        "legal": (n, 6),
        "died": (n, 6),
        "alive_steps": (n, 6),
        "observed": (n, 6),
        "q": (n, 6),
    }
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{table}: {key} expected {shape}, got {data[key].shape}")
    if data["fold"].shape != (n,) or data["decisive"].shape != (n,) or data["a_star"].shape != (n,):
        raise ValueError(f"{table}: row metadata shape mismatch")
    np.savez_compressed(path, schema=np.asarray(SCHEMA), table=np.asarray(table), **data)
    return path


def _safe_extract(archive: Path, destination: Path) -> Path:
    """Extract a bundle archive without admitting path traversal entries."""
    with tarfile.open(archive, "r:*") as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination.resolve()) + os.sep):
                raise ValueError(f"unsafe bundle archive member: {member.name}")
        stream.extractall(destination, filter="data")
    candidates = [p for p in destination.iterdir() if p.is_dir() and (p / "bundle.json").is_file()]
    if (destination / "bundle.json").is_file():
        return destination
    if len(candidates) != 1:
        raise ValueError("archive must contain one A2 bundle directory")
    return candidates[0]


def materialize_bundle(bundle: Path, output: Path, checkpoint: Path) -> dict[str, Path]:
    """Verify bundle bytes and create remote normalized T5/A5 NPZ tables."""
    bundle, output, checkpoint = Path(bundle), Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite materialization: {output}")
    if bundle.is_file():
        with tempfile.TemporaryDirectory(prefix="a2-bundle-") as temporary:
            return materialize_bundle(_safe_extract(bundle, Path(temporary)), output, checkpoint)
    receipt = _verify_bundle(bundle)
    opponent_root = bundle / "checkpoints"
    if any(not (opponent_root / name).is_file() for name in OPPONENT_CHECKPOINTS):
        raise ValueError("bundle lacks a frozen opponent checkpoint needed by the audited replay")
    # Must happen before the lazy strict-context import in replay_world.
    os.environ["SNAKE_CKPT_DIR"] = str(opponent_root.resolve())
    output.mkdir(parents=True)
    t5 = _save_table(output / "t5.npz", _t5_raw(bundle), "t5")
    a5_data, counts = _a5_raw(bundle, checkpoint)
    a5 = _save_table(output / "a5.npz", a5_data, "a5")
    metadata = {
        "schema": "a2-materialization/v1",
        "bundle_sha256": sha256(bundle / "bundle.json"),
        "a5": counts,
        "t5_states": 19976,
        "t5_legal_branches": 103264,
        "inputs": receipt["files"],
    }
    (output / "materialization.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return {"t5": t5, "a5": a5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path, metavar="OUTPUT")
    group.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(build_bundle(args.prepare), indent=2, sort_keys=True))
        return
    if args.output is None or args.checkpoint is None:
        parser.error("--bundle requires --output and --checkpoint")
    print(
        json.dumps(
            {
                k: str(v)
                for k, v in materialize_bundle(args.bundle, args.output, args.checkpoint).items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
