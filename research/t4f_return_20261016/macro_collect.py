#!/usr/bin/env python3
"""Build T4f's immutable macro-boost derivative of the T4e return bank.

The input bank is never altered.  Each source world is replayed from its planned
identity and every recorded state is checked byte-for-byte before only boost
branches are replaced.  The first macro frame takes the recorded boost action
(``k`` is the simulator burn cadence).  Later macro frames preserve B8's fresh
masked-greedy direction and force boost only when that variant is legal.  B8
then continues normally to D300.  Non-boost branch bytes are copied unchanged
into the v3 bank.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t4d_return_regression_20261016.collector import (
    DEPTH,
    _configure_device,
    _snapshot,
    branch_summary,
)
from research.t4e_return_20261016.collection import masked_greedy
from research.t4e_return_20261016.publisher import _sha256

WORKER_LIMIT = 12
SCHEMA = "t4f-return-bank/v3"
DEFAULT_SOURCE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016")
DEFAULT_OUT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4f-return-bank-v3-20261016")


def boost_cost_period() -> int:
    """Read the simulator's configured frames-per-burned-segment contract."""
    from src.simd_env.batch_sim import BatchSimConfig

    period = int(BatchSimConfig(num_envs=1, num_snakes=1).boost_length_cost_frames)
    if period < 1:
        raise ValueError("boost_length_cost_frames must be positive")
    return period


def sha256(path: Path) -> str:
    """Hash one immutable input or output file."""
    return _sha256(Path(path))


def load_jobs(plan_root: Path) -> dict[int, dict[str, Any]]:
    """Load and uniquely index original deterministic source-world identities."""
    jobs: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(plan_root).glob("cycle-*.json")):
        payload = json.loads(path.read_text())
        for raw in payload.get("jobs", []):
            job = dict(raw)
            index = int(job["plan_index"])
            if index in jobs:
                raise ValueError(f"duplicate source plan index: {index}")
            jobs[index] = job
    if not jobs:
        raise ValueError("no source plans found")
    return jobs


def verify_source(source: Path) -> dict[str, Any]:
    """Hash-check the complete immutable source before any macro replay starts."""
    root = Path(source)
    binding_path = root / "binding.json"
    if not binding_path.is_file():
        raise FileNotFoundError(f"source binding is missing: {binding_path}")
    binding = json.loads(binding_path.read_text())
    finals = binding.get("finals")
    if not isinstance(finals, Mapping):
        raise ValueError("source binding has no checkpoint identities")
    for arm, value in finals.items():
        path, expected = Path(value["path"]), value["sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"source B8 checkpoint hash mismatch: {arm}")
    plans = sorted((root / "plans").glob("cycle-*.json"))
    if not plans:
        raise ValueError("source has no immutable plans")
    banks: dict[str, Any] = {}
    for name in ("train", "holdout"):
        bank = root / "banks" / name
        manifest_path = bank / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("shards")
        actual = {path.name for path in bank.glob("*.npz")}
        if not isinstance(expected, Mapping) or set(expected) != actual:
            raise ValueError(f"source {name} shard set disagrees with its manifest")
        for shard_name, digest in expected.items():
            if sha256(bank / shard_name) != digest:
                raise RuntimeError(f"source {name} shard hash mismatch: {shard_name}")
        banks[name] = {"manifest_sha256": sha256(manifest_path), "shards": len(expected)}
    return {
        "binding_sha256": sha256(binding_path),
        "plan_sha256": {path.name: sha256(path) for path in plans},
        "banks": banks,
        "checkpoints": {arm: value["sha256"] for arm, value in sorted(finals.items())},
    }


def shard_plan_index(name: str) -> int:
    """Extract a source plan index from a T4e world shard filename."""
    prefix = "world-"
    if not name.startswith(prefix) or len(name) < len(prefix) + 5:
        raise ValueError(f"unrecognized T4e shard name: {name}")
    return int(name[len(prefix) : len(prefix) + 5])


def grouped_states(archive: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Group repeated branch rows into ordered original decision states."""
    required = {"state_packed", "state_scalars", "action", "source_decision", "source_seed"}
    missing = required - set(archive)
    if missing:
        raise ValueError(f"source shard lacks replay identity fields: {sorted(missing)}")
    result: list[dict[str, Any]] = []
    by_decision: dict[int, list[int]] = {}
    for index, decision in enumerate(np.asarray(archive["source_decision"], dtype=np.int64)):
        by_decision.setdefault(int(decision), []).append(index)
    for decision, indexes in sorted(by_decision.items()):
        first = indexes[0]
        packed = np.asarray(archive["state_packed"])[first]
        scalars = np.asarray(archive["state_scalars"])[first]
        if any(
            not np.array_equal(packed, np.asarray(archive["state_packed"])[index])
            or not np.array_equal(scalars, np.asarray(archive["state_scalars"])[index])
            for index in indexes[1:]
        ):
            raise ValueError("source decision has inconsistent state identity across branches")
        result.append(
            {"decision": decision, "indexes": indexes, "packed": packed, "scalars": scalars}
        )
    return result


def _terminal_observation(world: Any) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    """Capture the actual selector-state bootstrap inputs after a live branch."""
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    captured: dict[str, Any] = {}

    class StopAtSelector(Exception):
        pass

    def choose(prepared: Any) -> np.ndarray:
        world.runtime.prepare(prepared)
        captured["obs"] = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        captured["mask"] = prepared.get_resolved_action_mask()[0, 0].astype(bool).copy()
        raise StopAtSelector

    try:
        world.sim.step_with_policy(choose)
    except StopAtSelector:
        pass
    if set(captured) != {"obs", "mask"}:
        raise RuntimeError("failed to capture macro branch terminal selector state")
    return captured["obs"], captured["mask"]


def _force_b8_direction_boost(world: Any) -> int:
    """Take one fresh B8 selector phase with its direction boosted when legal.

    ``step_prepared`` consumes the snapshot that was already prepared when it
    was captured.  A second macro frame must instead enter ``step_with_policy``
    so runtime preparation, B8 selection, and opponent dispatch occur again.
    The selector's current B8 action supplies the relative direction.  Its
    boost variant is forced only when legal in the refreshed resolved mask;
    otherwise its normal direction is retained.

    Returns:
        The action executed for the hero in this macro frame.
    """
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, int] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        b8_direction = int(actions[0, 0]) % 3
        boost_action = b8_direction + 3
        action = boost_action if bool(masks[0, 0, boost_action]) else b8_direction
        actions[0, 0] = action
        selected["action"] = action
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return selected["action"]


def rollout_macro(snapshot: Mapping[str, Any], action: int, period: int) -> dict[str, Any]:
    """Take the branch boost once, then boosted B8 directions through frame k."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

    if int(action) not in set(np.asarray(snapshot["legal"], dtype=np.int64).tolist()) or action < 3:
        raise ValueError("macro rollout requires an advisory-legal boost action")
    branch = snapshot["branch"].clone()
    trace: list[tuple[int, bool]] = []
    for frame in range(min(period, DEPTH)):
        if frame == 0:
            step_prepared(branch, int(action))
        else:
            _force_b8_direction_boost(branch)
        alive = bool(branch.sim.alive[0, 0])
        trace.append((int(branch.sim.length[0, 0]) if alive else 0, alive))
        if not alive:
            break
    while len(trace) < DEPTH and bool(branch.sim.alive[0, 0]):
        choose_and_step(branch, capture_obs=False)
        alive = bool(branch.sim.alive[0, 0])
        trace.append((int(branch.sim.length[0, 0]) if alive else 0, alive))
    summary = branch_summary(int(snapshot["length"]), trace)
    terminal, terminal_mask = (
        _terminal_observation(branch) if not summary["died"] else (None, np.zeros(6, bool))
    )
    return {**summary, "terminal": terminal, "terminal_mask": terminal_mask}


def _snapshot_matches(snapshot: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Require exact source observation identity before deriving a target."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    packed = pack_ref(snapshot["obs"]["local"], snapshot["obs"]["global"], train_layout())[0]
    return bool(
        np.array_equal(packed, state["packed"])
        and np.array_equal(snapshot["obs"]["scalars"][0], state["scalars"])
    )


def _replay_shard(
    source: Path, job: Mapping[str, Any], period: int
) -> tuple[dict[int, tuple[int, dict[int, dict[str, Any]]]], int]:
    """Replay one source world once and derive all boost branches by decision."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world, choose_and_step

    with np.load(source, allow_pickle=False) as archive:
        states = grouped_states({name: archive[name] for name in archive.files})
        source_seed = np.asarray(archive["source_seed"], dtype=np.int64)
        if not np.all(source_seed == int(job["seed"])):
            raise ValueError("source shard seed disagrees with bound source plan")
    targets = {int(state["decision"]): state for state in states}
    world = build_world(str(job["arm"]), dict(job), hero_checkpoint=Path(job["student"]))
    result: dict[int, tuple[int, dict[int, dict[str, Any]]]] = {}
    for decision in range(max(targets) + 1):
        target = targets.get(decision)
        captured: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepare(sim: Any) -> None:
            original_prepare(sim)
            if target is not None:
                value = _snapshot(world, decision)
                if value is not None:
                    captured["snapshot"] = value

        world.runtime.prepare = prepare
        try:
            actions, _mask, _obs = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        if target is not None:
            snapshot = captured.get("snapshot")
            if snapshot is None or not _snapshot_matches(snapshot, target):
                raise RuntimeError(f"source state identity mismatch at decision {decision}")
            b8_action = masked_greedy(snapshot["q"], snapshot["legal"])
            if int(actions[0, 0]) != b8_action:
                raise RuntimeError("replayed B8 action disagrees with recorded Q/mask")
            boost_actions = [int(value) for value in snapshot["legal"] if int(value) >= 3]
            result[decision] = (
                b8_action,
                {action: rollout_macro(snapshot, action, period) for action in boost_actions},
            )
        if not bool(world.sim.alive[0, 0]) and decision < max(targets):
            raise RuntimeError("source world died before every recorded decision")
    return result, len(states)


def _pack_terminal(terminal: Mapping[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    if terminal is None:
        return np.zeros(train_layout().nbytes, np.uint8), np.zeros(60, np.float32)
    return pack_ref(terminal["local"], terminal["global"], train_layout())[0], np.asarray(
        terminal["scalars"][0], np.float32
    )


def derive_shard(
    source: Path, destination: Path, job: Mapping[str, Any], period: int, manifest_json: str
) -> dict[str, Any]:
    """Copy a shard into v3, replacing only its boost target outcomes."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite derived shard: {destination}")
    rolled, state_count = _replay_shard(source, job, period)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "manifest_json"}
    b8 = np.empty(len(arrays["action"]), dtype=np.int64)
    replaced = 0
    for index, action in enumerate(np.asarray(arrays["action"], dtype=np.int64)):
        decision = int(np.asarray(arrays["source_decision"])[index])
        b8_action, macro = rolled[decision]
        b8[index] = b8_action
        if action < 3:
            continue
        outcome = macro.get(int(action))
        if outcome is None:
            raise ValueError("source boost branch is not advisory legal on replay")
        arrays["T"][index] = int(outcome["T"])
        arrays["died"][index] = bool(outcome["died"])
        arrays["S"][index] = float(outcome["S"])
        arrays["L0"][index] = int(outcome["L0"])
        arrays["L_T"][index] = int(outcome["L_T"])
        packed, scalars = _pack_terminal(outcome["terminal"])
        arrays["terminal_packed"][index] = packed
        arrays["terminal_scalars"][index] = scalars
        arrays["terminal_mask"][index] = outcome["terminal_mask"]
        replaced += 1
    arrays["b8_action"] = b8
    arrays["manifest_json"] = np.asarray(manifest_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "shard": destination.name,
        "sha256": sha256(destination),
        "states": state_count,
        "rows": len(b8),
        "boost_rows_replaced": replaced,
    }


def _derive_call(args: tuple[str, str, dict[str, Any], int, str]) -> dict[str, Any]:
    _configure_device("cpu")
    return derive_shard(Path(args[0]), Path(args[1]), args[2], args[3], args[4])


def _manifest_template(source: Path, period: int, source_root: Path) -> dict[str, Any]:
    original = json.loads((source / "manifest.json").read_text())
    fields = list(original["fields"])
    fields.append("b8_action")
    return {
        **{
            key: value
            for key, value in original.items()
            if key not in {"shards", "publication", "fields", "schema_version", "collection_schema"}
        },
        # The on-disk field layout is unchanged from v2; collection_schema
        # carries the distinct macro-policy identity without breaking loaders.
        "schema_version": 2,
        "collection_schema": SCHEMA,
        "fields": fields,
        "macro": {
            "depth": DEPTH,
            "boost_cost_period_frames": period,
            "policy": "first_branch_boost_then_B8_direction_boost_if_legal_k_frames_then_B8",
        },
        "source_bank": {
            "path": str(source),
            "manifest_sha256": sha256(source / "manifest.json"),
            "root": str(source_root),
        },
    }


def _manifest(template: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Attach output shard hashes to the immutable per-shard manifest template."""
    rows = list(records)
    return {
        **template,
        "shards": {record["shard"]: record["sha256"] for record in rows},
        "publication": {
            "rows": sum(int(record["rows"]) for record in rows),
            "states": sum(int(record["states"]) for record in rows),
            "boost_rows_replaced": sum(int(record["boost_rows_replaced"]) for record in rows),
        },
    }


def _existing_record(
    source: Path, destination: Path, embedded: str, progress: Path
) -> dict[str, Any] | None:
    """Validate a completed shard so an interrupted collection can resume safely."""
    if not destination.is_file():
        return None
    receipt_path = progress / f"{destination.name}.json"
    if not receipt_path.is_file():
        raise ValueError(f"existing derived shard lacks durable receipt: {destination}")
    receipt = json.loads(receipt_path.read_text())
    source_digest = sha256(source)
    if receipt.get("source_sha256") != source_digest or receipt.get("sha256") != sha256(
        destination
    ):
        raise ValueError(f"existing derived shard receipt hash mismatch: {destination}")
    with np.load(destination, allow_pickle=False) as archive:
        if str(archive.get("manifest_json", np.asarray("")).item()) != embedded:
            raise ValueError(f"existing derived shard has a different macro binding: {destination}")
        required = {"action", "source_decision", "b8_action", "T", "died", "S", "L0", "L_T"}
        if not required <= set(archive.files):
            raise ValueError(f"existing derived shard is incomplete: {destination}")
        rows = len(archive["action"])
        if len(archive["b8_action"]) != rows or len(archive["source_decision"]) != rows:
            raise ValueError(f"existing derived shard has invalid row alignment: {destination}")
        replaced = int(np.sum(np.asarray(archive["action"], dtype=np.int64) >= 3))
    with np.load(source, allow_pickle=False) as original:
        if len(original["action"]) != rows:
            raise ValueError(
                f"existing derived shard row count disagrees with source: {destination}"
            )
        states = len(grouped_states({name: original[name] for name in original.files}))
    return {
        "shard": destination.name,
        "sha256": sha256(destination),
        "states": states,
        "rows": rows,
        "boost_rows_replaced": replaced,
        "resumed": True,
        "source_sha256": source_digest,
    }


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    """Durably record a completed shard without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != dict(value):
            raise ValueError(f"existing shard receipt disagrees: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError:
        if json.loads(path.read_text()) != dict(value):
            raise ValueError(f"concurrent shard receipt disagrees: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def derive_bank(
    source: Path,
    destination: Path,
    jobs: Mapping[int, Mapping[str, Any]],
    period: int,
    workers: int,
    source_root: Path,
) -> dict[str, Any]:
    """Derive one train or holdout v3 bank with a hash-bound source manifest."""
    shards = sorted(Path(source).glob("*.npz"))
    if not shards:
        raise ValueError(f"source bank has no shards: {source}")
    template = _manifest_template(source, period, source_root)
    embedded = json.dumps(template, sort_keys=True, separators=(",", ":"))
    destination.mkdir(parents=True, exist_ok=True)
    progress = destination / ".progress"
    tasks = []
    records = []
    for shard in shards:
        index = shard_plan_index(shard.name)
        if index not in jobs:
            raise ValueError(f"source shard has no planned world: {shard.name}")
        existing = _existing_record(shard, destination / shard.name, embedded, progress)
        if existing is None:
            tasks.append(
                (str(shard), str(destination / shard.name), dict(jobs[index]), period, embedded)
            )
        else:
            records.append(existing)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_derive_call, task): task for task in tasks}
            for future in as_completed(futures):
                record = future.result()
                source_shard = Path(futures[future][0])
                record["source_sha256"] = sha256(source_shard)
                _write_receipt(progress / f"{record['shard']}.json", record)
                records.append(record)
                print(
                    json.dumps(
                        {"status": "SHARD_COMPLETE", "bank": source.name, **record}, sort_keys=True
                    ),
                    flush=True,
                )
    records.sort(key=lambda row: row["shard"])
    manifest = _manifest(template, records)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                f"existing bank manifest disagrees with derived shards: {manifest_path}"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def target_gap(bank: Path) -> dict[str, float | int | None]:
    """Report mean boost-minus-normal realized target gaps by retained family."""
    from research.t4d_return_regression_20261016.bank import realized_returns

    values: dict[int, list[tuple[float, bool]]] = {0: [], 1: []}
    for shard in sorted(Path(bank).glob("*.npz")):
        with np.load(shard, allow_pickle=False) as archive:
            target = realized_returns(
                archive["T"],
                archive["died"],
                archive["S"],
                archive["L0"],
                archive["L_T"],
                0.998,
                1.0,
            )
            for value, action, family in zip(target, archive["action"], archive["family"]):
                values[int(family)].append((float(value), int(action) >= 3))
    result: dict[str, float | int | None] = {}
    for family, label in ((0, "decisive"), (1, "outside")):
        normal = [value for value, boost in values[family] if not boost]
        boost = [value for value, boost in values[family] if boost]
        result[f"{label}_boost_rows"] = len(boost)
        result[f"{label}_normal_rows"] = len(normal)
        result[f"{label}_boost_minus_normal"] = (
            float(np.mean(boost) - np.mean(normal)) if boost and normal else None
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create both v3 banks, preserving source bytes and source provenance."""
    if not 1 <= args.workers <= WORKER_LIMIT:
        raise ValueError("--workers must be from 1 through 12")
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    started = time.monotonic()
    period = boost_cost_period()
    source_identity = verify_source(source)
    jobs = load_jobs(source / "plans")
    binding = {
        "schema": SCHEMA,
        "source": str(source),
        "source_identity": source_identity,
        "boost_cost_period_frames": period,
        "policy": "first_branch_boost_then_B8_direction_boost_if_legal_k_frames_then_B8",
    }
    if out.exists():
        if not out.is_dir():
            raise FileExistsError("T4f output exists and is not a collection directory")
        existing = out / "macro_binding.json"
        if not existing.is_file() or json.loads(existing.read_text()) != binding:
            raise ValueError("existing T4f output has no matching immutable macro binding")
        if (out / "completion.json").is_file():
            return json.loads((out / "completion.json").read_text())
    else:
        out.mkdir(parents=True)
        (out / "macro_binding.json").write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n"
        )
    train = derive_bank(
        source / "banks" / "train", out / "banks" / "train", jobs, period, args.workers, source
    )
    holdout = derive_bank(
        source / "banks" / "holdout", out / "banks" / "holdout", jobs, period, args.workers, source
    )
    receipt = {
        "schema": SCHEMA,
        "source": str(source),
        "source_identity": source_identity,
        "boost_cost_period_frames": period,
        "workers": args.workers,
        "cpu_threads_per_worker": 1,
        "wall_seconds": time.monotonic() - started,
        "banks": {"train": train["publication"], "holdout": holdout["publication"]},
        "target_statistics": (
            "reported separately by target_stats.py with source B8 online/target bootstrap"
        ),
    }
    (out / "completion.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--out", type=Path, default=DEFAULT_OUT)
    value.add_argument("--workers", type=int, default=WORKER_LIMIT)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
