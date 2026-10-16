"""Reconstruct T3 decisive worlds on one CPU process; never invokes a learner."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.t4_snapshot_restart_20261016.snapshots import (
    SCHEMA,
    capture,
    restore,
    save_snapshot,
    sha256,
    step_recorded,
)


def state_hashes(sim: Any, runtime: Any) -> dict:
    """Hash all dynamic state, including raw storage, RNG and opponent memory."""
    import hashlib

    from research.t1_commitment_20261015.replay import _hash_value

    values = {k: v for k, v in vars(sim).items() if k != "_det"}
    values["runtime"] = {
        k: v
        for k, v in vars(runtime).items()
        if isinstance(v, np.ndarray) or k in {"memory", "fresh_rows", "carried_rows"}
    }
    result = {}
    for k, v in values.items():
        h = hashlib.sha256()
        _hash_value(h, v)
        result[k] = h.hexdigest()
    return result


def assert_state_equal(a: Any, ar: Any, b: Any, br: Any) -> None:
    """Compare all simulator state and carried policy state, excluding lazy helpers."""
    left, right = state_hashes(a, ar), state_hashes(b, br)
    mismatch = [k for k in left.keys() | right.keys() if left.get(k) != right.get(k)]
    if mismatch:
        raise AssertionError(f"snapshot full-state mismatch: {sorted(mismatch)}")


def parity(world: Any, snapshot: dict, frames: int = 300) -> None:
    """Step replay original and disk-restored state with independently recomputed policies."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

    sim, runtime, policies, groups = restore(snapshot)
    assert_state_equal(world.sim, world.runtime, sim, runtime)
    for _ in range(frames):
        left, _, _ = choose_and_step(world, capture_obs=False)

        def choose(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared)
            actions = np.ones((1, sim.S), np.int64)
            masks = sim.get_resolved_action_mask()
            alive = sim.get_alive()
            if alive[0, 0]:
                actions[0, 0] = world.hero.actions(masks[0, :1], sim, np.array([[0, 0]]))[0]
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(
                        masks[live[:, 0], live[:, 1]], sim, live
                    )
            return actions

        right = {}

        def select(prepared: Any) -> np.ndarray:
            right["actions"] = choose(prepared)
            return right["actions"]

        sim.step_with_policy(select)
        runtime.observe_step(sim)
        np.testing.assert_array_equal(left, right["actions"])
        assert_state_equal(world.sim, world.runtime, sim, runtime)


def reconstruct_world(
    bank: Path, receipt_path: Path, out: Path, limit: int, check_parity: bool
) -> tuple[list, dict]:
    from research.t3_rollout_labels_20261015 import bank_engine as be
    from research.t3_rollout_labels_20261015.calibrate_bank import _check_b8_checkpoint
    from research.t4_snapshot_restart_20261016.snapshots import load_snapshot

    receipt = json.loads(receipt_path.read_text())
    key = receipt["key"]
    rows_path = bank / "worlds" / f"{key}.npz"
    if receipt["status"] != "PASS" or sha256(rows_path) != receipt["rows_sha256"]:
        raise ValueError(f"invalid collector receipt: {key}")
    _check_b8_checkpoint(receipt, key)
    for path, digest in receipt["run_identity"]["roster_checkpoint_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"changed opponent checkpoint: {path}")
    with np.load(rows_path, allow_pickle=False) as z:
        rows = {k: z[k].copy() for k in z.files}
    indices = np.flatnonzero(rows["decisive"])[:limit]
    if not len(indices):
        return [], {}
    wanted = {int(rows["decision"][i]): int(i) for i in indices}
    specs = receipt["roster"]["specs"]
    started = time.perf_counter()
    world = be.build_world(receipt["arm"], receipt["roster"])
    initial = capture(world.sim, world.runtime, specs)
    tape, boundaries = [], {}
    for decision in range(max(wanted) + 1):
        if decision in wanted:
            boundaries[decision] = state_hashes(world.sim, world.runtime)
        selected = {}
        original_prepare = world.runtime.prepare

        def prepare(sim: Any) -> None:
            original_prepare(sim)
            if decision in wanted:
                selected["frame"] = int(sim.frame[0])
                selected["length"] = int(sim.length[0, 0])

        world.runtime.prepare = prepare
        try:
            actions, advisory, obs = be.choose_and_step(world, capture_obs=decision in wanted)
        finally:
            del world.runtime.prepare
        tape.append(actions.copy())
        if decision in wanted:
            i = wanted[decision]
            if selected["frame"] != int(rows["frame"][i]) or int(actions[0, 0]) != int(
                rows["action"][i]
            ):
                raise AssertionError(f"recorded decision mismatch: {key}:{decision}")
            np.testing.assert_array_equal(advisory, rows["mask"][i])
            for name in ("local", "global", "scalars"):
                np.testing.assert_array_equal(obs[name][0], rows[name][i])
            if "length" in rows and selected["length"] != int(rows["length"][i]):
                raise AssertionError("recorded length mismatch")
            if "q" in rows:
                np.testing.assert_array_equal(be.prepared_q(world, obs), rows["q"][i])
    tape_path = out / f"{key}.actions.npz"
    np.savez_compressed(tape_path, actions=np.asarray(tape, dtype=np.int8))
    # Reconstruct from seed state + the persisted, independently replayed full action tape.
    sim, runtime, _, _ = restore(initial)
    world.sim, world.runtime = sim, runtime
    for p in world.opponents:
        if hasattr(p, "runtime"):
            p.runtime = runtime
    entries = []
    parity_snapshot = None
    provenance = {
        "key": key,
        "receipt_sha256": sha256(receipt_path),
        "rows_sha256": sha256(rows_path),
        "tape": tape_path.name,
        "tape_sha256": sha256(tape_path),
        "arm": receipt["arm"],
        "roster": receipt["roster"],
        "hero_sha256": receipt["run_identity"]["b8_final_sha256"][receipt["arm"]],
    }
    with np.load(tape_path, allow_pickle=False) as z:
        recorded = z["actions"]
        for decision, actions in enumerate(recorded):
            if decision in wanted:
                if state_hashes(sim, runtime) != boundaries[decision]:
                    raise AssertionError(f"recorded action replay diverged: {key}:{decision}")
                snap = capture(sim, runtime, specs, {**provenance, "decision": decision})
                path = out / f"{key}-d{decision:05d}.npz"
                entries.append(save_snapshot(path, snap))
                if check_parity and parity_snapshot is None:
                    parity_snapshot = load_snapshot(path)
            step_recorded(sim, runtime, actions.astype(np.int64))
    elapsed = time.perf_counter() - started
    # Parity is timed separately and uses the reconstructed original, not two disk clones.
    parity_seconds = 0.0
    if parity_snapshot is not None:
        sim, runtime, _, _ = restore(initial)
        world.sim, world.runtime = sim, runtime
        for p in world.opponents:
            if hasattr(p, "runtime"):
                p.runtime = runtime
        for actions in tape[: int(parity_snapshot["sim"]["frame"][0])]:
            step_recorded(sim, runtime, actions)
        t = time.perf_counter()
        parity(world, parity_snapshot)
        parity_seconds = time.perf_counter() - t
    return entries, dict(
        key=key,
        mix=receipt["mix"],
        snapshots=len(entries),
        reconstruction_seconds=elapsed,
        parity_frames=300 if check_parity else 0,
        parity_seconds=parity_seconds,
        checkpoints=receipt["run_identity"]["roster_checkpoint_sha256"],
    )


def main() -> None:
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every decisive row")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit must be >= 0")
    bank = args.bank / "bank" if (args.bank / "bank" / "worlds").is_dir() else args.bank
    args.out.mkdir(parents=True, exist_ok=False)
    entries, reports, checkpoints, checked_mixes = [], [], {}, set()
    # A bounded measurement samples across arms/mixes instead of selecting a
    # lexicographic arm prefix. Full builds visit the identical ordered roster.
    import hashlib

    paths = sorted(
        (bank / "worlds").glob("*.json"), key=lambda p: hashlib.sha256(p.name.encode()).hexdigest()
    )
    for receipt_path in paths:
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("decisive_rows"):
            continue
        remaining = args.limit - len(entries) if args.limit else 2**31
        new, report = reconstruct_world(
            bank, receipt_path, args.out, remaining, receipt["mix"] not in checked_mixes
        )
        if new:
            entries.extend(new)
            checkpoints.update(report.pop("checkpoints"))
            reports.append(report)
            checked_mixes.add(receipt["mix"])
            print(json.dumps(report), flush=True)
        if args.limit and len(entries) >= args.limit:
            break
    if not entries:
        raise ValueError("no decisive snapshots found")
    seconds = sum(r["reconstruction_seconds"] for r in reports)
    manifest = dict(
        schema=SCHEMA,
        snapshots=entries,
        checkpoints=checkpoints,
        source_bank=str(bank.resolve()),
        worlds=reports,
        reconstruction_seconds=seconds,
        seconds_per_1000=seconds * 1000 / len(entries),
        processes=1,
        device="cpu",
        training=False,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("reconstruction_seconds", "seconds_per_1000")}))


if __name__ == "__main__":
    main()
