"""Collect real all-action D-step return-regression branches for T4d.

This module deliberately performs no training.  It replays a supplied student on
fresh planned worlds, then continues every advisory-legal first action with that
same student's greedy policy.  Candidate states are either in the final
300 decisions before an observed death or at least 300 observed decisions
from that death (including survivor worlds).  The latter are the required
outside-bank magnitude targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

# This module only starts one collector process.  Set native pool limits before
# importing NumPy or Torch so a caller cannot accidentally inherit a wide pool.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402

GAMMA = 0.998
KAPPA = 1.0
PHI_DIVISOR = 10.0
DEPTH = 300
MIN_LENGTH = 700
FAMILY_DEATH_WINDOW = np.uint8(0)
FAMILY_OUTSIDE = np.uint8(1)


def phi(length: int | float, alive: bool = True) -> float:
    """Return the bounded potential; terminal/dead potential is exactly zero."""
    return float(length) / PHI_DIVISOR if alive else 0.0


def branch_summary(initial_length: int, steps: Iterable[tuple[int, bool]]) -> dict[str, Any]:
    """Compute T4d's stored sufficient return statistics from branch steps.

    ``steps`` holds ``(next_length, alive_after_transition)`` in temporal
    order.  A dead transition stores no mass potential and terminates a branch.
    The helper is intentionally simulation-independent so its recurrence is
    directly unit-testable against a hand-computed branch.
    """
    values = list(steps)
    if not values:
        raise ValueError("a branch must contain at least one transition")
    total = 0.0
    died = False
    terminal_length = 0
    for index, (next_length, alive) in enumerate(values):
        if died:
            raise ValueError("a branch cannot step after death")
        if alive:
            terminal_length = int(next_length)
            total += GAMMA**index * phi(next_length)
        else:
            died = True
            terminal_length = 0
            if index != len(values) - 1:
                raise ValueError("dead transition must terminate the branch")
    return {
        "T": len(values),
        "died": died,
        "S": float(total),
        "L0": int(initial_length),
        "L_T": int(terminal_length),
    }


def realised_target(
    summary: dict[str, Any], *, gamma: float = GAMMA, kappa: float = KAPPA
) -> float:
    """Return the truncated reward target before the survivor value bootstrap."""
    T = int(summary["T"])
    if T < 1:
        raise ValueError("T must be positive")
    value = kappa * (1.0 - gamma) * float(summary["S"]) - phi(int(summary["L0"]))
    if bool(summary["died"]):
        return value - 3.0 * gamma ** (T - 1)
    return value + gamma**T * phi(int(summary["L_T"]))


def _configure_device(device: str) -> None:
    """Configure the requested collector inference device and one CPU thread."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already fixed by an embedding test runner
        pass
    from src.core.device_manager import DeviceManager

    requested = torch.device(device)
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    DeviceManager.override_device(requested)


def _snapshot(world: Any, decision: int) -> dict[str, Any] | None:
    """Capture one prepared state only when it can support a full D-step branch."""
    from research.t3_rollout_labels_20261015.bank_engine import HORIZON, prepared_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    sim = world.sim
    alive = bool(sim.alive[0, 0])
    length = int(sim.length[0, 0]) if alive else 0
    frame = int(sim.frame[0])
    if not alive or length < MIN_LENGTH or frame > HORIZON - DEPTH + 1:
        return None
    advisory = sim.get_advisory_action_mask()[0, 0].astype(bool)
    legal = np.flatnonzero(advisory).astype(np.int64)
    if not len(legal):
        return None
    obs = build_ego_raster(
        sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    q = prepared_q(world, obs)
    return {
        "decision": int(decision),
        "frame": frame,
        "length": length,
        "obs": obs,
        "legal": legal,
        "q": q,
        "branch": world.clone(),
    }


def _terminal_observation(world: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Materialize terminal survivor inputs at its actual next decision phase."""
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    # ``step_with_policy`` owns the next-decision lifecycle.  Capturing in its
    # selector hook (and aborting before an action) gives precisely the state
    # that the trainer will bootstrap from; calling runtime.prepare directly
    # would miss simulator lifecycle work before that selector.
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
        raise RuntimeError("failed to capture actual terminal selector state")
    return captured["obs"], captured["mask"]


def rollout_branch(snapshot: dict[str, Any], action: int, depth: int = DEPTH) -> dict[str, Any]:
    """Roll one actual forced-first-action branch and retain its exact endpoint."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

    if int(action) not in set(np.asarray(snapshot["legal"], dtype=np.int64).tolist()):
        raise ValueError("cannot roll an action that was not legal at the captured state")
    branch = snapshot["branch"].clone()
    trace: list[tuple[int, bool]] = []
    step_prepared(branch, int(action))
    alive = bool(branch.sim.alive[0, 0])
    trace.append((int(branch.sim.length[0, 0]) if alive else 0, alive))
    while len(trace) < depth and alive:
        choose_and_step(branch, capture_obs=False)
        alive = bool(branch.sim.alive[0, 0])
        trace.append((int(branch.sim.length[0, 0]) if alive else 0, alive))
    summary = branch_summary(int(snapshot["length"]), trace)
    if not summary["died"]:
        terminal, terminal_mask = _terminal_observation(branch)
    else:
        terminal, terminal_mask = None, np.zeros(6, dtype=bool)
    return {**summary, "terminal": terminal, "terminal_mask": terminal_mask}


def _pack(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pack current and survivor terminal observations using the M3C lossless layout."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    layout = train_layout()
    n = len(rows)
    locals_ = np.concatenate([row["obs"]["local"] for row in rows], axis=0)
    globals_ = np.concatenate([row["obs"]["global"] for row in rows], axis=0)
    state_packed = pack_ref(locals_, globals_, layout)
    state_scalars = np.concatenate([row["obs"]["scalars"] for row in rows], axis=0).astype(
        np.float32, copy=False
    )
    terminal_packed = np.zeros((n, layout.nbytes), dtype=np.uint8)
    terminal_scalars = np.zeros_like(state_scalars)
    for i, row in enumerate(rows):
        terminal = row["terminal"]
        if terminal is not None:
            terminal_packed[i] = pack_ref(terminal["local"], terminal["global"], layout)[0]
            terminal_scalars[i] = terminal["scalars"][0]
    return (
        state_packed,
        terminal_packed,
        {
            "packed_shape": [int(layout.nbytes)],
            "scalars_shape": list(state_scalars.shape[1:]),
            "packing_version": "ego2s-bv8/obs_pack-v1",
            "packing_layout": layout.to_dict(),
            "feature_order": (
                "ego2s-bv8 local,global packed; bv8 scalars as emitted by build_ego_raster"
            ),
        },
    )


def write_bank(
    path: Path, rows: Sequence[dict[str, Any]], provenance: dict[str, Any]
) -> dict[str, Any]:
    """Write the trainer-facing flattened real-branch bank and hash-bound manifest."""
    path = Path(path)
    if path.exists() or (path.parent / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite return bank: {path}")
    if not rows:
        raise ValueError("no qualifying real branches; refusing to manufacture an empty bank")
    state_packed, terminal_packed, packing = _pack(rows)
    state_scalars = np.concatenate([row["obs"]["scalars"] for row in rows], axis=0).astype(
        np.float32
    )
    terminal_scalars = np.zeros_like(state_scalars)
    for i, row in enumerate(rows):
        if row["terminal"] is not None:
            terminal_scalars[i] = row["terminal"]["scalars"][0]
    manifest = {
        "schema_version": 1,
        "gamma": GAMMA,
        "kappa": KAPPA,
        "phi": "length/10,dead=0",
        "reward_semantics": "mass+pbrs+death_no_overlays_or_kill",
        "fields": [
            "state_packed",
            "state_scalars",
            "action",
            "T",
            "died",
            "S",
            "L0",
            "L_T",
            "terminal_packed",
            "terminal_scalars",
            "terminal_mask",
            "row_weight",
            "family",
        ],
        "layout": packing.pop("packing_layout"),
        **packing,
        "provenance": provenance,
    }
    arrays = {
        "state_packed": state_packed,
        "state_scalars": state_scalars,
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "T": np.asarray([row["T"] for row in rows], dtype=np.int32),
        "died": np.asarray([row["died"] for row in rows], dtype=bool),
        "S": np.asarray([row["S"] for row in rows], dtype=np.float64),
        "L0": np.asarray([row["L0"] for row in rows], dtype=np.int32),
        "L_T": np.asarray([row["L_T"] for row in rows], dtype=np.int32),
        "terminal_packed": terminal_packed,
        "terminal_scalars": terminal_scalars,
        "terminal_mask": np.stack([row["terminal_mask"] for row in rows]).astype(bool),
        "row_weight": np.asarray([row.get("row_weight", 1.0) for row in rows], dtype=np.float32),
        "source_seed": np.asarray([row.get("source_seed", -1) for row in rows], dtype=np.int64),
        "source_decision": np.asarray(
            [row.get("source_decision", -1) for row in rows], dtype=np.int64
        ),
        "death_distance": np.asarray(
            [row.get("death_distance", -1) for row in rows], dtype=np.int64
        ),
        "family": np.asarray([row["family"] for row in rows], dtype=np.uint8),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if not np.isfinite(arrays["S"]).all() or not (arrays["row_weight"] > 0).all():
        raise ValueError("return-bank target statistics or weights are invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.parent / "manifest.json"
    sidecar.write_text(
        json.dumps({**manifest, "shards": {path.name: digest}}, indent=2, sort_keys=True) + "\n"
    )
    return {"path": str(path), "sha256": digest, "rows": len(rows), "manifest": str(sidecar)}


class OutsideMatcher:
    """Global fixed T4c strata quotas, with deterministic unique first-eligible sampling.

    Only fingerprints/counts persist across worlds. Once a state has 300 observed
    future decisions it can be rolled immediately: any later death is necessarily
    at least 300 decisions away. No global pool of simulator clones is needed.
    """

    def __init__(self, bank: Path) -> None:
        from research.t4c_20261016.data import bound_files, target_definition

        self.bank = Path(bank)
        self.bindings = bound_files(self.bank)
        self.cuts, self.quotas, self.excluded = target_definition(self.bank)
        self.counts: Counter[str] = Counter()
        self.seen: set[str] = set()

    def admit(self, snapshot: dict[str, Any]) -> bool:
        from research.t4a_posthoc_20261016.spillover import observation_fingerprints
        from research.t4c_20261016.data import strata

        obs = snapshot["obs"]
        fingerprint = str(observation_fingerprints(obs["local"], obs["global"], obs["scalars"])[0])
        key = str(strata(np.asarray([snapshot["length"]]), snapshot["q"][None, :], self.cuts)[0])
        if (
            fingerprint in self.excluded
            or fingerprint in self.seen
            or self.counts[key] >= self.quotas.get(key, 0)
        ):
            return False
        self.seen.add(fingerprint)
        self.counts[key] += 1
        snapshot["stratum"] = key
        return True

    def receipt(self) -> dict[str, Any]:
        from research.t4c_20261016.data import bound_files

        if bound_files(self.bank) != self.bindings:
            raise RuntimeError("matching reference changed during collection")
        return {
            "reference": {"path": str(self.bank), "files": self.bindings},
            "decile_cutpoints": self.cuts.tolist(),
            "target_counts": self.quotas,
            "selected_counts": dict(self.counts),
            "deficits": {
                key: quota - self.counts[key]
                for key, quota in self.quotas.items()
                if self.counts[key] < quota
            },
            "selection": "global first eligible unique state per fixed stratum; no replacement",
        }


def outside_eligible(snapshot_decision: int, observed_decision: int) -> bool:
    """Require 300 observed future decisions, including the fatal-boundary convention."""
    return observed_decision - snapshot_decision >= DEPTH


def collect_world(
    plan: dict[str, Any],
    student: Path,
    depth: int = DEPTH,
    matcher: OutsideMatcher | None = None,
    *,
    device: str = "cpu",
    max_states: int = 0,
    state_budget: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one world with <=61 retained clones and global matching quotas."""
    from research.t3_rollout_labels_20261015.bank_engine import (
        HORIZON,
        build_world,
        choose_and_step,
    )

    world = build_world("B8_s0", plan, hero_checkpoint=student)
    world.hero.net.to(device)
    pending: deque[dict[str, Any]] = deque()
    rows: list[dict[str, Any]] = []
    budget = state_budget if state_budget is not None else [0]
    captured_count = selected = peak_pending = 0
    branch_seconds = 0.0

    def capacity() -> bool:
        return not max_states or budget[0] < max_states

    def roll(snapshot: dict[str, Any], family: int, distance: int | None) -> None:
        nonlocal selected, branch_seconds
        started = time.perf_counter()
        # Match the tightened bank's effective state weights. All legal actions
        # are retained, even when the forced action dies immediately.
        weight = 0.5 if family == FAMILY_OUTSIDE else (0.3 if distance >= 100 else 1.0)
        for action in snapshot["legal"]:
            rows.append(
                {
                    "obs": snapshot["obs"],
                    "action": int(action),
                    "family": family,
                    "row_weight": weight,
                    "source_seed": int(plan["seed"]),
                    "source_decision": snapshot["decision"],
                    "death_distance": -1 if distance is None else distance,
                    **rollout_branch(snapshot, int(action), depth),
                }
            )
        branch_seconds += time.perf_counter() - started
        selected += 1
        budget[0] += 1

    for decision in range(HORIZON):
        chosen: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepared(sim: Any) -> None:
            original_prepare(sim)
            captured = _snapshot(world, decision) if decision % 5 == 0 and capacity() else None
            if captured is not None:
                chosen["snapshot"] = captured

        world.runtime.prepare = prepared
        try:
            choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        if "snapshot" in chosen:
            pending.append(chosen["snapshot"])
            captured_count += 1
            peak_pending = max(peak_pending, len(pending))
        while pending and outside_eligible(pending[0]["decision"], decision):
            snapshot = pending.popleft()
            if matcher is not None and capacity() and matcher.admit(snapshot):
                roll(snapshot, FAMILY_OUTSIDE, None)
        if not bool(world.sim.alive[0, 0]):
            break
    death_decision = None if bool(world.sim.alive[0, 0]) else decision
    if death_decision is not None:
        for snapshot in pending:
            distance = death_decision - snapshot["decision"]
            if capacity() and 0 <= distance < DEPTH:
                roll(snapshot, FAMILY_DEATH_WINDOW, distance)
    # A surviving world's last 300 decisions remain censored, never outside.
    return rows, {
        "seed": int(plan["seed"]),
        "mix": str(plan["mix"]),
        "decisions": decision + 1,
        "death_decision": death_decision,
        "captured_states": captured_count,
        "selected_states": selected,
        "branches": len(rows),
        "peak_pending_clones": peak_pending,
        "branch_seconds": branch_seconds,
    }


def collect(
    student: Path,
    out: Path,
    namespace: str,
    worlds: int,
    mixes: Sequence[str],
    depth: int,
    match_bank: Path | None,
    *,
    device: str = "cpu",
    max_states: int = 0,
    allow_partial: bool = False,
    smoke_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Collect serially into a fresh namespace; incomplete matching fails closed."""
    from research.t3_rollout_labels_20261015.bank_engine import sha256
    from research.t4d_return_regression_20261016.planning import plan

    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("bank.py").resolve(),
        Path(__file__).with_name("planning.py").resolve(),
    ]
    source_hashes = {
        str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p) for p in source_paths
    }
    setup_started = time.perf_counter()
    if smoke_seeds is not None and not (allow_partial and 0 < max_states <= 4 and worlds <= 2):
        raise ValueError("explicit fixture seeds require a <=2-world, <=4-state partial smoke")
    matcher = OutsideMatcher(match_bank) if match_bank else None
    plan_doc = plan(student, out, namespace, worlds, list(mixes), smoke_seeds=smoke_seeds)
    setup_seconds = time.perf_counter() - setup_started
    rows: list[dict[str, Any]] = []
    receipts = []
    budget = [0]
    started = time.perf_counter()
    for world_plan in plan_doc["plans"]:
        if sha256(student) != plan_doc["student"]["sha256"]:
            raise RuntimeError("student changed during collection")
        world_rows, receipt = collect_world(
            world_plan,
            student,
            depth,
            matcher,
            device=device,
            max_states=max_states,
            state_budget=budget,
        )
        rows.extend(world_rows)
        receipts.append(receipt)
    if sha256(student) != plan_doc["student"]["sha256"]:
        raise RuntimeError("student changed during collection")
    if any(
        sha256(p) != source_hashes[str(p.relative_to(Path(__file__).resolve().parents[2]))]
        for p in source_paths
    ):
        raise RuntimeError("collection source changed during execution")
    matching = matcher.receipt() if matcher else None
    branch_seconds = sum(r["branch_seconds"] for r in receipts)
    result = {
        "schema": "t4d-return-collection/v1",
        "student": plan_doc["student"],
        "seed_audit": plan_doc["seed_audit"],
        "namespace": namespace,
        "depth": depth,
        "worlds": receipts,
        "device": device,
        "max_states": max_states,
        "allow_partial": allow_partial,
        "matching": matching,
        "wall_seconds": time.perf_counter() - started,
        "setup_seconds": setup_seconds,
        "branch_seconds": branch_seconds,
        "branches": len(rows),
        "seconds_per_1000_branches": branch_seconds * 1000 / len(rows) if rows else None,
        "source_sha256": source_hashes,
    }
    incomplete = bool(matching and matching["deficits"])
    if incomplete and not allow_partial:
        result.update(bank=None, status="INCOMPLETE_MATCHING_NO_BANK")
    elif rows:
        result["status"] = "PARTIAL_SMOKE" if allow_partial else "COLLECTED"
        result["bank"] = write_bank(out / "return_bank.npz", rows, result)
    else:
        result.update(bank=None, status="NO_QUALIFYING_REAL_BRANCHES")
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Opt in to return collection; default invocation never starts work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--mixes", default="scripted")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--record-returns", action="store_true")
    parser.add_argument("--outside-returns", action="store_true")
    parser.add_argument("--match-bank", type=Path)
    parser.add_argument(
        "--max-states", type=int, default=0, help="global smoke cap; zero is uncapped"
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="write a labeled partial smoke bank"
    )
    parser.add_argument(
        "--smoke-seeds", help="explicit comma-separated fixture seeds; no freshness claim"
    )
    args = parser.parse_args(argv)
    if args.worlds < 1 or not 1 <= args.depth <= DEPTH or args.max_states < 0:
        parser.error("positive worlds, depth in [1,300], nonnegative max-states required")
    if not args.record_returns:
        parser.error("--record-returns is required; ordinary collection remains unchanged")
    if args.outside_returns != (args.match_bank is not None):
        parser.error("--outside-returns and --match-bank must be supplied together")
    if args.allow_partial and not args.max_states:
        parser.error("--allow-partial needs a positive --max-states smoke bound")
    mixes = [item for item in args.mixes.split(",") if item]
    if not mixes or any(item not in {"scripted", "mixed", "frozen"} for item in mixes):
        parser.error("--mixes must contain scripted, mixed, or frozen")
    _configure_device(args.device)
    result = collect(
        args.student.resolve(),
        args.out.resolve(),
        f"t4d-return-20261016/{args.round}",
        args.worlds,
        mixes,
        args.depth,
        args.match_bank.resolve() if args.match_bank else None,
        device=args.device,
        max_states=args.max_states,
        allow_partial=args.allow_partial,
        smoke_seeds=[int(x) for x in args.smoke_seeds.split(",")] if args.smoke_seeds else None,
    )
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "INCOMPLETE_MATCHING_NO_BANK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
