"""L2 fresh-bank replay of the frozen, unchanged T3 dense-search rule.

This is deliberately a development-only evaluator.  It has no connection to a
registered T34 verdict or to serving.  A shard contains paired bare-B8 and
dense-search results for an immutable bank partition; :mod:`analyze` is the
only consumer which combines shards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_served_search.chunks import (
    CHUNK_COUNT,
    chunk_rows,
    validate_chunk_rows,
    world_ids_sha256,
)
from research.bl_served_search.exports import export_json
from research.t1_commitment_20261015.replay import configure_cpu
from research.t3_rollout_labels_20261015.trigger import DenseTrigger

SCHEMA = "bl-served-search-l2-shard/v1"
ROW_SCHEMA = "bl-served-search-l2-row/v1"
HORIZON = 5000
DEPTH = 300
ARM = "b8_s3"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MIX_COUNTS = {"scripted": 32, "frozen": 64, "mixed": 64}


def canonical(value: Any) -> str:
    """Return a finite stable JSON form used for identity binding."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_runtime_parity(
    parity: Any, *, bank_sha256: str, source: dict[str, str], require_local_runtime: bool
) -> dict[str, Any]:
    """Bind parity evidence to this bank, code and the consuming runtime."""
    if not isinstance(parity, dict) or parity.get("status") != "VERIFIED":
        raise ValueError("runtime parity receipt is not VERIFIED")
    if (
        parity.get("bank_sha256") != bank_sha256
        or parity.get("b8_sha256") != B8_SHA256
        or parity.get("source_manifest") != source
    ):
        raise ValueError("runtime parity receipt binds different bank/B8/source")
    if not all(
        _sha256(parity.get(key)) for key in ("mac_reference_sha256", "runtime_verification_sha256")
    ):
        raise ValueError("runtime parity receipt lacks evidence hashes")
    runtime = parity.get("runtime_identity")
    if not isinstance(runtime, dict) or not all(
        isinstance(runtime.get(key), str) and runtime[key]
        for key in ("node", "machine", "colab_cli_root")
    ):
        raise ValueError("runtime parity receipt lacks a complete runtime identity")
    if require_local_runtime:
        expected = {
            "node": platform.node(),
            "machine": platform.machine(),
            "colab_cli_root": str(Path(os.environ["COLAB_CLI_ROOT"]).resolve()),
        }
        if runtime != expected:
            raise ValueError("runtime parity receipt belongs to a different runtime")
    return parity


def source_manifest() -> dict[str, str]:
    """Pin the read-only T3 implementation which gives L2 its meaning."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        root / "research/bl_served_search/chunks.py",
        root / "research/t3_rollout_labels_20261015/trigger.py",
        root / "research/t3_rollout_labels_20261015/bank_engine.py",
        root / "research/t3_rollout_labels_20261015/engine.py",
        root / "research/t1_commitment_20261015/replay.py",
        root / "src/simd_env/eval_engine.py",
        root / "src/simd_env/grid_sim.py",
        root / "src/simd_env/batch_sim.py",
    )
    return {str(path.relative_to(root)): digest_file(path) for path in paths}


def _legal(q: np.ndarray, advisory: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply B8's advisory fallback and its exact Q/lower-action ordering."""
    legal = np.asarray(advisory, dtype=bool).copy()
    if legal.shape != (6,):
        raise RuntimeError("malformed B8 advisory mask")
    if not legal.any():
        legal[:3] = True
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError("nonfinite or malformed B8 Q")
    # np.argmax uses the first (lowest action) member of an equal-Q tie.
    bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
    return legal, bare


def _outcome_score(outcome: dict[str, Any]) -> int:
    """T3's survival-first score, including its dead-branch longest-delay fallback."""
    if bool(outcome["survives"]):
        return DEPTH
    if "frames_survived" in outcome:
        return int(outcome["frames_survived"])
    # ``rollout_all``'s mixed/frozen serial adapter counts the fatal
    # transition in frames_executed; T3's batched engine counts only *alive*
    # branch frames.  Convert before applying the common ranking.
    return max(0, int(outcome.get("frames_executed", 0)) - 1)


def _choose(outcomes: dict[int, dict[str, Any]], bare: int, q: np.ndarray) -> int:
    """Keep the unchanged T3 ranking: survival, then bare, Q, lowest action."""
    best = max(_outcome_score(value) for value in outcomes.values())
    tied = [action for action, value in outcomes.items() if _outcome_score(value) == best]
    if bare in tied:
        return int(bare)
    return min(tied, key=lambda action: (-float(q[action]), int(action)))


def _step(
    world: Any, decision: int, trigger: DenseTrigger | None, *, dense: bool, horizon: int
) -> dict[str, Any]:
    """Advance one live world at the evaluator's Watch selector seam.

    The clone is made after the Vector61 runtime's sole ``prepare`` call and
    before dispatch, exactly as the T3 collector does.  Thus frozen/mixed
    opponent carry is never approximated by the fast-anchor-only T1 adapter.
    """
    from research.t3_rollout_labels_20261015.bank_engine import (
        prepared_q,
        rollout_all,
    )
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        alive = bool(prepared.alive[0, 0])
        length = int(prepared.length[0, 0]) if alive else 0
        advisory = prepared.get_advisory_action_mask()[0, 0]
        obs = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
        legal, bare = _legal(q, advisory)
        chosen, outcomes, batch_wall = bare, None, 0.0
        # Dense T3 does not expand the B8 fallback mask: an empty advisory mask
        # permits B8's normal-action fallback but is explicitly not searchable.
        eligible = bool(dense and trigger is not None and trigger.fires(decision, length, alive))
        # Calling fires is itself semantic: an empty advisory mask may occur on
        # the first length-eligible decision, and must still anchor the
        # episode-local every-fifth-decision clock.  Only the branch batch is
        # skipped for that decision.
        fired = bool(eligible and np.asarray(advisory, dtype=bool).any())
        if fired:
            started = time.perf_counter()
            # ``clone`` removes a temporary hook and carries the private
            # Vector61 state, including v8 opponent memory, independently.
            branch = world.clone()
            actions = [int(action) for action in np.flatnonzero(legal)]
            remaining = min(DEPTH, horizon - decision)
            outcomes = rollout_all(branch, actions, remaining, decision, record_outcomes=True)
            batch_wall = time.perf_counter() - started
            chosen = _choose(outcomes, bare, q)
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        # Dispatch produces B8's ordinary choice.  Assert it before replacing
        # the hero action so a changed mask/tie convention fails closed.
        if int(actions[0, 0]) != bare:
            raise RuntimeError(f"B8 dispatch/Q mismatch at decision {decision}")
        actions[0, 0] = chosen
        selected.update(
            alive=alive,
            length=length,
            fired=fired,
            bare=bare,
            chosen=chosen,
            outcomes=outcomes,
            batch_wall_seconds=batch_wall,
        )
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return selected


def run_arm(plan: dict[str, Any], *, dense: bool, b8_path: Path, horizon: int) -> dict[str, Any]:
    """Evaluate either bare B8 s3 or the unchanged dense T3 overlay."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    configure_cpu()
    world = build_world(ARM, plan, hero_checkpoint=b8_path)
    trigger = DenseTrigger(min_length=600, interval=5) if dense else None
    mass = alive_frames = decisions = 0
    mass_trace: list[int] = []
    alive_trace: list[bool] = []
    batches = changes = surviving_changes = delay_only_changes = residual_changes = 0
    d300_escape_changes = 0
    death_frame = death_length = death_cause = None
    rollout_env_frames = 0
    rollout_wall_seconds = 0.0
    started = time.perf_counter()
    for decision in range(horizon):
        selected = _step(world, decision, trigger, dense=dense, horizon=horizon)
        decisions += 1
        if selected["fired"]:
            batches += 1
            rollout_wall_seconds += float(selected["batch_wall_seconds"])
            outcomes = selected["outcomes"]
            assert outcomes is not None
            rollout_env_frames += sum(
                int(outcome.get("frames_executed", outcome.get("env_frames", 0)))
                for outcome in outcomes.values()
            )
            if selected["chosen"] != selected["bare"]:
                changes += 1
                chosen = outcomes[int(selected["chosen"])]
                bare = outcomes[int(selected["bare"])]
                if bool(chosen["survives"]) and not bool(bare["survives"]):
                    surviving_changes += 1
                    d300_escape_changes += 1
                else:
                    # Every non-rescue changed choice is a finite-horizon
                    # delay: both branches die, or the bare branch lasts to
                    # H300 and ranking broke a Q tie.  The latter is retained
                    # explicitly rather than silently folded into a rescue.
                    delay_only_changes += 1
        alive = bool(world.sim.alive[0, 0])
        alive_frames += int(alive)
        logical_mass = int(world.sim.length[0, 0]) if alive else 0
        mass += logical_mass
        mass_trace.append(logical_mass)
        alive_trace.append(alive)
        if not alive:
            death_frame = decision
            death_length = int(selected["length"])
            death_cause = int(world.sim.get_death_cause()[0, 0])
            break
    mass_trace.extend([0] * (horizon - len(mass_trace)))
    alive_trace.extend([False] * (horizon - len(alive_trace)))
    # A local H300 rescue that still ends in an episode death is disclosed as
    # a D300-only escape/re-death rather than counted as a productive rescue.
    d300_only = d300_escape_changes if death_frame is not None else 0
    return {
        "mass_integral": mass / horizon,
        "alive_frame_fraction": alive_frames / horizon,
        "no_death_episode": bool(world.sim.alive[0, 0]),
        "decisions": decisions,
        "batches": batches,
        "action_changes": changes,
        "horizon_surviving_changes": surviving_changes,
        "delay_only_changes": delay_only_changes,
        "residual_changes": residual_changes,
        "d300_escape_changes": d300_escape_changes,
        "d300_only_escapes": d300_only,
        "re_deaths": int(bool(d300_only)),
        "proposed_interventions": batches,
        "completed_interventions": batches,
        # L2' has no paced deadline/consumer.  It must never manufacture a
        # deadline-valid count from an unpaced replay.
        "consumed_before_deadline_interventions": None,
        "consumed_before_deadline_availability": "UNAVAILABLE_UNPACED_L2",
        "death_frame": death_frame,
        "death_length": death_length,
        "death_cause": death_cause,
        "rollout_env_frames": rollout_env_frames,
        "rollout_wall_seconds": rollout_wall_seconds,
        "wall_seconds": time.perf_counter() - started,
        "_mass_trace": mass_trace,
        "_alive_trace": alive_trace,
    }


def run_world(row: dict[str, Any], *, b8_path: Path) -> dict[str, Any]:
    """Return one paired world receipt; inputs are already collision-audited."""
    plan = row.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("bank row lacks a materialized roster plan")
    if int(plan.get("seed", -1)) != int(row["world_seed"]) or plan.get("mix") != row["mix"]:
        raise ValueError("bank row/roster identity mismatch")
    horizon = int(row["horizon"])
    baseline = run_arm(plan, dense=False, b8_path=b8_path, horizon=horizon)
    dense = run_arm(plan, dense=True, b8_path=b8_path, horizon=horizon)
    baseline_trace, dense_trace = baseline.pop("_mass_trace"), dense.pop("_mass_trace")
    baseline_alive, dense_alive = baseline.pop("_alive_trace"), dense.pop("_alive_trace")
    if len(baseline_trace) != horizon or len(dense_trace) != horizon:
        raise RuntimeError("full-horizon mass trace is incomplete")
    if len(baseline_alive) != horizon or len(dense_alive) != horizon:
        raise RuntimeError("full-horizon alive trace is incomplete")
    common_delta = dense_only = baseline_only = 0
    for base_mass, dense_mass in zip(baseline_trace, dense_trace):
        if base_mass and dense_mass:
            common_delta += dense_mass - base_mass
        elif dense_mass:
            dense_only += dense_mass
        elif base_mass:
            baseline_only -= base_mass
    ledger = {
        "common_alive_productivity_delta": common_delta / horizon,
        "dense_only_survival_mass": dense_only / horizon,
        "baseline_only_survival_mass": baseline_only / horizon,
        "total": (common_delta + dense_only + baseline_only) / horizon,
    }
    if not math.isclose(
        ledger["total"], dense["mass_integral"] - baseline["mass_integral"], abs_tol=1e-12
    ):
        raise RuntimeError("mass ledger does not reconcile to paired mass delta")
    baseline_last_length = int(baseline["death_length"] or 0)
    baseline_tail = (
        dense_trace[int(baseline["death_frame"]) :] if baseline["death_frame"] is not None else []
    )
    productive_rescue = bool(
        baseline["death_frame"] is not None
        and dense["death_frame"] is None
        and baseline_tail
        and sum(baseline_tail) / len(baseline_tail) >= 0.9 * baseline_last_length
    )
    common_alive_growth_loss = (
        sum(
            max(0, base_mass - dense_mass)
            for base_mass, dense_mass, base_alive, dense_alive in zip(
                baseline_trace, dense_trace, baseline_alive, dense_alive
            )
            if base_alive and dense_alive
        )
        / horizon
    )

    def sealer(arm: dict[str, Any]) -> str:
        # A death-cause code describes contact, not the topology/onset needed
        # for T1-style sealer attribution.  Preserve it separately and label
        # every unreconstructed death unresolved.
        return "none" if arm["death_frame"] is None else "unresolved"

    world_ledger = {
        "productive_rescue": productive_rescue,
        "d300_only_escape": int(dense["d300_only_escapes"]) > 0,
        "re_death": int(dense["re_deaths"]) > 0,
        "delay_only_changes": int(dense["delay_only_changes"]),
        "newly_caused_death": baseline["death_frame"] is None and dense["death_frame"] is not None,
        "common_alive_growth_loss": common_alive_growth_loss,
        "natural_world_denominator": 1,
        "baseline_died": baseline["death_frame"] is not None,
        "b8_death_frame": baseline["death_frame"],
        "b8_death_length": baseline["death_length"],
        "b8_death_cause": baseline["death_cause"],
        "baseline_sealer": sealer(baseline),
        "dense_sealer": sealer(dense),
        "interventions": {
            "proposed": int(dense["proposed_interventions"]),
            "completed": int(dense["completed_interventions"]),
            "consumed_before_deadline": dense["consumed_before_deadline_interventions"],
            "consumed_before_deadline_availability": dense["consumed_before_deadline_availability"],
        },
    }
    return {
        "schema": ROW_SCHEMA,
        "world_id": str(row["world_id"]),
        "world_seed": int(row["world_seed"]),
        "mix": str(row["mix"]),
        "horizon": horizon,
        "plan": plan,
        "baseline": baseline,
        "dense": dense,
        "paired_delta": {
            key: float(dense[key]) - float(baseline[key])
            for key in ("mass_integral", "alive_frame_fraction", "no_death_episode")
        },
        "mass_ledger": ledger,
        "trajectories": {"baseline_mass": baseline_trace, "dense_mass": dense_trace},
        "world_ledger": world_ledger,
    }


def _worker(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, b8_path = job
    return run_world(row, b8_path=Path(b8_path))


def portable_plan(
    plan: dict[str, Any], checkpoint_dir: Path, *, allow_absolute: bool = False
) -> dict[str, Any]:
    """Resolve bank-pinned checkpoint basenames and bind their bytes locally.

    Banks intentionally contain portable basenames rather than the owner's
    machine paths.  The ordered member digests remain beside them, so a Colab
    input mount cannot silently substitute a roster model.
    """
    specs = plan.get("specs")
    expected = plan.get("member_sha256s")
    if not isinstance(specs, list) or not isinstance(expected, list) or len(specs) != len(expected):
        raise ValueError("portable roster lacks ordered specs/member digests")
    resolved: list[list[Any]] = []
    for spec, member_sha in zip(specs, expected):
        if not isinstance(spec, list) or len(spec) != 2:
            raise ValueError("malformed portable roster specification")
        kind, reference = spec
        if kind != "checkpoint":
            resolved.append([kind, reference])
            continue
        name = Path(str(reference)).name
        if name != str(reference) and not allow_absolute:
            raise ValueError("portable checkpoint reference must be a basename")
        path = Path(str(reference)) if name != str(reference) else checkpoint_dir / name
        if not path.is_file() or digest_file(path) != str(member_sha):
            raise ValueError(f"roster checkpoint missing or digest mismatch: {name}")
        resolved.append([kind, str(path)])
    return {**plan, "specs": resolved}


def _write_progress(path: Path, *, completed: int, total: int, started: float) -> None:
    """Publish non-scientific operational telemetry without touching receipts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if completed else None
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        canonical(
            {
                "schema": "bl-served-search-l2-progress/v1",
                "completed": completed,
                "total": total,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }
        )
        + "\n"
    )
    os.replace(temporary, path)


def validate_bank(bank: dict[str, Any], *, fixture: bool = False) -> list[dict[str, Any]]:
    """Reject malformed/non-L2 banks before any world is constructed."""
    if bank.get("schema") != "bl-served-search-bank/v1":
        raise ValueError("unexpected L2 bank schema")
    rows = bank.get("excluded_preflight_rows") if fixture else bank.get("rows")
    required = 3 if fixture else sum(MIX_COUNTS.values())
    if not isinstance(rows, list) or len(rows) != required:
        raise ValueError(f"L2 bank must contain exactly {required} requested worlds")
    by_mix = {mix: 0 for mix in MIX_COUNTS}
    ids, seeds = set(), set()
    for row in rows:
        if not isinstance(row, dict) or row.get("mix") not in by_mix:
            raise ValueError("malformed L2 bank row")
        by_mix[row["mix"]] += 1
        identity = str(row.get("world_id"))
        seed = int(row.get("world_seed", -1))
        if identity in ids or seed in seeds or not 0 <= seed <= 2**32 - 1:
            raise ValueError("duplicate/invalid L2 bank world identity")
        ids.add(identity)
        seeds.add(seed)
        if not fixture and int(row.get("horizon", -1)) != HORIZON:
            raise ValueError("L2 selection worlds must be H5000")
    if not fixture and by_mix != MIX_COUNTS:
        raise ValueError(f"wrong L2 mix allocation: {by_mix}")
    return sorted(rows, key=lambda value: str(value["world_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--fixture", action="store_true", help="run three excluded preflight worlds"
    )
    parser.add_argument("--runtime-parity-receipt", type=Path)
    parser.add_argument("--b8-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("SNAKE_CKPT_DIR", "saved_snakes")),
    )
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="required after the committed registration is independently admitted",
    )
    args = parser.parse_args()
    if not args.fixture and not args.allow_production:
        parser.error("refusing L2 selection-world execution without --allow-production")
    parity: dict[str, Any] | None = None
    if not args.fixture:
        if args.runtime_parity_receipt is None:
            parser.error("production L2 requires a verified runtime parity receipt")
        from research.bl_served_search.exports import verify_export

        verify_export(args.runtime_parity_receipt)
        try:
            parity = validate_runtime_parity(
                json.loads(args.runtime_parity_receipt.read_text()),
                bank_sha256=digest_file(args.bank),
                source=source_manifest(),
                require_local_runtime=True,
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
    if not 1 <= args.workers <= 12:
        parser.error("workers must be 1..12")
    if args.fixture:
        if not 1 <= args.shard_count <= 3 or not 0 <= args.shard_index < args.shard_count:
            parser.error("fixture shard-count must be 1..3 with a valid index")
    elif args.shard_count != CHUNK_COUNT or not 0 <= args.shard_index < CHUNK_COUNT:
        parser.error("L2' requires exactly four immutable chunk receipts (indices 0..3)")
    if args.out.exists() or any(
        Path(str(args.out) + suffix).exists() for suffix in (".sha256", ".complete")
    ):
        parser.error("output is create-only and already exists")
    bank = json.loads(args.bank.read_text())
    rows = validate_bank(bank, fixture=args.fixture)
    if not args.b8_path.is_file() or digest_file(args.b8_path) != B8_SHA256:
        parser.error("--b8-path must name the supervisor-provided B8 checkpoint")
    selected = (
        [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
        if args.fixture
        else chunk_rows(rows, args.shard_index)
    )
    if not args.fixture:
        validate_chunk_rows(selected)
    selected = [
        {
            **row,
            "plan": portable_plan(
                dict(row["plan"]), args.checkpoint_dir, allow_absolute=args.fixture
            ),
        }
        for row in selected
    ]
    started = time.perf_counter()
    progress_path = args.out.parent / (args.out.name + ".progress.json")
    context = multiprocessing.get_context("spawn")
    # A parent-owned 120-second heartbeat is independent of a completed
    # world. Dense worlds can take many minutes, so completion-only logging
    # would make a live supervisor job look stalled.
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        pending = {pool.submit(_worker, (row, str(args.b8_path))) for row in selected}
        results = []
        _write_progress(progress_path, completed=0, total=len(selected), started=started)
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=120, return_when=concurrent.futures.FIRST_COMPLETED
            )
            if not done:
                _write_progress(
                    progress_path, completed=len(results), total=len(selected), started=started
                )
                continue
            for future in done:
                result = future.result()
                export_json(args.out.parent / "worlds" / f"{result['world_id']}.json", result)
                results.append(result)
            _write_progress(
                progress_path, completed=len(results), total=len(selected), started=started
            )
    selected_order = {str(row["world_id"]): index for index, row in enumerate(selected)}
    results.sort(key=lambda value: selected_order[str(value["world_id"])])
    payload = {
        "schema": SCHEMA if not args.fixture else "bl-served-search-l2-fixture-shard/v1",
        "status": "completed",
        "identity": {
            "bank_sha256": digest_file(args.bank),
            "bank_semantic_sha256": digest_value(bank),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "chunk_index": args.shard_index if not args.fixture else None,
            "chunk_count": CHUNK_COUNT if not args.fixture else None,
            "chunk_world_ids_sha256": world_ids_sha256(selected) if not args.fixture else None,
            "workers": args.workers,
            "arm": ARM,
            "b8_sha256": B8_SHA256,
            "horizon": HORIZON,
            "fixture": args.fixture,
            "runtime_parity": parity,
            "dense_rule": (
                "T3 DenseTrigger(L>=600, interval=5), 6 actions x H300, " "B8 greedy eps0 no veto"
            ),
            "source_manifest": source_manifest(),
        },
        "rows": results,
        "rows_sha256": digest_value(results),
        "wall_seconds": time.perf_counter() - started,
        "clean_exit": True,
    }
    export_json(args.out, payload)


if __name__ == "__main__":
    main()
