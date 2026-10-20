"""Bounded on-policy ExIt collection with deterministic death-window rewinds.

This is preparation machinery, not an evaluator or a promotion authority. All
exports describe the final, replayable trajectory; labels from abandoned
suffixes are removed when a world rewinds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_NETWORKS: dict[tuple[str, int, int], Any] = {}


def _share_network(path: str | Path, network: Any) -> Any:
    """Intern immutable source eval tensors before making device copies."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    return _NETWORKS.setdefault(key, network)


def prepare_world(world: Any) -> Any:
    """Stop at the evaluator's prepared selector seam exactly once."""
    if getattr(world.sim, "_policy_phase_prepared", False):
        return world

    class Prepared(Exception):
        pass

    def select(sim: Any) -> np.ndarray:
        world.runtime.prepare(sim)
        raise Prepared

    try:
        world.sim.step_with_policy(select)
    except Prepared:
        world.sim._policy_phase_prepared = True
        world.sim._policy_step_in_progress = False
    return world


def make_world_rows(config: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    """Derive portable prep worlds in a separate round/shard/attempt namespace.

    These are development fixtures. Campaign bank collision auditing and the
    preregistration belong to P7 and must precede any Round-0 data collection.
    """
    from research.apex_safety_20260926 import dev_screen
    from research.phase0_search_20261019.bank import _strict_lookup

    if count < 1:
        raise ValueError("count must be positive")
    namespace = str(config.get("namespace", "t34-exit-preflight/v1"))
    if not namespace or "/Users/" in namespace:
        raise ValueError("namespace must be nonempty and portable")
    mixes = config.get("mixes", ("scripted", "scripted", "frozen", "mixed"))
    if isinstance(mixes, str):
        mixes = mixes.split(",")
    if not mixes or any(m not in ("scripted", "frozen", "mixed") for m in mixes):
        raise ValueError("mixes must contain scripted, frozen, or mixed")
    seeds = dev_screen.screen_seeds(count, "t34-exit-b8-20261019/v1", namespace)
    if len(set(seeds)) != count:
        raise ValueError("duplicate prep-world seed")
    design = {(int(r["world_seed"]), r["mix"]): r for r in dev_screen._design_rows(seeds)}
    rows = []
    lookup = _strict_lookup()
    frozen = (
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
        dev_screen.CHAMPION[1],
    )
    for index, seed in enumerate(seeds):
        mix = str(mixes[index % len(mixes)])
        slots = design[(int(seed), mix)]["slots"]
        opponents = []
        for slot_index, slot in enumerate(slots):
            kind, reference = lookup[str(slot["member_sha256"])]
            if kind == "checkpoint":
                reference = frozen[(index + slot_index) % len(frozen)]
            opponents.append([kind, reference])
        row = {
            "namespace": namespace,
            "world_id": f"{namespace}/{mix}/{index:06d}",
            "world_seed": int(seed),
            "mix": mix,
            "opponents": opponents,
            "opponent_veto": "v8",
            "opponent_veto_lambda": 8.0,
            "horizon": 5000,
        }
        rows.append(row)
    return rows


def build_world(row: Mapping[str, Any], checkpoint: Path | str) -> Any:
    """Reuse the strict constructor, installing the plan's frozen v8 opponents."""
    requested_threads = int(os.environ.get("NUMBA_NUM_THREADS", "1"))
    requested_device = os.environ.get("SNAKE_DQN_DEVICE", "cpu")
    from research.phase0_search_20261019.teacher import build_world as base_build

    world = base_build(dict(row), checkpoint)
    # Phase-0's compatibility import forces CPU/one thread. It is used only as
    # a world constructor here; restore the explicit T34 recipe after that lazy
    # import and before any labels or continuations execute.
    import numba

    os.environ["NUMBA_NUM_THREADS"] = str(requested_threads)
    os.environ["SNAKE_DQN_DEVICE"] = requested_device
    numba.set_num_threads(requested_threads)
    world.hero.net = _share_network(checkpoint, world.hero.net)
    # Different snake slots own independent v8 bookkeeping; model tensors alone
    # are shared. Veto hooks are keyed by environment, so sharing one hook among
    # multiple slots would mix their diagnostic and memoized state.
    world.opponents = [copy.copy(policy) for policy in world.opponents]
    for policy in world.opponents:
        if hasattr(policy, "network") and hasattr(policy, "checkpoint_path"):
            policy.network = _share_network(policy.checkpoint_path, policy.network)
    if row.get("opponent_veto") == "v8":
        for policy in world.opponents:
            if hasattr(policy, "_configure_veto"):
                policy._configure_veto(
                    tuple(range(1, world.sim.S)),
                    "v8",
                    float(row.get("opponent_veto_lambda", 8.0)),
                    None,
                )
    elif row.get("opponent_veto") not in (None, "none"):
        raise ValueError("unsupported frozen opponent veto")
    return world


def _diag_dict(diag: Any) -> dict[str, Any]:
    value = asdict(diag) if is_dataclass(diag) else dict(diag)
    return value


def _snapshot_at_window_start(
    saved: tuple[int, Any] | None,
    world: Any,
    decision: int,
    actions: list[list[int]],
    window: int,
) -> tuple[int, Any]:
    """Maintain one lagging prepared snapshot by replaying recorded actions.

    Keeping 150 complete simulator clones per root exceeds a lane's RAM at
    W=256. Replaying the oldest trace row advances the private snapshot exactly
    one transition and keeps storage linear in resident roots instead.
    """
    if saved is None:
        snapshot = world.clone()
        # T3's world clone deliberately shares immutable model tensors. Its
        # optional v8 dictionaries also need independent mutable bookkeeping.
        for policy in getattr(snapshot, "opponents", []):
            for name in ("veto_counters", "boost_counters", "live_vetoes"):
                if hasattr(policy, name):
                    setattr(policy, name, {})
            if hasattr(policy, "last_veto"):
                policy.last_veto = None
        return decision, snapshot
    start, snapshot = saved
    wanted = max(0, decision - window + 1)
    while start < wanted:
        snapshot.sim.step(np.asarray(actions[start], dtype=np.int64).reshape(1, 6))
        snapshot.runtime.observe_step(snapshot.sim)
        prepare_world(snapshot)
        start += 1
    return start, snapshot


def collect_world(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect one terminal-hero world, searching at registered trigger points.

    ``config`` requires ``checkpoint`` and optionally a portable ``row``;
    limits are ``frames`` (5000), ``seconds`` (1500), and ``horizon`` (300).
    An externally supplied ``teacher`` permits benchmark/fit policy injection.
    """
    from research.t34_exit_b8.engine import (
        ExitTeacher,
        ExitTeacherConfig,
        TriggerConfig,
        mix_action,
    )
    from research.t34_exit_b8.exports import state_hash

    frames = int(config.get("frames", 5000))
    seconds = float(config.get("seconds", 1500))
    if frames < 1 or seconds <= 0:
        raise ValueError("frames and seconds must be positive")
    row = dict(config.get("row") or make_world_rows(config, 1)[0])
    checkpoint = Path(str(config["checkpoint"]))
    trigger = TriggerConfig(**dict(config.get("triggers", {})))
    teacher = config.get("teacher") or ExitTeacher(
        ExitTeacherConfig(
            horizon=int(config.get("horizon", 300)),
            beta=float(config.get("beta", 1.0)),
            trigger=trigger,
        ),
        device=str(config.get("device", "cpu")),
        policy=config.get("policy"),
    )
    world = config.get("world_builder", build_world)(row, checkpoint)
    seed = int(row.get("world_seed", row.get("seed")))
    actions: list[list[int]] = []
    labels: list[dict[str, Any]] = []
    hashes = {"0": state_hash(world.sim)}
    snapshot: tuple[int, Any] | None = None
    rewinds = 0
    death_window_end = -1
    decision = 0
    started = time.monotonic()
    heartbeat = started
    deadline = float(config.get("deadline", started + seconds))
    mass_sum = 0.0
    masses: list[float] = []
    timed_out = False
    while decision < frames and bool(world.sim.alive[0, 0]):
        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            break
        if now - heartbeat >= 110:
            print(
                json.dumps(
                    {
                        "event": "t34_progress",
                        "world_id": row["world_id"],
                        "done": decision,
                        "total": frames,
                        "rewinds": rewinds,
                        "elapsed_seconds": now - started,
                    }
                ),
                flush=True,
            )
            heartbeat = now
        prepare_world(world)
        if trigger.death_window and rewinds < trigger.max_rewinds:
            snapshot = _snapshot_at_window_start(
                snapshot, world, decision, actions, trigger.death_window
            )
        evaluation = teacher.evaluate(world)
        student = evaluation.greedy_action()
        length = int(world.sim.length[0, 0])
        reason = None
        if decision <= death_window_end:
            reason = "death_window"
        elif trigger.fires(decision=decision, length=length, seed=seed):
            reason = "confirm" if length >= trigger.confirm_min_length else "uniform"
        chosen = student
        # A label always means H complete lookahead transitions; the teacher
        # may continue beyond the collection horizon on its private clones.
        if reason is not None:
            recommended, diagnostics = teacher.choose([world])
            diag = _diag_dict(diagnostics[0])
            recommended_action = int(recommended[0])
            beta = float(config.get("beta", 1.0))
            chosen = mix_action(
                student, recommended_action, seed=seed, decision=decision, beta=beta
            )
            diag.update(
                decision=decision,
                reason=reason,
                length=length,
                action=recommended_action,
                student_action=student,
                executed_action=chosen,
            )
            labels.append(diag)
        grid = teacher.actions(world, chosen)
        world.sim.step(grid)
        world.runtime.observe_step(world.sim)
        actions.append(np.asarray(grid[0], dtype=np.int64).tolist())
        masses.append(float(world.sim.length[0, 0]) if world.sim.alive[0, 0] else 0.0)
        decision += 1
        if decision % 250 == 0:
            hashes[str(decision)] = state_hash(world.sim)
        if not world.sim.alive[0, 0] and snapshot is not None and rewinds < trigger.max_rewinds:
            death_window_end = decision - 1
            restart, world = snapshot
            rewinds += 1
            actions = actions[:restart]
            masses = masses[:restart]
            labels = [label for label in labels if int(label["decision"]) < restart]
            hashes = {key: value for key, value in hashes.items() if int(key) <= restart}
            decision = restart
            snapshot = None
    hashes[str(decision)] = state_hash(world.sim)
    mass_sum = sum(masses)
    profile = teacher.profile()
    result = {
        "schema": "t34-exit-world/v1",
        "world_id": row["world_id"],
        "seed": seed,
        "mix": row["mix"],
        "row": row,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "actions": actions,
        "hashes": hashes,
        "labels": labels,
        "rng_stream_ids": {
            "food": seed,
            "scripted": seed,
            "sample": "sha256:t34-exit-v1:seed:decision:low-length",
            "beta": "sha256:t34-exit-v1:seed:decision:beta",
        },
        "rewinds": rewinds,
        "decisions": decision,
        "survives": bool(world.sim.alive[0, 0]),
        "complete": not timed_out,
        "mass_integral": mass_sum / frames,
        "frames": frames,
        "beta": float(config.get("beta", 1.0)),
        "horizon": int(config.get("horizon", 300)),
        "triggers": asdict(trigger),
        "profile": profile,
    }
    return result


def run_worlds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect with up to W=256 resident roots and batch their triggered searches.

    Each root retains its own simulator/RNG and bounded rewind ring. The teacher
    receives all simultaneously triggered roots in one call, so benchmark and
    label workers exercise the same verify/expand kernel and batching schedule.
    """
    from research.t34_exit_b8.engine import (
        ExitTeacher,
        ExitTeacherConfig,
        TriggerConfig,
        mix_action,
    )
    from research.t34_exit_b8.exports import state_hash, write_world

    started = time.monotonic()
    seconds = float(config.get("seconds", 1500))
    rows = list(config.get("rows") or make_world_rows(config, int(config.get("max_worlds", 2))))
    frames = int(config.get("frames", 5000))
    capacity = int(config.get("worlds_per_process", config.get("worlds", 256)))
    if capacity < 1 or frames < 1 or seconds <= 0:
        raise ValueError("root capacity, frames and seconds must be positive")
    deadline = started + seconds
    trigger = TriggerConfig(**dict(config.get("triggers", {})))
    teacher = config.get("teacher") or ExitTeacher(
        ExitTeacherConfig(
            horizon=int(config.get("horizon", 300)),
            beta=float(config.get("beta", 1.0)),
            trigger=trigger,
        ),
        device=str(config.get("device", "cpu")),
        policy=config.get("policy"),
    )
    teacher.deadline = deadline
    checkpoint = Path(str(config["checkpoint"]))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rollout_checkpoint = Path(str(config.get("student_checkpoint") or checkpoint))
    rollout_sha = hashlib.sha256(rollout_checkpoint.read_bytes()).hexdigest()
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    done, decisions, active, resident, export_bytes = 0, 0, 0, 0, 0
    mixes = {str(row["mix"]): {"searched": 0, "expanded": 0, "worlds": 0} for row in rows}
    peak_roots = search_batch_peak = 0
    heartbeat = started
    root_steps = boost_steps = teacher_flips = roots_started = 0
    gpu_samples: list[float] = []
    indices = list(config.get("row_indices", range(len(rows))))
    if len(indices) != len(rows) or len(set(indices)) != len(indices):
        raise ValueError("export row indices must be unique and match the world bank")
    iterator = iter(zip(indices, rows))
    states: list[dict[str, Any]] = []
    exhausted = False

    def progress() -> None:
        nonlocal heartbeat
        now = time.monotonic()
        if now - heartbeat < 110:
            return
        elapsed = max(now - started, 1e-9)
        profile = teacher.profile()
        active_work = int(profile.get("active_row_steps", 0))
        resident_work = int(profile.get("resident_row_steps", 0))
        sampler = config.get("gpu_utilization_sampler")
        if sampler is not None:
            sample = sampler()
            if sample is not None:
                gpu_samples.append(float(sample))
        print(
            json.dumps(
                {
                    "event": "t34_progress",
                    "done": done,
                    "total": len(rows),
                    "resident_roots": len(states),
                    "elapsed_seconds": elapsed,
                    "active_row_steps_per_second": active_work / elapsed,
                    "resident_row_steps_per_second": resident_work / elapsed,
                    "active_resident_fraction": active_work / max(1, resident_work),
                    "expansion_fraction_per_mix": {
                        mix: m["expanded"] / max(m["searched"], 1) for mix, m in mixes.items()
                    },
                    "worlds_per_hour": done * 3600 / elapsed,
                    "boost_share": boost_steps / max(root_steps, 1),
                    "flips_per_world": teacher_flips / max(roots_started, 1),
                    "ram_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    * (1 if sys.platform == "darwin" else 1024),
                }
            ),
            flush=True,
        )
        heartbeat = now

    teacher.progress = progress
    while time.monotonic() < deadline:
        while not exhausted and len(states) < capacity and time.monotonic() < deadline:
            pair = next(iterator, None)
            if pair is None:
                exhausted = True
                break
            index, row = pair
            world = config.get("world_builder", build_world)(row, checkpoint)
            roots_started += 1
            states.append(
                {
                    "index": index,
                    "row": row,
                    "world": world,
                    "seed": int(row.get("world_seed", row.get("seed"))),
                    "decision": 0,
                    "actions": [],
                    "labels": [],
                    "hashes": {"0": state_hash(world.sim)},
                    "masses": [],
                    "snapshot": None,
                    "rewinds": 0,
                    "death_window_end": -1,
                    "active": 0,
                    "resident": 0,
                    "searched": 0,
                    "expanded": 0,
                }
            )
            progress()
        peak_roots = max(peak_roots, len(states))
        if not states:
            break
        if time.monotonic() >= deadline:
            break
        searching = []
        for state in states:
            world = prepare_world(state["world"])
            decision = state["decision"]
            if state["rewinds"] < trigger.max_rewinds:
                state["snapshot"] = _snapshot_at_window_start(
                    state["snapshot"],
                    world,
                    decision,
                    state["actions"],
                    trigger.death_window,
                )
            state["student"] = teacher.evaluate(world).greedy_action()
            length = int(world.sim.length[0, 0])
            state["length"] = length
            reason = "death_window" if decision <= state["death_window_end"] else None
            if reason is None and trigger.fires(
                decision=decision, length=length, seed=state["seed"]
            ):
                reason = "confirm" if length >= trigger.confirm_min_length else "uniform"
            state["reason"] = reason
            state["chosen"] = state["student"]
            if reason is not None:
                searching.append(state)
        search_batch_peak = max(search_batch_peak, len(searching))
        if searching:
            try:
                recommended, diagnostics = teacher.choose([state["world"] for state in searching])
            except TimeoutError:
                break
            for state, action, diagnostic in zip(searching, recommended, diagnostics):
                diag = _diag_dict(diagnostic)
                chosen = mix_action(
                    state["student"],
                    int(action),
                    seed=state["seed"],
                    decision=state["decision"],
                    beta=float(config.get("beta", 1.0)),
                )
                state["chosen"] = chosen
                diag.update(
                    decision=state["decision"],
                    reason=state["reason"],
                    length=state["length"],
                    action=int(action),
                    student_action=state["student"],
                    executed_action=chosen,
                )
                state["labels"].append(diag)
                state["searched"] += 1
                state["expanded"] += bool(diag.get("expanded", False))
                mixes[str(state["row"]["mix"])]["searched"] += 1
                mixes[str(state["row"]["mix"])]["expanded"] += bool(diag.get("expanded", False))
                state["active"] += int(diag.get("active_row_steps", 0))
                state["resident"] += int(diag.get("resident_row_steps", 0))
        remaining = []
        for state in states:
            world = state["world"]
            grid = teacher.actions(world, state["chosen"])
            root_steps += 1
            boost_steps += int(state["chosen"] >= 3)
            teacher_flips += int(state["chosen"] != state["student"])
            world.sim.step(grid)
            world.runtime.observe_step(world.sim)
            state["actions"].append(np.asarray(grid[0], dtype=np.int64).tolist())
            state["masses"].append(float(world.sim.length[0, 0]) if world.sim.alive[0, 0] else 0.0)
            state["decision"] += 1
            decision = state["decision"]
            if decision % 250 == 0:
                state["hashes"][str(decision)] = state_hash(world.sim)
            if (
                not world.sim.alive[0, 0]
                and state["snapshot"] is not None
                and state["rewinds"] < trigger.max_rewinds
            ):
                state["death_window_end"] = decision - 1
                restart, state["world"] = state["snapshot"]
                state["rewinds"] += 1
                state["actions"] = state["actions"][:restart]
                state["masses"] = state["masses"][:restart]
                state["labels"] = [d for d in state["labels"] if d["decision"] < restart]
                state["hashes"] = {k: v for k, v in state["hashes"].items() if int(k) <= restart}
                state["decision"] = restart
                state["snapshot"] = None
                remaining.append(state)
                continue
            if world.sim.alive[0, 0] and decision < frames:
                remaining.append(state)
                continue
            state["hashes"][str(decision)] = state_hash(world.sim)
            result = {
                "schema": "t34-exit-world/v1",
                "world_id": state["row"]["world_id"],
                "seed": state["seed"],
                "mix": state["row"]["mix"],
                "row": state["row"],
                "checkpoint_sha256": checkpoint_sha,
                "rollout_policy_sha256": rollout_sha,
                "actions": state["actions"],
                "hashes": state["hashes"],
                "labels": state["labels"],
                "rewinds": state["rewinds"],
                "frames": frames,
                "horizon": int(config.get("horizon", 300)),
                "beta": float(config.get("beta", 1.0)),
                "complete": True,
                "rng_stream_ids": {
                    "food": state["seed"],
                    "scripted": state["seed"],
                    "sample": "sha256:t34-exit-v1:seed:decision:low-length",
                    "beta": "sha256:t34-exit-v1:seed:decision:beta",
                },
                "profile": {
                    "active_row_steps": state["active"],
                    "resident_row_steps": state["resident"],
                },
            }
            path = output / f"world-{state['index']:06d}.json"
            write_world(path, result)
            export_bytes += path.stat().st_size
            done += 1
            decisions += decision
            active += state["active"]
            resident += state["resident"]
            entry = mixes.setdefault(
                str(state["row"]["mix"]), {"searched": 0, "expanded": 0, "worlds": 0}
            )
            entry["worlds"] += 1
            print(
                json.dumps(
                    {
                        "event": "t34_world_complete",
                        "done": done,
                        "total": len(rows),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        states = remaining
        progress()
    import numba

    elapsed = time.monotonic() - started
    return {
        "worlds": done,
        "requested_worlds": len(rows),
        "decisions": decisions,
        "active_row_steps": active,
        "resident_row_steps": resident,
        "elapsed_seconds": elapsed,
        "worlds_per_hour": done * 3600 / max(elapsed, 1e-9),
        "decisions_per_second": decisions / max(elapsed, 1e-9),
        "active_row_steps_per_second": active / max(elapsed, 1e-9),
        "active_resident_fraction": active / max(resident, 1),
        "mixes": mixes,
        "export_bytes": export_bytes,
        "export_bytes_per_world": export_bytes / max(done, 1),
        "world_rate_scope": "completed terminal-or-horizon worlds; unfinished roots are censored",
        "complete": done == len(rows),
        "configured_root_capacity": capacity,
        "peak_root_worlds": peak_roots,
        "search_batch_peak": search_batch_peak,
        "teacher_profile": teacher.profile(),
        "actual_policy_device": str(getattr(teacher.policy, "device", config.get("device", "cpu"))),
        "actual_numba_threads": int(numba.get_num_threads()),
        "boost_share": boost_steps / max(root_steps, 1),
        "flips_per_world": teacher_flips / max(roots_started, 1),
        "root_steps_including_rewinds": root_steps,
        "gpu_utilization_samples_percent": gpu_samples,
    }
