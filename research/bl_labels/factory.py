"""Retrospective exact-label collection for the F0 hindsight factory.

This module deliberately has no simulator of its own.  It drives the existing
continuous Watch6 implementation and restores its trusted B2 snapshots for all
counterfactual work.  A failed restore digest or a failed bare reroll makes the
whole unit invalid; callers must never export its partial rows as labels.
"""

from __future__ import annotations

import copy
import copyreg
import hashlib
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np

from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.b2_engine import FRP3, WatchDriver, attest, driver_stats
from research.bl_served_search.exports import digest
from research.t34_shadow.shadow import full_digest

HORIZON = 300
EXTENDED_HORIZON = 1440
SNAPSHOT_STRIDE = 20
ROOT_LOOKBACK = 300
ROOT_STRIDE = 5
LAST_CONTIGUOUS = 30
ROOT_CAP = 90
PRESERVATION_ROOTS = 8
SCHEMA = "bl-labels-f0-unit/v1"


class FactoryInvalid(RuntimeError):
    """An exactness invariant failed and the containing tranche is invalid."""


def _restore_terminal_sim(state: dict[str, Any]) -> Any:
    """Unpickle the evaluator's dynamically generated terminal GridBatchSim."""
    from src.simd_env import eval_engine

    cls = eval_engine._terminal_hero_sim_class("grid")
    value = cls.__new__(cls)
    value.__dict__.update(state)
    return value


def _reduce_terminal_sim(value: Any) -> tuple[Callable[..., Any], tuple[dict[str, Any]]]:
    return _restore_terminal_sim, (dict(value.__dict__),)


class _GateFrozen:
    """B2-compatible facade over the evaluator's real runtime/opponent carry."""

    def __init__(self, runtime: Any, opponents: list[Any]) -> None:
        self.runtime = runtime
        self.vector = {index + 1: policy for index, policy in enumerate(opponents)}

    def actions(self, sim: Any, actions: np.ndarray) -> None:
        from src.simd_env.eval_engine import _dispatch_actions

        self.runtime.prepare(sim)
        _dispatch_actions(
            sim, sim.get_resolved_action_mask(), actions, [], [list(self.vector.values())]
        )

    def observe_step(self, sim: Any) -> None:
        self.runtime.observe_step(sim)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["vector"] = {}
        memo: dict[int, Any] = {}
        for key, policy in self.vector.items():
            if id(policy) in memo:
                state["vector"][key] = memo[id(policy)]
                continue
            saved = copy.copy(policy)
            if hasattr(saved, "_torch"):
                saved._torch = None
            if hasattr(saved, "_rule_config_checked"):
                saved._rule_config_checked = None
            state["vector"][key] = saved
            memo[id(policy)] = saved
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import torch

        self.__dict__.update(state)
        for policy in self.vector.values():
            if hasattr(policy, "_torch"):
                policy._torch = torch


class GateDriver(WatchDriver):
    """Terminal H5000 T3 world exposed at the same prepared selector seam."""

    def __init__(self, plan: dict[str, Any], b8: Path, frp3: Path) -> None:
        from research.t3_rollout_labels_20261015.bank_engine import build_world
        from src.simd_env import eval_engine

        if digest(b8) != ceiling.B8_SHA256 or digest(frp3) != FRP3:
            raise ValueError("F0 requires pinned B8 s3 and FRP3-s12 identities")
        from research.redesign_scope_20261007 import grid_h5000_identity as gid
        from research.t1_commitment_20261015.replay import configure_cpu

        configure_cpu()
        plan = copy.deepcopy(plan)
        plan["specs"] = [
            [kind, Path(ref).name if kind == "checkpoint" else ref] for kind, ref in plan["specs"]
        ]
        plan = ceiling.portable_plan(plan, gid.CHECKPOINT_DIR)
        world = build_world("b8_s3", plan, hero_checkpoint=b8)
        from research.t34_shadow.shadow import clone_world_exact

        world = clone_world_exact(world)
        # The evaluator class is generated lazily; make that exact existing
        # class module-resolvable before trusted B2 snapshot pickling.
        cls = eval_engine._terminal_hero_sim_class("grid")
        setattr(eval_engine, cls.__name__, cls)
        copyreg.pickle(type(world.sim), _reduce_terminal_sim)
        self.sim = world.sim
        self.frozen = _GateFrozen(world.runtime, world.opponents)
        self.hero = world.hero
        self.stats = __import__("research.bl_k1.native", fromlist=["RowStats"]).RowStats()
        self.eligible_opponent_rows = 0
        self.decision = 0
        self.life = 0
        self.awaiting_respawn = False

    @classmethod
    def restore(cls, payload: bytes) -> "GateDriver":
        # B2's decoder restores trusted config/RNG/grid aliases.  Re-tagging is
        # safe because the payload contains the Gate facade, not a Watch roster.
        value = WatchDriver.restore(payload)
        value.__class__ = cls
        return value

    def step(self, forced: int | None = None) -> dict[str, Any]:
        # The frozen gate roster is inherited unchanged from L2'. Its policies
        # have no enabled v8 veto seats; count only actual eligible v8 rows.
        before = self.eligible_opponent_rows
        event = super().step(forced)
        self.eligible_opponent_rows = before
        return event


@dataclass(frozen=True)
class _TapeEntry:
    decision: int
    life: int
    frame: int
    length: int
    action: int
    q: tuple[float, ...]
    legal: tuple[int, ...]
    digest: str
    observation: dict[str, np.ndarray]
    actions: tuple[int, ...] = ()
    env_digest: str | None = None
    post_length: int = 0


def _platform_tag() -> str:
    """Return a concise platform label without claiming cross-host identity."""
    machine = platform.machine().lower()
    return (
        "mac-arm64"
        if machine in {"arm64", "aarch64"} and platform.system() == "Darwin"
        else machine
    )


def _selection_hash(namespace: str, *parts: object) -> str:
    payload = "/".join(("bl-labels-f0/v1", namespace, *(str(x) for x in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def _digest_driver(driver: WatchDriver, checkpoint: str) -> str:
    """Use the shared full digest with the B2 driver's actual carry objects.

    ``WatchDriver`` predates ``CapturedWorld`` and stores its five policies in
    ``frozen.vector``.  The adapter is intentionally structural only: it lets
    the established digest bind the same sim, hero, opponent carry, and global
    RNG, rather than creating a weaker parallel digest format.
    """
    import torch

    frozen = driver.frozen
    world = SimpleNamespace(
        sim=driver.sim,
        runtime=SimpleNamespace(
            frozen=frozen,
            decision=driver.decision,
            life=driver.life,
            awaiting_respawn=driver.awaiting_respawn,
        ),
        hero=driver.hero,
        opponents=list(frozen.vector.values()),
    )
    return full_digest(
        world,
        checkpoint={
            "b8": checkpoint,
            "numpy_rng": np.random.get_state(),
            "torch_rng": bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        },
    )


def _raw_observation(driver: WatchDriver) -> dict[str, np.ndarray]:
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    value = build_ego_raster(
        driver.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    return {key: np.asarray(value[key][0]).copy() for key in ("local", "global", "scalars")}


def _env_digest(sim: Any) -> str:
    """Use the established evaluator digest for replay-only platform audits."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    return str(gid.env_digest(sim, 0))


def _serial_observation(value: dict[str, np.ndarray]) -> dict[str, Any]:
    """Keep T5 raw planes and scalar dtype values; JSON encoding happens above us."""
    return {key: np.asarray(item).tolist() for key, item in value.items()}


def _restore_root(
    snapshots: dict[int, bytes],
    tape: dict[int, _TapeEntry],
    root: _TapeEntry,
    checkpoint: str,
    *,
    restore: Callable[[bytes], WatchDriver] = WatchDriver.restore,
) -> WatchDriver:
    """Restore the nearest prepared snapshot and replay the recorded bare tape."""
    anchors = [decision for decision in snapshots if decision <= root.decision]
    if not anchors:
        raise FactoryInvalid(f"missing snapshot before root {root.decision}")
    driver = restore(snapshots[max(anchors)])
    replayed = False
    while driver.decision < root.decision:
        source = tape.get(driver.decision)
        if source is None:
            raise FactoryInvalid(f"missing action tape at decision {driver.decision}")
        driver.step(source.action)
        replayed = True
        if driver.decision < root.decision:
            driver.prepare()
    if replayed:
        driver.prepare()
    observed = _digest_driver(driver, checkpoint)
    if observed != root.digest:
        raise FactoryInvalid(
            f"digest mismatch at root {root.decision}: expected {root.digest}, observed {observed}"
        )
    return driver


def _branch(
    snapshot: bytes,
    action: int,
    depth: int,
    *,
    restore: Callable[[bytes], WatchDriver] = WatchDriver.restore,
    pulse: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Evaluate one reacting exact branch through the existing B2 driver."""
    from research.bl_served_search.b2_engine import continue_branch

    masses: list[int] = []

    def recording_restore(payload: bytes) -> WatchDriver:
        driver = restore(payload)
        step = driver.step

        def record(forced: int | None = None) -> dict[str, Any]:
            event = step(forced)
            masses.append(int(event["hero_length"]))
            return event

        driver.step = record
        return driver

    outcome = continue_branch(snapshot, action, depth, restore=recording_restore, pulse=pulse)
    h300 = dict(outcome["H300"])
    end = outcome["continuation"] if h300["survives"] else outcome["death"]
    return {
        "survives": bool(h300["survives"]),
        "frames_executed": int(h300["frames_executed"]),
        "death_decision": None if h300["survives"] else int(end["decision"]),
        "final_length": int(masses[min(len(masses), HORIZON) - 1]),
        "mass_integral": sum(masses[:HORIZON]) / HORIZON,
        "native_v8": outcome["native_v8"],
        "extended": (
            {"survives": bool(outcome["continuation"] and outcome["continuation"]["survives"])}
            if depth == EXTENDED_HORIZON
            else None
        ),
    }


def _complete_outcomes(
    snapshot: bytes, actions: Iterable[int], depth: int
) -> dict[int, dict[str, Any]]:
    return {int(action): _branch(snapshot, int(action), depth) for action in sorted(set(actions))}


def _row(
    root: _TapeEntry,
    outcomes: dict[int, dict[str, Any]],
    *,
    death_decision: int,
    stream: str,
    platform_tag: str,
    forced_complete: bool,
    audit_selected: bool,
) -> dict[str, Any]:
    status, died, horizons, masses = [], [], [], []
    for action in range(6):
        outcome = outcomes.get(action)
        if outcome is None:
            status.append(exact_stop.NOT_EVALUATED)
            died.append(None)
            horizons.append(None)
            masses.append(None)
        else:
            status.append("EXACT")
            died.append(not bool(outcome["survives"]))
            horizons.append(int(outcome["frames_executed"]))
            masses.append(outcome.get("mass_integral"))
    survivors = [action for action, value in outcomes.items() if bool(value["survives"])]
    bare_died = bool(died[root.action]) if root.action in outcomes else True
    decisive = bare_died and bool(survivors)
    return {
        "schema": "bl-labels-t5-row/v1",
        "world": 0,
        "life": root.life,
        "frame": root.frame,
        "decision": root.decision,
        "length": root.length,
        "roster": stream,
        "platform": platform_tag,
        "legal": [action in root.legal for action in range(6)],
        "bare": root.action,
        "q": list(root.q),
        "died": died,
        "horizon": horizons,
        "mass": masses,
        "branch_death_decision": [
            outcomes[action].get("death_decision") if action in outcomes else None
            for action in range(6)
        ],
        "final_length": [
            outcomes[action].get("final_length") if action in outcomes else None
            for action in range(6)
        ],
        "alive_steps": [
            None if value is None else (int(value) if not dead else max(0, int(value) - 1))
            for value, dead in zip(horizons, died)
        ],
        "status": status,
        "decisive": decisive,
        "a_star": ceiling._choose(outcomes, root.action, np.asarray(root.q)),
        "death_decision": death_decision,
        "complete_six_action": forced_complete,
        "audit_selected": audit_selected,
        **_serial_observation(root.observation),
    }


def _select_roots(entries: list[_TapeEntry], death: _TapeEntry, namespace: str) -> list[_TapeEntry]:
    """Select doom roots at exclusive endpoint f=e+1; ordinary roots are separate."""
    endpoint = death.decision + 1
    life_entries = [
        entry for entry in entries if entry.life == death.life and entry.decision < endpoint
    ]
    doomed = [entry for entry in life_entries if entry.decision >= endpoint - ROOT_LOOKBACK]
    grid = [
        entry
        for entry in doomed
        if (entry.decision - (endpoint - ROOT_LOOKBACK)) % ROOT_STRIDE == 0
    ]
    tail = [entry for entry in doomed if entry.decision >= endpoint - LAST_CONTIGUOUS]
    selected = {entry.decision: entry for entry in [*grid, *tail]}
    return [selected[key] for key in sorted(selected)][:ROOT_CAP]


def make_driver(allocation: dict[str, Any], b8: Path, frp3: Path) -> WatchDriver:
    """Construct an untouched public population, with deterministic worker RNG."""
    import random

    import torch

    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    seed = int(allocation["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if allocation["stream"] == "watch6":
        return WatchDriver(seed, b8, frp3)
    if allocation["stream"] not in {"gate-frozen", "gate-mixed", "gate-scripted"}:
        raise ValueError("foreign F0 stream")
    return GateDriver(allocation["plan"], b8, frp3)


def collect_unit(
    allocation: dict[str, Any], b8: Path, frp3: Path, namespace: str
) -> dict[str, Any]:
    """Collect natural lives; restore only selected roots and roll alternatives.

    ``decision`` is the zero-based transition index. A fatal transition e ends
    the life at exclusive endpoint f=e+1, so the bare event interval is f-d.
    The loop includes maintenance/dead frames in the frozen session horizon.
    Full snapshots remain in a 420-decision ring plus eight safe reservoirs.
    Exported action audits rebuild the original world, avoiding repeated model
    weights in exports. No counterfactual uses an opponent action tape.
    """
    if digest(b8) != ceiling.B8_SHA256 or digest(frp3) != FRP3:
        raise ValueError("F0 requires pinned B8 s3 and FRP3-s12 identities")
    if not namespace or type(allocation.get("seed")) is not int:
        raise ValueError("F0 requires namespace and integer seed")
    public = {
        key: copy.deepcopy(value)
        for key, value in allocation.items()
        if key not in {"publish_life", "progress", "deadline_monotonic", "skip_lives"}
    }
    stream = public["stream"]
    selection_namespace = namespace + "/" + public["allocation_id"]
    parent_split = (
        "audit-sealed"
        if int(_selection_hash(namespace, "split", public["allocation_id"]), 16) % 5 == 0
        else "train"
    )
    horizon = int(public.get("horizon", 14400 if stream == "watch6" else 5000))
    if horizon <= 0:
        raise ValueError("positive F0 horizon required")
    publish = allocation.get("publish_life", lambda _value: None)
    progress = allocation.get("progress", lambda _value: None)
    skip = set(allocation.get("skip_lives", ()))
    deadline = allocation.get("deadline_monotonic")
    started = last_progress = time.monotonic()
    stages = dict.fromkeys(
        ("play", "restore_replay", "doom_alternatives", "ordinary", "audit", "export"), 0.0
    )
    driver = make_driver(public, Path(b8), Path(frp3))
    stages["play"] += time.monotonic() - started
    checkpoint = digest(b8)
    restore = type(driver).restore
    ring: dict[int, _TapeEntry] = {}
    snapshots: dict[int, bytes] = {}
    # Safe root -> (entry, anchor snapshot, prefix tape, natural H300 outcome).
    safe: dict[int, tuple[_TapeEntry, int, bytes, dict[int, _TapeEntry], dict[str, Any]]] = {}
    all_actions: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    audit_candidates: dict[str, dict[str, Any]] = {}
    counts = dict.fromkeys(
        (
            "deaths",
            "roots",
            "digest_checks",
            "bare_rerolls",
            "bare_audit_mismatches",
            "h1440",
            "reversals",
            "rescue_bearing_lives",
            "decisive_rows",
        ),
        0,
    )
    max_snapshots = 0
    eligible_safe = 0
    native_totals: dict[str, int] = {}
    snapshot_events: list[dict[str, Any]] = []

    def pulse() -> None:
        nonlocal last_progress
        now = time.monotonic()
        if deadline is not None and now >= float(deadline):
            raise RuntimeError("INCOMPLETE F0 unit wall cap")
        if now - last_progress >= 110:
            progress({"done_decisions": driver.decision, "total_decisions": horizon})
            last_progress = now

    def outcome_from_tape(root: _TapeEntry, end: int, fatal: bool) -> dict[str, Any]:
        stop = min(end, root.decision + HORIZON)
        values = [ring[d].post_length for d in range(root.decision, stop)]
        return {
            "survives": not fatal,
            "frames_executed": stop - root.decision,
            "death_decision": end - 1 if fatal else None,
            "final_length": values[-1],
            "mass_integral": sum(values) / HORIZON,
        }

    def remember_audit(root: _TapeEntry) -> None:
        key = _selection_hash(
            namespace, public["allocation_id"], "audit-root", root.life, root.decision
        )
        if len(audit_candidates) >= 64 and key >= max(audit_candidates):
            return
        audit_candidates[key] = {
            "root_id": key,
            "initial_allocation": public,
            "checkpoint_sha256": checkpoint,
            "root_decision": root.decision,
            "root_digest": root.digest,
            "tape_end": len(all_actions),
        }
        if len(audit_candidates) > 64:
            del audit_candidates[max(audit_candidates)]

    while driver.decision < horizon:
        pulse()
        play_start = time.monotonic()
        driver.prepare()
        if stream != "watch6" and not bool(driver.sim.alive[0, 0]):
            break
        alive = bool(driver.sim.alive[0, 0])
        entry = None
        if alive:
            legal, bare = driver.legal_and_bare()
            entry = _TapeEntry(
                driver.decision,
                driver.life,
                int(driver.sim.frame[0]),
                int(driver.sim.length[0, 0]),
                bare,
                tuple(float(x) for x in driver.last_root_q),
                tuple(legal),
                _digest_driver(driver, checkpoint),
                _raw_observation(driver),
            )
            ring[entry.decision] = entry
            if driver.decision % SNAPSHOT_STRIDE == 0 or not snapshots:
                snapshots[driver.decision] = driver.snapshot()
            # At prepared d, root d-300 has exactly 300 living transitions.
            old = ring.get(driver.decision - HORIZON)
            if old is not None and old.life == driver.life and old.life not in skip:
                eligible_safe += 1
                key = _selection_hash(selection_namespace, "preserve", old.life, old.decision)
                worst = (
                    max(
                        safe,
                        key=lambda d: _selection_hash(selection_namespace, "preserve", old.life, d),
                    )
                    if safe
                    else None
                )
                if len(safe) < PRESERVATION_ROOTS or key < _selection_hash(
                    selection_namespace, "preserve", old.life, worst
                ):
                    anchor = max(d for d in snapshots if d <= old.decision)
                    prefix = {d: value for d, value in ring.items() if anchor <= d <= old.decision}
                    safe[old.decision] = (
                        old,
                        anchor,
                        snapshots[anchor],
                        prefix,
                        outcome_from_tape(old, old.decision + HORIZON, False),
                    )
                    if len(safe) > PRESERVATION_ROOTS:
                        del safe[worst]
        event = driver.step()
        all_actions.append(
            {
                "decision": int(event["decision"]),
                "actions": event["actions"],
                "hero_q": list(entry.q) if entry else None,
                "env_digest": _env_digest(driver.sim),
                "legal": list(entry.legal) if entry else [],
                "length": entry.length if entry else 0,
                "full_digest": (
                    entry.digest if entry and entry.decision % SNAPSHOT_STRIDE == 0 else None
                ),
            }
        )
        if entry is not None:
            entry = replace(
                entry,
                actions=tuple(event["actions"]),
                env_digest=all_actions[-1]["env_digest"],
                post_length=int(event["hero_length"]),
            )
            ring[entry.decision] = entry
        stages["play"] += time.monotonic() - play_start
        # Ring plus ordinary reservoirs: storage independent of session length.
        cutoff = driver.decision - 420
        ring = {d: value for d, value in ring.items() if d >= cutoff}
        snapshots = {d: value for d, value in snapshots.items() if d >= cutoff - SNAPSHOT_STRIDE}
        max_snapshots = max(max_snapshots, len(snapshots) + len(safe))
        if entry is None or event["death_cause"] == "none":
            continue
        counts["deaths"] += 1
        native_before = driver_stats(driver)
        for key, value in native_before.items():
            if type(value) is int:
                native_totals[key] = native_totals.get(key, 0) + value
        death_payload = driver.snapshot()
        snapshot_events.append(
            {
                "decision": driver.decision,
                "life": entry.life,
                "sha256": hashlib.sha256(death_payload).hexdigest(),
                "kind": "death",
            }
        )
        life_rows: list[dict[str, Any]] = []
        life_counts = dict.fromkeys(counts, 0)
        try:
            if entry.life not in skip:
                doomed = _select_roots(list(ring.values()), entry, namespace)
                ordinary_count = len(safe)
                for root in doomed:
                    pulse()
                    t = time.monotonic()
                    restored = _restore_root(snapshots, ring, root, checkpoint, restore=restore)
                    payload = restored.snapshot()
                    stages["restore_replay"] += time.monotonic() - t
                    life_counts["digest_checks"] += 1
                    natural = outcome_from_tape(root, entry.decision + 1, True)
                    complete_audit = (
                        int(
                            _selection_hash(
                                selection_namespace, "complete", root.life, root.decision
                            ),
                            16,
                        )
                        % 10
                        == 0
                    )
                    complete = (
                        root.decision >= entry.decision + 1 - LAST_CONTIGUOUS or complete_audit
                    )
                    actions = sorted(set((*root.legal, root.action)))
                    t = time.monotonic()

                    def evaluate(a: int) -> dict[str, Any]:
                        if a == root.action:
                            return natural
                        return _branch(payload, a, HORIZON, restore=restore, pulse=pulse)

                    if complete:
                        outcomes = {a: evaluate(a) for a in actions}
                        chosen = ceiling._choose(outcomes, root.action, np.asarray(root.q))
                    else:
                        result = exact_stop.run(
                            actions, root.action, np.asarray(root.q), evaluate, ceiling._choose
                        )
                        outcomes = {a: v for a, v in result["outcomes"].items() if "survives" in v}
                        chosen = result["chosen"]
                    stages["doom_alternatives"] += time.monotonic() - t
                    row = _row(
                        root,
                        outcomes,
                        death_decision=entry.decision,
                        stream=stream,
                        platform_tag=_platform_tag(),
                        forced_complete=complete,
                        audit_selected=complete_audit,
                    )
                    row.update(
                        a_star=int(chosen),
                        digest=root.digest,
                        sampling_prob=1.0,
                        sampling_kind="doom",
                        natural_prevalence_weight=1.0,
                    )
                    t = time.monotonic()
                    if (
                        int(
                            _selection_hash(
                                selection_namespace, "bare-audit", root.life, root.decision
                            ),
                            16,
                        )
                        % 10
                        == 0
                    ):
                        life_counts["bare_rerolls"] += 1
                        actual = _branch(
                            payload, root.action, HORIZON, restore=restore, pulse=pulse
                        )
                        if any(
                            actual[k] != natural[k]
                            for k in (
                                "survives",
                                "frames_executed",
                                "death_decision",
                                "final_length",
                            )
                        ):
                            raise FactoryInvalid(f"bare audit mismatch at {root.decision}")
                    if (
                        row["decisive"]
                        and int(
                            _selection_hash(
                                selection_namespace, "extended", root.life, root.decision
                            ),
                            16,
                        )
                        % 10
                        == 0
                    ):
                        long = _branch(
                            payload, int(chosen), EXTENDED_HORIZON, restore=restore, pulse=pulse
                        )
                        row["h1440"] = {
                            "action": int(chosen),
                            "survives": long["extended"]["survives"],
                        }
                        life_counts["h1440"] += 1
                        life_counts["reversals"] += int(not row["h1440"]["survives"])
                    stages["audit"] += time.monotonic() - t
                    remember_audit(root)
                    life_rows.append(row)
                # Every safe root in this life has the same bottom-k hash inclusion
                # probability. No outcome-based acceptance or prefix mutation.
                for root, anchor, payload, prefix, natural in safe.values():
                    pulse()
                    t = time.monotonic()
                    restored = _restore_root(
                        {anchor: payload}, prefix, root, checkpoint, restore=restore
                    )
                    restored_payload = restored.snapshot()
                    life_counts["digest_checks"] += 1
                    outcomes = {root.action: natural}
                    for action in root.legal:
                        if action != root.action:
                            outcomes[action] = _branch(
                                restored_payload, action, HORIZON, restore=restore, pulse=pulse
                            )
                    row = _row(
                        root,
                        outcomes,
                        death_decision=entry.decision,
                        stream=stream,
                        platform_tag=_platform_tag(),
                        forced_complete=True,
                        audit_selected=False,
                    )
                    row.update(
                        a_star=ceiling._choose(outcomes, root.action, np.asarray(root.q)),
                        digest=root.digest,
                        sampling_prob=min(1.0, ordinary_count / max(1, eligible_safe)),
                        sampling_kind="ordinary",
                        natural_prevalence_weight=min(5.0, eligible_safe / max(1, ordinary_count)),
                    )
                    life_rows.append(row)
                    stages["ordinary"] += time.monotonic() - t
                life_counts.update(
                    deaths=1,
                    roots=len(life_rows),
                    decisive_rows=sum(row["decisive"] for row in life_rows),
                    rescue_bearing_lives=int(any(row["decisive"] for row in life_rows)),
                )
                for row in life_rows:
                    row.update(
                        world=public["allocation_id"],
                        parent_world=public["allocation_id"],
                        namespace=namespace,
                        stream="S-Watch6" if stream == "watch6" else "S-gate",
                        roster=stream,
                        code_sha=__import__("os").environ.get("F0_COMMIT", "local-smoke"),
                        b8_sha256=checkpoint,
                        split=parent_split,
                    )
                for key, value in life_counts.items():
                    if key != "deaths":
                        counts[key] += value
                t = time.monotonic()
                publish(
                    {
                        "schema": "bl-labels-f0-life/v1",
                        "status": "COMPLETE",
                        "namespace": namespace,
                        "allocation_id": public["allocation_id"],
                        "allocation": public,
                        "life": entry.life,
                        "death_decision": entry.decision,
                        "rows": life_rows,
                        "counters": life_counts,
                        "provenance": {
                            "platform": _platform_tag(),
                            "b8_sha256": checkpoint,
                            "native_v8": native_before,
                        },
                        "timing_seconds": dict(stages),
                    }
                )
                stages["export"] += time.monotonic() - t
                all_rows.extend(life_rows)
        finally:
            driver = restore(death_payload)
            safe.clear()
            ring.clear()
            snapshots.clear()
            eligible_safe = 0
        if stream != "watch6":
            break
    remaining = driver_stats(driver)
    attest(remaining)
    for key, value in remaining.items():
        if type(value) is int:
            native_totals[key] = native_totals.get(key, 0) + value
    attest(native_totals)
    suffix = [entry for entry in ring.values() if driver.decision - entry.decision < HORIZON]
    unknown = [
        {
            "decision": entry.decision,
            "life": entry.life,
            "status": "UNKNOWN",
            "reason": "alive_end_of_session_suffix_shorter_than_H300",
        }
        for entry in suffix
    ]
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "namespace": namespace,
        "allocation": public,
        "rows": all_rows,
        "audit_roots": list(audit_candidates.values()),
        "audit_tape": all_actions,
        "snapshots": snapshot_events,
        "counters": counts,
        "unknown_suffix": unknown,
        "timing_seconds": stages,
        "provenance": {
            "b8_sha256": checkpoint,
            "frp3_sha256": digest(frp3),
            "platform": _platform_tag(),
            "native_v8": native_totals,
            "snapshot_stride": 20,
            "max_live_snapshots": max_snapshots,
            "root_lookback": 300,
            "root_cap": 90,
            "digest_schema": "t34-shadow-full-digest/v3+F0-B2-carry",
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    publish(
        {
            **result,
            "schema": "bl-labels-f0-session/v1",
            "allocation_id": public["allocation_id"],
            "rows": [],
        }
    )
    return result
