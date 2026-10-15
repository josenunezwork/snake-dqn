"""Part D collection helpers for dense rollout labels.

This module deliberately owns the collection-side evaluation wrapper.  It does
not alter the evaluator, T1 replay, or the learner: mixed/frozen worlds retain
the evaluator's Vector61 runtime and call its ``prepare``/``observe_step``
hooks around every transition.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.b8_death_census_20261014 import census

HORIZON = 5000
DEPTH = 300
NAMESPACE = "t3-rollout-labels-dense/v1"
SEED_DOMAIN = "t3-rollout-labels-20261015"
MIX_CYCLE = ("scripted",) * 27 + ("mixed",) * 9 + ("frozen",) * 4


def sha256(path: Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh_seeds(count: int) -> list[int]:
    """Derive a dedicated Part-D namespace, never a numbered dev-probe round."""
    from research.apex_safety_20260926 import dev_screen

    values = dev_screen.screen_seeds(count, SEED_DOMAIN, NAMESPACE)
    if len(values) != len(set(values)):
        raise RuntimeError("Part-D namespace has an internal seed collision")
    return [int(value) for value in values]


def concrete_seed_hits(root: Path, candidates: Iterable[int]) -> set[int]:
    """Return exact candidate seeds already present anywhere below ``root``.

    This deliberately has no declaration exclusions.  A candidate cannot enter
    the Part-D allocation when its concrete integer appears in either the
    repository or artifact tree, irrespective of why it appeared there.
    """
    import os
    import subprocess
    import tempfile

    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"concrete-seed scan root is not a directory: {root}")
    values = [int(seed) for seed in candidates]
    if len(values) != len(set(values)):
        raise ValueError("concrete-seed scan candidates must be unique")
    if not values:
        return set()
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    try:
        handle.write("\n".join(str(seed) for seed in values) + "\n")
        handle.close()
        result = subprocess.run(
            [
                "rg",
                "--no-filename",
                "--only-matching",
                "--fixed-strings",
                "--word-regexp",
                "--file",
                handle.name,
                str(root),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"concrete-seed scan failed for {root}: exit {result.returncode}")
        tokens = {str(seed): seed for seed in values}
        output = set(result.stdout.splitlines())
        unexpected = output - set(tokens)
        if unexpected:
            raise RuntimeError(
                f"concrete-seed scan returned unexpected values for {root}: {unexpected}"
            )
        matches = {tokens[token] for token in output}
        return matches
    finally:
        if not handle.closed:
            handle.close()
        os.unlink(handle.name)


def allocate_unseen_seeds(
    count: int, scan_roots: Sequence[Path]
) -> tuple[list[int], dict[str, Any]]:
    """Choose ``count`` dedicated seeds absent from every pre-allocation scan root.

    Candidate prefixes are derived deterministically from the Part-D namespace.
    The initial twofold prefix absorbs ordinary 32-bit collisions; if needed,
    later prefixes grow deterministically without ever reusing a rejected seed.
    """
    if count < 1:
        raise ValueError("seed allocation count must be positive")
    roots = [Path(root).resolve() for root in scan_roots]
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("seed allocation requires distinct scan roots")
    prefix, considered, selected = count * 2, set(), []
    rejected: dict[int, set[str]] = {}
    batches = []
    while len(selected) < count:
        derived = fresh_seeds(prefix)
        if len(derived) != prefix or len(derived) != len(set(derived)):
            raise RuntimeError("dedicated seed derivation did not return a unique full prefix")
        candidates = [seed for seed in derived if seed not in considered]
        if not candidates:
            raise RuntimeError("seed allocation made no deterministic candidate progress")
        considered.update(candidates)
        root_hits = {str(root): concrete_seed_hits(root, candidates) for root in roots}
        for seed in candidates:
            hit_roots = {root for root, values in root_hits.items() if seed in values}
            if hit_roots:
                rejected.setdefault(seed, set()).update(hit_roots)
            else:
                selected.append(seed)
                if len(selected) == count:
                    break
        batches.append(
            {
                "candidate_prefix": prefix,
                "new_candidates": len(candidates),
                "matched_by_root": {
                    root: sorted(int(seed) for seed in values) for root, values in root_hits.items()
                },
            }
        )
        prefix *= 2
    provenance = {
        "method": "rg-o-fixed-word-concrete-seed-rejection/v1",
        "scan_roots": [str(root) for root in roots],
        "requested_count": count,
        "candidate_batches": batches,
        "rejected_seeds": [
            {"seed": seed, "roots": sorted(hit_roots)}
            for seed, hit_roots in sorted(rejected.items())
        ],
        "selected_seed_count": len(selected),
    }
    return selected, provenance


def verify_fresh_namespace(
    root: Path, seeds: Iterable[int], *, declaration_exclusions: Iterable[Path] = ()
) -> dict[str, Any]:
    """Refuse a namespace whose source string or concrete seeds already appear.

    This is intentionally conservative: generated collection can only start
    after the repository/artifact search has a recorded zero-hit receipt.  The
    optional exclusions are exact declaration/transcript files, never output
    directories: world receipts, namespace receipts, and label data remain
    searchable even when they sit beside an excluded handoff document.
    """
    import os
    import subprocess
    import tempfile

    root = Path(root).resolve()
    excluded: dict[Path, str] = {}
    for raw_path in declaration_exclusions:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"freshness declaration exclusion is not a file: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"freshness declaration exclusion is outside scan root: {path}"
            ) from exc
        excluded[path] = str(relative)
    seeds = [int(seed) for seed in seeds]
    needles = [NAMESPACE, *(str(seed) for seed in seeds)]
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    try:
        handle.write("\n".join(needles) + "\n")
        handle.close()
        result = subprocess.run(
            [
                "rg",
                "-l",
                "--fixed-strings",
                "--word-regexp",
                "--file",
                handle.name,
                str(root),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"namespace search failed: exit {result.returncode}")
        matches = [
            Path(line).resolve() for line in result.stdout.splitlines() if ".git" not in line
        ]
    finally:
        if not handle.closed:
            handle.close()
        os.unlink(handle.name)
    found = [str(path.relative_to(root)) for path in matches if path not in excluded]
    excluded_matches = [str(path.relative_to(root)) for path in matches if path in excluded]
    hits = {"matching_files": found[:20]} if found else {}
    if hits:
        raise RuntimeError(f"Part-D namespace is not fresh: {hits}")
    from research.redesign_m2_20261008 import ni_spec

    exclusions = {
        "ni": ni_spec.ni_seeds(),
        "dev_round_52": ni_spec.distill_seeds(len(seeds), round_index=52),
        "dev_round_60": ni_spec.distill_seeds(len(seeds), round_index=60),
    }
    overlaps = {name: sorted(set(seeds) & set(values)) for name, values in exclusions.items()}
    if any(overlaps.values()):
        raise RuntimeError(f"Part-D seeds overlap an excluded namespace: {overlaps}")
    return {
        "namespace": NAMESPACE,
        "seed_domain": SEED_DOMAIN,
        "seed_count": len(seeds),
        "hits": {},
        "declaration_exclusions": [
            {"path": relative, "sha256": sha256(path)}
            for path, relative in sorted(excluded.items(), key=lambda item: item[1])
        ],
        "excluded_declaration_matches": sorted(excluded_matches),
        "excluded_namespace_overlap": overlaps,
    }


def cycle_jobs(cycles: int) -> list[dict[str, Any]]:
    """Return a balanced 40-world cycle for each of B8_s0..B8_s4."""
    if cycles < 1:
        raise ValueError("cycles must be positive")
    jobs = []
    for cycle in range(cycles):
        for arm in census.B8_ARMS:
            for index, mix in enumerate(MIX_CYCLE):
                jobs.append({"arm": arm, "mix": mix, "cycle": cycle, "index": index})
    return jobs


def full_horizon(frame: int, depth: int = DEPTH, horizon: int = HORIZON) -> bool:
    """Only label a selector snapshot with all D transitions before H5000.

    Watch selection has already advanced ``sim.frame`` by one, so a snapshot
    at frame 4701 is decision 4700 and its 300 transitions end at frame 5000.
    """
    return int(frame) <= int(horizon) - int(depth) + 1


def dense_eligible(trigger: Any, decision: int, frame: int, length: int, alive: bool) -> bool:
    """The addendum's L700/frame2000 dense trigger plus a full-horizon guard."""
    if not alive:
        return False
    fired = trigger.fires(decision, length, alive)
    return bool(fired and int(frame) >= 2000 and full_horizon(frame))


def classify(outcomes: dict[int, dict], b8_action: int) -> tuple[bool, np.ndarray]:
    """Return ``(decisive, surviving_set)`` for one all-legal-action rollout."""
    surviving = np.zeros(6, dtype=bool)
    for action, outcome in outcomes.items():
        surviving[int(action)] = bool(outcome["survives"])
    if int(b8_action) not in outcomes:
        raise ValueError("B8 action was not included in the tested legal outcomes")
    alternatives = [int(action) for action in outcomes if int(action) != int(b8_action)]
    decisive = not bool(surviving[int(b8_action)]) and any(
        surviving[action] for action in alternatives
    )
    counterweight = bool(surviving[int(b8_action)]) and any(
        not surviving[action] for action in alternatives
    )
    return decisive, surviving if decisive or counterweight else np.zeros(6, dtype=bool)


def label_set(surviving: np.ndarray, policy: str = "surviving-set") -> np.ndarray:
    """Apply a collection label policy without changing the rollout outcome set.

    ``prefer-nonboost`` is deliberately a smallest-sufficient-set policy: when
    normal-speed actions survived, boost actions are omitted; a boost-only
    escape remains fully represented.  The raw per-action outcomes (when
    requested) are never affected by this presentation choice.
    """
    values = np.asarray(surviving, dtype=bool)
    if values.shape != (6,):
        raise ValueError("surviving set must have shape (6,)")
    if policy == "surviving-set":
        return values.copy()
    if policy == "prefer-nonboost":
        normal = values.copy()
        if normal[:3].any():
            normal[3:] = False
        return normal
    raise ValueError(f"unknown label policy: {policy}")


def outcome_arrays(outcomes: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pack optional all-action branch outcomes with null deaths encoded as -1."""
    survived = np.zeros(6, dtype=bool)
    recorded = np.zeros(6, dtype=bool)
    death_frame = np.full(6, -1, dtype=np.int64)
    final_length = np.zeros(6, dtype=np.int64)
    for action, outcome in outcomes.items():
        if not isinstance(action, int):
            continue
        if not 0 <= action < 6:
            raise ValueError("outcome action outside [0, 5]")
        if "final_length" not in outcome:
            raise ValueError("recorded outcome requires final_length")
        recorded[action] = True
        survived[action] = bool(outcome["survives"])
        death = outcome.get("death_decision")
        if death is not None:
            death_frame[action] = int(death)
        final_length[action] = int(outcome.get("final_length", 0))
    return {
        "outcome_recorded": recorded,
        "outcome_survived": survived,
        "outcome_death_frame": death_frame,
        "outcome_final_length": final_length,
    }


def validate_b8_controls(
    controls: Sequence[tuple[int, bool, int]], death_decision: int | None, depth: int = DEPTH
) -> int:
    """Match every B8 first-action branch to the unchanged collection trajectory.

    A death at decision d+depth is outside the d..d+depth-1 continuation.
    This checks both survival and the exact fatal decision without an extra rollout.
    """
    for decision, survives, executed in controls:
        expected = death_decision is None or death_decision >= decision + depth
        if survives != expected:
            raise RuntimeError(f"B8 control survival mismatch at decision {decision}")
        if survives and executed != depth:
            raise RuntimeError(f"B8 control horizon mismatch at decision {decision}")
        if not survives and decision + executed - 1 != death_decision:
            raise RuntimeError(f"B8 control death frame mismatch at decision {decision}")
    return len(controls)


def counterweight_trim(
    rows: Sequence[dict[str, np.ndarray]], ratio: float = 1.0
) -> list[dict[str, np.ndarray]]:
    """Keep every decisive row and a fixed-seed uniform global counterweight sample."""
    decisive = [row for row in rows if bool(row["decisive"])]
    nondecisive = [row for row in rows if not bool(row["decisive"])]
    if not 0 <= ratio <= 1:
        raise ValueError("counterweight ratio must lie in [0, 1]")
    count = min(int(len(decisive) * ratio), len(nondecisive))
    if not count:
        return decisive
    indices = np.sort(
        np.random.default_rng(20261015).choice(len(nondecisive), count, replace=False)
    )
    return decisive + [nondecisive[int(index)] for index in indices]


@dataclass
class CapturedWorld:
    """A single freshly built evaluator world and its private runtime state."""

    sim: Any
    runtime: Any
    hero: Any
    opponents: list[Any]
    seed: int
    mix: str
    kinds: list[str]

    def clone(self) -> "CapturedWorld":
        """Clone sim, policy RNG state, and carried Vector61 state for one branch."""
        clone_runtime = copy.copy(self.runtime)
        # The collector temporarily installs an instance-level prepare hook to
        # snapshot a selector.  Branches must recover the class method, not
        # inherit that closure (which closes over the live world).
        if "prepare" in vars(clone_runtime):
            delattr(clone_runtime, "prepare")
        for name, value in vars(self.runtime).items():
            if isinstance(value, np.ndarray):
                setattr(clone_runtime, name, value.copy())
            elif name == "memory":
                setattr(clone_runtime, name, copy.deepcopy(value))
        clone = CapturedWorld(
            census.clone_sim(self.sim),
            clone_runtime,
            copy.copy(self.hero),
            [],
            self.seed,
            self.mix,
            list(self.kinds),
        )
        clone.sim._policy_phase_prepared = True
        clone.sim._policy_step_in_progress = False
        if hasattr(clone.sim, "_active_env_mask"):
            del clone.sim._active_env_mask
        # Policies only carry immutable eval models plus the vector runtime or
        # deterministic profile anchor state.  Copy their dictionaries without
        # cloning Torch modules/kernels, then attach the cloned runtime.
        for policy in [clone.hero, *(copy.copy(policy) for policy in self.opponents)]:
            if hasattr(policy, "__dict__"):
                policy.__dict__ = dict(policy.__dict__)
            if hasattr(policy, "runtime"):
                policy.runtime = clone.runtime
            clone.opponents.append(policy) if policy is not clone.hero else None
        return clone


def planned_rosters(
    context: dict[str, Any], seeds: Sequence[int], mixes: Sequence[str]
) -> list[dict[str, Any]]:
    """Materialize the entire ordered bank once so pool rotation stays intact."""
    if len(seeds) != len(mixes) or len(seeds) != len(set(int(seed) for seed in seeds)):
        raise ValueError("planned roster seeds/mixes must be ordered one-to-one")
    rows = context["ds"]._design_rows(list(seeds))
    wanted = {(int(seed), str(mix)) for seed, mix in zip(seeds, mixes)}
    matches = {
        (int(row["world_seed"]), row["mix"]): row
        for row in rows
        if (int(row["world_seed"]), row["mix"]) in wanted
    }
    if len(matches) != len(wanted):
        raise RuntimeError("fresh world roster construction was incomplete")
    planned = []
    for world_index, (seed, mix) in enumerate(zip(seeds, mixes)):
        row = matches[(int(seed), str(mix))]
        # Keep the persisted plan JSON-stable: evaluator agent specs accept a
        # two-item list just as they accept the source tuple.
        specs = [list(context["lookup"][slot["member_sha256"]]) for slot in row["slots"]]
        planned.append(
            {
                "world_index": world_index,
                "seed": int(seed),
                "mix": str(mix),
                "specs": specs,
                "member_sha256s": [str(slot["member_sha256"]) for slot in row["slots"]],
            }
        )
    return planned


def build_world(
    arm: str, plan: dict[str, Any], *, hero_checkpoint: Path | None = None
) -> CapturedWorld:
    """Capture an untouched evaluator world through its public construction path."""
    # The census context alone has only scripted rows.  The grid identity
    # context is the pinned all-mix roster/profile authority.
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env import eval_engine as ee

    context = gid._context(1)
    seed, mix, roster = int(plan["seed"]), str(plan["mix"]), list(plan["specs"])
    cls = ee._terminal_hero_sim_class("grid")
    original = cls.step_with_policy
    captured: dict[str, Any] = {}

    class StopCapture(Exception):
        pass

    def capture(sim: Any, selector: Any, *args: Any, **kwargs: Any) -> None:
        cells = dict(
            zip(
                selector.__code__.co_freevars, (cell.cell_contents for cell in selector.__closure__)
            )
        )
        captured.update(sim=sim, **cells)
        raise StopCapture

    cls.step_with_policy = capture
    try:
        ee.run_simd_eval(
            context["lookup"][gid.FRP3_S12[1]],
            roster,
            HORIZON,
            [seed],
            profile=context["profile"],
            opponent_specs_by_world={seed: roster},
            mix_id=mix,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            hero_ego2s=str(hero_checkpoint if hero_checkpoint is not None else census.b8_path(arm)),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
            vector61_backend="numba",
            scripted_anchors="fast",
        )
    except StopCapture:
        pass
    finally:
        cls.step_with_policy = original
    if not captured:
        raise RuntimeError("evaluator did not reach Part-D capture seam")
    return CapturedWorld(
        captured["sim"],
        captured["vector61_runtime"],
        captured["hero_policies"][0],
        captured["opp_policies"][0],
        seed,
        mix,
        [census.spec_kind(spec) for spec in roster],
    )


def step_prepared(world: CapturedWorld, forced: int) -> None:
    """Consume a clone captured inside Watch selection with a forced hero action."""
    from src.simd_env.eval_engine import _dispatch_actions

    sim = world.sim
    if not getattr(sim, "_policy_phase_prepared", False):
        raise RuntimeError("forced continuation must start from a prepared selector snapshot")
    actions = np.ones((1, sim.S), dtype=np.int64)
    _dispatch_actions(sim, sim.get_resolved_action_mask(), actions, [world.hero], [world.opponents])
    actions[0, 0] = int(forced)
    sim.step(actions)
    world.runtime.observe_step(sim)


def prepared_q(world: CapturedWorld, obs: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate stored decision-time observations with the exact rowwise B8 forward."""
    from src.simd_env.ego2s_policy import forward_q

    q = forward_q(world.hero.net, obs, "rowwise").numpy()[0].copy()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError("nonfinite or malformed B8 Q vector")
    return q


def choose_and_step(
    world: CapturedWorld, *, capture_obs: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """Take one ordinary exact Watch transition, respecting runtime hooks."""
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        advisory = prepared.get_advisory_action_mask()[0, 0].astype(bool)
        obs = (
            build_ego_raster(
                prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            if capture_obs
            else None
        )
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        selected.update(actions=actions, advisory=advisory, obs=obs)
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return selected["actions"], selected["advisory"], selected["obs"]


def rollout_all(
    world: CapturedWorld,
    legal: Sequence[int],
    depth: int = DEPTH,
    decision: int = 0,
    *,
    record_outcomes: bool = False,
) -> dict[int, dict[str, Any]]:
    """Exact D-step continuations for all legal first actions, including vector opponents."""
    if world.mix == "scripted":
        # Scripted evaluation worlds have no controlled Vector61 rows; this is
        # exactly the already-fidelity-tested Part-A K-way batch engine.
        from research.t1_commitment_20261015.replay import Replay
        from research.t3_rollout_labels_20261015.engine import batched_rollout

        prepared = Replay(world.sim, world.hero, [world.opponents], [world.seed], [world.kinds])
        outcomes = batched_rollout(
            prepared, legal, decision, depth, record_outcomes=record_outcomes
        )
        outcomes.pop("_batch_receipt")
        return outcomes
    outcomes: dict[int, dict[str, Any]] = {}
    for action in legal:
        branch = world.clone()
        executed = 0
        step_prepared(branch, int(action))
        executed += 1
        while executed < depth and bool(branch.sim.alive[0, 0]):
            choose_and_step(branch, capture_obs=False)
            executed += 1
        outcome = {
            "survives": bool(branch.sim.alive[0, 0]) and executed == depth,
            "frames_executed": executed,
        }
        if record_outcomes:
            outcome.update(
                death_decision=(
                    None if bool(branch.sim.alive[0, 0]) else int(decision) + executed - 1
                ),
                # Dead heroes contribute zero logical mass at the branch horizon.
                final_length=int(branch.sim.length[0, 0]) if bool(branch.sim.alive[0, 0]) else 0,
            )
        outcomes[int(action)] = outcome
    return outcomes


def write_rows(path: Path, rows: Sequence[dict[str, Any]], *, packed: bool = False) -> None:
    """Atomically write one loader-compatible raw or packed bv8 shard."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.stack([row[key] for row in rows])
        for key in ("local", "global", "scalars", "mask", "surviving")
    }
    arrays["action"] = np.asarray([row["action"] for row in rows], dtype=np.int64)
    arrays["weight"] = np.ones(len(rows), dtype=np.float32)
    for key, dtype in (
        ("q", np.float32),
        ("length", np.int64),
        ("outcome_recorded", bool),
        ("outcome_survived", bool),
        ("outcome_death_frame", np.int64),
        ("outcome_final_length", np.int64),
    ):
        if all(key in row for row in rows):
            arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
    if packed:
        from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

        layout = train_layout()
        arrays["packed"] = pack_ref(arrays.pop("local"), arrays.pop("global"), layout)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def write_world_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomically persist untrimmed rows, including the merge-only class bit."""
    if not rows:
        # An empty world still needs a durable, valid receipt payload.
        arrays = {
            "local": np.zeros((0, 7, 31, 31), np.uint8),
            "global": np.zeros((0, 4, 37, 37), np.uint8),
            "scalars": np.zeros((0, 60), np.float32),
            "mask": np.zeros((0, 6), bool),
            "surviving": np.zeros((0, 6), bool),
            "action": np.zeros(0, np.int64),
            "decisive": np.zeros(0, bool),
            "decision": np.zeros(0, np.int64),
            "frame": np.zeros(0, np.int64),
            "length": np.zeros(0, np.int64),
            "q": np.zeros((0, 6), np.float32),
        }
    else:
        arrays = {
            key: np.stack([row[key] for row in rows])
            for key in ("local", "global", "scalars", "mask", "surviving")
        }
        arrays["action"] = np.asarray([row["action"] for row in rows], dtype=np.int64)
        arrays["decisive"] = np.asarray([row["decisive"] for row in rows], dtype=bool)
        arrays["decision"] = np.asarray([row["decision"] for row in rows], dtype=np.int64)
        arrays["frame"] = np.asarray([row["frame"] for row in rows], dtype=np.int64)
        for key, dtype in (
            ("q", np.float32),
            ("length", np.int64),
            ("outcome_recorded", bool),
            ("outcome_survived", bool),
            ("outcome_death_frame", np.int64),
            ("outcome_final_length", np.int64),
        ):
            if all(key in row for row in rows):
                arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The Part-D report fields, computed after global counterweight trimming."""
    n = len(rows)
    survive = np.stack([row["surviving"] for row in rows]) if rows else np.zeros((0, 6), bool)
    actions = np.asarray([row["action"] for row in rows], dtype=np.int64)
    return {
        "rows": n,
        "decisive_rows": int(sum(bool(row["decisive"]) for row in rows)),
        "counterweight_rows": int(sum(not bool(row["decisive"]) for row in rows)),
        "surviving_action_density": float(survive.mean()) if n else 0.0,
        "rows_with_boost_survivor_fraction": float(survive[:, 3:].any(axis=1).mean()) if n else 0.0,
        "surviving_boost_share": float(survive[:, 3:].sum() / max(1, survive.sum())) if n else 0.0,
        "b8_agree": float(survive[np.arange(n), actions].mean()) if n else 0.0,
    }


def manifest(out: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Build a hash-bound bank manifest after all shard writes complete."""
    shards = sorted(out.glob("*.npz"))
    files = sorted([*shards, *(out.glob("bank-*.provenance.jsonl"))])
    for name in (
        "namespace.json",
        "legacy-resume.json",
        "prefilter.json",
        "progress.json",
        "death-window-resume.json",
        "death-window-validation.json",
        "death-window-validation-v2.json",
        "death-window-retained.json",
    ):
        if (out / name).exists():
            files.append(out / name)
    if (out / "calibration").is_dir():
        files.extend(sorted((out / "calibration").glob("*")))
    if (out / "source-history").is_dir():
        files.extend(sorted((out / "source-history").rglob("*")))
    receipts = sorted((out / "worlds").glob("*.json")) if (out / "worlds").is_dir() else []
    value = {
        "schema": "t3-label-bank/v1",
        "provenance": provenance,
        "shards": {p.name: sha256(p) for p in shards},
        "files": {p.relative_to(out).as_posix(): sha256(p) for p in files if p.is_file()},
        "world_receipts": {p.name: sha256(p) for p in receipts},
    }
    (out / "manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value
