"""Deterministic verify-then-expand continuation teacher for ExIt-B8.

This is intentionally an exact-simulator controller.  It takes a T3/T33
captured ``World`` at a prepared Watch boundary, verifies the student's greedy
continuation for H steps, and only expands legal alternatives when it dies.
The continuation loop advances all living branches once per tick and removes
finished branches from the active set, so metrics distinguish active work from
resident branch capacity.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from research.t34_exit_b8.policy import B8Policy, PolicyEvaluation, RolloutPolicy, action_grid

NUM_ACTIONS = 6


def _merged_policy_key(policy: Any, source_id: int, slot: int) -> tuple[Any, ...]:
    """Identify one source-slot policy without conflating its veto configuration."""
    return (
        int(source_id),
        type(policy),
        getattr(policy, "checkpoint_path", id(policy)),
        int(slot),
        getattr(policy, "veto_spec", None),
        tuple(sorted(getattr(policy, "veto_slots", frozenset()))),
    )


def _clone_merged_policy(policy: Any, runtime: Any, branch_envs: np.ndarray) -> Any:
    """Bind a source policy to a merged runtime with independent per-branch carry."""
    import copy

    clone = copy.copy(policy)
    state = dict(clone.__dict__)
    try:
        clone.__dict__ = state
    except AttributeError:
        clone.__dict__.clear()
        clone.__dict__.update(state)
    clone.runtime = runtime
    # Native veto dispatch must keep the source forward recipe.  Non-veto
    # policies retain the historical batched forward used by this branch path.
    if not getattr(policy, "veto_slots", frozenset()):
        clone.forward = "batched"
    for name in ("veto_counters", "boost_counters", "live_vetoes"):
        carry = getattr(policy, name, None)
        if not isinstance(carry, dict):
            continue
        setattr(
            clone,
            name,
            {
                int(env): copy.deepcopy(carry[0])
                for env in np.asarray(branch_envs, dtype=np.int64)
                if 0 in carry
            },
        )
    if hasattr(policy, "last_veto"):
        clone.last_veto = copy.deepcopy(policy.last_veto)
    return clone


@dataclass(frozen=True)
class TriggerConfig:
    """Configurable trajectory admission rules from FINAL_PLAN section 3."""

    confirm_interval: int = 8
    confirm_min_length: int = 600
    low_length_probability: float = 0.03
    death_window: int = 150
    max_rewinds: int = 2

    def __post_init__(self) -> None:
        if self.confirm_interval < 1 or self.confirm_min_length < 0:
            raise ValueError("trigger interval and length must be non-negative")
        if not 0.0 <= self.low_length_probability <= 1.0:
            raise ValueError("low_length_probability must be in [0, 1]")
        if self.death_window < 1 or not 0 <= self.max_rewinds <= 2:
            raise ValueError("invalid death-window rewind limits")

    def fires(self, *, decision: int, length: int, seed: int) -> bool:
        """Return a schedule-stable confirmation or 3% low-length trigger."""
        if length >= self.confirm_min_length:
            return decision % self.confirm_interval == 0
        return _unit_interval(seed, decision, "low-length") < self.low_length_probability


@dataclass(frozen=True)
class ExitTeacherConfig:
    """Teacher search settings.  H=300 and fp32 are the governed defaults."""

    horizon: int = 300
    beta: float = 1.0
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    fp32: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.beta not in (0.0, 0.5, 1.0):
            raise ValueError("beta must be one of 0, 0.5, 1")
        if not self.fp32:
            raise ValueError("T34 labels require fp32 policy evaluation")


@dataclass(frozen=True)
class ActionOutcome:
    """One action's exact H-step outcome; mass is zero after hero death."""

    survives: bool
    alive_steps: int
    mass: float
    row_steps: int = 0


@dataclass(frozen=True)
class DecisionDiagnostic:
    """Serializable label and work receipt for one searched student decision."""

    student_action: int
    action: int
    q: tuple[float, ...]
    legal: tuple[bool, ...]
    verified_survives: bool
    expanded: bool
    forbidden: tuple[int, ...]
    outcomes: dict[int, ActionOutcome]
    active_row_steps: int
    resident_row_steps: int
    nominal_row_steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without NumPy scalar leakage."""
        return asdict(self)


def _unit_interval(seed: int, decision: int, purpose: str) -> float:
    digest = hashlib.sha256(f"t34-exit-v1:{seed}:{decision}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def mix_action(
    student_action: int, teacher_action: int, *, seed: int, decision: int, beta: float
) -> int:
    """Apply the R1/R2/R3 teacher-action schedule without mutable RNG state."""
    if beta not in (0.0, 0.5, 1.0):
        raise ValueError("beta must be one of 0, 0.5, 1")
    if not (0 <= student_action < NUM_ACTIONS and 0 <= teacher_action < NUM_ACTIONS):
        raise ValueError("actions must be in [0, 5]")
    return teacher_action if _unit_interval(seed, decision, "beta") < beta else student_action


class ExitTeacher:
    """Exact H=300 verify-first teacher over prepared T3/T33 worlds.

    ``choose`` returns the *label* a-star.  Applying beta is deliberately left
    to trajectory collection through :func:`mix_action`, which makes the bank's
    behavioural policy explicit in its trace receipt.
    """

    def __init__(
        self,
        config: ExitTeacherConfig = ExitTeacherConfig(),
        *,
        policy: RolloutPolicy | None = None,
        device: str = "cpu",
    ) -> None:
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps, or cuda")
        self.config, self.policy, self.device = config, (policy or B8Policy(device=device)), device
        self._opponents = (
            self.policy if isinstance(self.policy, B8Policy) else B8Policy(device=device)
        )
        self.deadline: float | None = None
        self._profile: dict[str, int | float] = {
            "decisions": 0,
            "verifications": 0,
            "expansions": 0,
            "active_row_steps": 0,
            "resident_row_steps": 0,
            "nominal_row_steps": 0,
            "policy_s": 0.0,
            "sim_s": 0.0,
        }

    def profile(self, *, reset: bool = False) -> dict[str, int | float]:
        """Return additive active/resident counters and phase timing."""
        result = dict(self._profile)
        if reset:
            for key, value in self._profile.items():
                self._profile[key] = 0 if isinstance(value, int) else 0.0
        return result

    def evaluate(self, world: Any) -> PolicyEvaluation:
        """Expose the current student's exact decision-time Q values and legal mask."""
        started = time.perf_counter()
        result = self.policy.evaluate(world)
        self._profile["policy_s"] += time.perf_counter() - started
        return result

    def q(self, world: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return raw Q and B8's advisory selection mask for trajectory collection."""
        evaluation = self.evaluate(world)
        select = evaluation.legal if evaluation.select is None else evaluation.select
        return np.asarray(evaluation.q, dtype=np.float32), np.asarray(select, dtype=bool)

    def actions(self, world: Any, hero_action: int) -> np.ndarray:
        """Expose T3's six-snake dispatcher with the hero action overridden."""
        return action_grid(world, hero_action, device=self.device, cache=self._opponents)

    def choose(self, worlds: Sequence[Any]) -> tuple[np.ndarray, list[DecisionDiagnostic]]:
        """Verify each prepared world, expanding only those whose student action dies."""
        worlds = list(worlds)
        if not worlds:
            return np.empty(0, dtype=np.int64), []
        self._check_deadline()
        if any(hasattr(world.sim, "cfg") for world in worlds) and not callable(
            getattr(self.policy, "evaluate_rows", None)
        ):
            raise TypeError("rollout policy must implement evaluate and evaluate_rows")
        networks = {
            id(world.hero.net): world.hero.net for world in worlds if hasattr(world.hero, "net")
        }
        if len(networks) > 1:
            from research.t33_batched_teacher.teacher import _model_digest

            digests = {_model_digest(net) for net in networks.values()}
            if len(digests) != 1:
                raise ValueError("all roots must use identical student checkpoint weights")
        evaluations = [self.evaluate(world) for world in worlds]
        students = [evaluation.greedy_action() for evaluation in evaluations]
        if all(hasattr(world.sim, "cfg") for world in worlds):
            outcomes = self._rollout_merged(worlds, [[student] for student in students])
        else:
            outcomes = [self._rollout(world, [student]) for world, student in zip(worlds, students)]
        failed = [
            i
            for i, (result, student) in enumerate(zip(outcomes, students))
            if not result[student].survives
        ]
        if failed:
            choices = [
                [int(a) for a in np.flatnonzero(evaluations[i].legal) if int(a) != students[i]]
                for i in failed
            ]
            if all(hasattr(worlds[i].sim, "cfg") for i in failed):
                expanded = self._rollout_merged([worlds[i] for i in failed], choices)
            else:
                expanded = [
                    self._rollout(worlds[i], actions) for i, actions in zip(failed, choices)
                ]
            for index, result in zip(failed, expanded):
                outcomes[index].update(result)
        actions, diagnostics = [], []
        for i, (evaluation, student, result) in enumerate(zip(evaluations, students, outcomes)):
            action = self._select(evaluation, student, result)
            forbidden = tuple(
                a
                for a, outcome in sorted(result.items())
                if not outcome.survives and result[action].survives
            )
            active = sum(outcome.row_steps for outcome in result.values())
            resident = active
            diagnostics.append(
                DecisionDiagnostic(
                    student_action=student,
                    action=action,
                    q=tuple(float(q) for q in evaluation.q),
                    legal=tuple(bool(a) for a in evaluation.legal),
                    verified_survives=result[student].survives,
                    expanded=i in failed,
                    forbidden=forbidden,
                    outcomes=result,
                    active_row_steps=active,
                    resident_row_steps=resident,
                    nominal_row_steps=len(result) * self.config.horizon,
                )
            )
            actions.append(action)
        self._profile["decisions"] += len(worlds)
        self._profile["verifications"] += len(worlds)
        self._profile["expansions"] += len(failed)
        return np.asarray(actions, dtype=np.int64), diagnostics

    def choose_one(self, world: Any) -> tuple[int, DecisionDiagnostic]:
        """Return a-star and a complete per-action receipt without mutating ``world``."""
        actions, diagnostics = self.choose([world])
        return int(actions[0]), diagnostics[0]

    def _clone(self, world: Any) -> Any:
        if not hasattr(world, "clone"):
            raise TypeError("ExitTeacher requires a captured world with clone()")
        # Branches begin from the source's actual vector/v8 carry.  ``World.clone``
        # intentionally makes only a shallow policy-state copy for old T3 uses;
        # the shadow helper deep-copies mutable hook state while retaining frozen
        # network tensors by identity.
        from research.t34_shadow.shadow import clone_world_exact

        return clone_world_exact(world)

    def _rollout(self, world: Any, actions: Iterable[int]) -> dict[int, ActionOutcome]:
        """Run requested first-actions in one T33 merged batch when available.

        The fallback exists solely for hermetic tiny protocol fixtures; real
        captured GridBatchSim worlds always take the merged, active-mask path.
        """
        requested = [int(action) for action in actions]
        if hasattr(world.sim, "cfg"):
            return self._rollout_merged([world], [requested])[0]
        return self._rollout_serial({action: self._clone(world) for action in requested})

    def _rollout_serial(self, branches: dict[int, Any]) -> dict[int, ActionOutcome]:
        """Advance branch leaders in lockstep and compact dead branches each tick."""
        remaining = dict(branches)
        alive_steps = {action: 0 for action in branches}
        mass = {action: 0.0 for action in branches}
        steps = {action: 0 for action in branches}
        self._profile["nominal_row_steps"] += len(branches) * self.config.horizon
        first = True
        for _ in range(self.config.horizon):
            self._check_deadline()
            if not remaining:
                break
            next_remaining: dict[int, Any] = {}
            for action, branch in remaining.items():
                steps[action] += 1
                self._profile["active_row_steps"] += 1
                self._profile["resident_row_steps"] += 1
                started = time.perf_counter()
                if first:
                    self._first_step(branch, action)
                else:
                    self._policy_step(branch)
                self._profile["sim_s"] += time.perf_counter() - started
                if bool(branch.sim.alive[0, 0]):
                    alive_steps[action] += 1
                    mass[action] += float(branch.sim.length[0, 0])
                    next_remaining[action] = branch
            remaining = next_remaining
            first = False
        return {
            action: ActionOutcome(
                survives=action in remaining,
                alive_steps=alive_steps[action],
                mass=mass[action],
                row_steps=steps[action],
            )
            for action in branches
        }

    def _rollout_merged(
        self, worlds: list[Any], choices: list[list[int]]
    ) -> list[dict[int, ActionOutcome]]:
        """Batch all requested (source, action) rows with cloned CRN streams.

        T33's assembler supplies isolated simulator/runtime carry. Disposable
        source policy copies suppress its no-veto admission check only during
        assembly; the true frozen configurations and their carried hook state
        are then restored on private T34 policies before any transition.
        """
        from research.t33_batched_teacher.teacher import merge_worlds
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

        results: list[dict[int, ActionOutcome]] = [{} for _ in worlds]
        masks = []
        sources = []
        for world, requested in zip(worlds, choices):
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            mask[requested] = True
            masks.append(mask)
            source = self._clone(world)
            for policy in source.opponents:
                if hasattr(policy, "veto_slots"):
                    policy.veto_slots = frozenset()
            sources.append(source)
        count = sum(len(actions) for actions in choices)
        if not count:
            return results
        merged = merge_worlds(sources, 1, masks, horizon=self.config.horizon)
        # All six first-action rows exist only during audited T33 assembly.
        # VERIFY becomes precisely one physical row per source before stepping.
        merged, _ = self._compact_merged(merged, np.flatnonzero(merged.layout.active))
        seeds = [int(worlds[int(source)].seed) for source in merged.layout.source]
        cache: dict[tuple, Any] = {}
        branch_envs = {
            source_id: np.flatnonzero(merged.layout.source == source_id)
            for source_id in range(len(worlds))
        }
        roster = []
        for source_id in merged.layout.source:
            row = []
            for slot, policy in enumerate(worlds[int(source_id)].opponents, 1):
                if isinstance(policy, FastProfileAnchorSimdPolicy):
                    key = ("anchor", policy.kind)
                    if key not in cache:
                        cache[key] = FastProfileAnchorSimdPolicy(policy.kind, seeds)
                else:
                    # A source has one carried vector policy per slot.  Its
                    # candidate-action branches each receive an independent
                    # copy of that source's env-0 hook/counter carry.  Sharing
                    # across sources would make a v8 hook cross-contaminate
                    # rows even when its immutable checkpoint is identical.
                    key = _merged_policy_key(policy, int(source_id), slot)
                    if key not in cache:
                        cache[key] = _clone_merged_policy(
                            policy, merged.runtime, branch_envs[int(source_id)]
                        )
                row.append(cache[key])
            roster.append(row)
        merged.opponents = roster
        alive_steps = np.zeros((len(worlds), NUM_ACTIONS), dtype=np.int64)
        row_steps = np.zeros_like(alive_steps)
        mass = np.zeros_like(alive_steps, dtype=np.float64)
        survives = np.zeros_like(alive_steps, dtype=bool)
        self._profile["nominal_row_steps"] += count * self.config.horizon
        for tick in range(self.config.horizon):
            self._check_deadline()
            sim = merged.sim
            if not sim.E:
                break
            self._profile["active_row_steps"] += int(sim.E)
            self._profile["resident_row_steps"] += int(sim.E)
            self._profile["peak_active_rows"] = max(
                int(self._profile.get("peak_active_rows", 0)), int(sim.E)
            )
            grid = np.ones((sim.E, sim.S), dtype=np.int64)
            source_ids, action_ids = merged.layout.source, merged.layout.action
            row_steps[source_ids, action_ids] += 1

            def select(prepared: Any) -> np.ndarray:
                merged.runtime.prepare(prepared)
                self._dispatch_merged(merged, grid, np.ones(sim.E, dtype=bool))
                rows = np.column_stack((np.arange(sim.E), np.zeros(sim.E, dtype=np.int64)))
                q = self.policy.evaluate_rows(prepared, rows, worlds[0].hero.net)
                legal = prepared.get_legal_action_mask()[:, 0].astype(bool)
                advisory = prepared.get_advisory_action_mask()[:, 0].astype(bool)
                missing = ~advisory.any(axis=1)
                advisory[missing, :3] = legal[missing, :3]
                grid[:, 0] = np.where(advisory, q, -np.inf).argmax(axis=1)
                return grid

            started = time.perf_counter()
            if tick == 0:
                self._dispatch_merged(merged, grid, np.ones(sim.E, dtype=bool))
                grid[:, 0] = action_ids
                sim.step(grid)
            else:
                sim.step_with_policy(select)
            merged.runtime.observe_step(sim)
            self._profile["sim_s"] += time.perf_counter() - started
            live = sim.alive[:, 0].astype(bool)
            alive_steps[source_ids[live], action_ids[live]] += 1
            mass[source_ids[live], action_ids[live]] += sim.length[live, 0]
            survives[source_ids, action_ids] = live
            if not live.any():
                break
            if not live.all() and tick + 1 < self.config.horizon:
                self._profile["compactions"] = int(self._profile.get("compactions", 0)) + 1
                merged, _ = self._compact_merged(merged, np.flatnonzero(live))
        for index, requested in enumerate(choices):
            results[index] = {
                action: ActionOutcome(
                    bool(survives[index, action]),
                    int(alive_steps[index, action]),
                    float(mass[index, action]),
                    int(row_steps[index, action]),
                )
                for action in requested
            }
        return results

    @staticmethod
    def _compact_merged(merged: Any, keep: np.ndarray) -> tuple[Any, np.ndarray]:
        """Stable-gather every environment-leading mutable field after deaths.

        This is deliberately local to T34: T33 keeps a fixed slab for a CUDA
        graph, whereas ExIt needs active-row compaction for its cost contract.
        ``bind`` recreates geometry caches, then gathered carry state is put
        back so the next Vector61 decision remains an exact continuation.
        """
        from dataclasses import replace

        from research.t3_rollout_labels_20261015.engine import _restore_grid_views

        sim, runtime = merged.sim, merged.runtime
        old_e = int(sim.E)
        keep = np.asarray(keep, dtype=np.int64)

        def gather(owner: Any) -> dict[str, Any]:
            saved: dict[str, Any] = {}
            if owner is None:
                return saved
            for name, value in vars(owner).items():
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == old_e:
                    saved[name] = value[keep].copy()
                elif isinstance(value, list) and len(value) == old_e:
                    saved[name] = [value[int(index)] for index in keep]
            return saved

        sim_values = gather(sim)
        runtime_values = gather(runtime)
        memory_values = gather(getattr(runtime, "memory", None))
        controlled = np.asarray(runtime.controlled, dtype=bool)[keep].copy()
        for name, value in sim_values.items():
            setattr(sim, name, value)
        sim.E = len(keep)
        try:
            sim.cfg = replace(sim.cfg, num_envs=sim.E)
        except TypeError:
            pass
        sim._env_index = np.arange(sim.E, dtype=np.int64)[:, None]
        sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
        sim._det = None
        if hasattr(sim, "_active_env_mask"):
            del sim._active_env_mask
        _restore_grid_views(sim)
        runtime.bind(sim, controlled)
        for name, value in runtime_values.items():
            if hasattr(runtime, name):
                setattr(runtime, name, value)
        if getattr(runtime, "memory", None) is not None:
            for name, value in memory_values.items():
                if hasattr(runtime.memory, name):
                    setattr(runtime.memory, name, value)
        old_rows = merged.opponents
        merged.opponents = [old_rows[int(index)] for index in keep]
        unique = {id(policy): policy for row in merged.opponents for policy in row}
        for policy in unique.values():
            for name in ("veto_counters", "boost_counters", "live_vetoes"):
                values = getattr(policy, name, None)
                if isinstance(values, dict):
                    setattr(
                        policy,
                        name,
                        {
                            new_env: values[int(old_env)]
                            for new_env, old_env in enumerate(keep)
                            if int(old_env) in values
                        },
                    )
            if hasattr(policy, "_world_seeds"):
                policy._world_seeds = tuple(policy._world_seeds[int(index)] for index in keep)
        layout = merged.layout
        merged.layout = type(layout)(
            layout.source[keep].copy(),
            layout.action[keep].copy(),
            layout.rollout[keep].copy(),
            np.ones(len(keep), dtype=bool),
        )
        return merged, np.ones(len(keep), dtype=bool)

    def _dispatch_merged(self, merged: Any, grid: np.ndarray, active: np.ndarray) -> None:
        """Batch every frozen policy and retain Vector61's exact veto path."""
        del active
        self._opponents.dispatch(merged.sim, merged.runtime, merged.opponents, grid)

    def _check_deadline(self) -> None:
        progress = getattr(self, "progress", None)
        if progress is not None:
            progress()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("ExitTeacher deadline reached")

    def _first_step(self, world: Any, action: int) -> None:
        sim = world.sim
        if not getattr(sim, "_policy_phase_prepared", False):
            raise RuntimeError("teacher continuation requires a prepared Watch snapshot")
        sim.step(self.actions(world, action))
        world.runtime.observe_step(sim)

    def _policy_step(self, world: Any) -> None:
        def choose(prepared: Any) -> np.ndarray:
            world.runtime.prepare(prepared)
            evaluation = self.evaluate(world)
            return self.actions(world, evaluation.greedy_action())

        world.sim.step_with_policy(choose)
        world.runtime.observe_step(world.sim)

    @staticmethod
    def _select(
        evaluation: PolicyEvaluation,
        student: int,
        outcomes: dict[int, ActionOutcome],
    ) -> int:
        survivors = [action for action, outcome in outcomes.items() if outcome.survives]
        if survivors:
            best_q = max(float(evaluation.q[action]) for action in survivors)
            tied = [action for action in survivors if float(evaluation.q[action]) == best_q]
            return student if student in tied else min(tied)
        best_steps = max(outcome.alive_steps for outcome in outcomes.values())
        tied = [action for action, outcome in outcomes.items() if outcome.alive_steps == best_steps]
        return student if student in tied else min(tied)
